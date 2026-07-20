"""
Circuit service (Feature 13): import, per-SAE compatibility, activation with
the evidence gate, slice-fallback, and lossless export.

Division of labour:
  * THIS service owns the circuit artifact — validation, storage, the
    per-referenced-SAE compatibility matrix, the rung gate, and the choice
    between full multi-SAE serving and the per-layer cluster-slice fallback.
  * Feature 12 (``SAEService.set_circuit_steering``) owns the actual serving
    math once a circuit is deemed fully serveable.
  * Feature 8 (``ClusterService.import_definition``) owns the slice path — a
    per-layer slice IS an ordinary ``cluster-definition/v1``, so the fallback
    reuses that code unchanged rather than forking it.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import ValidationError as PydanticValidationError

from millm.api.schemas.circuit import (
    MAX_CIRCUIT_IMPORT_BYTES,
    CircuitDefinitionV1,
    CircuitMember,
)
from millm.api.schemas.cluster import MAX_NAME as CLUSTER_MAX_NAME
from millm.core.circuit_evidence import (
    EvidenceRung,
    circuit_rung,
    is_validated,
    rung_language,
    rung_next_step,
)
from millm.core.errors import (
    CircuitNotFoundError,
    UnvalidatedCircuitError,
    ValidationError,
)
from millm.db.models.circuit import Circuit
from millm.db.repositories.circuit_repository import CircuitRepository

logger = structlog.get_logger()

MAX_NAME_DEDUPE_ATTEMPTS = 50

#: Per-referenced-SAE compatibility verdicts (mirrors the cluster matrix).
VERDICT_BIND = "bind"
VERDICT_WARN = "warn"
VERDICT_BLOCK = "block"
VERDICT_UNBOUND = "unbound"


class CircuitService:
    """Import, activate and export ``mistudio.circuit-definition/v1`` documents."""

    def __init__(
        self,
        repository: CircuitRepository,
        sae_service: Any = None,
        cluster_service: Any = None,
    ) -> None:
        self.repository = repository
        self._sae_service = sae_service
        self._cluster_service = cluster_service

    # ── Import ─────────────────────────────────────────────────────────────

    async def import_definition(
        self,
        payload: dict[str, Any],
        *,
        raw_bytes: int | None = None,
        on_conflict: str = "rename",
    ) -> Circuit:
        """Validate and store a circuit definition.

        The RAW document is stored verbatim (``circuit_meta``) so re-export is
        lossless — pydantic's parsed form would drop additive producer fields.

        Raises:
            ValidationError: payload too large, unknown kind, bad schema.
        """
        if raw_bytes is not None and raw_bytes > MAX_CIRCUIT_IMPORT_BYTES:
            raise ValidationError(
                f"Circuit definition exceeds the {MAX_CIRCUIT_IMPORT_BYTES} byte cap",
                details={"bytes": raw_bytes, "max_bytes": MAX_CIRCUIT_IMPORT_BYTES},
            )

        kind = (payload or {}).get("kind")
        if kind != "mistudio.circuit-definition":
            raise ValidationError(
                f"Unknown kind {kind!r} — expected 'mistudio.circuit-definition'",
                details={"kind": kind, "code": "UNKNOWN_KIND"},
            )

        definition = CircuitDefinitionV1.model_validate(payload)

        verdicts = self.assess_compatibility(definition)
        serveable = bool(verdicts) and all(
            v["verdict"] in (VERDICT_BIND, VERDICT_WARN) for v in verdicts
        )

        edge_rungs = [e.rung for e in definition.edges]
        rung = int(circuit_rung(edge_rungs))
        name = await self._dedupe_name(definition.name, on_conflict)

        circuit = await self.repository.create(
            id=f"circ_{uuid.uuid4().hex[:12]}",
            name=name,
            description=definition.narrative,
            circuit_meta=payload,  # RAW document — lossless re-export
            rung=rung,
            edge_count=len(definition.edges),
            layers=definition.layers(),
            per_sae_warnings=verdicts,
            serveable=serveable,
            provenance={
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "origin": "file",
            },
        )
        logger.info(
            "circuit_imported",
            circuit_id=circuit.id,
            name=name,
            rung=rung,
            layers=circuit.layers,
            serveable=serveable,
        )
        return circuit

    def assess_compatibility(
        self, definition: CircuitDefinitionV1
    ) -> list[dict[str, Any]]:
        """Per-REFERENCED-SAE compatibility verdicts.

        Evaluated independently for every layer the circuit references (unlike
        the cluster matrix, which has a single SAE to judge). A circuit is
        fully serveable only when EVERY referenced layer binds; anything else
        degrades to the per-layer slice fallback rather than serving a member
        through a mismatched basis.

        Verdicts: ``bind`` (attached + compatible), ``warn`` (attached,
        non-fatal difference), ``block`` (attached but feature-space mismatch —
        indices would be meaningless), ``unbound`` (nothing attached there).
        """
        from millm.services.sae_service import AttachedSAEState

        state = AttachedSAEState()
        verdicts: list[dict[str, Any]] = []
        for layer in definition.layers():
            ref = definition.sae_for_layer(layer)
            entry = state.by_layer(layer)
            if entry is None:
                verdicts.append(
                    {
                        "layer": layer,
                        "sae_id": ref.mistudio_sae_id if ref else None,
                        "verdict": VERDICT_UNBOUND,
                        "reason": (
                            f"No unique SAE attached at L{layer} — this layer cannot "
                            "serve until one is attached"
                        ),
                    }
                )
                continue

            declared = ref.n_features if ref else None
            if declared is not None and int(declared) != int(entry.sae.d_sae):
                verdicts.append(
                    {
                        "layer": layer,
                        "sae_id": entry.sae_id,
                        "verdict": VERDICT_BLOCK,
                        "reason": (
                            f"Feature-space mismatch at L{layer}: definition declares "
                            f"n_features={declared}, attached SAE has {entry.sae.d_sae} "
                            "— member indices would be meaningless"
                        ),
                    }
                )
                continue

            entry_verdict: dict[str, Any] = {
                "layer": layer,
                "sae_id": entry.sae_id,
                "verdict": VERDICT_BIND,
            }
            if ref and ref.mistudio_sae_id and ref.mistudio_sae_id != entry.sae_id:
                entry_verdict["verdict"] = VERDICT_WARN
                entry_verdict["reason"] = (
                    f"L{layer} was authored against SAE '{ref.mistudio_sae_id}' but "
                    f"'{entry.sae_id}' is attached — a different feature basis"
                )
            verdicts.append(entry_verdict)
        return verdicts

    # ── Read ───────────────────────────────────────────────────────────────

    async def get(self, circuit_id: str) -> Circuit:
        circuit = await self.repository.get(circuit_id)
        if circuit is None:
            raise CircuitNotFoundError(
                f"Circuit '{circuit_id}' not found", details={"circuit_id": circuit_id}
            )
        return circuit

    async def list_circuits(
        self,
        *,
        min_rung: int | None = None,
        serveable: bool | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = await self.repository.get_all(
            min_rung=min_rung, serveable=serveable, limit=limit, offset=offset
        )
        return [self.summarize(r) for r in rows]

    async def get_active(self) -> dict[str, Any] | None:
        row = await self.repository.get_active()
        return self.summarize(row) if row else None

    def summarize(self, circuit: Circuit) -> dict[str, Any]:
        """List/detail row. Rung language is ALWAYS server-rendered from the
        ladder so no client can re-phrase an evidence claim."""
        return {
            "id": circuit.id,
            "name": circuit.name,
            "description": circuit.description,
            "rung": circuit.rung,
            "rung_language": rung_language(circuit.rung),
            "rung_next_step": rung_next_step(circuit.rung),
            "validated": is_validated(circuit.rung),
            "edge_count": circuit.edge_count,
            "layers": circuit.layers,
            "serveable": circuit.serveable,
            "is_active": circuit.is_active,
            "serving_mode": circuit.serving_mode,
            "intensity": circuit.intensity,
            "per_sae_warnings": circuit.per_sae_warnings or [],
            "created_at": circuit.created_at,
            "updated_at": circuit.updated_at,
        }

    # ── Activation ─────────────────────────────────────────────────────────

    async def activate(
        self, circuit_id: str, *, acknowledge_unvalidated: bool = False
    ) -> dict[str, Any]:
        """Serve a circuit.

        Gate order matters:
          1. **Evidence gate** — a circuit below rung 2 is not causally
             validated; activating it requires an explicit acknowledgement so
             nobody steers production traffic on a merely-associated circuit
             by accident.
          2. **SAE-set gate** — all referenced SAEs bound ⇒ full multi-SAE
             serving via Feature 12; otherwise degrade to the per-layer cluster
             slice (never a mismatched-basis serve).
        """
        circuit = await self.get(circuit_id)

        if not is_validated(circuit.rung) and not acknowledge_unvalidated:
            raise UnvalidatedCircuitError(
                f"Circuit '{circuit.name}' is {rung_language(circuit.rung)} "
                f"(rung {circuit.rung}), not causally validated. Re-send with "
                f"acknowledge_unvalidated=true to steer with it anyway.",
                details={
                    "circuit_id": circuit.id,
                    "rung": circuit.rung,
                    "rung_language": rung_language(circuit.rung),
                    "next_step": rung_next_step(circuit.rung),
                },
            )

        definition = self._parse_stored(circuit)
        verdicts = self.assess_compatibility(definition)
        bound_layers = [
            v["layer"] for v in verdicts if v["verdict"] in (VERDICT_BIND, VERDICT_WARN)
        ]
        all_bound = bool(verdicts) and len(bound_layers) == len(verdicts)

        # Co-tenancy guard (F12 R2/R3 finding): circuit serving CLEARS each
        # target layer's SAE before applying, so an active cluster/profile
        # steering one of those layers would be silently wiped while its row
        # still reported "active". Release it explicitly — but only for the
        # layers we ACTUALLY serve (a slice fallback touches one layer, not
        # every bindable one), and only AFTER the serve succeeds, so a failed
        # activation never leaves the user with nothing steering.
        served_layers = bound_layers if all_bound else bound_layers[:1]

        if all_bound:
            result = await self._serve_full(circuit, definition)
        else:
            result = await self._serve_slices(circuit, definition, bound_layers, verdicts)

        co_tenant_warnings = await self._release_co_tenants(served_layers)

        # Refresh `serveable` too: it was a snapshot of attachment state at
        # IMPORT time, so a circuit that became fully bindable since then kept
        # reporting "not serveable" while actively serving — and was filtered
        # out of ?serveable=true queries.
        try:
            await self.repository.update(
                circuit.id, per_sae_warnings=verdicts, serveable=all_bound
            )
            await self.repository.set_active(
                circuit.id, serving_mode=result["serving_mode"]
            )
        except Exception:
            # The steering is already applied; if we cannot record it, clear it
            # rather than leave the model steering with no active row to stop it.
            try:
                self._sae_service.clear_circuit_steering()
            except Exception:  # pragma: no cover - defensive
                logger.error("circuit_activate_rollback_clear_failed", circuit_id=circuit.id)
            raise
        refreshed = await self.repository.get(circuit.id)
        result["warnings"] = co_tenant_warnings + list(result.get("warnings") or [])
        return {
            **self.summarize(refreshed),
            **result,
            "acknowledged_unvalidated": bool(
                not is_validated(circuit.rung) and acknowledge_unvalidated
            ),
        }

    def _parse_stored(self, circuit: Circuit) -> CircuitDefinitionV1:
        """Re-validate a stored document, surfacing corruption structurally.

        ``circuit_meta`` is the RAW payload, validated at IMPORT time. If the
        contract tightened since, or the row was hand-edited/partially written,
        a bare ``model_validate`` would raise a pydantic error that is not a
        ``MiLLMError`` — landing as an opaque 500 with no circuit id and no
        recovery path. Convert it to a structured, actionable failure.
        """
        try:
            return CircuitDefinitionV1.model_validate(circuit.circuit_meta)
        except PydanticValidationError as e:
            raise ValidationError(
                f"Circuit '{circuit.name}' has a stored definition that no longer "
                f"validates against the v1 contract ({e.error_count()} error(s)) — "
                "re-import it from miStudio",
                details={
                    "circuit_id": circuit.id,
                    "errors": json.loads(e.json())[:5],
                },
            ) from e

    async def _release_co_tenants(self, target_layers: list[int]) -> list[str]:
        """Deactivate an active cluster/profile that steers a target layer.

        Circuit serving takes exclusive ownership of the layers it applies to
        (it clears each SAE before writing). Rather than silently clobbering a
        co-located cluster, deactivate it and return a user-visible warning.
        Best-effort: a lookup failure must not block activation.
        """
        if self._cluster_service is None or not target_layers:
            return []
        try:
            active = await self._cluster_service.get_active_cluster()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("circuit_co_tenant_lookup_failed", error=str(e))
            return []
        if active is None:
            return []

        active_layer = getattr(active, "layer", None)
        # Deactivate when the cluster steers one of our target layers, or when
        # its layer is unknown (we cannot prove it is safe to leave running).
        if active_layer is not None and int(active_layer) not in target_layers:
            return []
        try:
            await self._cluster_service.deactivate(active.id)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "circuit_co_tenant_deactivate_failed", profile_id=active.id, error=str(e)
            )
            return [
                f"Cluster '{getattr(active, 'name', active.id)}' also steers this "
                "layer and could not be deactivated — its steering may be overwritten"
            ]
        logger.info("circuit_co_tenant_deactivated", profile_id=active.id)
        return [
            f"Deactivated cluster '{getattr(active, 'name', active.id)}' — a circuit "
            "takes exclusive ownership of the layers it steers"
        ]

    async def _serve_full(
        self, circuit: Circuit, definition: CircuitDefinitionV1
    ) -> dict[str, Any]:
        """All referenced SAEs bound — delegate to Feature 12 serving."""
        members = self._serving_members(definition)
        edges = [e.model_dump(mode="json") for e in definition.edges]
        intensity = (
            definition.budget.intensity if definition.budget else circuit.intensity
        )
        outcome = self._sae_service.set_circuit_steering(
            members, intensity, edges=edges
        )
        return {
            "serving_mode": "full",
            "bound_layers": definition.layers(),
            "applied_per_layer": outcome.applied_per_layer,
            "hazards": outcome.hazards,
            "warnings": outcome.clamp_warnings,
        }

    async def _serve_slices(
        self,
        circuit: Circuit,
        definition: CircuitDefinitionV1,
        bound_layers: list[int],
        verdicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Incomplete SAE set — serve the per-layer cluster slice instead.

        A slice is an ordinary ``cluster-definition/v1``, so it goes through the
        UNCHANGED Feature 8 import path. The partial-rendering marker rides in
        display-only fields, so a slice can never be mistaken for the circuit.
        """
        if not bound_layers:
            from millm.core.errors import SAESetIncompleteError

            offenders = [
                {
                    "layer": v["layer"],
                    "sae_id": v.get("sae_id"),
                    "reason": v.get("verdict"),
                }
                for v in verdicts
            ]
            raise SAESetIncompleteError(offenders)

        # Serve the first bound layer's slice (single-active semantics: one
        # cluster profile steers at a time).
        layer = bound_layers[0]
        slice_doc = self.to_layer_slice(definition, layer, circuit_rung_value=circuit.rung)
        # ClusterService.import_definition takes a VALIDATED model (the raw
        # payload rides alongside for lossless storage) — passing the bare dict
        # crashed on `.name`. Validating here also means a malformed projection
        # surfaces as a structured error instead of an AttributeError.
        from millm.api.schemas.cluster import ClusterDefinitionV1

        try:
            slice_model = ClusterDefinitionV1.model_validate(slice_doc)
        except PydanticValidationError as e:
            raise ValidationError(
                f"Circuit '{circuit.name}' could not be projected to an L{layer} "
                f"cluster slice: {e.error_count()} contract error(s)",
                details={"layer": layer, "errors": json.loads(e.json())[:5]},
            ) from e
        item = await self._cluster_service.import_definition(
            slice_model,
            raw_payload=slice_doc,
            activate=True,
            on_conflict="rename",
        )
        return {
            "serving_mode": "slice_fallback",
            "bound_layers": bound_layers,
            "slice_layer": layer,
            "slice_profile_id": getattr(item, "profile_id", None),
            "warnings": [
                f"Only L{sorted(bound_layers)} of {definition.layers()} bound — serving "
                f"the L{layer} slice, a PARTIAL rendering of this circuit, not the "
                "whole circuit"
            ],
        }

    def to_layer_slice(
        self,
        definition: CircuitDefinitionV1,
        layer: int,
        *,
        circuit_rung_value: int | None = None,
    ) -> dict[str, Any]:
        """Project ONE layer of a circuit as a valid ``cluster-definition/v1``.

        Mirrors miStudio's ``to_layer_slice``: the partial-rendering marker
        travels ONLY in display fields (name suffix + ``provenance.source_note``)
        so the result is schema-identical to an ordinary cluster definition and
        the Feature 8 importer consumes it unchanged.
        """
        ref = definition.sae_for_layer(layer)
        members: list[dict[str, Any]] = []
        seen_idx: set[int] = set()
        for m in definition.members:
            if m.layer != layer:
                continue
            # A cluster_ref contributes its frozen expansion AND (if present)
            # its own feature — taking only one would silently drop the other.
            sources = list(m.expanded_members or [])
            if m.feature is not None:
                sources.append(m.feature)
            for feat in sources:
                # Dedupe: the same feature may appear standalone and inside a
                # referenced cluster; a cluster definition is keyed by index, so
                # a duplicate would last-write-win downstream.
                if feat.feature_idx in seen_idx:
                    continue
                seen_idx.add(feat.feature_idx)
                members.append(feat.model_dump(mode="json"))
        if not members:
            raise ValidationError(
                f"Circuit has no serveable members on L{layer}",
                details={"layer": layer},
            )

        # Always carry the circuit's GLOBAL intensity onto the slice, even when
        # the document declares no per-layer budget entry — dropping it would
        # silently serve the slice at the cluster default (λ=1.0) instead of the
        # authored λ.
        budget = None
        if definition.budget:
            per_layer = definition.budget.layers.get(str(layer))
            budget = per_layer.model_dump(mode="json") if per_layer else {}
            budget["intensity"] = definition.budget.intensity
            budget["intensity_range"] = definition.budget.intensity_range

        rung_value = (
            circuit_rung_value
            if circuit_rung_value is not None
            else int(circuit_rung([e.rung for e in definition.edges]))
        )
        note = (
            f"projection_of='{definition.name}' circuit; parent_rung={rung_value}; "
            "partial_rendering=true — a slice is NOT the circuit"
        )[:500]

        # The cluster contract caps names at MAX_NAME (120) while a circuit name
        # may be 200 — truncate the base so the projection always validates
        # (a long name must not make the fallback impossible), leaving room for
        # the marker suffix and any " (2)" dedupe suffix.
        suffix = f" [L{layer} slice]"
        headroom = CLUSTER_MAX_NAME - len(suffix) - 6
        base = definition.name if len(definition.name) <= headroom else (
            definition.name[: max(headroom, 1)]
        )

        return {
            "kind": "mistudio.cluster-definition",
            "schema_version": "1",
            "name": f"{base}{suffix}",
            "narrative": definition.narrative,
            "model": definition.model.model_dump(mode="json"),
            "sae": ref.model_dump(mode="json") if ref else {"layer": layer},
            "members": members,
            "budget": budget,
            "provenance": {
                **definition.provenance.model_dump(mode="json"),
                "source_note": note,
            },
        }

    def _serving_members(self, definition: CircuitDefinitionV1) -> list[CircuitMember]:
        """Flatten the circuit's members into the Feature 12 serving shape.

        A ``cluster_ref`` contributes its frozen ``expanded_members`` AND its own
        ``feature`` when both are present — taking only one silently dropped
        authored members from the intervention. Duplicates on a
        ``(layer, feature_idx)`` are collapsed because the serving path rejects
        a repeated key outright.
        """
        out: list[CircuitMember] = []
        seen: set[tuple[int, int]] = set()
        for m in definition.members:
            ref = definition.sae_for_layer(m.layer)
            sae_id = ref.mistudio_sae_id if ref else None
            sources = list(m.expanded_members or [])
            if m.feature is not None:
                sources.append(m.feature)
            for feat in sources:
                key = (m.layer, feat.feature_idx)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    CircuitMember(
                        feature_idx=feat.feature_idx,
                        layer=m.layer,
                        budget=feat.strength,
                        sign=feat.sign,
                        sae_id=sae_id,
                        label=feat.label,
                    )
                )
        return out

    async def deactivate(self, circuit_id: str) -> dict[str, Any]:
        """Stop serving a circuit and clear its steering."""
        circuit = await self.get(circuit_id)
        if self._sae_service is not None:
            try:
                self._sae_service.clear_circuit_steering()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("circuit_clear_steering_failed", error=str(e))
        await self.repository.deactivate(circuit.id)
        refreshed = await self.repository.get(circuit.id)
        return {**self.summarize(refreshed), "cleared_steering": True}

    async def set_intensity(self, circuit_id: str, intensity: float) -> dict[str, Any]:
        """Set the global λ for a circuit, re-applying if it is serving.

        The stored document is parsed BEFORE the DB write: a corrupt document
        must not leave the persisted intensity changed while the model keeps
        steering at the old λ (a silent DB/GPU divergence).
        """
        circuit = await self.get(circuit_id)
        serving_full = circuit.is_active and circuit.serving_mode == "full"
        definition = self._parse_stored(circuit) if serving_full else None

        await self.repository.update(circuit.id, intensity=float(intensity))
        refreshed = await self.repository.get(circuit.id)

        reapplied = False
        warnings: list[str] = []
        if serving_full and definition is not None:
            self._sae_service.set_circuit_steering(
                self._serving_members(definition),
                float(intensity),
                edges=[e.model_dump(mode="json") for e in definition.edges],
            )
            reapplied = True
        elif refreshed.is_active and refreshed.serving_mode == "slice_fallback":
            # The slice is served by a cluster profile that owns its own λ, so
            # this dial did NOT reach the model. Say so rather than reporting a
            # new intensity the steering never received.
            warnings.append(
                "This circuit is serving a per-layer slice; the slice's cluster "
                "profile keeps its own intensity, so this dial was recorded but "
                "not applied. Adjust the slice cluster's intensity instead."
            )
        return {
            **self.summarize(refreshed),
            "reapplied": reapplied,
            "warnings": warnings,
        }

    async def delete(self, circuit_id: str) -> dict[str, Any]:
        circuit = await self.get(circuit_id)
        if circuit.is_active:
            await self.deactivate(circuit.id)
        await self.repository.delete(circuit.id)
        return {"circuit_id": circuit_id, "deleted": True}

    # ── Export ─────────────────────────────────────────────────────────────

    async def export_definition(self, circuit_id: str) -> dict[str, Any]:
        """Re-export the circuit LOSSLESSLY (the raw stored document)."""
        circuit = await self.get(circuit_id)
        return circuit.circuit_meta

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _dedupe_name(self, name: str, on_conflict: str) -> str:
        existing = await self.repository.get_by_name(name)
        if existing is None:
            return name
        if on_conflict == "fail":
            raise ValidationError(
                f"A circuit named '{name}' already exists", details={"name": name}
            )
        for n in range(2, MAX_NAME_DEDUPE_ATTEMPTS + 2):
            candidate = f"{name} ({n})"
            if await self.repository.get_by_name(candidate) is None:
                return candidate
        raise ValidationError(
            f"Could not find a free name for '{name}' after "
            f"{MAX_NAME_DEDUPE_ATTEMPTS} attempts",
            details={"name": name},
        )
