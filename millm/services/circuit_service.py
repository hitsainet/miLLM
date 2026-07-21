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
    SAESetIncompleteError,
    UnvalidatedCircuitError,
    ValidationError,
)
from millm.db.models.circuit import Circuit
from millm.db.repositories.circuit_repository import CircuitRepository

from millm.ml.circuit_steering import CircuitSteeringEngine
from millm.services.sae_service import AttachedSAEState

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

        # Feature 15: arm edge sensing AFTER the serve and after co-tenant
        # release — arming earlier would arm against SAEs that a co-tenant
        # release is about to detach. Best-effort: an observation surface must
        # never fail an activation.
        self._arm_edge_sensing(circuit, definition, served_layers)

        # Refresh `serveable` too: it was a snapshot of attachment state at
        # IMPORT time, so a circuit that became fully bindable since then kept
        # reporting "not serveable" while actively serving — and was filtered
        # out of ?serveable=true queries.
        try:
            await self.repository.update(
                circuit.id,
                per_sae_warnings=verdicts,
                serveable=all_bound,
                provenance={
                    **(circuit.provenance or {}),
                    # Remember which cluster profile backs a slice serve so
                    # deactivate() can tear down the thing actually steering.
                    "slice_profile_id": result.get("slice_profile_id"),
                },
            )
            await self.repository.set_active(
                circuit.id, serving_mode=result["serving_mode"]
            )
        except Exception:
            # The steering is already applied; if we cannot record it, undo it
            # rather than leave the model steering with no active row to stop it.
            # For a slice serve the steering belongs to a CLUSTER profile, so
            # clearing circuit steering alone would not undo it.
            try:
                slice_profile_id = result.get("slice_profile_id")
                if slice_profile_id and self._cluster_service is not None:
                    await self._cluster_service.deactivate(slice_profile_id)
                if self._sae_service is not None:
                    self._sae_service.clear_circuit_steering()
            except Exception:  # pragma: no cover - defensive
                logger.error("circuit_activate_rollback_clear_failed", circuit_id=circuit.id)
            raise
        refreshed = await self.repository.get(circuit.id)
        result["warnings"] = co_tenant_warnings + list(result.get("warnings") or [])
        # Feature 16: activation is an authoritative steering write
        AttachedSAEState().bump_steering_epoch('circuit_activate')
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
        # F18: ONE derivation. The MEMBERS and the INTENSITY come from the
        # plan, so this path and the dial cannot resolve either differently
        # (F14-R1-01 was exactly that divergence).
        #
        # R1-07: an earlier version of this comment also claimed the response's
        # `bound_layers` could no longer drift from the apply. It does not:
        # that field still reports `definition.layers()`, deliberately. The
        # FTASKS is explicit that any response-shape delta is a defect, not a
        # feature, and `bound_layers` is a CONTRACT field describing the
        # document's declared layers — not the claim set.
        #
        # The claim-set identity is real and is what the DIAL relies on for its
        # snapshot (F14-R2-01); it simply is not what this response field
        # reports. The comment asserted a guarantee the code below did not make,
        # which would have convinced the next reader that a drift was
        # impossible here.
        plan = CircuitSteeringEngine(self._sae_service._sae_state).plan_for(
            definition, circuit
        )
        members = plan.members
        edges = [e.model_dump(mode="json") for e in definition.edges]
        intensity = plan.intensity
        # R3 finding 2: `activate` bumps for the whole logical action, so this
        # write must NOT bump as well — one activation advanced the epoch by 2,
        # and any request whose snapshot landed BETWEEN the two saw a spurious
        # mismatch, skipped its restore and stranded its transient λ in global
        # state permanently. Identical to the defect R1 fixed for
        # `set_intensity` and did not apply here.
        outcome = self._sae_service.set_circuit_steering(
            members, intensity, edges=edges, authoritative=False
        )
        return {
            "serving_mode": "full",
            "bound_layers": definition.layers(),
            "applied_per_layer": outcome.applied_per_layer,
            # R3 finding 7: carry the epoch OUR write produced rather than
            # discarding it — without this the activate path has nothing to
            # compare and cannot report supersession at all.
            "applied_epoch": outcome.applied_epoch,
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
        slice_doc = self.to_layer_slice(
            definition,
            layer,
            circuit_rung_value=circuit.rung,
            fallback_intensity=circuit.intensity,
        )
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

        # ClusterService REPORTS its outcome, it does not raise: an incompatible
        # slice comes back as `imported_unbound` (activation explicitly skipped)
        # or `error`. Treating that as a successful serve would mark the circuit
        # active while the model runs completely unsteered — exactly the failure
        # the cluster path itself fixed in its own review. Fail closed.
        status = getattr(item, "status", None)
        profile_id = getattr(item, "profile_id", None)
        item_warnings = list(getattr(item, "warnings", []) or [])
        # A successful IMPORT is not a successful ACTIVATION: ClusterService
        # keeps status='imported' when activation itself raised, recording the
        # failure only as a warning. Checking the status alone would let the
        # circuit claim to serve while the model runs unsteered.
        activation_failed = any(
            "activation failed" in w.lower() or "activation requested but skipped" in w.lower()
            for w in item_warnings
        )
        if status != "imported" or not profile_id or activation_failed:
            raise SAESetIncompleteError(
                [
                    {
                        "layer": layer,
                        "sae_id": None,
                        "reason": (
                            "slice_activation_failed"
                            if activation_failed
                            else f"slice_import_{status or 'failed'}"
                        ),
                        "detail": getattr(item, "error", None) or "; ".join(item_warnings),
                    }
                ]
            )
        return {
            "serving_mode": "slice_fallback",
            "bound_layers": bound_layers,
            "slice_layer": layer,
            "slice_profile_id": profile_id,
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
        fallback_intensity: float | None = None,
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
        if definition.budget is not None:
            per_layer = definition.budget.layers.get(str(layer))
            budget = per_layer.model_dump(mode="json") if per_layer else {}
            budget["intensity"] = definition.budget.intensity
            budget["intensity_range"] = definition.budget.intensity_range
        elif fallback_intensity is not None:
            # No budget block at all: _serve_full would use circuit.intensity,
            # so the slice must too — otherwise the two modes silently disagree
            # on λ (the slice would take the cluster default of 1.0).
            budget = {"intensity": float(fallback_intensity)}

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

    # F18: `_serving_members` was DELETED here. Its rules now live in
    # `CircuitSteeringEngine.serving_members` — the ONE derivation — and every
    # call site consumes a `ServingPlan` rather than re-deriving. No shim: a
    # forwarding stub would leave two names for one thing and invite the next
    # caller to pick the wrong one, which is how four derivations happened.

    async def deactivate(self, circuit_id: str) -> dict[str, Any]:
        """Stop serving a circuit and clear whatever is ACTUALLY steering.

        In ``slice_fallback`` mode the live steering belongs to the cluster
        profile the slice created, not to circuit steering — clearing only the
        latter would report success while the slice kept running.
        """
        circuit = await self.get(circuit_id)
        cleared = False

        if circuit.serving_mode == "slice_fallback":
            slice_profile_id = (circuit.provenance or {}).get("slice_profile_id")
            if slice_profile_id and self._cluster_service is not None:
                try:
                    await self._cluster_service.deactivate(slice_profile_id)
                    cleared = True
                except Exception as e:
                    logger.warning(
                        "circuit_slice_profile_deactivate_failed",
                        profile_id=slice_profile_id,
                        error=str(e),
                    )
        if self._sae_service is not None:
            try:
                # R3 finding 3: `deactivate` bumps for the whole logical
                # action; bumping here too advanced the epoch by 2 for one
                # deactivation, stranding any restore snapshotted between them.
                self._sae_service.clear_circuit_steering(authoritative=False)
                cleared = True
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("circuit_clear_steering_failed", error=str(e))

        self._disarm_edge_sensing()

        await self.repository.deactivate(circuit.id)
        refreshed = await self.repository.get(circuit.id)
        # Feature 16: deactivation is authoritative
        AttachedSAEState().bump_steering_epoch('circuit_deactivate')
        return {**self.summarize(refreshed), "cleared_steering": cleared}

    @staticmethod
    def _edge_sensing_layer_saes(layers: list[int]) -> dict:
        """layer -> LoadedSAE for the served layers, resolved once.

        ``by_layer`` returns None for an ambiguous layer (zero or more than
        one SAE attached), so an edge on that layer is reported unsensable
        rather than watched against a guessed basis.
        """
        from millm.services.sae_service import AttachedSAEState

        state = AttachedSAEState()
        out: dict = {}
        for layer in layers:
            entry = state.by_layer(layer)
            if entry is not None:
                out[layer] = entry.sae
        return out

    def _arm_edge_sensing(self, circuit, definition, served_layers: list[int]) -> None:
        """Arm edge sensing when the operator has enabled it for this circuit."""
        if not getattr(circuit, "sensing_enabled", False):
            return
        try:
            from millm.api.dependencies import get_circuit_sensing_service

            layer_saes = self._edge_sensing_layer_saes(served_layers)
            if not layer_saes:
                return
            unsensable = get_circuit_sensing_service().arm_for_circuit(
                circuit, definition, layer_saes
            )
            if unsensable:
                logger.info(
                    "circuit_edge_sensing_partial",
                    circuit_id=circuit.id,
                    unsensable=len(unsensable),
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "circuit_edge_sensing_arm_failed",
                circuit_id=getattr(circuit, "id", None),
                error=str(e),
            )

    def _disarm_edge_sensing(self) -> None:
        try:
            import millm.api.dependencies as deps

            service = deps._circuit_sensing_service
            if service is None or not service.is_armed:
                return
            from millm.services.sae_service import AttachedSAEState

            state = AttachedSAEState()
            layer_saes = {e.layer: e.sae for e in state.entries()}
            service.disarm(layer_saes)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("circuit_edge_sensing_disarm_failed", error=str(e))

    async def set_intensity(
        self,
        circuit_id: str,
        intensity: float,
        *,
        acknowledge_unvalidated: bool = False,
    ) -> dict[str, Any]:
        """Set the global λ for a circuit, re-applying if it is serving.

        The stored document is parsed BEFORE the DB write: a corrupt document
        must not leave the persisted intensity changed while the model keeps
        steering at the old λ (a silent DB/GPU divergence).
        """
        circuit = await self.get(circuit_id)
        serving_full = circuit.is_active and circuit.serving_mode == "full"

        # The evidence gate must hold by CONSTRUCTION, not merely because
        # activation checked it once: re-applying steering here is a fresh arm
        # of the circuit (e.g. after a restart left a stale active row), so an
        # unvalidated circuit must not reach the model without an ack.
        if serving_full and not is_validated(circuit.rung) and not acknowledge_unvalidated:
            raise UnvalidatedCircuitError(
                f"Circuit '{circuit.name}' is {rung_language(circuit.rung)} "
                f"(rung {circuit.rung}), not causally validated. Re-send with "
                f"acknowledge_unvalidated=true to re-apply its steering.",
                details={
                    "circuit_id": circuit.id,
                    "rung": circuit.rung,
                    "rung_language": rung_language(circuit.rung),
                    "next_step": rung_next_step(circuit.rung),
                },
            )

        definition = self._parse_stored(circuit) if serving_full else None

        await self.repository.update(circuit.id, intensity=float(intensity))
        refreshed = await self.repository.get(circuit.id)

        reapplied = False
        warnings: list[str] = []
        if serving_full and definition is not None:
            # R2: the DB write above already committed the new intensity, so a
            # raise here leaves persisted λ diverging from live steering — the
            # very divergence the parse-before-write ordering exists to prevent.
            # Report it rather than letting the exception imply nothing landed.
            try:
                _outcome = self._sae_service.set_circuit_steering(
                    CircuitSteeringEngine.serving_members(definition),
                    float(intensity),
                    edges=[e.model_dump(mode="json") for e in definition.edges],
                )
                reapplied = True
                # R2 finding 12: take the epoch OUR write produced from the
                # OUTCOME. Re-reading `steering_epoch` afterwards names
                # whoever wrote LAST, so a second operator landing in that gap
                # made `still_current` compare equal and report success for a
                # value already superseded. A stub outcome without the field
                # degrades to a live read (the weaker pre-R2 behaviour).
                _epoch = getattr(_outcome, "applied_epoch", None)
                # Must be an int: a stub/mock outcome yields a truthy sentinel
                # that would never equal the real counter, silently reporting
                # every clean write as superseded. Fall back to a live read
                # (the weaker pre-R2 behaviour) rather than trusting it.
                applied_epoch = (
                    _epoch if isinstance(_epoch, int) and not isinstance(_epoch, bool)
                    else AttachedSAEState().steering_epoch
                )
            except Exception as exc:
                logger.warning(
                    "circuit_set_intensity_apply_failed",
                    circuit_id=circuit.id, error=str(exc),
                )
                warnings.append(
                    f"The intensity was recorded but could not be applied to "
                    f"the model: {exc}. Persisted and live steering now differ."
                )
        elif refreshed.is_active and refreshed.serving_mode == "slice_fallback":
            # The slice is served by a cluster profile that owns its own λ, so
            # this dial did NOT reach the model. Say so rather than reporting a
            # new intensity the steering never received.
            warnings.append(
                "This circuit is serving a per-layer slice; the slice's cluster "
                "profile keeps its own intensity, so this dial was recorded but "
                "not applied. Adjust the slice cluster's intensity instead."
            )
        # Feature 16: an intensity change is authoritative. R1: when the
        # steering call above ran it ALREADY bumped, so bumping again made one
        # operator action advance the epoch by 2 and any snapshot taken between
        # the two saw a spurious mismatch. Bump here only when nothing else
        # did (the slice-fallback and no-op branches), and either way capture
        # the epoch OUR action produced so the check below tests whether
        # something landed AFTER us rather than reporting our own bump.
        if not reapplied:
            # Nothing above bumped (slice-fallback / no-op), so this action is
            # the authoritative write and owns the bump.
            applied_epoch = AttachedSAEState().bump_steering_epoch(
                "circuit_set_intensity"
            )
        # `reapplied: true` used to be unconditional whenever the steering call
        # was made — an affirmative claim that survived an in-flight request
        # restoring the pre-request snapshot over it moments later. It now
        # means what a caller reads it to mean: the value is live.
        # R2: the epoch GUARD is what makes this truthful, not a ledger.
        #
        # R1 added a "revert ledger" so this could detect a restore having
        # overwritten us. That mechanism was structurally incapable of working:
        # the restore only recorded when saved == current (i.e. when NOTHING
        # bumped), while applied_epoch here is always post-bump — so the two
        # conditions were mutually exclusive by construction, and the ledger
        # also fired FALSE POSITIVES on ordinary idle traffic.
        #
        # It was also unnecessary. Once the guard works, an in-flight restore
        # CANNOT revert us: our bump advances the epoch, the restore sees the
        # mismatch, and it skips. The only way this write stops being live is
        # another AUTHORITATIVE write landing after ours — which is exactly
        # what the epoch comparison below detects.
        still_current = AttachedSAEState().steering_epoch == applied_epoch
        if reapplied and not still_current:
            warnings.append(
                "The intensity was applied but immediately superseded by "
                "another authoritative steering change; it is not live."
            )
        return {
            **self.summarize(refreshed),
            "reapplied": reapplied and still_current,
            "superseded": bool(reapplied and not still_current),
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
