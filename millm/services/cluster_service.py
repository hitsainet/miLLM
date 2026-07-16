"""
Cluster import service (Feature 8).

Materializes `mistudio.cluster-definition/v1` documents as cluster-typed
steering profiles: validation and compatibility assessment at import,
lambda-scaled activation with a hard bounds gate, lossless re-export from
the stored definition. Bundle imports are per-item isolated.
"""

import uuid
from typing import Any

import structlog

from millm.api.schemas.cluster import (
    ClusterDefinitionV1,
    ClusterImportItem,
    ClusterImportResult,
    ClusterSummary,
    ClusterBundleV1,
)
from millm.core.errors import ProfileNotFoundError, ValidationError
from millm.core.steering_range import STEERING_RANGE, would_clamp
from millm.db.models.profile import Profile
from millm.db.repositories.profile_repository import ProfileRepository
from millm.services.profile_service import ProfileService
from millm.services.sae_service import AttachedSAEState, SAEService

logger = structlog.get_logger()

MAX_NAME_DEDUPE_ATTEMPTS = 50


class ClusterService:
    """Import / list / activate / export cluster definitions as profiles."""

    def __init__(
        self,
        profile_service: ProfileService,
        repository: ProfileRepository,
        sae_service: SAEService,
    ) -> None:
        self.profile_service = profile_service
        self.repository = repository
        self.sae_service = sae_service

    # ── Import ───────────────────────────────────────────────────────────

    async def import_definition(
        self,
        definition: ClusterDefinitionV1,
        *,
        on_conflict: str = "rename",
        hub_ref: dict[str, Any] | None = None,
        activate: bool = False,
    ) -> ClusterImportItem:
        """
        Import one definition as a cluster-typed profile.

        Compatibility is assessed against the attached SAE (warn-level at
        import; activation is the hard gate). Never raises for a merely
        incompatible definition — outcomes are reported per item.
        """
        sae_id, warnings = self._assess_compatibility(definition)
        warnings += self._range_warnings(definition)

        try:
            name = await self._dedupe_name(definition.name, on_conflict)
        except ValidationError as e:
            return ClusterImportItem(
                name=definition.name, status="error", error=str(e), warnings=warnings
            )

        # Steering stored at lambda=1 basis; the member's sign folds into the
        # stored value (definition strengths are magnitudes with a sign field).
        steering = {
            str(m.feature_idx): float(m.sign) * float(m.strength)
            for m in definition.members
        }
        meta = definition.model_dump(mode="json")
        meta["warnings"] = warnings
        if hub_ref:
            meta["hub_ref"] = hub_ref

        profile = await self.repository.create(
            profile_id=f"prof_{uuid.uuid4().hex[:12]}",
            name=name,
            description=definition.narrative,
            model_id=definition.model.hf_id or definition.model.mistudio_model_id,
            sae_id=sae_id,
            layer=definition.sae.layer,
            steering=steering,
            source_kind="cluster",
            cluster_meta=meta,
            intensity=definition.budget.intensity if definition.budget else 1.0,
        )

        status = "imported" if sae_id else "imported_unbound"
        logger.info(
            "cluster_imported",
            profile_id=profile.id,
            name=name,
            status=status,
            members=len(definition.members),
            warnings=len(warnings),
        )

        if activate and sae_id:
            try:
                await self.activate(profile.id)
            except Exception as e:  # activation failure must not undo the import
                warnings.append(f"Imported but activation failed: {e}")
                status = "imported"

        return ClusterImportItem(
            name=name, status=status, profile_id=profile.id, warnings=warnings
        )

    async def import_bundle(
        self, bundle: ClusterBundleV1, *, on_conflict: str = "rename"
    ) -> ClusterImportResult:
        """Import every definition in a bundle; one bad item never poisons the rest."""
        results: list[ClusterImportItem] = []
        for definition in bundle.definitions:
            try:
                results.append(
                    await self.import_definition(definition, on_conflict=on_conflict)
                )
            except Exception as e:
                logger.exception("cluster_bundle_item_failed", name=definition.name)
                results.append(
                    ClusterImportItem(name=definition.name, status="error", error=str(e))
                )
        return ClusterImportResult(
            results=results,
            imported=sum(r.status in ("imported", "imported_unbound") for r in results),
            blocked=sum(r.status == "blocked" for r in results),
            errors=sum(r.status == "error" for r in results),
        )

    # ── Listing / activation / intensity / export ────────────────────────

    async def list_clusters(self) -> list[ClusterSummary]:
        rows = [p for p in await self.repository.get_all() if p.source_kind == "cluster"]
        return [self._summarize(p) for p in rows]

    async def activate(self, profile_id: str) -> dict[str, Any]:
        """
        Activate a cluster profile with the hard compatibility gate.

        Gate (in order): row must be a cluster; an SAE must be attached; the
        definition's declared n_features (when present) must equal the attached
        SAE's d_sae; every member index must be within [0, d_sae). Unbound rows
        that pass the gate are bound to the attached SAE as a side effect.
        """
        profile = await self._get_cluster(profile_id)

        sae = AttachedSAEState().attached_sae
        if sae is None:
            # ProfileService raises SAENotAttachedError with the house message;
            # delegate so the error shape matches manual profiles.
            return await self.profile_service.activate_profile(profile_id)

        declared = ((profile.cluster_meta or {}).get("sae") or {}).get("n_features")
        if declared is not None and int(declared) != sae.d_sae:
            raise ValidationError(
                f"Cluster '{profile.name}' was authored for an SAE with "
                f"{declared} features; the attached SAE has {sae.d_sae}. "
                "Member indices would be meaningless — activation blocked.",
                details={"profile_id": profile_id, "declared_n_features": declared,
                         "attached_d_sae": sae.d_sae},
            )

        bad = [int(k) for k in (profile.steering or {}) if not 0 <= int(k) < sae.d_sae]
        if bad:
            raise ValidationError(
                f"Cluster '{profile.name}' references feature indices out of range "
                f"[0, {sae.d_sae}) for the attached SAE: {sorted(bad)[:8]} — "
                "activation blocked.",
                details={"profile_id": profile_id, "bad_indices": sorted(bad)[:20],
                         "attached_d_sae": sae.d_sae},
            )

        # Late binding: an unbound import that passes the gate binds to the
        # SAE it is now steering against (provenance keeps the original ref).
        attachment = self.sae_service.get_attachment_status()
        if profile.sae_id is None and attachment.sae_id:
            await self.repository.update(
                profile_id, sae_id=attachment.sae_id, layer=attachment.layer
            )

        return await self.profile_service.activate_profile(profile_id)

    async def deactivate(self, profile_id: str) -> dict[str, Any]:
        await self._get_cluster(profile_id)
        return await self.profile_service.deactivate_profile(profile_id)

    async def set_intensity(
        self, profile_id: str, intensity: float, *, reapply: bool = True
    ) -> dict[str, Any]:
        """Persist the lambda dial; re-apply steering when the cluster is active."""
        profile = await self._get_cluster(profile_id)
        await self.repository.update(profile_id, intensity=float(intensity))
        reapplied = False
        if reapply and profile.is_active:
            await self.activate(profile_id)
            reapplied = True
        return {"profile_id": profile_id, "intensity": float(intensity),
                "reapplied": reapplied}

    async def export_definition(self, profile_id: str) -> ClusterDefinitionV1:
        """Re-emit the lossless original definition from cluster_meta."""
        profile = await self._get_cluster(profile_id)
        meta = dict(profile.cluster_meta or {})
        meta.pop("warnings", None)
        meta.pop("hub_ref", None)
        return ClusterDefinitionV1.model_validate(meta)

    # ── Internals ────────────────────────────────────────────────────────

    async def _get_cluster(self, profile_id: str) -> Profile:
        profile = await self.repository.get(profile_id)
        if profile is None:
            raise ProfileNotFoundError(
                f"Cluster profile '{profile_id}' not found",
                details={"profile_id": profile_id},
            )
        if profile.source_kind != "cluster":
            raise ValidationError(
                f"Profile '{profile_id}' is not an imported cluster",
                details={"profile_id": profile_id, "source_kind": profile.source_kind},
            )
        return profile

    def _assess_compatibility(
        self, definition: ClusterDefinitionV1
    ) -> tuple[str | None, list[str]]:
        """
        bind (attached SAE, compatible) / warn-bind (model or layer differ) /
        unbound (no SAE attached, or declared feature space differs — the
        activation gate is the hard backstop either way).
        """
        warnings: list[str] = []
        state = AttachedSAEState()
        sae = state.attached_sae
        if sae is None:
            warnings.append("No SAE attached — imported unbound; bind by activating "
                            "once a compatible SAE is attached")
            return None, warnings

        ref = definition.sae
        if ref.n_features is not None and int(ref.n_features) != sae.d_sae:
            warnings.append(
                f"Feature-space mismatch: definition declares n_features="
                f"{ref.n_features}, attached SAE has {sae.d_sae} — imported "
                "unbound; activation will be blocked against this SAE"
            )
            return None, warnings

        if ref.layer is not None and state.attached_layer is not None \
                and int(ref.layer) != int(state.attached_layer):
            warnings.append(
                f"Layer mismatch: definition targets L{ref.layer}, attached SAE "
                f"is on L{state.attached_layer}"
            )
        return state.attached_sae_id, warnings

    def _range_warnings(self, definition: ClusterDefinitionV1) -> list[str]:
        rng = definition.budget.intensity_range if definition.budget else None
        lam_max = float(rng[1]) if rng and len(rng) == 2 else 2.0
        hot = [m.feature_idx for m in definition.members
               if would_clamp(float(m.sign) * float(m.strength) * lam_max)]
        if not hot:
            return []
        return [
            f"Members {sorted(hot)} exceed ±{STEERING_RANGE:g} at λ_max={lam_max:g}; "
            "effective values clamp at apply time"
        ]

    async def _dedupe_name(self, name: str, on_conflict: str) -> str:
        if not await self.repository.name_exists(name):
            return name
        if on_conflict == "fail":
            raise ValidationError(
                f"A profile named '{name}' already exists",
                details={"name": name},
            )
        for n in range(2, MAX_NAME_DEDUPE_ATTEMPTS + 2):
            candidate = f"{name} ({n})"
            if not await self.repository.name_exists(candidate):
                return candidate
        raise ValidationError(
            f"Could not find a free name for '{name}' after "
            f"{MAX_NAME_DEDUPE_ATTEMPTS} attempts",
            details={"name": name},
        )

    def _summarize(self, profile: Profile) -> ClusterSummary:
        meta = profile.cluster_meta or {}
        return ClusterSummary(
            id=profile.id,
            name=profile.name,
            description=profile.description,
            model_id=profile.model_id,
            sae_id=profile.sae_id,
            layer=profile.layer,
            is_active=profile.is_active,
            intensity=profile.intensity,
            sensing_enabled=profile.sensing_enabled,
            member_count=len(meta.get("members", []) or profile.steering or {}),
            display_token=meta.get("display_token"),
            bound=profile.sae_id is not None,
            warnings=list(meta.get("warnings", [])),
            hub_ref=meta.get("hub_ref"),
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )
