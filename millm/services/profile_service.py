"""
Profile service for managing steering configuration profiles.

This service handles creating, loading, saving, and applying steering profiles
that persist configuration across sessions.
"""

import uuid
from typing import Any, Optional

import structlog

from millm.core.errors import (
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    SAENotAttachedError,
    ValidationError,
)
from millm.core.steering_range import clamp_steering, would_clamp
from millm.db.models.profile import Profile
from millm.db.repositories.profile_repository import ProfileRepository
from millm.services.sae_service import AttachedSAEState, SAEService

logger = structlog.get_logger()


class ProfileService:
    """
    Service for managing steering configuration profiles.

    Coordinates between the profile repository and SAE service to manage
    persistent steering configurations.
    """

    def __init__(
        self,
        repository: ProfileRepository,
        sae_service: SAEService,
    ) -> None:
        """
        Initialize the profile service.

        Args:
            repository: Profile database repository.
            sae_service: SAE service for applying steering.
        """
        self.repository = repository
        self.sae_service = sae_service

        logger.debug("ProfileService initialized")

    # =========================================================================
    # Listing Methods
    # =========================================================================

    async def list_profiles(self) -> list[Profile]:
        """
        Get all profiles from the database.

        Returns:
            List of all profiles ordered by name.
        """
        return await self.repository.get_all()

    async def get_profile(self, profile_id: str) -> Profile:
        """
        Get a single profile by ID.

        Args:
            profile_id: The profile's database ID.

        Returns:
            The Profile if found.

        Raises:
            ProfileNotFoundError: If profile doesn't exist.
        """
        profile = await self.repository.get(profile_id)
        if not profile:
            raise ProfileNotFoundError(
                f"Profile with ID '{profile_id}' not found",
                details={"profile_id": profile_id},
            )
        return profile

    async def get_active_profile(self) -> Optional[Profile]:
        """
        Get the currently active profile.

        Returns:
            The active Profile or None if no profile is active.
        """
        return await self.repository.get_active()

    # =========================================================================
    # Create Methods
    # =========================================================================

    async def create_profile(
        self,
        name: str,
        description: Optional[str] = None,
        steering: Optional[dict[int, float]] = None,
        model_id: Optional[str] = None,
        sae_id: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> Profile:
        """
        Create a new profile.

        Args:
            name: Unique display name for the profile.
            description: Optional description of the profile's purpose.
            steering: Dict mapping feature indices to steering values.
            model_id: Optional model identifier the profile was designed for.
            sae_id: Optional SAE ID the steering is for.
            layer: Optional layer the SAE targets.

        Returns:
            The created Profile instance.

        Raises:
            ProfileAlreadyExistsError: If a profile with this name already exists.
        """
        # Check for duplicate name
        if await self.repository.name_exists(name):
            raise ProfileAlreadyExistsError(
                f"Profile with name '{name}' already exists",
                details={"name": name},
            )

        # Generate unique ID
        profile_id = f"prof_{uuid.uuid4().hex[:12]}"

        # Convert steering keys to strings for JSONB storage
        steering_dict: dict[str, Any] = {}
        if steering:
            steering_dict = {str(k): v for k, v in steering.items()}

        profile = await self.repository.create(
            profile_id=profile_id,
            name=name,
            description=description,
            model_id=model_id,
            sae_id=sae_id,
            layer=layer,
            steering=steering_dict,
        )

        logger.info(
            "profile_created",
            profile_id=profile_id,
            name=name,
            feature_count=len(steering_dict),
        )

        return profile

    async def save_current_steering(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Profile:
        """
        Save current SAE steering configuration as a new profile.

        Args:
            name: Name for the new profile.
            description: Optional description.

        Returns:
            The created Profile instance.

        Raises:
            ProfileAlreadyExistsError: If a profile with this name exists.
            SAENotAttachedError: If no SAE is attached.
        """
        # Get current attachment status
        attachment = self.sae_service.get_attachment_status()
        if not attachment.is_attached:
            raise SAENotAttachedError(
                "Cannot save steering: no SAE is attached",
            )

        # Get current steering values
        steering = self.sae_service.get_steering_values()

        # Create profile with current state
        return await self.create_profile(
            name=name,
            description=description,
            steering=steering,
            sae_id=attachment.sae_id,
            layer=attachment.layer,
        )

    # =========================================================================
    # Update Methods
    # =========================================================================

    async def update_profile(
        self,
        profile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        steering: Optional[dict[int, float]] = None,
        model_id: Optional[str] = None,
        sae_id: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> Profile:
        """
        Update an existing profile.

        Args:
            profile_id: The profile's ID.
            name: New name (optional).
            description: New description (optional).
            steering: New steering values (optional, replaces existing).
            model_id: New model identifier (optional).
            sae_id: New SAE ID (optional).
            layer: New layer (optional).

        Returns:
            The updated Profile instance.

        Raises:
            ProfileNotFoundError: If profile doesn't exist.
            ProfileAlreadyExistsError: If new name conflicts with existing.
        """
        # Verify profile exists
        profile = await self.get_profile(profile_id)

        # Cluster rows carry lambda=1-basis steering + a lossless stored
        # definition; free-form steering edits here would silently double-
        # scale on activation and diverge from the exported artifact
        # (review find). Clusters change via re-import or the Clusters page.
        if steering is not None and profile.source_kind == "cluster":
            raise ValidationError(
                f"Profile '{profile.name}' is an imported cluster — its "
                "steering cannot be edited directly. Re-import an updated "
                "definition or adjust the intensity dial instead.",
                details={"profile_id": profile_id, "source_kind": "cluster"},
            )

        # Check name uniqueness if changing name
        if name and name != profile.name:
            if await self.repository.name_exists(name, exclude_id=profile_id):
                raise ProfileAlreadyExistsError(
                    f"Profile with name '{name}' already exists",
                    details={"name": name},
                )

        # Build update kwargs
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if steering is not None:
            # Convert keys to strings for JSONB
            updates["steering"] = {str(k): v for k, v in steering.items()}
        if model_id is not None:
            updates["model_id"] = model_id
        if sae_id is not None:
            updates["sae_id"] = sae_id
        if layer is not None:
            updates["layer"] = layer

        if not updates:
            return profile

        updated = await self.repository.update(profile_id, **updates)
        if not updated:
            raise ProfileNotFoundError(
                f"Profile with ID '{profile_id}' not found",
                details={"profile_id": profile_id},
            )

        logger.info(
            "profile_updated",
            profile_id=profile_id,
            updates=list(updates.keys()),
        )

        return updated

    # =========================================================================
    # Activation Methods
    # =========================================================================

    def _validate_activation(self, profile: Profile) -> None:
        """
        Hard activation gate, shared by every activation path.

        Cluster rows (Feature 8): the definition's declared n_features must
        match the attached SAE's feature space. All rows: every steering
        index must be within [0, d_sae). Runs BEFORE any live-steering
        mutation so a refused activation leaves the current steering intact.
        """
        from millm.services.sae_service import AttachedSAEState

        sae = AttachedSAEState().attached_sae
        if sae is None:
            return  # caller raises SAENotAttachedError with the house message

        if profile.source_kind == "cluster":
            declared = ((profile.cluster_meta or {}).get("sae") or {}).get("n_features")
            if declared is not None and int(declared) != sae.d_sae:
                raise ValidationError(
                    f"Cluster '{profile.name}' was authored for an SAE with "
                    f"{declared} features; the attached SAE has {sae.d_sae}. "
                    "Member indices would be meaningless — activation blocked.",
                    details={"profile_id": profile.id,
                             "declared_n_features": declared,
                             "attached_d_sae": sae.d_sae},
                )

        bad = [int(k) for k in (profile.steering or {})
               if not 0 <= int(k) < sae.d_sae]
        if bad:
            raise ValidationError(
                f"Profile '{profile.name}' references feature indices out of "
                f"range [0, {sae.d_sae}) for the attached SAE: "
                f"{sorted(bad)[:8]} — activation blocked.",
                details={"profile_id": profile.id,
                         "bad_indices": sorted(bad)[:20],
                         "attached_d_sae": sae.d_sae},
            )

    async def _release_active_circuit(self) -> list[str]:
        """Deactivate an active circuit row before a profile takes the layers.

        The circuit path releases co-tenant clusters when it activates; this is
        the symmetric half. Best-effort by design — a bookkeeping failure must
        not block the user's activation — but the row is reconciled so nothing
        reports "serving" while a profile has taken over its layers.
        """
        warnings: list[str] = []
        try:
            from millm.db.repositories.circuit_repository import CircuitRepository

            repo = CircuitRepository(self.repository.session)
            active = await repo.get_active()
            if active is None:
                return []
            await repo.deactivate(active.id)
            logger.info("circuit_released_for_profile_activation",
                        circuit_id=active.id)
            warnings.append(
                f"Deactivated circuit '{active.name}' — a profile takes over the "
                "layers it was steering"
            )
        except Exception as e:
            logger.warning("circuit_release_failed", error=str(e))
        return warnings

    async def activate_profile(
        self,
        profile_id: str,
        apply_steering: bool = True,
    ) -> dict[str, Any]:
        """
        Activate a profile, optionally applying its steering values.

        Args:
            profile_id: The profile's ID.
            apply_steering: Whether to apply steering to the current SAE.

        Returns:
            Dict with activation status and applied feature count.

        Raises:
            ProfileNotFoundError: If profile doesn't exist.
            SAENotAttachedError: If apply_steering is True but no SAE attached.
        """
        profile = await self.get_profile(profile_id)

        applied_steering = False
        feature_count = 0
        circuit_warnings: list[str] = []

        # Single-active invariant across manual / cluster / CIRCUIT (Feature 13).
        # A circuit releases an active cluster when it activates; without the
        # symmetric release, activating a profile here would clear/overwrite the
        # layer while the circuit row kept reporting is_active + serving_mode —
        # an "active" circuit that steers nothing, which is exactly the class of
        # lie the circuit path fails closed on.
        if apply_steering:
            circuit_warnings = await self._release_active_circuit()

        # Apply steering if requested
        if apply_steering:
            attachment = self.sae_service.get_attachment_status()
            # Validate EVERYTHING before touching live steering: this is the
            # single choke point every activation path goes through (Profiles
            # route, Clusters route, MCP). Runs even for empty-steering rows —
            # a poisoned cluster_meta must not activate+late-bind unchecked
            # (round-2 find).
            self._validate_activation(profile)
            if profile.steering:
                # Profile has steering values — require SAE to be attached
                if not attachment.is_attached:
                    raise SAENotAttachedError(
                        "Cannot apply steering: no SAE is attached",
                    )
                # Convert string keys back to int and apply. Steering values are
                # stored at lambda=1 basis (Feature 8): scale by the profile's
                # intensity dial and clamp to the supported range — imported
                # cluster strengths (contract allows ±300) times lambda (≤2)
                # can exceed miLLM's ±200 steering range.
                lam = profile.intensity if profile.intensity is not None else 1.0
                steering = {
                    int(k): clamp_steering(float(v) * lam)
                    for k, v in profile.steering.items()
                }
                clamped = [
                    int(k) for k, v in profile.steering.items()
                    if would_clamp(float(v) * lam)
                ]
                if clamped:
                    logger.warning(
                        "profile_activation_values_clamped",
                        profile_id=profile_id,
                        intensity=lam,
                        clamped_features=clamped,
                    )
                self.sae_service.clear_steering()
                self.sae_service.set_steering_batch(steering)
                self.sae_service.enable_steering(True)
                applied_steering = True
                feature_count = len(steering)
            elif attachment.is_attached:
                # Profile has no steering values — clear any existing steering so
                # the profile's (empty) steering state is correctly reflected.
                self.sae_service.clear_steering()
                self.sae_service.enable_steering(False)
                applied_steering = True
                feature_count = 0

        # Warn when the caller explicitly requested steering but nothing was applied
        # (profile has no features AND no SAE is attached).  This is not an error —
        # the profile's empty-steering state is correct — but silence here surprises
        # operators who forget to attach an SAE first.
        if apply_steering and not applied_steering and not attachment.is_attached:
            logger.info(
                "profile_activation_apply_steering_no_op",
                profile_id=profile_id,
                reason="no_sae_attached_and_profile_has_no_steering_values",
            )

        # Set profile as active
        await self.repository.set_active(profile_id)

        # Sensing lifecycle (Feature 11): arm when the newly active cluster
        # has sensing enabled and an SAE is attached; anything else disarms
        # (activating profile B must never keep sensing profile A).
        sensing_armed = self._sync_sensing_arm_state(profile)

        logger.info(
            "profile_activated",
            profile_id=profile_id,
            applied_steering=applied_steering,
            feature_count=feature_count,
        )

        # Feature 16: the Feature 10 path has the identical window
        AttachedSAEState().bump_steering_epoch('profile_activate')
        return {
            "profile_id": profile_id,
            "applied_steering": applied_steering,
            "feature_count": feature_count,
            # 011 R2: an arm refusal (bad thresholds, mismatched SAE) was a
            # log-only event — callers now see whether sensing engaged.
            "sensing_armed": sensing_armed,
            # 013 R3: if activating this profile took the layers from an active
            # circuit, say so — the circuit row was reconciled, not silently
            # left claiming to serve.
            "warnings": circuit_warnings,
        }

    async def deactivate_profile(
        self,
        profile_id: str,
        clear_steering: bool = True,
    ) -> dict[str, Any]:
        """
        Deactivate a profile, optionally clearing steering values.

        Args:
            profile_id: The profile's ID.
            clear_steering: Whether to clear current SAE steering.

        Returns:
            Dict with deactivation status.

        Raises:
            ProfileNotFoundError: If profile doesn't exist.
        """
        profile = await self.get_profile(profile_id)

        # Captured BEFORE repository.deactivate mutates this identity-mapped
        # row (011 R1: reading profile.is_active afterwards is always False,
        # which made the disarm condition dead code).
        was_active = bool(profile.is_active)
        cleared_steering = False

        # Clear steering if requested — but ONLY when the row being
        # deactivated is the one whose steering is live. Deactivating an
        # inactive profile must never wipe the active profile's steering
        # (review find: POST /deactivate on any id used to clear globally).
        if clear_steering and was_active:
            attachment = self.sae_service.get_attachment_status()
            if attachment.is_attached:
                self.sae_service.clear_steering()
                cleared_steering = True

        # Deactivate profile
        await self.repository.deactivate(profile_id)

        # Sensing lifecycle (Feature 11): deactivating the sensed profile
        # disarms — regardless of the clear_steering flag; deactivating
        # another row leaves the armed state alone.
        if was_active:
            self._sync_sensing_arm_state(None)

        logger.info(
            "profile_deactivated",
            profile_id=profile_id,
            cleared_steering=cleared_steering,
        )

        # Feature 16: the Feature 10 path has the identical window
        AttachedSAEState().bump_steering_epoch('profile_deactivate')
        return {
            "profile_id": profile_id,
            "cleared_steering": cleared_steering,
        }

    # =========================================================================
    # Delete Methods
    # =========================================================================

    async def delete_profile(self, profile_id: str) -> dict[str, Any]:
        """
        Delete a profile.

        Args:
            profile_id: The profile's ID.

        Returns:
            Dict with deletion status.

        Raises:
            ProfileNotFoundError: If profile doesn't exist.
        """
        profile = await self.get_profile(profile_id)
        was_active = profile.is_active

        # Deactivate first if active
        if was_active:
            # Deleting the active profile must also clear its LIVE steering —
            # otherwise the model keeps steering with orphaned values that no
            # profile owns and no UI can see or dial off.
            attachment = self.sae_service.get_attachment_status()
            if attachment.is_attached:
                self.sae_service.clear_steering()
            await self.repository.deactivate(profile_id)
            # Sensing lifecycle (011 R1): deleting the armed cluster must
            # disarm — otherwise the runtime keeps sensing into a dead FK
            # (an IntegrityError per request, silently swallowed).
            self._sync_sensing_arm_state(None)

        # Delete profile
        deleted = await self.repository.delete(profile_id)
        if not deleted:
            raise ProfileNotFoundError(
                f"Profile with ID '{profile_id}' not found",
                details={"profile_id": profile_id},
            )

        logger.info(
            "profile_deleted",
            profile_id=profile_id,
            was_active=was_active,
        )

        return {
            "profile_id": profile_id,
            "was_active": was_active,
        }

    def _sync_sensing_arm_state(self, active_profile) -> bool:
        """
        Arm/disarm co-activation sensing to match the active profile
        (Feature 11). Never raises — sensing is an observation feature and
        must not break activation. Returns True when sensing armed.
        """
        try:
            import millm.api.dependencies as deps
            from millm.services.sae_service import AttachedSAEState

            service = deps.get_sensing_service()
            sae = AttachedSAEState().attached_sae
            should_arm = (
                active_profile is not None
                and getattr(active_profile, "source_kind", None) == "cluster"
                and bool(getattr(active_profile, "sensing_enabled", False))
                and sae is not None
            )
            if should_arm:
                service.arm_for_profile(active_profile, sae)
                return True
            service.disarm(sae)
        except Exception:
            logger.warning("sensing_arm_sync_failed", exc_info=True)
        return False
