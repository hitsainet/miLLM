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
)
from millm.db.models.profile import Profile
from millm.db.repositories.profile_repository import ProfileRepository
from millm.services.sae_service import SAEService

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

        # Apply steering if requested
        if apply_steering and profile.steering:
            attachment = self.sae_service.get_attachment_status()
            if not attachment.is_attached:
                raise SAENotAttachedError(
                    "Cannot apply steering: no SAE is attached",
                )

            # Convert string keys back to int and apply
            steering = {int(k): v for k, v in profile.steering.items()}
            self.sae_service.clear_steering()
            self.sae_service.set_steering_batch(steering)
            self.sae_service.enable_steering(True)

            applied_steering = True
            feature_count = len(steering)

        # Set profile as active
        await self.repository.set_active(profile_id)

        logger.info(
            "profile_activated",
            profile_id=profile_id,
            applied_steering=applied_steering,
            feature_count=feature_count,
        )

        return {
            "profile_id": profile_id,
            "applied_steering": applied_steering,
            "feature_count": feature_count,
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

        cleared_steering = False

        # Clear steering if requested
        if clear_steering:
            attachment = self.sae_service.get_attachment_status()
            if attachment.is_attached:
                self.sae_service.clear_steering()
                cleared_steering = True

        # Deactivate profile
        await self.repository.deactivate(profile_id)

        logger.info(
            "profile_deactivated",
            profile_id=profile_id,
            cleared_steering=cleared_steering,
        )

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
            await self.repository.deactivate(profile_id)

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
