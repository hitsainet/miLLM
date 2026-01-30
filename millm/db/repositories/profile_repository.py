"""
Repository for Profile database operations.

Provides CRUD operations for steering configuration profiles.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.profile import Profile


class ProfileRepository:
    """
    Repository for Profile CRUD operations.

    All methods are async and use the provided session.
    The repository does not manage transactions - that's the caller's responsibility.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy async session for database operations.
        """
        self.session = session

    # ==========================================================================
    # Profile CRUD Operations
    # ==========================================================================

    async def create(
        self,
        profile_id: str,
        name: str,
        description: str | None = None,
        model_id: str | None = None,
        sae_id: str | None = None,
        layer: int | None = None,
        steering: dict[str, Any] | None = None,
    ) -> Profile:
        """
        Create a new Profile record.

        Args:
            profile_id: Unique ID for the profile.
            name: Display name for the profile.
            description: Optional description.
            model_id: Optional model identifier the profile was created for.
            sae_id: Optional SAE ID the steering is for.
            layer: Optional layer the SAE targets.
            steering: Dict mapping feature index to steering value.

        Returns:
            The created Profile instance.
        """
        profile = Profile(
            id=profile_id,
            name=name,
            description=description,
            model_id=model_id,
            sae_id=sae_id,
            layer=layer,
            steering=steering or {},
        )
        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def get(self, profile_id: str) -> Profile | None:
        """
        Get a Profile by its ID.

        Args:
            profile_id: The profile's primary key.

        Returns:
            The Profile instance or None if not found.
        """
        return await self.session.get(Profile, profile_id)

    async def get_by_name(self, name: str) -> Profile | None:
        """
        Get a Profile by its name.

        Args:
            name: The profile's unique name.

        Returns:
            The Profile instance or None if not found.
        """
        result = await self.session.execute(
            select(Profile).where(Profile.name == name)
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> list[Profile]:
        """
        Get all Profiles, ordered by name.

        Returns:
            List of all Profile instances.
        """
        result = await self.session.execute(
            select(Profile).order_by(Profile.name)
        )
        return list(result.scalars().all())

    async def get_active(self) -> Profile | None:
        """
        Get the currently active profile (if any).

        Returns:
            The active Profile instance or None.
        """
        result = await self.session.execute(
            select(Profile).where(Profile.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def update(self, profile_id: str, **kwargs: Any) -> Profile | None:
        """
        Update a Profile's attributes.

        Args:
            profile_id: The profile's primary key.
            **kwargs: Attributes to update.

        Returns:
            The updated Profile instance or None if not found.
        """
        profile = await self.get(profile_id)
        if profile is None:
            return None

        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def update_steering(
        self,
        profile_id: str,
        steering: dict[str, Any],
    ) -> Profile | None:
        """
        Update a Profile's steering values.

        Args:
            profile_id: The profile's primary key.
            steering: New steering dict mapping feature index to value.

        Returns:
            The updated Profile instance or None if not found.
        """
        profile = await self.get(profile_id)
        if profile is None:
            return None

        profile.steering = steering
        profile.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def set_active(self, profile_id: str) -> Profile | None:
        """
        Set a profile as the active profile.

        Note: Only one active profile is allowed (enforced by database constraint).
        This method first deactivates any currently active profile.

        Args:
            profile_id: The profile's ID to set as active.

        Returns:
            The activated Profile instance or None if not found.
        """
        # First, deactivate any currently active profile
        await self.deactivate_all()

        # Then activate the requested profile
        profile = await self.get(profile_id)
        if profile is None:
            return None

        profile.is_active = True
        profile.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def deactivate(self, profile_id: str) -> Profile | None:
        """
        Deactivate a specific profile.

        Args:
            profile_id: The profile's ID.

        Returns:
            The deactivated Profile or None if not found.
        """
        profile = await self.get(profile_id)
        if profile is None:
            return None

        profile.is_active = False
        profile.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def deactivate_all(self) -> int:
        """
        Deactivate all active profiles.

        Returns:
            Number of profiles deactivated.
        """
        result = await self.session.execute(
            select(Profile).where(Profile.is_active == True)  # noqa: E712
        )
        profiles = result.scalars().all()

        count = 0
        for profile in profiles:
            profile.is_active = False
            profile.updated_at = datetime.utcnow()
            count += 1

        if count > 0:
            await self.session.commit()

        return count

    async def delete(self, profile_id: str) -> bool:
        """
        Delete a Profile by ID.

        Args:
            profile_id: The profile's primary key.

        Returns:
            True if deleted, False if not found.
        """
        profile = await self.get(profile_id)
        if profile is None:
            return False

        await self.session.delete(profile)
        await self.session.commit()
        return True

    async def exists(self, profile_id: str) -> bool:
        """
        Check if a Profile exists.

        Args:
            profile_id: The profile's primary key.

        Returns:
            True if exists, False otherwise.
        """
        profile = await self.get(profile_id)
        return profile is not None

    async def name_exists(self, name: str, exclude_id: str | None = None) -> bool:
        """
        Check if a profile name already exists.

        Args:
            name: The profile name to check.
            exclude_id: Optional profile ID to exclude (for updates).

        Returns:
            True if name exists, False otherwise.
        """
        query = select(Profile).where(Profile.name == name)
        if exclude_id:
            query = query.where(Profile.id != exclude_id)

        result = await self.session.execute(query)
        return result.scalar_one_or_none() is not None
