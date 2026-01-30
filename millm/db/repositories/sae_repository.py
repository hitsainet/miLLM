"""
Repository for SAE database operations.

Provides CRUD operations for SAE and SAEAttachment ORM classes.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.sae import SAE, SAEAttachment, SAEStatus


class SAERepository:
    """
    Repository for SAE CRUD operations.

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
    # SAE CRUD Operations
    # ==========================================================================

    async def create(self, **kwargs: Any) -> SAE:
        """
        Create a new SAE record.

        Args:
            **kwargs: SAE attributes to set.

        Returns:
            The created SAE instance.
        """
        sae = SAE(**kwargs)
        self.session.add(sae)
        await self.session.commit()
        await self.session.refresh(sae)
        return sae

    async def create_downloading(
        self,
        sae_id: str,
        repository_id: str,
        revision: str = "main",
        cache_path: str = "",
    ) -> SAE:
        """
        Create SAE record in downloading state.

        Args:
            sae_id: Unique ID for the SAE.
            repository_id: HuggingFace repository ID.
            revision: Git revision (branch, tag, commit).
            cache_path: Path where SAE will be cached.

        Returns:
            The created SAE instance in downloading state.
        """
        sae = SAE(
            id=sae_id,
            repository_id=repository_id,
            revision=revision,
            name=repository_id.split("/")[-1],  # Use repo name as initial name
            status=SAEStatus.DOWNLOADING,
            d_in=0,  # Updated after download
            d_sae=0,
            cache_path=cache_path,
        )
        self.session.add(sae)
        await self.session.commit()
        await self.session.refresh(sae)
        return sae

    async def get(self, sae_id: str) -> SAE | None:
        """
        Get an SAE by its ID.

        Args:
            sae_id: The SAE's primary key.

        Returns:
            The SAE instance or None if not found.
        """
        return await self.session.get(SAE, sae_id)

    async def get_all(self) -> list[SAE]:
        """
        Get all SAEs, ordered by created_at descending.

        Returns:
            List of all SAE instances.
        """
        result = await self.session.execute(
            select(SAE).order_by(SAE.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_status(self, status: SAEStatus) -> list[SAE]:
        """
        Get all SAEs with a specific status.

        Args:
            status: The SAEStatus to filter by.

        Returns:
            List of SAE instances with the given status.
        """
        result = await self.session.execute(
            select(SAE).where(SAE.status == status).order_by(SAE.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_repository(
        self, repository_id: str, revision: str = "main"
    ) -> SAE | None:
        """
        Get SAE by repository and revision.

        Args:
            repository_id: HuggingFace repository ID.
            revision: Git revision.

        Returns:
            The SAE instance or None if not found.
        """
        result = await self.session.execute(
            select(SAE).where(
                SAE.repository_id == repository_id,
                SAE.revision == revision,
            )
        )
        return result.scalar_one_or_none()

    async def update(self, sae_id: str, **kwargs: Any) -> SAE | None:
        """
        Update an SAE's attributes.

        Args:
            sae_id: The SAE's primary key.
            **kwargs: Attributes to update.

        Returns:
            The updated SAE instance or None if not found.
        """
        sae = await self.get(sae_id)
        if sae is None:
            return None

        for key, value in kwargs.items():
            if hasattr(sae, key):
                setattr(sae, key, value)

        sae.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(sae)
        return sae

    async def update_downloaded(
        self,
        sae_id: str,
        cache_path: str,
        d_in: int,
        d_sae: int,
        trained_on: str | None = None,
        trained_layer: int | None = None,
        file_size_bytes: int | None = None,
        name: str | None = None,
    ) -> SAE | None:
        """
        Update SAE after successful download.

        Args:
            sae_id: The SAE's ID.
            cache_path: Path to cached SAE files.
            d_in: Input dimension.
            d_sae: SAE feature dimension.
            trained_on: Model the SAE was trained on.
            trained_layer: Layer the SAE was trained for.
            file_size_bytes: Size on disk.
            name: Display name (optional).

        Returns:
            The updated SAE instance or None if not found.
        """
        sae = await self.get(sae_id)
        if sae is None:
            return None

        sae.status = SAEStatus.CACHED
        sae.cache_path = cache_path
        sae.d_in = d_in
        sae.d_sae = d_sae
        sae.trained_on = trained_on
        sae.trained_layer = trained_layer
        sae.file_size_bytes = file_size_bytes
        sae.updated_at = datetime.utcnow()

        if name:
            sae.name = name
        elif trained_on and trained_layer is not None:
            sae.name = f"{trained_on} Layer {trained_layer} SAE"

        await self.session.commit()
        await self.session.refresh(sae)
        return sae

    async def update_status(
        self,
        sae_id: str,
        status: SAEStatus,
        error_message: str | None = None,
    ) -> SAE | None:
        """
        Update an SAE's status and optionally its error message.

        Args:
            sae_id: The SAE's primary key.
            status: The new SAEStatus.
            error_message: Optional error message (set only for ERROR status).

        Returns:
            The updated SAE instance or None if not found.
        """
        sae = await self.get(sae_id)
        if sae is None:
            return None

        sae.status = status
        sae.updated_at = datetime.utcnow()

        if status == SAEStatus.ERROR:
            sae.error_message = error_message
        elif error_message is None:
            sae.error_message = None

        await self.session.commit()
        await self.session.refresh(sae)
        return sae

    async def delete(self, sae_id: str) -> bool:
        """
        Delete an SAE by ID (cascades to attachments).

        Args:
            sae_id: The SAE's primary key.

        Returns:
            True if deleted, False if not found.
        """
        sae = await self.get(sae_id)
        if sae is None:
            return False

        await self.session.delete(sae)
        await self.session.commit()
        return True

    async def exists(self, sae_id: str) -> bool:
        """
        Check if an SAE exists.

        Args:
            sae_id: The SAE's primary key.

        Returns:
            True if exists, False otherwise.
        """
        sae = await self.get(sae_id)
        return sae is not None

    # ==========================================================================
    # SAE Attachment Operations
    # ==========================================================================

    async def get_active_attachment(self) -> SAEAttachment | None:
        """
        Get the currently active attachment (if any).

        Returns:
            The active SAEAttachment instance or None.
        """
        result = await self.session.execute(
            select(SAEAttachment).where(SAEAttachment.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def get_attachments_for_sae(self, sae_id: str) -> list[SAEAttachment]:
        """
        Get all attachments for a specific SAE.

        Args:
            sae_id: The SAE's primary key.

        Returns:
            List of SAEAttachment instances.
        """
        result = await self.session.execute(
            select(SAEAttachment)
            .where(SAEAttachment.sae_id == sae_id)
            .order_by(SAEAttachment.attached_at.desc())
        )
        return list(result.scalars().all())

    async def create_attachment(
        self,
        sae_id: str,
        model_id: int,
        layer: int,
        memory_usage_mb: int | None = None,
    ) -> SAEAttachment:
        """
        Create new attachment record.

        Note: Only one active attachment is allowed (enforced by database constraint).

        Args:
            sae_id: The SAE's ID.
            model_id: The model's ID.
            layer: Layer where SAE is attached.
            memory_usage_mb: GPU memory used.

        Returns:
            The created SAEAttachment instance.
        """
        attachment = SAEAttachment(
            sae_id=sae_id,
            model_id=model_id,
            layer=layer,
            memory_usage_mb=memory_usage_mb,
            is_active=True,
        )
        self.session.add(attachment)
        await self.session.commit()
        await self.session.refresh(attachment)
        return attachment

    async def deactivate_attachment(self, sae_id: str) -> SAEAttachment | None:
        """
        Mark the active attachment for an SAE as inactive.

        Args:
            sae_id: The SAE's ID.

        Returns:
            The deactivated attachment or None if none was active.
        """
        result = await self.session.execute(
            select(SAEAttachment).where(
                SAEAttachment.sae_id == sae_id,
                SAEAttachment.is_active == True,  # noqa: E712
            )
        )
        attachment = result.scalar_one_or_none()

        if attachment is None:
            return None

        attachment.is_active = False
        attachment.detached_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(attachment)
        return attachment

    async def deactivate_all_attachments(self) -> int:
        """
        Deactivate all active attachments.

        Returns:
            Number of attachments deactivated.
        """
        result = await self.session.execute(
            select(SAEAttachment).where(SAEAttachment.is_active == True)  # noqa: E712
        )
        attachments = result.scalars().all()

        count = 0
        for attachment in attachments:
            attachment.is_active = False
            attachment.detached_at = datetime.utcnow()
            count += 1

        if count > 0:
            await self.session.commit()

        return count

    async def is_sae_attached(self, sae_id: str) -> bool:
        """
        Check if an SAE is currently attached.

        Args:
            sae_id: The SAE's ID.

        Returns:
            True if the SAE has an active attachment.
        """
        result = await self.session.execute(
            select(SAEAttachment).where(
                SAEAttachment.sae_id == sae_id,
                SAEAttachment.is_active == True,  # noqa: E712
            )
        )
        return result.scalar_one_or_none() is not None
