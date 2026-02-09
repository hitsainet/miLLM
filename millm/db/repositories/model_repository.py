"""
Repository for Model database operations.

Provides CRUD operations for the Model ORM class.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.model import Model, ModelStatus, QuantizationType


class ModelRepository:
    """
    Repository for Model CRUD operations.

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

    async def create(self, **kwargs: Any) -> Model:
        """
        Create a new model record.

        Args:
            **kwargs: Model attributes to set.

        Returns:
            The created Model instance with id populated.
        """
        model = Model(**kwargs)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get_by_id(self, model_id: int) -> Model | None:
        """
        Get a model by its ID.

        Args:
            model_id: The model's primary key.

        Returns:
            The Model instance or None if not found.
        """
        return await self.session.get(Model, model_id)

    async def get_all(self) -> list[Model]:
        """
        Get all models, ordered by created_at descending.

        Returns:
            List of all Model instances.
        """
        result = await self.session.execute(
            select(Model).order_by(Model.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_status(self, status: ModelStatus) -> list[Model]:
        """
        Get all models with a specific status.

        Args:
            status: The ModelStatus to filter by.

        Returns:
            List of Model instances with the given status.
        """
        result = await self.session.execute(
            select(Model).where(Model.status == status).order_by(Model.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_loaded_model(self) -> Model | None:
        """
        Get the currently loaded model (if any).

        Returns:
            The loaded Model instance or None.
        """
        result = await self.session.execute(
            select(Model).where(Model.status == ModelStatus.LOADED)
        )
        return result.scalar_one_or_none()

    async def find_by_name(self, name: str) -> Model | None:
        """
        Find a model by its display name.

        Args:
            name: The model's display name.

        Returns:
            The Model instance or None if not found.
        """
        result = await self.session.execute(
            select(Model).where(Model.name == name)
        )
        return result.scalar_one_or_none()

    async def get_locked_model(self) -> Model | None:
        """
        Get the currently locked model (if any).

        Returns:
            The locked Model instance or None.
        """
        result = await self.session.execute(
            select(Model).where(Model.locked == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def get_available_models(self) -> list[Model]:
        """
        Get all models that are available for use (READY, LOADED, or LOADING).

        Returns:
            List of available Model instances.
        """
        result = await self.session.execute(
            select(Model)
            .where(Model.status.in_([ModelStatus.READY, ModelStatus.LOADED, ModelStatus.LOADING]))
            .order_by(Model.created_at.desc())
        )
        return list(result.scalars().all())

    async def find_by_repo_quantization(
        self, repo_id: str, quantization: QuantizationType
    ) -> Model | None:
        """
        Find a model by repository ID and quantization level.

        Args:
            repo_id: The HuggingFace repository ID.
            quantization: The quantization level.

        Returns:
            The Model instance or None if not found.
        """
        result = await self.session.execute(
            select(Model).where(
                Model.repo_id == repo_id,
                Model.quantization == quantization,
            )
        )
        return result.scalar_one_or_none()

    async def find_by_local_path(self, local_path: str) -> Model | None:
        """
        Find a model by local path.

        Args:
            local_path: The local filesystem path.

        Returns:
            The Model instance or None if not found.
        """
        result = await self.session.execute(
            select(Model).where(Model.local_path == local_path)
        )
        return result.scalar_one_or_none()

    async def update(self, model_id: int, **kwargs: Any) -> Model | None:
        """
        Update a model's attributes.

        Args:
            model_id: The model's primary key.
            **kwargs: Attributes to update.

        Returns:
            The updated Model instance or None if not found.
        """
        model = await self.get_by_id(model_id)
        if model is None:
            return None

        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)

        model.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def update_status(
        self,
        model_id: int,
        status: ModelStatus,
        error_message: str | None = None,
    ) -> Model | None:
        """
        Update a model's status and optionally its error message.

        Args:
            model_id: The model's primary key.
            status: The new ModelStatus.
            error_message: Optional error message (cleared if not provided and status isn't ERROR).

        Returns:
            The updated Model instance or None if not found.
        """
        model = await self.get_by_id(model_id)
        if model is None:
            return None

        model.status = status
        model.updated_at = datetime.utcnow()

        if status == ModelStatus.ERROR:
            model.error_message = error_message
        elif error_message is None:
            model.error_message = None

        if status == ModelStatus.LOADED:
            model.loaded_at = datetime.utcnow()
        elif status != ModelStatus.LOADING:
            model.loaded_at = None

        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def delete(self, model_id: int) -> bool:
        """
        Delete a model by ID.

        Args:
            model_id: The model's primary key.

        Returns:
            True if deleted, False if not found.
        """
        model = await self.get_by_id(model_id)
        if model is None:
            return False

        await self.session.delete(model)
        await self.session.commit()
        return True

    async def exists(self, model_id: int) -> bool:
        """
        Check if a model exists.

        Args:
            model_id: The model's primary key.

        Returns:
            True if exists, False otherwise.
        """
        model = await self.get_by_id(model_id)
        return model is not None
