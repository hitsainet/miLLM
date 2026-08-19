"""
Repository for Model database operations.

Provides CRUD operations for the Model ORM class.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select, update
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
        Get the currently locked model, if one is actually holding the lock.

        A LOCK ONLY COUNTS WHILE THE MODEL IS LOADED. Both things the lock does
        — keep `/v1/models` pinned to one entry, and stop an inference request
        auto-unloading it — are meaningless for a model that is not resident,
        and `lock_model` refuses to set the flag on anything but a LOADED model.
        So a locked row in any other state is not a lock; it is debris, and the
        read is where that has to be decided.

        This is deliberately belt-and-braces with the startup reset in
        `main.py`, which clears the flag outright. That reset is the cure; this
        is the immunity. A lock that leaks by some route nobody has thought of
        yet — a crash between two writes, a hand-edited row, a restore from a
        backup taken mid-steer — costs a stale flag rather than a catalogue
        that has silently collapsed to a single model. Which is exactly what
        happened: `gemma-2-2b-it` held the lock from 2026-05-12 until
        2026-08-19, and for three months `/v1/models` advertised it alone while
        thirteen other ready models were invisible to every OpenAI client.

        `.first()` rather than `scalar_one_or_none()`: two locked rows are not
        supposed to be reachable — `set_exclusive_lock` is the only writer and
        it clears the others in the same statement — but the previous form
        RAISED `MultipleResultsFound` if they ever were, turning a stale flag
        into a 500 on the models listing for every client at once. A listing
        must degrade to "wrong entry" and never to "no service".
        """
        result = await self.session.execute(
            select(Model)
            .where(Model.locked.is_(True), Model.status == ModelStatus.LOADED)
            .order_by(Model.id)
        )
        return result.scalars().first()

    async def clear_locks(self, except_model_id: int | None = None) -> int:
        """Release every model lock, optionally sparing one. Returns rows changed."""
        stmt = update(Model).where(Model.locked.is_(True)).values(locked=False)
        if except_model_id is not None:
            stmt = stmt.where(Model.id != except_model_id)
        result = await self.session.execute(stmt)
        return int(result.rowcount or 0)

    async def set_exclusive_lock(self, model_id: int) -> Model | None:
        """Lock this model and release every other lock, together.

        THE ONLY WRITER THAT MAY SET THE FLAG. "Only one model can be locked at
        a time" was documented on `lock_model` and enforced only there, by a
        read-then-check that two callers in `sae_service` bypassed entirely by
        writing `repository.update(..., locked=True)` straight to the row. An
        invariant that a caller can skip by picking a different method is a
        convention, not an invariant; making the exclusive write the only write
        is what turns it back into one.
        """
        await self.clear_locks(except_model_id=model_id)
        return await self.update(model_id, locked=True)

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
