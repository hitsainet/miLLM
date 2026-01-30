"""
Repository layer for database operations.

Repositories provide CRUD operations for ORM models,
keeping database logic separate from business logic.
"""

from millm.db.repositories.model_repository import ModelRepository

__all__ = ["ModelRepository"]
