"""
Conftest for tests/integration/api/.

These tests use TestClient(create_app()) and define their own 'client' fixture,
which means we cannot use app.dependency_overrides on an already-created app.

Instead we patch async_session_factory at the source so that get_db in
dependencies.py picks up the mock on every call. FastAPI resolves get_db
lazily (per request), so the patch only needs to be in place when requests
are made, not at app-creation time.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def mock_async_session_factory(monkeypatch):
    """
    Replace async_session_factory with a no-op async context manager.

    get_db in dependencies.py does:
        async with async_session_factory() as session:
            yield session

    We use @asynccontextmanager so async_session_factory() returns a proper
    async context manager that yields an AsyncMock session.

    Patch target is millm.api.dependencies.async_session_factory because
    dependencies.py imports it with `from millm.db.base import async_session_factory`
    — that creates a local name binding which is what get_db resolves at call time.
    """
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()

    @asynccontextmanager
    async def _mock_factory():
        yield mock_session

    import millm.api.dependencies as deps_module
    monkeypatch.setattr(deps_module, "async_session_factory", _mock_factory)
