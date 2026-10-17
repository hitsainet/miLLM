"""
Integration test infrastructure.

Patches Docker-specific cache paths (MODEL_CACHE_DIR=/app/..., SAE_CACHE_DIR=/app/...)
to temp directories so that ModelDownloader and SAEDownloader can initialize
without PermissionError in development environments that don't have /app.

Also overrides the get_db dependency so tests that use create_app() don't need
a live PostgreSQL instance — most integration/api tests return early (503, 422)
before executing any database queries.

Tests that build their own minimal FastAPI app with dependency_overrides are
unaffected by the autouse fixture (it still runs but the dep override dominates).
"""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def redirect_cache_dirs(tmp_path, monkeypatch):
    """
    Redirect MODEL_CACHE_DIR and SAE_CACHE_DIR to a temp directory and
    override the database session dependency with a no-op async mock.

    Clears the lru_cache on singleton factory functions so they reinitialize
    with the patched paths on next call. Restored after each test.
    """
    from millm.core.config import settings
    from millm.api import dependencies

    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path / "saes"))

    # Clear singletons that captured the old paths at startup
    dependencies.get_model_downloader.cache_clear()
    dependencies.get_model_loader.cache_clear()
    dependencies.get_inference_service.cache_clear()

    yield

    dependencies.get_model_downloader.cache_clear()
    dependencies.get_model_loader.cache_clear()
    dependencies.get_inference_service.cache_clear()


@pytest.fixture(autouse=True)
def mock_db_session(request):
    """
    Override the get_db dependency on the app used in integration/api/ tests.

    Tests in tests/integration/api/ use TestClient(create_app()) and never
    hit real DB queries (they return 503/422 before any SELECT). Patching
    get_db avoids the asyncpg connection attempt to a non-existent dev DB.

    Tests in tests/integration/ that build their own FastAPI() with
    dependency_overrides are unaffected because they don't import create_app.
    """
    # Only apply for tests that rely on the full create_app() client fixture.
    # We detect this by checking if the test's module is under integration/api/.
    if "integration/api" not in str(request.fspath) and "integration\\api" not in str(request.fspath):
        yield
        return

    # Patch get_db at the source so any app created in this test gets the mock
    from millm.api import dependencies

    async def _mock_get_db():
        session = AsyncMock()
        yield session

    original = dependencies.get_db
    dependencies.get_db = _mock_get_db

    yield

    dependencies.get_db = original
