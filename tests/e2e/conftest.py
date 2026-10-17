"""
E2E test configuration and fixtures.
"""

import asyncio
import pytest
import tempfile
import shutil
from typing import AsyncGenerator

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from millm.main import create_app
from millm.db.base import Base, get_db
from millm.core.config import get_settings


# Test database URL - use SQLite for E2E tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_cache_dir():
    """Create temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="millm_e2e_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session


@pytest.fixture
async def client(
    test_engine,
    test_session,
    temp_cache_dir
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with test database and mocked cache directory."""
    settings = get_settings()

    async def override_get_db():
        yield test_session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    # Patch cache directory
    with patch.object(settings, 'model_cache_dir', temp_cache_dir):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_model_loader():
    """Mock the model loader to avoid GPU operations."""
    with patch('millm.services.model_service.ModelLoader') as mock:
        from millm.ml.model_loader import LoadedModel
        from datetime import datetime

        mock_instance = mock.return_value
        mock_instance.load.return_value = LoadedModel(
            model_id=1,
            model=None,
            tokenizer=None,
            loaded_at=datetime.utcnow(),
            memory_used_mb=100,
        )
        mock_instance.unload.return_value = True

        yield mock_instance


@pytest.fixture
def mock_model_downloader():
    """Mock the model downloader to avoid network operations."""
    with patch('millm.services.model_service.ModelDownloader') as mock:
        from millm.ml.model_downloader import ModelInfo

        mock_instance = mock.return_value
        mock_instance.get_model_info.return_value = ModelInfo(
            repo_id="test/model",
            model_name="model",
            author="test",
            params="7B",
            disk_size_mb=1000,
            estimated_memory_mb=8000,
        )
        mock_instance.download.return_value = "/tmp/test/model"
        mock_instance.delete_cached_model.return_value = True

        yield mock_instance
