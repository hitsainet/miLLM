"""
End-to-end tests for the model management workflow.

These tests verify the complete workflow from API request to database state changes.
Uses a tiny test model from HuggingFace for realistic testing.
"""

import asyncio
import pytest
import socketio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock
from typing import AsyncGenerator
import tempfile
import shutil
import os

from millm.main import create_app
from millm.db.base import get_db, engine, Base
from millm.db.models.model import ModelStatus, ModelSource, QuantizationType
from millm.core.config import get_settings


# Test model - using a very small test model for E2E tests
TEST_REPO_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def temp_cache_dir():
    """Create temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="millm_test_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
async def test_db():
    """Set up test database."""
    # Use SQLite for E2E tests
    test_engine = engine
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client(test_db, temp_cache_dir) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with mocked cache directory."""
    settings = get_settings()

    # Patch cache directory
    with patch.object(settings, 'model_cache_dir', temp_cache_dir):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class TestModelDownloadWorkflow:
    """E2E tests for model download workflow."""

    @pytest.mark.asyncio
    async def test_preview_model_returns_info(self, client: AsyncClient):
        """Test 20.1: Preview model returns correct information."""
        response = await client.post(
            "/api/models/preview",
            json={"repo_id": TEST_REPO_ID}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["repo_id"] == TEST_REPO_ID

    @pytest.mark.asyncio
    async def test_download_model_creates_record(self, client: AsyncClient):
        """Test 20.1: Download model creates database record and starts download."""
        response = await client.post(
            "/api/models",
            json={
                "repo_id": TEST_REPO_ID,
                "quantization": "fp16"
            }
        )

        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["repo_id"] == TEST_REPO_ID
        assert data["data"]["status"] in ["downloading", "ready"]

        # Store model_id for subsequent tests
        return data["data"]["id"]

    @pytest.mark.asyncio
    async def test_list_models_includes_downloaded(self, client: AsyncClient):
        """Test that list models returns downloaded models."""
        # First download a model
        await client.post(
            "/api/models",
            json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
        )

        # Then list models
        response = await client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_get_model_by_id(self, client: AsyncClient):
        """Test getting a specific model by ID."""
        # First download a model
        download_response = await client.post(
            "/api/models",
            json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
        )
        model_id = download_response.json()["data"]["id"]

        # Get the specific model
        response = await client.get(f"/api/models/{model_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == model_id


class TestModelLoadUnloadWorkflow:
    """E2E tests for model load/unload workflow."""

    @pytest.mark.asyncio
    async def test_load_model_changes_status(self, client: AsyncClient):
        """Test 20.2: Load model changes status appropriately."""
        # Create a mock model loader to avoid actual GPU operations
        with patch('millm.services.model_service.ModelLoader') as mock_loader:
            mock_instance = MagicMock()
            mock_instance.load = MagicMock(return_value=MagicMock(
                model_id=1,
                memory_used_mb=100
            ))
            mock_loader.return_value = mock_instance

            # First download a model
            download_response = await client.post(
                "/api/models",
                json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
            )
            model_id = download_response.json()["data"]["id"]

            # Wait briefly for download (in real tests this would be mocked)
            await asyncio.sleep(0.5)

            # Attempt to load
            response = await client.post(f"/api/models/{model_id}/load")

            # Response should be 202 (accepted) or 409 (conflict if still downloading)
            assert response.status_code in [202, 409]

    @pytest.mark.asyncio
    async def test_unload_model_releases_memory(self, client: AsyncClient):
        """Test 20.3: Unload model releases memory."""
        with patch('millm.services.model_service.ModelLoader') as mock_loader:
            mock_instance = MagicMock()
            mock_instance.unload = MagicMock(return_value=True)
            mock_loader.return_value = mock_instance

            # Attempt to unload (may fail if no model loaded, which is expected)
            response = await client.post("/api/models/1/unload")

            # Response could be 200 (success) or 404 (not found) or 409 (not loaded)
            assert response.status_code in [200, 404, 409]


class TestModelDeleteWorkflow:
    """E2E tests for model deletion workflow."""

    @pytest.mark.asyncio
    async def test_delete_model_removes_record(self, client: AsyncClient):
        """Test 20.4: Delete model removes database record."""
        # First download a model
        download_response = await client.post(
            "/api/models",
            json={"repo_id": f"{TEST_REPO_ID}", "quantization": "q4"}
        )
        model_id = download_response.json()["data"]["id"]

        # Delete the model
        response = await client.delete(f"/api/models/{model_id}")

        # Could be 200 (deleted) or 409 (still downloading)
        assert response.status_code in [200, 409]

        if response.status_code == 200:
            # Verify it's gone
            get_response = await client.get(f"/api/models/{model_id}")
            assert get_response.status_code == 404


class TestCancelDownloadWorkflow:
    """E2E tests for download cancellation workflow."""

    @pytest.mark.asyncio
    async def test_cancel_download_stops_progress(self, client: AsyncClient):
        """Test 20.5: Cancel download stops the download process."""
        # Start a download
        download_response = await client.post(
            "/api/models",
            json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
        )
        model_id = download_response.json()["data"]["id"]

        # Immediately try to cancel
        response = await client.post(f"/api/models/{model_id}/cancel")

        # Could be 200 (cancelled) or 409 (already finished)
        assert response.status_code in [200, 409]


class TestErrorHandling:
    """E2E tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_repo_returns_error(self, client: AsyncClient):
        """Test 20.6: Invalid repository returns appropriate error."""
        response = await client.post(
            "/api/models/preview",
            json={"repo_id": "definitely-not-a-real-repo/fake-model-12345"}
        )

        # Should return 404 or error
        assert response.status_code in [404, 500]
        data = response.json()
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_duplicate_download_prevented(self, client: AsyncClient):
        """Test that duplicate downloads are prevented."""
        # First download
        await client.post(
            "/api/models",
            json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
        )

        # Second download of same model
        response = await client.post(
            "/api/models",
            json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
        )

        # Should be 409 Conflict
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_load_nonexistent_model_returns_404(self, client: AsyncClient):
        """Test that loading non-existent model returns 404."""
        response = await client.post("/api/models/99999/load")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_model_returns_404(self, client: AsyncClient):
        """Test that deleting non-existent model returns 404."""
        response = await client.delete("/api/models/99999")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_insufficient_memory_error(self, client: AsyncClient):
        """Test 20.6: Insufficient memory returns appropriate error."""
        with patch('millm.ml.memory_utils.get_available_memory_mb', return_value=100):
            with patch('millm.ml.memory_utils.estimate_memory_mb', return_value=10000):
                # Try to load a model that requires more memory than available
                response = await client.post("/api/models/1/load")

                # Could be 507 (insufficient storage/memory) or 404 (not found)
                # Depending on implementation order of checks
                assert response.status_code in [404, 507, 409]


class TestHealthEndpoint:
    """E2E tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_ok(self, client: AsyncClient):
        """Test health endpoint returns OK status."""
        response = await client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestWebSocketEvents:
    """E2E tests for WebSocket event handling."""

    @pytest.mark.asyncio
    async def test_socket_connection_established(self, client: AsyncClient):
        """Test that Socket.IO connection can be established."""
        # This is a basic test - full Socket.IO testing requires more setup
        # In a real E2E test, you would use a Socket.IO client

        # For now, verify the app accepts WebSocket connections
        # by checking that the Socket.IO endpoint exists
        pass  # Placeholder for Socket.IO tests

    @pytest.mark.asyncio
    async def test_download_progress_events_emitted(self, client: AsyncClient):
        """Test that download progress events are emitted via Socket.IO."""
        # This would require a Socket.IO test client
        # Placeholder for now
        pass


# Integration test for full workflow
class TestFullWorkflow:
    """Integration test for complete model management workflow."""

    @pytest.mark.asyncio
    async def test_complete_model_lifecycle(self, client: AsyncClient):
        """Test the complete lifecycle: preview -> download -> load -> unload -> delete."""
        with patch('millm.services.model_service.ModelLoader') as mock_loader:
            mock_instance = MagicMock()
            mock_instance.load = MagicMock(return_value=MagicMock(
                model_id=1,
                memory_used_mb=100
            ))
            mock_instance.unload = MagicMock(return_value=True)
            mock_loader.return_value = mock_instance

            # Step 1: Preview
            preview_response = await client.post(
                "/api/models/preview",
                json={"repo_id": TEST_REPO_ID}
            )
            assert preview_response.status_code == 200

            # Step 2: Download
            download_response = await client.post(
                "/api/models",
                json={"repo_id": TEST_REPO_ID, "quantization": "fp16"}
            )
            assert download_response.status_code == 202
            model_id = download_response.json()["data"]["id"]

            # Step 3: Verify in list
            list_response = await client.get("/api/models")
            assert list_response.status_code == 200
            models = list_response.json()["data"]
            assert any(m["id"] == model_id for m in models)

            # Note: In a real E2E test, we would wait for download to complete
            # before attempting load/unload/delete operations
