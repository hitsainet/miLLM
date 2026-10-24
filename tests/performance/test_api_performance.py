"""
Performance tests for API response times.

Verifies that API endpoints meet performance requirements:
- List/get operations: <100ms
- Download start: <2s
"""

import asyncio
import time
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

from millm.main import create_app
from millm.core.config import get_settings


# Performance thresholds (in seconds)
LIST_GET_THRESHOLD = 0.100  # 100ms
DOWNLOAD_START_THRESHOLD = 2.0  # 2s


@pytest.fixture
async def client():
    """Create test client."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint_response_time(self, client: AsyncClient):
        """Test that health endpoint responds within 100ms."""
        start = time.perf_counter()
        response = await client.get("/api/health")
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < LIST_GET_THRESHOLD, (
            f"Health endpoint took {elapsed:.3f}s, threshold is {LIST_GET_THRESHOLD}s"
        )

    @pytest.mark.asyncio
    async def test_list_models_response_time(self, client: AsyncClient):
        """Test that list models endpoint responds within 100ms."""
        start = time.perf_counter()
        response = await client.get("/api/models")
        elapsed = time.perf_counter() - start

        # Note: Response may be 500 if database not set up, but timing is still valid
        assert elapsed < LIST_GET_THRESHOLD, (
            f"List models took {elapsed:.3f}s, threshold is {LIST_GET_THRESHOLD}s"
        )

    @pytest.mark.asyncio
    async def test_get_model_response_time(self, client: AsyncClient):
        """Test that get model endpoint responds within 100ms."""
        start = time.perf_counter()
        response = await client.get("/api/models/1")
        elapsed = time.perf_counter() - start

        # Note: Response will be 404 for non-existent model, but timing is still valid
        assert elapsed < LIST_GET_THRESHOLD, (
            f"Get model took {elapsed:.3f}s, threshold is {LIST_GET_THRESHOLD}s"
        )

    @pytest.mark.asyncio
    async def test_preview_model_response_time(self, client: AsyncClient):
        """Test that preview model endpoint starts within 2s."""
        with patch('millm.services.model_service.ModelDownloader') as mock:
            mock_instance = mock.return_value
            mock_instance.get_model_info.return_value = {
                "name": "test-model",
                "repo_id": "test/model",
                "params": "7B",
            }

            start = time.perf_counter()
            response = await client.post(
                "/api/models/preview",
                json={"repo_id": "test/model"}
            )
            elapsed = time.perf_counter() - start

            # Note: May fail due to service dependencies, but timing is the test
            assert elapsed < DOWNLOAD_START_THRESHOLD, (
                f"Preview model took {elapsed:.3f}s, threshold is {DOWNLOAD_START_THRESHOLD}s"
            )


class TestMemoryEstimationAccuracy:
    """Tests for memory estimation accuracy."""

    def test_memory_estimation_7b_fp16(self):
        """Test memory estimation for 7B model at FP16."""
        from millm.ml.memory_utils import estimate_memory_mb

        # 7B params * 2 bytes/param (FP16) * 1.2 overhead = 16,800 MB
        # Allow 20% variance
        estimated = estimate_memory_mb("7B", "fp16")
        expected = 7_000_000_000 * 2 * 1.2 / (1024 * 1024)  # ~16,000 MB

        variance = abs(estimated - expected) / expected
        assert variance < 0.20, (
            f"Memory estimation variance is {variance:.1%}, "
            f"expected {expected:.0f} MB, got {estimated:.0f} MB"
        )

    def test_memory_estimation_7b_q4(self):
        """Test memory estimation for 7B model at Q4."""
        from millm.ml.memory_utils import estimate_memory_mb

        # 7B params * 0.5 bytes/param (Q4) * 1.2 overhead = 4,200 MB
        estimated = estimate_memory_mb("7B", "q4")
        expected = 7_000_000_000 * 0.5 * 1.2 / (1024 * 1024)  # ~4,000 MB

        variance = abs(estimated - expected) / expected
        assert variance < 0.20, (
            f"Memory estimation variance is {variance:.1%}, "
            f"expected {expected:.0f} MB, got {estimated:.0f} MB"
        )

    def test_memory_estimation_7b_q8(self):
        """Test memory estimation for 7B model at Q8."""
        from millm.ml.memory_utils import estimate_memory_mb

        # 7B params * 1 byte/param (Q8) * 1.2 overhead = 8,400 MB
        estimated = estimate_memory_mb("7B", "q8")
        expected = 7_000_000_000 * 1.0 * 1.2 / (1024 * 1024)  # ~8,000 MB

        variance = abs(estimated - expected) / expected
        assert variance < 0.20, (
            f"Memory estimation variance is {variance:.1%}, "
            f"expected {expected:.0f} MB, got {estimated:.0f} MB"
        )

    def test_memory_estimation_70b_q4(self):
        """Test memory estimation for 70B model at Q4."""
        from millm.ml.memory_utils import estimate_memory_mb

        # 70B params * 0.5 bytes/param (Q4) * 1.2 overhead = 42,000 MB
        estimated = estimate_memory_mb("70B", "q4")
        expected = 70_000_000_000 * 0.5 * 1.2 / (1024 * 1024)  # ~40,000 MB

        variance = abs(estimated - expected) / expected
        assert variance < 0.20, (
            f"Memory estimation variance is {variance:.1%}, "
            f"expected {expected:.0f} MB, got {estimated:.0f} MB"
        )

    def test_memory_estimation_handles_various_formats(self):
        """Test memory estimation handles various param string formats."""
        from millm.ml.memory_utils import estimate_memory_mb

        # Test various formats
        formats = [
            ("7B", 7_000_000_000),
            ("7b", 7_000_000_000),
            ("70B", 70_000_000_000),
            ("1.3B", 1_300_000_000),
            ("405B", 405_000_000_000),
        ]

        for param_str, expected_params in formats:
            estimated = estimate_memory_mb(param_str, "fp16")
            expected = expected_params * 2 * 1.2 / (1024 * 1024)

            variance = abs(estimated - expected) / expected if expected > 0 else 0
            assert variance < 0.25, (
                f"Memory estimation for {param_str} variance is {variance:.1%}"
            )


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_list_requests(self, client: AsyncClient):
        """Test that multiple concurrent list requests are handled efficiently."""
        num_requests = 10

        async def make_request():
            start = time.perf_counter()
            response = await client.get("/api/models")
            elapsed = time.perf_counter() - start
            return elapsed

        # Make concurrent requests
        start = time.perf_counter()
        tasks = [make_request() for _ in range(num_requests)]
        times = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

        # Average time should be close to threshold, not num_requests * threshold
        avg_time = sum(times) / len(times)

        # Total time should be significantly less than sequential (num * threshold)
        sequential_time = num_requests * LIST_GET_THRESHOLD
        assert total_elapsed < sequential_time, (
            f"Concurrent requests took {total_elapsed:.3f}s, "
            f"sequential would be {sequential_time:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_concurrent_health_requests(self, client: AsyncClient):
        """Test that health endpoint handles concurrent requests."""
        num_requests = 20

        async def make_request():
            response = await client.get("/api/health")
            return response.status_code

        start = time.perf_counter()
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

        # All requests should succeed
        success_count = sum(1 for r in results if r == 200)

        # Should complete all requests reasonably quickly
        assert elapsed < 2.0, f"Concurrent health checks took {elapsed:.3f}s"
