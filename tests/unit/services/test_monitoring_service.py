"""
Unit tests for MonitoringService.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.services.monitoring_service import (
    ActivationEntry,
    FeatureStats,
    MonitoringService,
)


@pytest.fixture
def mock_sae_service():
    """Create a mock SAE service."""
    service = MagicMock()
    service.get_attachment_status.return_value = MagicMock(
        is_attached=True,
        sae_id="test-sae-123",
        monitoring_enabled=False,
    )
    return service


@pytest.fixture
def mock_emitter():
    """Create a mock progress emitter."""
    return MagicMock()


@pytest.fixture
def monitoring_service(mock_sae_service, mock_emitter):
    """Create a monitoring service for testing."""
    return MonitoringService(
        sae_service=mock_sae_service,
        emitter=mock_emitter,
        history_size=10,
        throttle_ms=0,  # No throttling for tests
    )


class TestFeatureStats:
    """Tests for FeatureStats dataclass."""

    def test_mean_calculation(self):
        """Mean is calculated correctly."""
        stats = FeatureStats()
        stats.update(2.0)
        stats.update(4.0)
        stats.update(6.0)
        assert stats.mean == pytest.approx(4.0)

    def test_std_calculation(self):
        """Standard deviation is calculated correctly."""
        stats = FeatureStats()
        for val in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            stats.update(val)
        # Standard deviation should be approximately 2.0
        assert stats.std == pytest.approx(2.0, rel=0.1)

    def test_min_max_tracking(self):
        """Min and max are tracked correctly."""
        stats = FeatureStats()
        stats.update(5.0)
        stats.update(1.0)
        stats.update(10.0)
        stats.update(3.0)
        assert stats.min == 1.0
        assert stats.max == 10.0

    def test_active_ratio(self):
        """Active ratio tracks non-zero values."""
        stats = FeatureStats()
        stats.update(1.0)
        stats.update(0.0)
        stats.update(2.0)
        stats.update(0.0)
        assert stats.active_ratio == 0.5

    def test_empty_stats(self):
        """Empty stats return safe values."""
        stats = FeatureStats()
        assert stats.mean == 0.0
        assert stats.std == 0.0
        assert stats.active_ratio == 0.0


class TestMonitoringServiceState:
    """Tests for monitoring state management."""

    def test_get_state_with_sae(self, monitoring_service, mock_sae_service):
        """Get state returns correct values when SAE is attached."""
        state = monitoring_service.get_state()
        assert state["sae_attached"] is True
        assert state["sae_id"] == "test-sae-123"
        assert state["history_size"] == 10

    def test_get_state_without_sae(self, mock_sae_service, mock_emitter):
        """Get state handles no SAE attached."""
        mock_sae_service.get_attachment_status.return_value = MagicMock(
            is_attached=False,
            sae_id=None,
            monitoring_enabled=False,
        )
        service = MonitoringService(mock_sae_service, mock_emitter)
        state = service.get_state()
        assert state["sae_attached"] is False

    def test_configure_updates_settings(self, monitoring_service, mock_sae_service):
        """Configure updates monitoring settings."""
        monitoring_service.configure(
            enabled=True,
            features=[0, 1, 2],
            history_size=50,
        )

        # Verify SAE service was called
        mock_sae_service.enable_monitoring.assert_called_with(True, [0, 1, 2])

        # Verify internal state
        state = monitoring_service.get_state()
        assert state["monitored_features"] == [0, 1, 2]
        assert state["history_size"] == 50


class TestMonitoringServiceHistory:
    """Tests for activation history."""

    def test_on_activation_adds_to_history(self, monitoring_service):
        """Activation records are added to history."""
        activations = torch.tensor([0.5, 0.0, 1.0, 0.0, 2.0])
        monitoring_service.on_activation(
            activations=activations,
            request_id="req-123",
            token_position=0,
        )

        history = monitoring_service.get_history()
        assert len(history) == 1
        assert history[0]["request_id"] == "req-123"

    def test_history_ring_buffer(self, monitoring_service):
        """History respects max size (ring buffer)."""
        # Add more entries than history_size (10)
        for i in range(15):
            activations = torch.tensor([float(i)])
            monitoring_service.on_activation(
                activations=activations,
                request_id=f"req-{i}",
            )

        history = monitoring_service.get_history()
        assert len(history) == 10

        # Newest should be first
        assert history[0]["request_id"] == "req-14"

    def test_history_filter_by_request_id(self, monitoring_service):
        """History can be filtered by request ID."""
        for i in range(5):
            activations = torch.tensor([1.0])
            monitoring_service.on_activation(
                activations=activations,
                request_id=f"req-{i % 2}",  # Alternating request IDs
            )

        history = monitoring_service.get_history(request_id="req-0")
        assert all(r["request_id"] == "req-0" for r in history)

    def test_clear_history(self, monitoring_service):
        """History can be cleared."""
        activations = torch.tensor([1.0])
        monitoring_service.on_activation(activations=activations)
        monitoring_service.on_activation(activations=activations)

        count = monitoring_service.clear_history()
        assert count == 2
        assert len(monitoring_service.get_history()) == 0


class TestMonitoringServiceStatistics:
    """Tests for feature statistics."""

    def test_statistics_updated_on_activation(self, monitoring_service):
        """Statistics are updated when activations are recorded."""
        monitoring_service._monitored_features = [0, 1, 2]

        for _ in range(5):
            activations = torch.tensor([1.0, 2.0, 3.0])
            monitoring_service.on_activation(activations=activations)

        stats = monitoring_service.get_statistics()
        assert stats["total_activations"] == 5
        assert len(stats["features"]) > 0

    def test_get_top_features(self, monitoring_service):
        """Can get top features by metric."""
        monitoring_service._monitored_features = [0, 1, 2]

        # Add some activations with different values
        for _ in range(10):
            activations = torch.tensor([1.0, 5.0, 3.0])
            monitoring_service.on_activation(activations=activations)

        top = monitoring_service.get_top_features(k=2, metric="mean")
        assert len(top) == 2
        # Feature 1 has highest mean (5.0)
        assert top[0]["feature_idx"] == 1

    def test_reset_statistics(self, monitoring_service):
        """Statistics can be reset."""
        activations = torch.tensor([1.0, 2.0])
        monitoring_service.on_activation(activations=activations)

        count = monitoring_service.reset_statistics()
        assert count >= 0

        stats = monitoring_service.get_statistics()
        assert stats["total_activations"] == 0


class TestMonitoringServiceEmitter:
    """Tests for WebSocket event emission."""

    def test_emit_activation_throttled(self, mock_sae_service, mock_emitter):
        """Activation events are throttled."""
        service = MonitoringService(
            sae_service=mock_sae_service,
            emitter=mock_emitter,
            throttle_ms=1000,  # 1 second throttle
        )

        activations = torch.tensor([1.0])

        # First activation should emit
        service.on_activation(activations=activations)

        # Second immediate activation should NOT emit due to throttle
        service.on_activation(activations=activations)

        # Should only emit once due to throttling
        assert mock_emitter.emit_activation_update.call_count <= 1
