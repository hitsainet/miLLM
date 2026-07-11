"""
Feature monitoring service.

Provides activation history tracking, statistics computation, and real-time updates.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from millm.services.sae_service import SAEService
    from millm.sockets.progress import ProgressEmitter

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Running statistics for a single feature."""

    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    active_count: int = 0

    @property
    def mean(self) -> float:
        """Calculate mean activation."""
        return self.sum / self.count if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Calculate variance using Welford's algorithm."""
        if self.count < 2:
            return 0.0
        return (self.sum_sq - (self.sum ** 2) / self.count) / (self.count - 1)

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return self.variance ** 0.5

    @property
    def active_ratio(self) -> float:
        """Calculate ratio of non-zero activations."""
        return self.active_count / self.count if self.count > 0 else 0.0

    def update(self, value: float) -> None:
        """Update statistics with new value."""
        self.count += 1
        self.sum += value
        self.sum_sq += value ** 2
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        if value > 0:
            self.active_count += 1


@dataclass
class ActivationEntry:
    """Entry in activation history."""

    timestamp: datetime
    request_id: Optional[str]
    token_position: int
    activations: dict[int, float]
    top_features: list[tuple[int, float]]


class MonitoringService:
    """
    Service for feature activation monitoring.

    Manages:
    - Activation history (ring buffer)
    - Running statistics per feature
    - WebSocket event emission (throttled)
    - Configuration of monitoring parameters

    Thread-safety:
    - History access should be thread-safe via deque
    - Statistics updates are not locked (eventual consistency OK)

    Attributes:
        sae_service: SAE service for accessing loaded SAE.
        emitter: Progress emitter for WebSocket events.
    """

    def __init__(
        self,
        sae_service: Optional["SAEService"] = None,
        emitter: Optional["ProgressEmitter"] = None,
        history_size: int = 100,
        throttle_ms: int = 100,
    ) -> None:
        """
        Initialize monitoring service.

        Args:
            sae_service: Optional SAE service (used in tests for mocking).
                When None the service accesses AttachedSAEState directly,
                which avoids capturing a per-request DB session in a singleton.
            emitter: WebSocket event emitter.
            history_size: Max entries in history buffer.
            throttle_ms: Min milliseconds between WebSocket events.
        """
        self._sae_service = sae_service
        self._emitter = emitter
        self._history_size = history_size
        self._throttle_ms = throttle_ms

        # History buffer
        self._history: deque[ActivationEntry] = deque(maxlen=history_size)

        # Per-feature statistics
        self._stats: dict[int, FeatureStats] = {}
        self._stats_start: Optional[datetime] = None
        self._total_activations: int = 0

        # Throttling
        self._last_emit_time: float = 0

        # Monitored features (None = all)
        self._monitored_features: Optional[list[int]] = None

        # Top-K: number of highest-activated features to capture per forward pass
        self._top_k: int = 10

        logger.debug(f"MonitoringService initialized with history_size={history_size}")

    # ==========================================================================
    # SAE state access
    # ==========================================================================

    def _get_attachment_status(self):
        """Return attachment status from sae_service (tests) or singleton (production)."""
        if self._sae_service is not None:
            return self._sae_service.get_attachment_status()
        from millm.services.sae_service import AttachedSAEState
        state = AttachedSAEState()
        # Build a lightweight status object matching the real AttachmentStatus shape
        sae = state.attached_sae
        class _Status:
            is_attached = state.is_attached
            sae_id = state.attached_sae_id
            monitoring_enabled = sae.is_monitoring_enabled if sae else False
        return _Status()

    def _enable_monitoring(self, enabled: bool, features: Optional[list[int]]) -> None:
        """Enable/disable monitoring via sae_service (tests) or singleton (production)."""
        if self._sae_service is not None:
            self._sae_service.enable_monitoring(enabled, features)
            return
        from millm.services.sae_service import AttachedSAEState
        sae = AttachedSAEState().attached_sae
        if sae is not None:
            sae.enable_monitoring(enabled, features)

    # ==========================================================================
    # Configuration
    # ==========================================================================

    def configure(
        self,
        enabled: bool = True,
        features: Optional[list[int]] = None,
        history_size: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Configure monitoring parameters.

        Args:
            enabled: Whether to enable monitoring on SAE.
            features: Specific features to monitor (None = all).
            history_size: New history buffer size.
            top_k: Number of top-activated features to capture per forward pass.
        """
        # Update top_k if provided
        if top_k is not None and top_k > 0:
            self._top_k = top_k

        # Update history size if changed
        if history_size is not None and history_size != self._history_size:
            self._history_size = history_size
            # Recreate deque with new maxlen
            old_history = list(self._history)
            self._history = deque(old_history[-history_size:], maxlen=history_size)

        # Store monitored features
        self._monitored_features = features

        # Configure SAE monitoring
        attachment = self._get_attachment_status()
        if attachment.is_attached:
            self._enable_monitoring(enabled, features)

        if enabled and self._stats_start is None:
            self._stats_start = datetime.utcnow()

        logger.info(
            f"Monitoring configured: enabled={enabled}, "
            f"features={'all' if features is None else len(features)}, "
            f"history_size={self._history_size}, top_k={self._top_k}"
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable monitoring."""
        attachment = self._get_attachment_status()
        if attachment.is_attached:
            self._enable_monitoring(enabled, self._monitored_features)

        if enabled and self._stats_start is None:
            self._stats_start = datetime.utcnow()

    # ==========================================================================
    # State
    # ==========================================================================

    def get_state(self) -> dict[str, Any]:
        """
        Get current monitoring state.

        Returns:
            Dictionary with monitoring state.
        """
        attachment = self._get_attachment_status()

        return {
            "enabled": attachment.monitoring_enabled,
            "sae_attached": attachment.is_attached,
            "sae_id": attachment.sae_id,
            "monitored_features": self._monitored_features,
            "history_size": self._history_size,
            "history_count": len(self._history),
            "top_k": self._top_k,
        }

    # ==========================================================================
    # Activation Recording
    # ==========================================================================

    def on_activation(
        self,
        activations: Tensor,
        request_id: Optional[str] = None,
        token_position: int = 0,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Record activation from forward pass.

        Called by the inference service once after a generation completes.

        Scope note (important): the SAE hook overwrites its captured activations
        on every forward pass and the inference service reads them only once,
        after generation.  What survives is therefore the activations of the
        *final* forward pass — i.e. the last generated token's residual stream,
        not the whole sequence.  This method reduces the captured tensor to that
        last position accordingly.  Prompt-token and intermediate-decode-step
        activations are not currently captured.  ``token_position`` reflects the
        caller's best estimate (0 when unknown) and should be read as "last
        token", not an absolute sequence index.

        Args:
            activations: Feature activations tensor (seq_len, d_sae) or (d_sae,).
                When monitoring a subset of features it is positionally compacted
                to (seq_len, len(monitored_features)); this method maps positions
                back to real feature indices via self._monitored_features.
            request_id: Associated inference request ID.
            token_position: Position label for the recorded activation (see note).
            top_k: Number of top features to capture (None = use service-level _top_k).
        """
        # Use service-level top_k unless overridden per-call
        effective_top_k = top_k if top_k is not None else self._top_k

        # Flatten to 1D (take last position along each leading dimension)
        # Handles (batch, seq_len, d_sae), (seq_len, d_sae), or (d_sae,)
        while activations.dim() > 1:
            activations = activations[-1]

        # Convert to dict and compute top-k
        acts_dict: dict[int, float] = {}
        top_features: list[tuple[int, float]] = []

        # Handle monitored features.  The captured tensor is positionally
        # compacted to the monitored-feature list, so position i corresponds to
        # real feature index self._monitored_features[i].  Guard against a
        # length mismatch (e.g. monitoring reconfigured mid-flight) rather than
        # IndexError-ing inside the post-generation path.
        if self._monitored_features is not None:
            width = activations.shape[0] if activations.dim() >= 1 else 0
            for i, idx in enumerate(self._monitored_features):
                if i >= width:
                    break
                value = activations[i].item()
                acts_dict[idx] = value
        else:
            # All features - only store non-zero for efficiency
            nonzero_mask = activations > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
            for idx in nonzero_indices:
                idx_int = idx.item()
                value = activations[idx_int].item()
                acts_dict[idx_int] = value

        # Compute top-k using the effective value
        if len(acts_dict) > 0:
            sorted_acts = sorted(acts_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_acts[:effective_top_k]

        # Create entry
        entry = ActivationEntry(
            timestamp=datetime.utcnow(),
            request_id=request_id,
            token_position=token_position,
            activations=acts_dict,
            top_features=top_features,
        )

        # Add to history
        self._history.append(entry)

        # Update statistics
        self._update_stats(acts_dict)

        # Emit WebSocket event (throttled)
        self._maybe_emit_event(entry)

    def _update_stats(self, activations: dict[int, float]) -> None:
        """Update running statistics for features."""
        self._total_activations += 1

        for idx, value in activations.items():
            if idx not in self._stats:
                self._stats[idx] = FeatureStats()
            self._stats[idx].update(value)

    def _maybe_emit_event(self, entry: ActivationEntry) -> None:
        """Emit WebSocket event if throttle allows."""
        if self._emitter is None:
            return

        now = time.time() * 1000
        if now - self._last_emit_time < self._throttle_ms:
            return

        self._last_emit_time = now

        # Emit event
        self._emitter.emit_activation_update(
            timestamp=entry.timestamp.isoformat(),
            features=entry.top_features,
            request_id=entry.request_id,
            position=entry.token_position,
        )

    # ==========================================================================
    # History
    # ==========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get activation history.

        Args:
            limit: Max entries to return.
            request_id: Filter by request ID.

        Returns:
            List of activation records (newest first).
        """
        records = list(self._history)

        # Filter by request_id
        if request_id is not None:
            records = [r for r in records if r.request_id == request_id]

        # Reverse for newest first
        records = records[::-1]

        # Apply limit
        if limit is not None:
            records = records[:limit]

        # Convert to dicts with activations as array of {feature_index, activation}
        return [
            {
                "timestamp": r.timestamp,
                "request_id": r.request_id,
                "token_position": r.token_position,
                "activations": [
                    {"feature_index": idx, "activation": val}
                    for idx, val in sorted(
                        r.activations.items(), key=lambda x: x[1], reverse=True
                    )
                ],
                "top_k": [
                    {"feature_index": idx, "activation": val}
                    for idx, val in r.top_features
                ],
            }
            for r in records
        ]

    def clear_history(self) -> int:
        """
        Clear activation history.

        Returns:
            Number of entries cleared.
        """
        count = len(self._history)
        self._history.clear()
        logger.info(f"Cleared {count} history entries")
        return count

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_statistics(
        self,
        feature_indices: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """
        Get feature statistics.

        Args:
            feature_indices: Specific features to get stats for (None = all).

        Returns:
            Statistics response dict.
        """
        features: list[dict[str, Any]] = []

        indices = feature_indices if feature_indices else list(self._stats.keys())

        for idx in indices:
            if idx in self._stats:
                stats = self._stats[idx]
                features.append({
                    "feature_idx": idx,
                    "count": stats.count,
                    "mean": round(stats.mean, 4),
                    "std": round(stats.std, 4),
                    "min": round(stats.min, 4) if stats.min != float("inf") else 0.0,
                    "max": round(stats.max, 4) if stats.max != float("-inf") else 0.0,
                    "active_ratio": round(stats.active_ratio, 4),
                })

        return {
            "features": features,
            "total_activations": self._total_activations,
            "since": self._stats_start,
        }

    def get_top_features(
        self,
        k: int = 10,
        metric: str = "mean",
    ) -> list[dict[str, Any]]:
        """
        Get top features by metric.

        Args:
            k: Number of features to return.
            metric: Metric to rank by (mean, max, active_ratio, count).

        Returns:
            List of top feature statistics.
        """
        if not self._stats:
            return []

        # Sort by metric
        def get_metric_value(item: tuple[int, FeatureStats]) -> float:
            _, stats = item
            if metric == "mean":
                return stats.mean
            elif metric == "max":
                return stats.max if stats.max != float("-inf") else 0.0
            elif metric == "active_ratio":
                return stats.active_ratio
            elif metric == "count":
                return float(stats.count)
            return stats.mean

        sorted_stats = sorted(
            self._stats.items(),
            key=get_metric_value,
            reverse=True,
        )

        result = []
        for idx, stats in sorted_stats[:k]:
            result.append({
                "feature_idx": idx,
                "count": stats.count,
                "mean": round(stats.mean, 4),
                "std": round(stats.std, 4),
                "min": round(stats.min, 4) if stats.min != float("inf") else 0.0,
                "max": round(stats.max, 4) if stats.max != float("-inf") else 0.0,
                "active_ratio": round(stats.active_ratio, 4),
            })

        return result

    def reset_statistics(self) -> int:
        """
        Reset all statistics.

        Returns:
            Number of features cleared.
        """
        count = len(self._stats)
        self._stats.clear()
        self._total_activations = 0
        self._stats_start = datetime.utcnow()
        logger.info(f"Reset statistics for {count} features")
        return count
