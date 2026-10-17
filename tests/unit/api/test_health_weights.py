"""`/api/health/detailed` carries a typed `weights` summary (031 FR-31.16.3, FTASKS 7.2-7.3).

Run through the real `detailed_health_check`; the summary never fails the endpoint.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from millm.api.routes.system import health as health_mod
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier


def _health():
    return asyncio.run(health_mod.detailed_health_check(model_loader=MagicMock(is_loaded=False)))


def test_the_summary_field_set_is_pinned():
    weights = _health().weights
    assert set(weights.model_dump()) == {
        "weights_state", "barrier", "attached", "attachment_state", "error",
    }
    assert (weights.weights_state, weights.barrier, weights.attached) == ("clean", "serving", False)
    assert weights.attachment_state is None and weights.error is None


def test_it_reads_the_live_barrier():
    barrier = WeightBarrier()
    barrier.mark_torn("test")
    set_weight_barrier(barrier)
    assert _health().weights.weights_state == "torn"


def test_a_read_error_is_reported_and_the_endpoint_still_answers():
    with patch("millm.services.weight_barrier.get_weight_barrier", side_effect=RuntimeError("x")):
        result = _health()
    assert result.weights.error == "RuntimeError: x" and result.weights.weights_state is None
    assert result.version
