"""Feature 29 task 1.5: the Retry-After value policy and the lease error codes.

Every value is asserted to be its OWN code's number, never the fallback (10), so a code that
fell through to the default branch fails here rather than looking like "a Retry-After was set".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from millm.api.routes.openai.errors import ERROR_STATUS_MAP
from millm.core import backpressure
from millm.core.backpressure import (
    backlog_rows,
    estimate_wait_seconds,
    register_backlog_provider,
    retry_after_for,
)
from millm.core.config import Settings, settings
from millm.core.errors import (
    InvalidLeaseRequestError,
    LeaseExpiredError,
    LeaseNotFoundError,
    ModelBusyError,
    ModelLeasedError,
    ModelLockedError,
    ModelNotResidentError,
)


class _Queue:
    """A queue stand-in whose numbers DISAGREE with each other, so a formula that used the wrong
    one gives a different answer."""

    def __init__(self, median, pending=0, holding=0, background=0, max_concurrent=1):
        self._median = median
        self.pending_count = pending
        self.holding_count = holding
        self.background_holding_count = background
        self.max_concurrent = max_concurrent

    def median_hold_seconds(self):
        return self._median


def _inference(queue, cbm=False):
    return SimpleNamespace(request_queue=queue, _use_cbm=lambda: cbm)


@pytest.fixture
def patch_inference(monkeypatch):
    def install(queue, cbm=False):
        monkeypatch.setattr(
            "millm.api.dependencies.get_inference_service", lambda: _inference(queue, cbm)
        )

    return install


class TestValuePerCode:
    def test_fallback_is_distinct_from_every_code_value(self):
        """The middleware's fallback must not coincide with a real code's value, or a forgotten
        builder would be indistinguishable from a correct one."""
        own = {
            settings.RETRY_AFTER_QUEUE_DEFAULT_S,
            settings.RETRY_AFTER_LOAD_S,
            settings.RETRY_AFTER_UNLOAD_S,
            settings.RETRY_AFTER_NOT_LOADED_S,
            settings.RETRY_AFTER_READINESS_S,
        }
        assert settings.RETRY_AFTER_FALLBACK_S not in own

    def test_model_busy_on_a_load_is_the_load_value(self):
        exc = ModelBusyError("loading", details={"loading_model_id": 2})
        assert retry_after_for(exc.code, exc.details) == 15

    def test_model_busy_on_an_unload_is_the_unload_value(self):
        exc = ModelBusyError("unloading", details={"loading_model_id": 2, "unloading_model_id": 1})
        assert retry_after_for(exc.code, exc.details) == 5

    def test_the_inference_unloading_mark_is_an_unload(self):
        assert retry_after_for("MODEL_BUSY", {"model_id": 1, "unloading": True}) == 5

    def test_model_loading_follows_the_same_rule(self):
        assert retry_after_for("MODEL_LOADING", {"loading_model_id": 2}) == 15
        assert retry_after_for("model_loading", {"unloading_model_id": 1}) == 5

    def test_not_loaded_memory_readiness(self):
        assert retry_after_for("MODEL_NOT_LOADED") == 30
        assert retry_after_for("INSUFFICIENT_MEMORY") == 30
        assert retry_after_for("READINESS") == 5

    def test_unknown_code_is_the_fallback(self):
        assert retry_after_for("SOMETHING_ELSE") == 10
        assert retry_after_for(None) == 10

    @pytest.mark.parametrize(
        "code", ["QUEUE_FULL", "MODEL_BUSY", "MODEL_LOADING", "MODEL_NOT_LOADED",
                 "INSUFFICIENT_MEMORY", "HUB_UNAVAILABLE", "READINESS", "X"],
    )
    def test_every_value_is_a_whole_number_at_least_one(self, code):
        value = retry_after_for(code, {})
        assert isinstance(value, int) and value >= 1


class TestQueueFull:
    def test_default_without_an_estimate(self, patch_inference):
        patch_inference(_Queue(median=None, pending=10, holding=1))
        assert retry_after_for("QUEUE_FULL") == 5

    def test_estimate_rounded_up(self, patch_inference):
        # median 2.2 s × (9 waiting + 1 holding) / 1 = 22.0 → 22
        patch_inference(_Queue(median=2.2, pending=10, holding=1))
        assert retry_after_for("QUEUE_FULL") == 22

    def test_estimate_clamped_at_the_maximum(self, patch_inference):
        patch_inference(_Queue(median=30.0, pending=10, holding=1))
        assert retry_after_for("QUEUE_FULL") == 60

    def test_small_estimate_is_at_least_one(self, patch_inference):
        patch_inference(_Queue(median=0.01, pending=1, holding=1))
        assert retry_after_for("QUEUE_FULL") == 1

    def test_cbm_running_gives_the_default(self, patch_inference):
        patch_inference(_Queue(median=2.0, pending=10, holding=1), cbm=True)
        assert retry_after_for("QUEUE_FULL") == 5

    def test_a_failed_lookup_still_answers(self, monkeypatch):
        def boom():
            raise RuntimeError("no service")

        monkeypatch.setattr("millm.api.dependencies.get_inference_service", boom)
        assert retry_after_for("QUEUE_FULL") == 5


class TestHubBreakerRemainder:
    def test_open_breaker_remaining_time(self):
        from millm.core.resilience import CircuitState
        from millm.services.cluster_hub_service import cluster_hub_circuit

        state = cluster_hub_circuit.state
        saved = (state.state, state.last_failure_time)
        try:
            state.state = CircuitState.OPEN
            state.last_failure_time = 1000.0
            # recovery 60 s, failed at 1000, now 1017.5 → 42.5 left → 43
            assert backpressure._hub_seconds(now=lambda: 1017.5) == 43
            # past the recovery time it is still at least 1
            assert backpressure._hub_seconds(now=lambda: 2000.0) == 1
        finally:
            state.state, state.last_failure_time = saved

    def test_closed_breaker_is_one(self):
        from millm.core.resilience import CircuitState
        from millm.services.cluster_hub_service import cluster_hub_circuit

        assert cluster_hub_circuit.state.state == CircuitState.CLOSED
        assert retry_after_for("HUB_UNAVAILABLE") == 1


class TestEstimate:
    def test_formula_with_disagreeing_counts(self):
        # pending 5 (3 waiting + 2 holding), 1 background chunk, max_concurrent 2, median 4:
        # 4 × (3 + 2 + 1) / 2 = 12
        q = _Queue(median=4.0, pending=5, holding=2, background=1, max_concurrent=2)
        assert estimate_wait_seconds(q, cbm_running=False) == 12.0

    def test_none_below_three_samples(self):
        assert estimate_wait_seconds(_Queue(median=None, pending=3, holding=1), False) is None

    def test_none_while_cbm_runs(self):
        assert estimate_wait_seconds(_Queue(median=4.0, pending=3, holding=1), True) is None


class TestBacklogProvider:
    def teardown_method(self):
        register_backlog_provider(None)

    def test_unregistered_is_none_not_zero(self):
        register_backlog_provider(None)
        assert backlog_rows() is None

    def test_registered_value(self):
        register_backlog_provider(lambda: 1234)
        assert backlog_rows() == 1234

    def test_a_raising_provider_is_none(self):
        def boom():
            raise RuntimeError("runner gone")

        register_backlog_provider(boom)
        assert backlog_rows() is None


class TestLeaseErrors:
    def test_model_leased_is_not_model_locked(self):
        assert not issubclass(ModelLeasedError, ModelLockedError)

    def test_v1_code_is_model_leased(self):
        exc = ModelLeasedError.for_lease(
            model_id=1, model_name="m1", holder="midataworks", reason="run 7",
            expires_at="2026-10-06T12:00:00+00:00", operation="auto_load", target_model_id=2,
        )
        assert exc.code.lower() == "model_leased"
        assert ERROR_STATUS_MAP[exc.code] == (409, "invalid_request_error")
        assert exc.details == {
            "holder": "midataworks", "reason": "run 7",
            "expires_at": "2026-10-06T12:00:00+00:00", "leased_model_id": 1,
            "leased_model_name": "m1", "operation": "auto_load", "target_model_id": 2,
        }
        assert "midataworks" in exc.message and "2026-10-06T12:00:00+00:00" in exc.message

    @pytest.mark.parametrize("cls, status", [
        (ModelLeasedError, 409), (ModelNotResidentError, 409), (LeaseNotFoundError, 404),
        (LeaseExpiredError, 409), (InvalidLeaseRequestError, 400),
    ])
    def test_status_and_map_row(self, cls, status):
        assert cls.status_code == status
        assert ERROR_STATUS_MAP[cls.code] == (status, "invalid_request_error")


class TestLeaseSettingsValidator:
    def test_defaults(self):
        assert settings.LEASE_DEFAULT_TTL_SECONDS == 7200
        assert settings.LEASE_MAX_TTL_SECONDS == 7200

    def test_default_above_maximum_is_refused(self):
        with pytest.raises(ValueError, match="LEASE_DEFAULT_TTL_SECONDS"):
            Settings(LEASE_DEFAULT_TTL_SECONDS=7201, LEASE_MAX_TTL_SECONDS=7200)

    def test_default_below_maximum_is_accepted(self):
        assert Settings(LEASE_DEFAULT_TTL_SECONDS=600).LEASE_DEFAULT_TTL_SECONDS == 600


class TestModelNotResidentBuilder:
    def test_names_requested_and_resident_and_the_lease(self):
        import json

        from millm.api.routes.openai.errors import model_not_resident_error

        resp = model_not_resident_error(
            "m2", "m1",
            {"holder": "midataworks", "expires_at": "2026-10-06T12:00:00+00:00", "reason": "r"},
        )
        body = json.loads(resp.body)["error"]
        assert resp.status_code == 409
        assert body["code"] == "model_not_resident"
        assert body["type"] == "invalid_request_error"
        assert "'m2'" in body["message"] and "'m1'" in body["message"]
        assert "midataworks" in body["message"]

    def test_nothing_resident_says_none(self):
        import json

        from millm.api.routes.openai.errors import model_not_resident_error

        body = json.loads(model_not_resident_error("m2", None).body)["error"]
        assert "'none'" in body["message"]
