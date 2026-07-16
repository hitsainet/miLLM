"""
Feature 10 (OWUI Cluster Dial) unit tests: _resolve_intensity resolution
matrix, _apply_request_steering dial semantics (request-lambda overrides
stored lambda, lambda=0 disable, live-values base, no-op guards), and clamp
parity with the shared steering-range helper.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.core.errors import InvalidFeatureIndexError
from millm.services.inference_service import InferenceService


def make_sae(d_sae=16384, values=None, enabled=False):
    sae = MagicMock()
    sae.d_sae = d_sae
    sae.get_steering_values.return_value = dict(values or {})
    sae.is_steering_enabled = enabled
    return sae


def make_profile(steering=None, intensity=1.0, source_kind="manual",
                 cluster_meta=None, name="p"):
    profile = MagicMock()
    profile.steering = steering
    profile.intensity = intensity
    profile.source_kind = source_kind
    profile.cluster_meta = cluster_meta
    profile.name = name
    return profile


@pytest.fixture
def service():
    return InferenceService(model_service=MagicMock())


def apply_ctx(sae, by_name=None, active=None):
    """Patch the SAE state + profile repository the apply path reads."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch("millm.services.sae_service.AttachedSAEState") as MockState, \
             patch("millm.db.base.async_session_factory"), \
             patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo:
            MockState.return_value.attached_sae = sae
            MockRepo.return_value.get_by_name = AsyncMock(return_value=by_name)
            MockRepo.return_value.get_active = AsyncMock(return_value=active)
            yield MockRepo.return_value

    return _ctx()


class TestResolveIntensity:
    """Symbolic-to-numeric resolution matrix (Task 2.1)."""

    def test_none_passes_through(self):
        assert InferenceService._resolve_intensity(None, None) is None

    def test_numeric_passthrough(self):
        assert InferenceService._resolve_intensity(1.3, None) == 1.3
        assert InferenceService._resolve_intensity(0, None) == 0.0

    def test_off_is_zero_regardless_of_range(self):
        profile = make_profile(cluster_meta={"budget": {"intensity_range": [0.7, 1.4]}})
        assert InferenceService._resolve_intensity("off", profile) == 0.0

    def test_min_max_from_cluster_intensity_range(self):
        profile = make_profile(cluster_meta={"budget": {"intensity_range": [0.7, 1.4]}})
        assert InferenceService._resolve_intensity("min", profile) == 0.7
        assert InferenceService._resolve_intensity("max", profile) == 1.4

    def test_config_fallback_without_profile(self):
        from millm.core.config import settings

        assert InferenceService._resolve_intensity("min", None) == settings.CLUSTER_INTENSITY_MIN
        assert InferenceService._resolve_intensity("max", None) == settings.CLUSTER_INTENSITY_MAX

    def test_config_fallback_with_malformed_range(self):
        from millm.core.config import settings

        profile = make_profile(cluster_meta={"budget": {"intensity_range": [1.0]}})
        assert InferenceService._resolve_intensity("max", profile) == settings.CLUSTER_INTENSITY_MAX


class TestDialSemantics:
    """_apply_request_steering with the dial (Tasks 2.2/2.6/2.7)."""

    async def test_request_lambda_overrides_stored_lambda(self, service):
        """Pitfall 1: absolute override, never a multiplier."""
        sae = make_sae()
        profile = make_profile(steering={"10": 100.0}, intensity=0.5)
        with apply_ctx(sae, by_name=profile):
            saved = await service._apply_request_steering("p", 2.0)
        assert saved is not None
        # 100 * 2.0 = 200 (NOT 100 * 0.5 * 2.0 = 100)
        sae.set_steering_batch.assert_called_once_with({10: 200.0})

    async def test_absent_dial_falls_back_to_stored_lambda(self, service):
        sae = make_sae()
        profile = make_profile(steering={"10": 100.0}, intensity=0.5)
        with apply_ctx(sae, by_name=profile):
            await service._apply_request_steering("p", None)
        sae.set_steering_batch.assert_called_once_with({10: 50.0})

    async def test_lambda_zero_disables_steering_for_request(self, service):
        sae = make_sae(values={5: 3.0}, enabled=True)
        profile = make_profile(steering={"10": 100.0})
        with apply_ctx(sae, by_name=profile):
            saved = await service._apply_request_steering("p", 0.0)
        assert saved == {"values": {5: 3.0}, "enabled": True}
        sae.enable_steering.assert_called_once_with(False)
        sae.set_steering_batch.assert_not_called()

    async def test_symbolic_off_via_apply(self, service):
        sae = make_sae(values={5: 3.0}, enabled=True)
        profile = make_profile(steering={"10": 100.0})
        with apply_ctx(sae, by_name=profile):
            await service._apply_request_steering("p", "off")
        sae.enable_steering.assert_called_once_with(False)

    async def test_dial_only_uses_active_profile_as_base(self, service):
        # enabled=True: an active profile implies steering is on; dial-only
        # requests refuse to enable disabled steering (R1 fix, tested below)
        sae = make_sae(enabled=True)
        active = make_profile(
            steering={"7": 50.0}, intensity=1.0, source_kind="cluster",
            cluster_meta={"budget": {"intensity_range": [0.5, 1.5]}},
        )
        with apply_ctx(sae, active=active) as repo:
            await service._apply_request_steering(None, "max")
            repo.get_by_name.assert_not_called()
        sae.set_steering_batch.assert_called_once_with({7: 75.0})
        sae.enable_steering.assert_called_with(True)

    async def test_dial_only_no_profile_scales_live_values(self, service):
        """Pitfall 3: live values become the lambda=1 base."""
        sae = make_sae(values={3: 10.0}, enabled=True)
        with apply_ctx(sae, active=None):
            saved = await service._apply_request_steering(None, 1.5)
        assert saved == {"values": {3: 10.0}, "enabled": True}
        sae.set_steering_batch.assert_called_once_with({3: 15.0})

    async def test_dial_never_enables_unconfigured_steering(self, service):
        """Pitfall 3: steering disabled + dial>0 is a no-op, not an enable."""
        sae = make_sae(values={3: 10.0}, enabled=False)
        with apply_ctx(sae, active=None):
            result = await service._apply_request_steering(None, 1.5)
        assert result is None
        sae.set_steering_batch.assert_not_called()
        sae.enable_steering.assert_not_called()

    async def test_dial_zero_with_nothing_running_is_noop(self, service):
        sae = make_sae(values={}, enabled=False)
        with apply_ctx(sae, active=None):
            result = await service._apply_request_steering(None, 0.0)
        assert result is None
        sae.enable_steering.assert_not_called()

    async def test_dial_zero_enabled_but_empty_honors_off(self, service):
        sae = make_sae(values={}, enabled=True)
        with apply_ctx(sae, active=None):
            saved = await service._apply_request_steering(None, 0.0)
        assert saved == {"values": {}, "enabled": True}
        sae.enable_steering.assert_called_once_with(False)

    async def test_dial_clamps_via_shared_helper(self, service):
        """Clamp parity: base * lambda beyond +-200 clamps, never rejects."""
        sae = make_sae(values={3: 150.0}, enabled=True)
        with apply_ctx(sae, active=None):
            await service._apply_request_steering(None, 2.0)
        sae.set_steering_batch.assert_called_once_with({3: 200.0})

    async def test_cluster_gate_applies_to_dialed_active_profile(self, service):
        """Reapplying the active cluster at a new lambda still runs the
        declared-feature-space gate."""
        sae = make_sae(d_sae=100)
        active = make_profile(
            steering={"7": 50.0}, source_kind="cluster",
            cluster_meta={"sae": {"n_features": 16384}},
        )
        with apply_ctx(sae, active=active):
            with pytest.raises(InvalidFeatureIndexError):
                await service._apply_request_steering(None, 1.0)
        sae.set_steering_batch.assert_not_called()

    async def test_no_sae_returns_none_with_dial(self, service):
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = None
            result = await service._apply_request_steering(None, 1.5)
        assert result is None


class TestRoutingCondition:
    """Dialed requests must never reach CBM (Task 2.4)."""

    def test_dial_forces_serial(self, service):
        backend = MagicMock()
        backend.is_running = True
        backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_backend = backend
        service._cbm_force_serial_monitoring = False

        assert service._use_cbm_for_request(
            temperature=None, top_p=None, has_steering_override=True
        ) is False
        assert service._use_cbm_for_request(
            temperature=None, top_p=None, has_steering_override=False
        ) is True


class TestResolveRequestIntensityEcho:
    """Route-level resolution for the X-miLLM-Steering-Intensity echo —
    best-effort and honest: None (no header) when nothing can apply."""

    async def test_numeric_fast_path_no_db(self, service):
        request = MagicMock(steering_intensity=1.3, profile=None)
        with apply_ctx(make_sae()), \
             patch("millm.db.base.async_session_factory") as factory:
            assert await service.resolve_request_intensity(request) == 1.3
            factory.assert_not_called()

    async def test_absent_field_is_none(self, service):
        request = MagicMock(steering_intensity=None)
        assert await service.resolve_request_intensity(request) is None

    async def test_symbolic_resolves_against_active(self, service):
        request = MagicMock(steering_intensity="max", profile=None)
        active = make_profile(cluster_meta={"budget": {"intensity_range": [0.5, 1.5]}})
        with apply_ctx(make_sae(), active=active):
            assert await service.resolve_request_intensity(request) == 1.5

    async def test_symbolic_resolves_against_named_profile(self, service):
        request = MagicMock(steering_intensity="min", profile="p")
        named = make_profile(cluster_meta={"budget": {"intensity_range": [0.7, 1.4]}})
        with apply_ctx(make_sae(), by_name=named):
            assert await service.resolve_request_intensity(request) == 0.7

    async def test_no_sae_suppresses_echo(self, service):
        """R1 fix: apply will no-op — an echoed lambda would be a lie."""
        request = MagicMock(steering_intensity=1.5, profile=None)
        with apply_ctx(None):
            assert await service.resolve_request_intensity(request) is None

    async def test_missing_named_profile_suppresses_echo(self, service):
        """R1 fix: apply will 404 — don't emit a confident header first."""
        request = MagicMock(steering_intensity="max", profile="ghost")
        with apply_ctx(make_sae(), by_name=None):
            assert await service.resolve_request_intensity(request) is None

    async def test_db_failure_degrades_to_no_header(self, service):
        """R1 fix: a symbolic echo must never 500 the whole request."""
        request = MagicMock(steering_intensity="max", profile=None)
        with patch("millm.services.sae_service.AttachedSAEState") as MockState, \
             patch("millm.db.base.async_session_factory",
                   side_effect=RuntimeError("db down")):
            MockState.return_value.attached_sae = make_sae()
            assert await service.resolve_request_intensity(request) is None


class TestReviewRound1Fixes:
    """Pins for the R1 findings fixed in _apply_request_steering."""

    async def test_named_empty_profile_with_dial_is_noop_not_live_scaling(
        self, service
    ):
        """R1: profile-with-no-steering + dial must NOT fall through to
        scaling the active cluster's live values."""
        sae = make_sae(values={3: 10.0}, enabled=True)
        profile = make_profile(steering=None, name="neutral")
        with apply_ctx(sae, by_name=profile):
            result = await service._apply_request_steering("neutral", 1.5)
        assert result is None
        sae.set_steering_batch.assert_not_called()

    async def test_empty_cluster_profile_still_gated(self, service):
        """R1: the n_features gate runs even when the cluster's steering is
        empty (pre-010 ordering restored)."""
        sae = make_sae(d_sae=100)
        profile = make_profile(
            steering=None, source_kind="cluster", name="hollow",
            cluster_meta={"sae": {"n_features": 32768}},
        )
        with apply_ctx(sae, by_name=profile):
            with pytest.raises(InvalidFeatureIndexError):
                await service._apply_request_steering("hollow", 1.2)

    async def test_dial_only_never_reenables_disabled_active_profile(
        self, service
    ):
        """R1: operator disabled steering globally; a dial>0 on the active
        profile must not switch it back on."""
        sae = make_sae(values={7: 50.0}, enabled=False)
        active = make_profile(steering={"7": 50.0})
        with apply_ctx(sae, active=active):
            result = await service._apply_request_steering(None, 1.5)
        assert result is None
        sae.enable_steering.assert_not_called()
        sae.set_steering_batch.assert_not_called()

    async def test_named_profile_may_still_enable(self, service):
        """Explicitly naming a profile keeps pre-010 enable semantics."""
        sae = make_sae(values={}, enabled=False)
        profile = make_profile(steering={"7": 50.0})
        with apply_ctx(sae, by_name=profile):
            saved = await service._apply_request_steering("p", None)
        assert saved == {"values": {}, "enabled": False}
        sae.enable_steering.assert_called_with(True)

    async def test_numeric_dial_clamped_to_authored_range(self, service):
        """R1: /v1 is unauthenticated — a numeric lambda must not overdrive
        past the cluster's declared intensity_range."""
        sae = make_sae()
        profile = make_profile(
            steering={"10": 100.0}, source_kind="cluster",
            cluster_meta={"budget": {"intensity_range": [0.5, 1.2]}},
        )
        with apply_ctx(sae, by_name=profile):
            await service._apply_request_steering("p", 2.0)
        sae.set_steering_batch.assert_called_once_with({10: 120.0})

    async def test_dial_to_zero_bypasses_range_floor(self, service):
        """Dialing to 0 stays always allowed (set_intensity parity)."""
        sae = make_sae(values={10: 100.0}, enabled=True)
        profile = make_profile(
            steering={"10": 100.0}, source_kind="cluster",
            cluster_meta={"budget": {"intensity_range": [0.5, 1.2]}},
        )
        with apply_ctx(sae, by_name=profile):
            saved = await service._apply_request_steering("p", 0.0)
        assert saved is not None
        sae.enable_steering.assert_called_once_with(False)

    async def test_swapped_authored_range_is_normalized(self, service):
        """R1: a hand-authored [hi, lo] range must not invert min/max."""
        profile = make_profile(
            cluster_meta={"budget": {"intensity_range": [1.5, 0.5]}})
        assert InferenceService._resolve_intensity("min", profile) == 0.5
        assert InferenceService._resolve_intensity("max", profile) == 1.5

    async def test_garbage_authored_range_falls_back_to_config(self, service):
        from millm.core.config import settings

        profile = make_profile(
            cluster_meta={"budget": {"intensity_range": [None, "x"]}})
        assert (InferenceService._resolve_intensity("max", profile)
                == settings.CLUSTER_INTENSITY_MAX)

    async def test_streaming_setup_failure_restores_steering(self, service):
        """R1 top finding: an exception between apply and the streaming
        try/finally (tokenize/context-check/thread-start) must restore."""
        import torch as _torch  # noqa: F401

        sae = make_sae(values={5: 3.0}, enabled=True)
        request = MagicMock()
        request.profile = None
        request.steering_intensity = 0.0
        request.messages = [MagicMock(role="user", content="hi")]
        request.temperature = None
        request.top_p = None

        restored = []
        original_restore = service._restore_request_profile

        def spy_restore(saved):
            restored.append(saved)
            original_restore(saved)

        with apply_ctx(sae, active=None), \
             patch.object(service, "_restore_request_profile",
                          side_effect=spy_restore), \
             patch.object(service, "_format_chat_messages", return_value="hi"), \
             patch.object(type(service), "_tokenizer",
                          property(lambda self: (_ for _ in ()).throw(
                              RuntimeError("tokenizer exploded")))):
            with pytest.raises(RuntimeError, match="tokenizer exploded"):
                async for _ in service.stream_chat_completion(request):
                    pass
        assert restored == [{"values": {5: 3.0}, "enabled": True}]

    async def test_call_site_passes_dial_into_routing(self, service):
        """R1: pin the actual call-site expression — a dial-only request must
        reach _use_cbm_for_request with has_steering_override=True."""
        assert service._has_steering_override(
            MagicMock(profile=None, steering_intensity=1.5)) is True
        assert service._has_steering_override(
            MagicMock(profile="p", steering_intensity=None)) is True
        assert service._has_steering_override(
            MagicMock(profile=None, steering_intensity=None)) is False
        assert service._has_steering_override(object()) is False  # no fields

        seen = {}
        original = service._use_cbm_for_request

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        request = MagicMock(profile=None, steering_intensity="off",
                            temperature=None, top_p=None)
        request.messages = [MagicMock(role="user", content="hi")]
        with patch.object(service, "_use_cbm_for_request", side_effect=spy), \
             patch.object(service, "_format_chat_messages", return_value="hi"):
            gen = service.stream_chat_completion(request)
            try:
                await gen.__anext__()
            except Exception:
                pass
            await gen.aclose()
        assert seen.get("has_steering_override") is True


class TestDialConcurrency:
    """Task 4.2: interleaved dialed requests serialize on the request queue —
    each apply sees its own lambda and every restore returns the SAE to the
    pre-request state."""

    async def test_two_dials_apply_independently_and_restore_cleanly(self, service):
        import asyncio

        # Real steering state (dict-backed fake SAE) shared across "requests".
        state = {"values": {3: 10.0}, "enabled": True}
        sae = MagicMock()
        sae.d_sae = 16384
        sae.get_steering_values.side_effect = lambda: dict(state["values"])
        sae.set_steering_batch.side_effect = lambda v: state.update(values=dict(v))
        sae.enable_steering.side_effect = lambda e: state.update(enabled=e)
        sae.clear_steering.side_effect = lambda: state.update(values={})
        type(sae).is_steering_enabled = property(lambda self: state["enabled"])

        observed = []

        async def one_request(lam):
            async with service._request_queue.acquire():
                with apply_ctx(sae, active=None):
                    saved = await service._apply_request_steering(None, lam)
                    observed.append((lam, dict(state["values"]), state["enabled"]))
                    await asyncio.sleep(0)  # yield inside the critical section
                    service._restore_request_profile(saved)

        await asyncio.gather(one_request(2.0), one_request(0.0))

        # Each request saw ONLY its own dial while holding the semaphore.
        for lam, values, enabled in observed:
            if lam == 2.0:
                assert values == {3: 20.0} and enabled is True
            else:
                assert enabled is False
        # And the global state came back exactly as it started.
        assert state == {"values": {3: 10.0}, "enabled": True}
