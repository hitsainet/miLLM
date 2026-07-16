"""
Feature 11 Task 3.8: SensingService unit tests — config build from
cluster_meta, context-window slicing edges, summary builder, ambient rules,
and the arm/disarm lifecycle sync.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from millm.core.config import settings
from millm.ml.sae_wrapper import SensedHit
from millm.services.sensing_service import SensingService


def make_profile(members=None, sensing=None, name="fear cluster",
                 profile_id="prof_s1", display_token="fear"):
    profile = MagicMock()
    profile.id = profile_id
    profile.name = name
    meta = {
        "display_token": display_token,
        "members": members if members is not None else [
            {"feature_idx": 7, "strength": 1.0, "max_activation": 40.0,
             "label": "fear of drowning"},
            {"feature_idx": 9, "strength": -0.5, "max_activation": 20.0},
            {"feature_idx": 12, "strength": 0.8},  # no max_activation
        ],
    }
    if sensing is not None:
        meta["sensing"] = sensing
    profile.cluster_meta = meta
    profile.source_kind = "cluster"
    profile.sensing_enabled = True
    return profile


def make_hit(pos_start=5, pos_end=6, phase="decode",
             fired=((7, 8.4), (9, 2.2)), score=2.1):
    return SensedHit(pos_start=pos_start, pos_end=pos_end, phase=phase,
                     fired=list(fired), fired_count=len(fired), score=score)


@pytest.fixture
def service():
    return SensingService()


class TestConfigBuild:
    def test_thresholds_epsilon_max_with_inf_fallback(self, service):
        config = service.build_config(make_profile())
        assert config.member_indices == [7, 9, 12]
        # theta = max(floor, 0.1 * max_act); member 12 missing + zero floor
        # -> INFINITE threshold (R1 fix: floor 0 fired on any positive act)
        assert config.thresholds.tolist()[:2] == pytest.approx([4.0, 2.0])
        assert config.thresholds.tolist()[2] == float("inf")
        assert config.threshold_mode == "epsilon_max"
        assert config.min_k == 2  # max(2, ceil(0.3*3)) = 2

    def test_floor_only_mode_needs_positive_floor(self, service):
        members = [{"feature_idx": 1, "strength": 1.0},
                   {"feature_idx": 2, "strength": 1.0}]
        # All members missing max_activation + default zero floor: refuse
        # to arm rather than fire on every positive activation (R1 fix).
        with pytest.raises(ValueError, match="no usable"):
            service.build_config(make_profile(members=members))
        # With a positive authored floor, floor_only mode works.
        config = service.build_config(make_profile(
            members=members, sensing={"theta_floor": 0.5}))
        assert config.threshold_mode == "floor_only"
        assert config.thresholds.tolist() == [0.5, 0.5]

    def test_document_overrides(self, service):
        config = service.build_config(make_profile(
            sensing={"epsilon": 0.5, "min_k": 3, "context_tokens": 8}))
        assert config.thresholds[0].item() == pytest.approx(20.0)  # 0.5*40
        assert config.min_k == 3
        assert config.context_tokens == 8

    def test_hostile_overrides_degrade_to_defaults(self, service):
        config = service.build_config(make_profile(
            sensing={"epsilon": "loud", "min_k": "many",
                     "context_tokens": 9999}))
        assert config.thresholds[0].item() == pytest.approx(
            settings.SENSING_EPSILON * 40.0)
        assert config.min_k == 2
        assert config.context_tokens == 64  # hard max

    def test_min_k_clamped_to_member_count(self, service):
        members = [{"feature_idx": 1, "strength": 1.0, "max_activation": 5.0}]
        config = service.build_config(make_profile(
            members=members, sensing={"min_k": 5}))
        assert config.min_k == 1

    def test_empty_members_raises(self, service):
        with pytest.raises(ValueError):
            service.build_config(make_profile(members=[]))


class TestContextSlicing:
    def setup_method(self):
        self.tokenizer = MagicMock()
        self.tokenizer.decode.side_effect = (
            lambda ids, **kw: " ".join(f"t{i}" for i in ids))

    def test_window_around_span(self, service):
        ids = torch.arange(100).unsqueeze(0)  # (1, 100)
        text, window = SensingService._context(
            ids, make_hit(pos_start=50, pos_end=52), k=3, tokenizer=self.tokenizer)
        assert window == list(range(47, 56))
        assert text.startswith("t47")

    def test_event_at_position_zero(self, service):
        ids = torch.arange(10).unsqueeze(0)
        _, window = SensingService._context(
            ids, make_hit(pos_start=0, pos_end=0), k=4, tokenizer=self.tokenizer)
        assert window == [0, 1, 2, 3, 4]

    def test_event_at_end_of_sequence(self, service):
        ids = torch.arange(10)  # 1-D also accepted
        _, window = SensingService._context(
            ids, make_hit(pos_start=9, pos_end=9), k=4, tokenizer=self.tokenizer)
        assert window == [5, 6, 7, 8, 9]

    def test_k_zero_keeps_event_without_text(self, service):
        text, window = SensingService._context(
            torch.arange(10), make_hit(), k=0, tokenizer=self.tokenizer)
        assert text is None and window is None

    def test_missing_ids_degrades(self, service):
        text, window = SensingService._context(
            None, make_hit(), k=4, tokenizer=self.tokenizer)
        assert text is None and window is None

    def test_decode_failure_degrades(self, service):
        self.tokenizer.decode.side_effect = RuntimeError("boom")
        text, window = SensingService._context(
            torch.arange(10), make_hit(pos_start=2, pos_end=2), k=2,
            tokenizer=self.tokenizer)
        assert text is None and window is None


class TestSummary:
    def test_format_and_label(self, service):
        sae = MagicMock()
        sae.d_sae = 16384
        service.arm_for_profile(make_profile(), sae)
        summary = service._summary(make_hit())
        assert summary.startswith("fear: 2/3 members fired")
        assert "F7 'fear of drowning'" in summary
        assert "during decode @ 5–6" in summary

    def test_single_position_span(self, service):
        service.arm_for_profile(make_profile(), MagicMock(d_sae=16384))
        assert "@ 5" in service._summary(make_hit(pos_end=5))

    def test_hard_cap_300(self, service):
        profile = make_profile(display_token="x" * 400)
        service.arm_for_profile(profile, MagicMock(d_sae=16384))
        assert len(service._summary(make_hit())) <= 300


class TestLifecycle:
    def test_arm_and_disarm_track_state(self, service):
        sae = MagicMock()
        sae.d_sae = 16384
        service.arm_for_profile(make_profile(), sae)
        assert service.is_armed and service.armed_profile_id == "prof_s1"
        sae.arm_sensing.assert_called_once()
        service.disarm(sae)
        assert not service.is_armed
        sae.disarm_sensing.assert_called_once()

    def test_disarm_with_detached_sae(self, service):
        service.arm_for_profile(make_profile(), MagicMock(d_sae=16384))
        service.disarm(None)  # SAE already gone — must not raise
        assert not service.is_armed

    def test_overhead_warn_threshold(self, service):
        with patch("millm.services.sensing_service.logger") as mock_log:
            service.note_request_overhead(settings.SENSING_MAX_OVERHEAD_MS + 1)
        mock_log.warning.assert_called_once()
        assert service.status()["last_request_overhead_ms"] > 0

    def test_status_shape(self, service):
        status = service.status()
        assert status["armed"] is False
        sae = MagicMock(d_sae=16384)
        sae.is_sensing_armed = True
        service.arm_for_profile(make_profile(), sae)
        # status() reconciles against the attached SAE (R3 fix)
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            status = service.status()
        assert status["member_count"] == 3
        assert status["threshold_mode"] == "epsilon_max"
        assert status["retention"]["max_events_per_cluster"] == \
            settings.SENSING_MAX_EVENTS_PER_CLUSTER


class TestRecord:
    async def test_record_persists_prunes_and_emits(self, service, test_session):
        from millm.db.models.profile import Profile

        row = Profile(id="prof_s1", name="fear cluster", steering={"7": 1.0},
                      source_kind="cluster")
        test_session.add(row)
        await test_session.flush()

        service.arm_for_profile(make_profile(), MagicMock(d_sae=16384))
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: "ctx"

        emitted = []
        factory = MagicMock(return_value=test_session)
        test_session.commit = AsyncMock()

        class _Ctx:
            async def __aenter__(self):
                return test_session

            async def __aexit__(self, *a):
                return False

        with patch("millm.db.base.async_session_factory",
                   return_value=_Ctx()), \
             patch.object(service, "_emit_events",
                          side_effect=lambda p: emitted.extend(p)):
            payloads = await service.record(
                "req-1", [make_hit()], False,
                torch.arange(20).unsqueeze(0), tokenizer)

        assert len(payloads) == 1
        assert payloads[0]["context_text"] == "ctx"
        assert payloads[0]["fired_members"] == [[7, 8.4], [9, 2.2]]
        assert emitted and emitted[0]["summary"].startswith("fear:")

    async def test_record_noop_when_unarmed_or_empty(self, service):
        assert await service.record("req-1", [], False, None, None) == []
        assert await service.record("req-1", [make_hit()], False, None, None) == []

    def test_ws_payload_excludes_context(self, service):
        """FTID pitfall 6: WS payloads carry no user content."""
        sent = []
        emitter = MagicMock()
        emitter.emit_sensing_event.side_effect = lambda p: sent.append(p)
        with patch("millm.sockets.progress.progress_emitter", emitter):
            service._emit_events([{"id": 1, "summary": "s",
                                   "context_text": "SECRET",
                                   "context_token_ids": [1, 2]}])
        assert sent == [{"id": 1, "summary": "s"}]


class TestInferenceWiring:
    """011 R1 (finder B): the actual begin/flush wiring in InferenceService
    was untested for every generation path."""

    def _service(self):
        from millm.services.inference_service import InferenceService

        return InferenceService(model_service=MagicMock())

    def _armed_sae(self):
        sae = MagicMock()
        sae.is_sensing_armed = True
        sae._sensing = MagicMock(profile_id="prof_s1")
        sae._sensing_overhead_ms = 0.5
        sae.collect_sensing_hits.return_value = ("req-1", [make_hit()], False)
        return sae

    def test_begin_snapshots_profile_id(self):
        service = self._service()
        sae = self._armed_sae()
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            ctx = service._sensing_begin("req-1")
        assert ctx == (sae, "prof_s1")
        sae.begin_sensing_request.assert_called_once_with("req-1")

    def test_begin_skips_under_speculative_decoding(self):
        """R1 fix: verification passes break absolute-position accounting."""
        service = self._service()
        service._speculative_model_id = "draft-model"
        sae = self._armed_sae()
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            assert service._sensing_begin("req-1") is None
        sae.begin_sensing_request.assert_not_called()

    def test_begin_none_when_unarmed(self):
        service = self._service()
        sae = MagicMock()
        sae.is_sensing_armed = False
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            assert service._sensing_begin("req-1") is None

    async def test_notify_records_with_snapshotted_profile(self):
        service = self._service()
        sae = self._armed_sae()
        sensing_service = MagicMock()
        sensing_service.record = AsyncMock(return_value=[])
        ids = torch.arange(10)
        with patch("millm.api.dependencies._sensing_service", sensing_service), \
             patch.object(type(service), "is_model_loaded",
                          lambda self: False):
            await service._notify_sensing((sae, "prof_SNAPSHOT"), ids)
        kwargs = sensing_service.record.await_args.kwargs
        assert kwargs["profile_id"] == "prof_SNAPSHOT"
        sensing_service.note_request_overhead.assert_called_once_with(0.5)

    async def test_notify_survives_record_failure(self):
        """A DB outage in the flush must never break generation."""
        service = self._service()
        sae = self._armed_sae()
        sensing_service = MagicMock()
        sensing_service.record = AsyncMock(side_effect=RuntimeError("db down"))
        with patch("millm.api.dependencies._sensing_service", sensing_service), \
             patch.object(type(service), "is_model_loaded",
                          lambda self: False):
            await service._notify_sensing((sae, "p"), None)  # no raise

    async def test_notify_noop_without_ctx_or_service(self):
        service = self._service()
        await service._notify_sensing(None, None)
        sae = self._armed_sae()
        with patch("millm.api.dependencies._sensing_service", None):
            await service._notify_sensing((sae, "p"), None)
        sae.collect_sensing_hits.assert_called_once()  # boundary still closed


class TestAmbientCounts:
    """R1 (finder B): _ambient_counts rules were completely untested."""

    def _sae(self, monitoring=True, compacted=False, offset=7, acts=None):
        sae = MagicMock()
        sae.is_monitoring_enabled = monitoring
        sae._monitored_features = [1, 2] if compacted else None
        sae._sensing_token_offset = offset
        sae.get_feature_activations_for_item.return_value = acts
        return sae

    def test_counts_only_last_position_events(self):
        from millm.services.inference_service import InferenceService

        acts = torch.zeros(3, 16)
        acts[-1, :5] = 2.0  # 5 ambient features fired at the last position
        sae = self._sae(acts=acts)
        hits = [make_hit(pos_start=2, pos_end=6),   # includes last pos (6)
                make_hit(pos_start=1, pos_end=3)]   # doesn't
        counts = InferenceService._ambient_counts(sae, hits)
        assert counts == {0: 5}

    def test_none_when_monitoring_off_or_compacted(self):
        from millm.services.inference_service import InferenceService

        hits = [make_hit(pos_end=6)]
        assert InferenceService._ambient_counts(
            self._sae(monitoring=False), hits) is None
        assert InferenceService._ambient_counts(
            self._sae(compacted=True), hits) is None

    def test_none_when_no_capture(self):
        from millm.services.inference_service import InferenceService

        assert InferenceService._ambient_counts(
            self._sae(acts=None), [make_hit(pos_end=6)]) is None


class TestGenerationPathWiring:
    """R1 (finder B): begin/flush placement on the REAL generation paths."""

    def _service_with_model(self):
        from millm.services.inference_service import InferenceService

        service = InferenceService(model_service=MagicMock())
        return service

    async def test_nonstreaming_chat_begins_and_flushes(self):
        import torch as _torch
        from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage

        service = self._service_with_model()
        request = ChatCompletionRequest(
            model="m", messages=[ChatMessage(role="user", content="hi")])

        begin_calls = []
        notify_calls = []

        with patch.object(service, "_sensing_begin",
                          side_effect=lambda rid: begin_calls.append(rid) or
                          ("SAE", "prof")) as _, \
             patch.object(service, "_notify_sensing",
                          side_effect=lambda ctx, ids:
                          notify_calls.append((ctx, ids)) or _async_none()), \
             patch.object(service, "_format_chat_messages", return_value="hi"), \
             patch.object(service, "_generate_sync",
                          return_value=_torch.tensor([[1, 2, 3, 4]])), \
             patch.object(type(service), "_tokenizer",
                          property(lambda self: _make_tokenizer())), \
             patch.object(service, "_check_context_length"), \
             patch.object(service, "_build_generate_kwargs",
                          return_value={}), \
             patch.object(service, "get_loaded_model_info",
                          return_value=_model_info("m")):
            await service.create_chat_completion(request)

        assert len(begin_calls) == 1
        assert len(notify_calls) == 1
        ctx, full_ids = notify_calls[0]
        assert ctx == ("SAE", "prof")
        assert _torch.equal(full_ids, _torch.tensor([1, 2, 3, 4]))

    async def test_n_gt_1_goes_unsensed(self):
        import torch as _torch
        from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage

        service = self._service_with_model()
        request = ChatCompletionRequest(
            model="m", messages=[ChatMessage(role="user", content="hi")], n=2)

        with patch.object(service, "_sensing_begin") as begin, \
             patch.object(service, "_notify_sensing",
                          side_effect=lambda ctx, ids: _async_none()), \
             patch.object(service, "_format_chat_messages", return_value="hi"), \
             patch.object(service, "_generate_sync",
                          return_value=_torch.tensor([[1, 2, 3]])), \
             patch.object(type(service), "_tokenizer",
                          property(lambda self: _make_tokenizer())), \
             patch.object(service, "_check_context_length"), \
             patch.object(service, "_build_generate_kwargs", return_value={}), \
             patch.object(service, "get_loaded_model_info",
                          return_value=_model_info("m")):
            await service.create_chat_completion(request)
        begin.assert_not_called()

    def test_force_serial_false_leaves_cbm_eligible(self):
        """EC-11.3: with forcing off, armed requests stay CBM-eligible (and
        go unsensed there — begin only exists on the serial paths)."""
        from millm.core.config import settings
        from millm.services.inference_service import InferenceService

        service = InferenceService(model_service=MagicMock())
        backend = MagicMock()
        backend.is_running = True
        backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_backend = backend
        service._cbm_force_serial_monitoring = False

        armed_sae = MagicMock()
        armed_sae.is_sensing_armed = True
        with patch("millm.services.sae_service.AttachedSAEState") as MockState, \
             patch.object(settings, "SENSING_FORCE_SERIAL", False):
            MockState.return_value.attached_sae = armed_sae
            assert service._use_cbm_for_request(
                temperature=None, top_p=None, has_steering_override=False
            ) is True


def _async_none():
    import asyncio

    future = asyncio.get_event_loop().create_future()
    future.set_result(None)
    return future


def _make_tokenizer():
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.input_ids = torch.tensor([[1, 2]])
    encoded.to.return_value = encoded
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = "hello"
    tokenizer.eos_token_id = 0
    return tokenizer


def _model_info(name: str):
    info = MagicMock()
    info.name = name  # name= is a reserved MagicMock kwarg
    return info


class TestReviewRound3Pins:
    """011 R3: pins for the R2 fixes and named mutation survivors."""

    def test_status_reports_ws_dropped_and_first_flush_emits(self, service):
        assert service.status()["ws_events_dropped"] == 0
        # first flush must emit (throttle initialized to -inf, R2 fix 2)
        sent = []
        emitter = MagicMock()
        emitter.emit_sensing_event.side_effect = lambda p: sent.append(p)
        with patch("millm.sockets.progress.progress_emitter", emitter):
            service._emit_events([{"id": 1, "summary": "s"}])
        assert len(sent) == 1

    def test_throttle_drops_are_counted_and_observable(self, service):
        emitter = MagicMock()
        with patch("millm.sockets.progress.progress_emitter", emitter):
            service._emit_events([{"id": i, "summary": "s"} for i in range(8)])
            service._emit_events([{"id": 9, "summary": "s"}])  # within 100ms
        # 8 - _WS_MAX_PER_FLUSH(5) = 3 dropped from flush 1; flush 2 fully dropped
        assert service.status()["ws_events_dropped"] == 4
        assert emitter.emit_sensing_event.call_count == 5

    async def test_rearm_mismatch_does_not_stomp_new_cluster_state(
        self, service
    ):
        """R3 #1 (the regression R3 found): a snapshot flush after a re-arm
        must render neutrally WITHOUT mutating the armed cluster's state."""
        sae_a = MagicMock(d_sae=16384)
        service.arm_for_profile(make_profile(profile_id="prof_A"), sae_a)
        service.arm_for_profile(
            make_profile(profile_id="prof_B", display_token="anger",
                         name="anger cluster"),
            MagicMock(d_sae=16384))

        rows = {}

        class _Repo:
            def __init__(self, session):
                pass

            async def create_many(self, events):
                rows["events"] = events
                out = []
                for event in events:
                    row = MagicMock()
                    row.to_dict.return_value = dict(event, id=1)
                    out.append(row)
                return out

            async def prune(self, *a, **kw):
                return 0

        class _Ctx:
            async def __aenter__(self):
                session = MagicMock()

                async def _commit():
                    return None

                session.commit = _commit
                return session

            async def __aexit__(self, *a):
                return False

        with patch("millm.db.base.async_session_factory",
                   return_value=_Ctx()), \
             patch("millm.db.repositories.sensing_repository.SensingRepository",
                   _Repo), \
             patch.object(service, "_emit_events"):
            await service.record("req-1", [make_hit()], False, None, None,
                                 profile_id="prof_A")

        # A's rows render neutrally under the snapshot id...
        assert rows["events"][0]["summary"].startswith("prof_A:")
        # ...and B's armed formatting is untouched
        assert service._display_token == "anger"

    def test_zero_max_activation_treated_as_missing(self, service):
        """R3 #2: max_activation 0.0 must not produce theta=0."""
        members = [
            {"feature_idx": 1, "strength": 1.0, "max_activation": 0.0},
            {"feature_idx": 2, "strength": 1.0, "max_activation": 10.0},
        ]
        config = service.build_config(make_profile(members=members))
        assert config.thresholds.tolist()[0] == float("inf")

    def test_negative_epsilon_override_degrades_to_default(self, service):
        from millm.core.config import settings

        config = service.build_config(make_profile(
            sensing={"epsilon": -1.0}))
        assert config.thresholds[0].item() == pytest.approx(
            settings.SENSING_EPSILON * 40.0)

    def test_status_reconciles_stale_armed_state(self, service):
        """R3: swallowed disarm (e.g. detach failure) must not report armed
        forever."""
        service.arm_for_profile(make_profile(), MagicMock(d_sae=16384))
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = None  # SAE gone
            status = service.status()
        assert status["armed"] is False

    async def test_truncated_last_event_only_persisted(self, service):
        """R3 mutation pin: the cut point must be recoverable from rows."""
        service.arm_for_profile(make_profile(), MagicMock(d_sae=16384))
        captured = {}

        class _Repo:
            def __init__(self, session):
                pass

            async def create_many(self, events):
                captured["events"] = events
                out = []
                for event in events:
                    row = MagicMock()
                    row.to_dict.return_value = dict(event, id=1)
                    out.append(row)
                return out

            async def prune(self, *a, **kw):
                return 0

        class _Ctx:
            async def __aenter__(self):
                session = MagicMock()

                async def _commit():
                    return None

                session.commit = _commit
                return session

            async def __aexit__(self, *a):
                return False

        with patch("millm.db.base.async_session_factory",
                   return_value=_Ctx()), \
             patch("millm.db.repositories.sensing_repository.SensingRepository",
                   _Repo), \
             patch.object(service, "_emit_events"):
            await service.record(
                "req-1",
                [make_hit(pos_start=1, pos_end=1),
                 make_hit(pos_start=5, pos_end=5)],
                True, None, None)
        flags = [e["truncated"] for e in captured["events"]]
        assert flags == [False, True]

    def test_events_recorded_counter_increments(self, service):
        assert service.status()["events_recorded_since_start"] == 0
