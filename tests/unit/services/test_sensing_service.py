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
        service.arm_for_profile(make_profile(), MagicMock(d_sae=16384))
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
