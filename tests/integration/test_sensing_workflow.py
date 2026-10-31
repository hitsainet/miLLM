"""
Feature 11 Task 6.0: end-to-end sensing workflow — arm a real SensingService
config against a deterministic LoadedSAE, drive forward passes through the
REAL hook, flush through SensingService.record into a real (SQLite) session,
and verify spans/quorum/context/persistence. Plus lifecycle + routing safety.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.sensing_service import SensingService

D_IN = 8
D_SAE = 32


def make_sae() -> LoadedSAE:
    W_enc = torch.zeros(D_IN, D_SAE)
    for j in range(D_SAE):
        W_enc[j % D_IN, j] = 1.0
    config = SAEConfig(d_in=D_IN, d_sae=D_SAE, model_name="test",
                       hook_name="test", hook_layer=1)
    return LoadedSAE(
        W_enc=W_enc, b_enc=torch.zeros(D_SAE),
        W_dec=torch.zeros(D_SAE, D_IN), b_dec=torch.zeros(D_IN),
        config=config, device="cpu",
    )


def make_profile():
    profile = MagicMock()
    profile.id = "prof_ws1"
    profile.name = "fear cluster"
    profile.source_kind = "cluster"
    profile.sensing_enabled = True
    profile.is_active = True
    profile.cluster_meta = {
        "display_token": "fear",
        "members": [
            {"feature_idx": 0, "strength": 1.0, "max_activation": 10.0},
            {"feature_idx": 1, "strength": 1.0, "max_activation": 10.0},
            {"feature_idx": 2, "strength": 1.0, "max_activation": 10.0},
        ],
        # min_k pinned at 2: the default quorum is now ALL sensable
        # members (goal item 3) and this fixture co-fires 2 of 3
        "sensing": {"context_tokens": 3, "min_k": 2},
    }
    return profile


def pos(active: dict[int, float]) -> list[float]:
    row = [0.0] * D_IN
    for d, v in active.items():
        row[d] = v
    return row


def hidden(rows) -> torch.Tensor:
    return torch.tensor(rows).unsqueeze(0)


class _SessionCtx:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return False


@pytest.fixture
async def db_profile(test_session):
    from millm.db.models.profile import Profile

    row = Profile(id="prof_ws1", name="fear cluster", steering={"0": 1.0},
                  source_kind="cluster", sensing_enabled=True)
    test_session.add(row)
    await test_session.flush()
    test_session.commit = AsyncMock()  # workflow commits; fixture owns txn
    return row


class TestArmGenerateFlush:
    async def test_full_workflow_events_persisted_with_context(
        self, test_session, db_profile
    ):
        """Arm → hook-driven passes on a known co-firing pattern → flush →
        rows in the DB with correct span, quorum, context, summary + WS."""
        sae = make_sae()
        service = SensingService()
        service.arm_for_profile(make_profile(), sae)

        # theta = 0.1 * 10 = 1.0; quorum = max(2, ceil(0.9)) = 2
        hook_fn = SAEHooker()._create_hook_fn(sae)
        sae.begin_sensing_request("req-e2e")

        # prefill: positions 0-4; members 0+1 co-fire at positions 2,3
        hot = pos({0: 3.0, 1: 2.5})
        hook_fn(None, None, hidden([pos({}), pos({0: 2.0}), hot, hot, pos({})]))
        # decode steps: quiet, then co-fire at absolute position 6
        hook_fn(None, None, hidden([pos({})]))
        hook_fn(None, None, hidden([hot]))

        request_id, hits, truncated = sae.collect_sensing_hits()
        assert request_id == "req-e2e" and truncated is False
        assert [(h.pos_start, h.pos_end, h.phase) for h in hits] == [
            (2, 3, "prefill"), (6, 6, "decode"),
        ]

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = (
            lambda ids, **kw: " ".join(f"t{i}" for i in ids))
        full_ids = torch.arange(100, 107).unsqueeze(0)  # 7 tokens

        emitted = []
        with patch("millm.db.base.async_session_factory",
                   return_value=_SessionCtx(test_session)), \
             patch.object(service, "_emit_events",
                          side_effect=lambda p: emitted.extend(p)):
            payloads = await service.record(
                request_id, hits, truncated, full_ids, tokenizer)

        assert len(payloads) == 2
        first = payloads[0]
        assert first["phase"] == "prefill"
        assert first["fired_count"] == 2
        # context: +-3 around span 2-3 -> ids 100..106 clipped
        assert first["context_token_ids"] == [100, 101, 102, 103, 104, 105, 106]
        assert "fear: 2/3 members fired" in first["summary"]
        assert emitted and len(emitted) == 2

        # Persisted and queryable through the repository
        from millm.db.repositories.sensing_repository import SensingRepository

        repo = SensingRepository(test_session)
        rows = await repo.list_events(profile_id="prof_ws1")
        assert len(rows) == 2
        assert rows[0].summary.startswith("fear:")

    async def test_unarmed_request_zero_delta(self):
        """SEN-S3: without arming, the hook path is unchanged and collect
        yields nothing."""
        sae = make_sae()
        hook_fn = SAEHooker()._create_hook_fn(sae)
        x = hidden([pos({0: 5.0, 1: 5.0})])
        out = hook_fn(None, None, x)
        assert torch.equal(out, x)
        assert sae.collect_sensing_hits() == ("", [], False)
        assert sae._sensing_overhead_ms == 0.0


class TestRoutingSafety:
    def test_armed_sensing_forces_serial(self):
        """SEN-S1: a CBM-eligible request routes serial while sensing is
        armed (SENSING_FORCE_SERIAL default true)."""
        from millm.services.inference_service import InferenceService

        service = InferenceService(model_service=MagicMock())
        backend = MagicMock()
        backend.is_running = True
        backend.sampling_params_match = MagicMock(return_value=True)
        service._cbm_backend = backend
        service._cbm_force_serial_monitoring = False

        sae = make_sae()
        sensing_service = SensingService()
        sensing_service.arm_for_profile(make_profile(), sae)

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            assert service._use_cbm_for_request(
                temperature=None, top_p=None, has_steering_override=False
            ) is False
            sae.disarm_sensing()
            assert service._use_cbm_for_request(
                temperature=None, top_p=None, has_steering_override=False
            ) is True


class TestLifecycle:
    def test_activate_arms_and_deactivate_disarms(self):
        """Profile activation syncs the arm state (task 3.7)."""
        import millm.api.dependencies as deps
        from millm.services.profile_service import ProfileService

        deps._sensing_service = None
        profile_service = ProfileService.__new__(ProfileService)
        sae = make_sae()

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            profile_service._sync_sensing_arm_state(make_profile())
            assert sae.is_sensing_armed
            assert deps.get_sensing_service().is_armed

            profile_service._sync_sensing_arm_state(None)
            assert not sae.is_sensing_armed
            assert not deps.get_sensing_service().is_armed
        deps._sensing_service = None

    def test_activating_non_sensing_profile_disarms_previous(self):
        import millm.api.dependencies as deps
        from millm.services.profile_service import ProfileService

        deps._sensing_service = None
        profile_service = ProfileService.__new__(ProfileService)
        sae = make_sae()
        manual = MagicMock()
        manual.source_kind = "manual"
        manual.sensing_enabled = False

        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.attached_sae = sae
            profile_service._sync_sensing_arm_state(make_profile())
            assert sae.is_sensing_armed
            profile_service._sync_sensing_arm_state(manual)
            assert not sae.is_sensing_armed
        deps._sensing_service = None

    def test_sae_detach_would_disarm(self):
        """The detach path calls disarm with the outgoing SAE."""
        import millm.api.dependencies as deps

        deps._sensing_service = None
        service = deps.get_sensing_service()
        sae = make_sae()
        service.arm_for_profile(make_profile(), sae)
        assert service.is_armed
        service.disarm(sae)  # what detach_sae invokes
        assert not service.is_armed and not sae.is_sensing_armed
        deps._sensing_service = None


class TestStreamingIdCapture:
    def test_id_capture_criteria_stores_reference(self):
        from millm.services.inference_service import _make_id_capture_criteria

        capture = _make_id_capture_criteria()
        assert capture is not None
        ids1 = torch.tensor([[1, 2, 3]])
        ids2 = torch.tensor([[1, 2, 3, 4]])
        assert capture(ids1, None) is False  # never stops generation
        capture(ids2, None)
        assert torch.equal(capture.latest_ids, ids2)  # latest wins, zero-copy
