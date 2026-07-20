"""Feature 16: the steering epoch.

The defect: a per-request steering override saves, applies, and restores
UNCONDITIONALLY. An operator who changed live steering while a request was
generating had their change written back over by that restore — and
`set_intensity` had already told them it succeeded (`"reapplied": true`).

The fix: a monotonic epoch bumped by every authoritative writer. A restore that
finds it advanced SKIPS. Last authoritative writer wins.
"""

from unittest.mock import MagicMock

import pytest

from millm.services.inference_service import InferenceService
from millm.services.sae_service import AttachedSAEState


@pytest.fixture(autouse=True)
def clean():
    state = AttachedSAEState()
    state._entries.clear()
    state._steering_epoch = 0
    state._reverted_epochs.clear()
    yield
    state._entries.clear()
    state._steering_epoch = 0
    state._reverted_epochs.clear()


def make_sae(values=None, enabled=True):
    sae = MagicMock()
    applied = dict(values or {})
    sae.d_sae = 8192
    sae.is_steering_enabled = enabled
    sae.set_steering_batch.side_effect = applied.update
    sae.clear_steering.side_effect = lambda idx=None: (
        applied.clear() if idx is None else applied.pop(idx, None)
    )
    sae.get_steering_values.side_effect = lambda: dict(applied)
    sae.enable_steering.side_effect = lambda v: setattr(sae, "is_steering_enabled", v)
    sae._applied = applied
    return sae


def service():
    return InferenceService.__new__(InferenceService)


class TestTheCounter:
    def test_starts_at_zero_and_is_monotonic(self):
        state = AttachedSAEState()
        assert state.steering_epoch == 0
        state.bump_steering_epoch("a")
        state.bump_steering_epoch("b")
        assert state.steering_epoch == 2

    def test_the_singleton_shares_one_counter(self):
        AttachedSAEState().bump_steering_epoch("a")
        assert AttachedSAEState().steering_epoch == 1

    def test_bump_returns_the_epoch_it_produced(self):
        """set_intensity needs ITS OWN epoch to test whether something landed
        after it — not merely whether anything is newer than a snapshot, which
        would report superseded for its own bump."""
        state = AttachedSAEState()
        assert state.bump_steering_epoch("x") == 1
        assert state.bump_steering_epoch("y") == 2

    def test_clearing_attachments_does_not_reset_it(self):
        """A counter that goes backwards can collide with a saved value and
        silently permit a stale restore."""
        state = AttachedSAEState()
        state.bump_steering_epoch("a")
        state._entries.clear()
        assert state.steering_epoch == 1


class TestRestoreSkipsWhenSuperseded:
    def _saved(self, sae, epoch):
        return {"values": {1: 40.0}, "enabled": True, "epoch": epoch}

    def test_restore_proceeds_when_the_epoch_is_unchanged(self):
        """EC-16.1 — the overwhelmingly common path must be unchanged."""
        sae = make_sae({1: 99.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)
        svc = service()
        svc._restore_request_profile(self._saved(sae, 0))
        assert sae._applied == {1: 40.0}

    def test_restore_skips_when_an_operator_wrote_in_between(self):
        """The headline behaviour: the operator's change survives."""
        sae = make_sae({1: 99.0})   # 99.0 == what the operator just set
        AttachedSAEState().set(sae, "sae-10", 10, None)
        saved = self._saved(sae, 0)

        AttachedSAEState().bump_steering_epoch("operator_set_intensity")

        service()._restore_request_profile(saved)
        assert sae._applied == {1: 99.0}, (
            "the request's restore overwrote the operator's change"
        )

    def test_a_snapshot_without_an_epoch_proceeds(self):
        """Older saved state, and the apply-failure rollback, carry no epoch
        and must behave exactly as before (EC-16.6)."""
        sae = make_sae({1: 99.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)
        AttachedSAEState().bump_steering_epoch("something")
        service()._restore_request_profile({"values": {1: 40.0}, "enabled": True})
        assert sae._applied == {1: 40.0}

    def test_the_guard_covers_the_circuit_branch_too(self):
        sae = make_sae({1: 99.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)
        saved = {
            "circuit": True,
            "epoch": 0,
            "layers": [{"sae_id": "sae-10", "layer": 10,
                        "values": {1: 40.0}, "enabled": True}],
        }
        AttachedSAEState().bump_steering_epoch("operator")
        service()._restore_request_profile(saved)
        assert sae._applied == {1: 99.0}

    def test_supersession_is_logged(self):
        """FR-16.3 — "my change vanished" and "my change won" must be
        distinguishable after the fact. structlog does not propagate to
        caplog, so the emitter is captured directly."""
        from unittest.mock import patch

        sae = make_sae({1: 99.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)
        saved = self._saved(sae, 0)
        AttachedSAEState().bump_steering_epoch("operator")

        import millm.services.inference_service as mod

        with patch.object(mod.logger, "info") as info:
            service()._restore_request_profile(saved)

        events = [c for c in info.call_args_list
                  if c.args and "superseded" in str(c.args[0])]
        assert events, "a skipped restore left no trace"
        kwargs = events[0].kwargs
        assert kwargs["saved_epoch"] == 0 and kwargs["current_epoch"] == 1, (
            "both epochs must be logged or the skip is undiagnosable"
        )


class TestTheGuardIsAboveBothBranches:
    def test_the_comparison_precedes_the_branch_demultiplex(self):
        """A saved shape added later must inherit the guard by default rather
        than by someone remembering to add it."""
        import inspect

        src = inspect.getsource(InferenceService._restore_request_profile)
        guard = src.index("saved_epoch")
        circuit_branch = src.index('saved.get("circuit")', guard)
        # The first mention of the circuit branch after the guard is the guard's
        # OWN log line; the demultiplex must come later still.
        demux = src.index("Feature 14", guard)
        assert guard < circuit_branch < demux


class TestAuthoritativeWritersBump:
    """FR-16.1 — a new writer added without a bump is the realistic regression,
    so this enumerates them rather than spot-checking."""

    EXPECTED = [
        ("millm/services/sae_service.py", "set_circuit_steering"),
        ("millm/services/sae_service.py", "clear_circuit_steering"),
        ("millm/services/sae_service.py", "attach_set"),
        ("millm/services/sae_service.py", "detach_sae"),
        ("millm/services/circuit_service.py", "circuit_activate"),
        ("millm/services/circuit_service.py", "circuit_deactivate"),
        ("millm/services/circuit_service.py", "circuit_set_intensity"),
        ("millm/services/profile_service.py", "profile_activate"),
        ("millm/services/profile_service.py", "profile_deactivate"),
    ]

    def test_every_authoritative_writer_bumps(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
        for rel, reason in self.EXPECTED:
            src = (repo / rel).read_text()
            # Normalise whitespace: a bump wrapped across lines by the
            # formatter is still a bump, and a test that only matches the
            # single-line form would fail for a formatting reason and invite
            # someone to loosen the assertion instead of reading it.
            flat = " ".join(src.split())
            assert (f'bump_steering_epoch( "{reason}"' in flat
                    or f'bump_steering_epoch("{reason}"' in flat
                    or f"bump_steering_epoch( '{reason}'" in flat
                    or f"bump_steering_epoch('{reason}'" in flat), \
                f"{rel} does not bump for {reason}"

    def test_the_low_level_batch_write_does_NOT_bump(self):
        """set_steering_batch is the write used BY the per-request apply path.
        Bumping there would make every request supersede its own restore,
        silently disabling per-request isolation entirely."""
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
        src = (repo / "millm/ml/sae_wrapper.py").read_text()
        assert "bump_steering_epoch" not in src


class TestWritersBumpBehaviourally:
    """R1 finding 3: the enumeration test grepped SOURCE TEXT, so all nine
    writers could be commented out or wrapped in `if False:` and it stayed
    green — the TestRingPruningIsWired anti-pattern this project has shipped
    before. These call the writers and observe the counter."""

    def test_operator_set_steering_bumps(self):
        from millm.services.sae_service import SAEService

        svc = SAEService.__new__(SAEService)
        svc._sae_state = AttachedSAEState()
        sae = make_sae()
        AttachedSAEState().set(sae, "sae-10", 10, None)

        before = AttachedSAEState().steering_epoch
        svc.set_steering(1, 40.0)
        assert AttachedSAEState().steering_epoch == before + 1

    def test_operator_batch_and_enable_each_bump(self):
        from millm.services.sae_service import SAEService

        svc = SAEService.__new__(SAEService)
        svc._sae_state = AttachedSAEState()
        AttachedSAEState().set(make_sae(), "sae-10", 10, None)

        before = AttachedSAEState().steering_epoch
        svc.set_steering_batch({1: 10.0})
        svc.enable_steering(True)
        assert AttachedSAEState().steering_epoch == before + 2

    def test_a_no_op_circuit_clear_does_NOT_bump(self):
        """R1 finding 8: an unconditional bump made every no-op clear
        supersede all in-flight restores, stranding their transient values in
        global state with no compensating write."""
        from millm.services.sae_service import SAEService

        svc = SAEService.__new__(SAEService)
        svc._sae_state = AttachedSAEState()
        before = AttachedSAEState().steering_epoch
        svc.clear_circuit_steering()          # nothing attached -> cleared == []
        assert AttachedSAEState().steering_epoch == before

    def test_a_per_request_apply_does_NOT_bump(self):
        """TID pitfall 2: a per-request apply that bumped would make every
        request supersede its OWN restore, silently disabling isolation."""
        from millm.services.sae_service import SAEService
        from millm.api.schemas.circuit import CircuitMember

        svc = SAEService.__new__(SAEService)
        svc._sae_state = AttachedSAEState()
        AttachedSAEState().set(make_sae(), "sae-10", 10, None)
        member = CircuitMember(feature_idx=1, layer=10, budget=10.0,
                               sign=1, sae_id="sae-10")

        before = AttachedSAEState().steering_epoch
        svc.set_circuit_steering([member], 1.0, authoritative=False)
        assert AttachedSAEState().steering_epoch == before, (
            "the per-request apply bumped and will supersede its own restore"
        )

        svc.set_circuit_steering([member], 1.0, authoritative=True)
        assert AttachedSAEState().steering_epoch == before + 1


class TestCaptureHappensAtSnapshotTime:
    """R1 finding 1: the circuit epoch was read at RETURN, after the apply —
    so an operator write landing during the apply window was absorbed and the
    restore reverted them. The TID forbids the late read by name."""

    def test_the_capture_precedes_the_apply(self):
        import inspect

        src = inspect.getsource(
            InferenceService._apply_request_circuit_steering
        )
        capture = src.index("saved_epoch = state.steering_epoch")
        apply_call = src.index("set_circuit_steering(")
        assert capture < apply_call, (
            "the epoch is captured after the apply; an operator write during "
            "the apply window would be absorbed and then reverted"
        )


class TestReappliedTruthfulness:
    """R1 finding 2: FR-16.4 — the feature's stated reason for existing — had
    ZERO tests. Reverting it to the original defect left 1609/1609 green."""

    def test_a_reverted_write_is_not_reported_as_reapplied(self):
        state = AttachedSAEState()
        applied = state.bump_steering_epoch("operator_set_intensity")

        # An in-flight restore proceeds and writes over that epoch. It cannot
        # bump (that would make every request supersede itself), so it records.
        state.note_restore_reverted(applied)

        assert state.was_reverted(applied) is True, (
            "set_intensity cannot detect the case it exists for"
        )

    def test_an_unreverted_write_is_clean(self):
        state = AttachedSAEState()
        applied = state.bump_steering_epoch("operator_set_intensity")
        assert state.was_reverted(applied) is False

    def test_the_ledger_is_bounded(self):
        """It must not grow without bound over a long-lived process."""
        state = AttachedSAEState()
        for _ in range(1000):
            e = state.bump_steering_epoch("x")
            state.note_restore_reverted(e)
        assert len(state._reverted_epochs) <= 256

    def test_superseded_is_a_plain_bool_on_the_response_model(self):
        """R1 findings 1 and 9: the field was computed by the service, omitted
        from the response model (so Pydantic silently DROPPED it), and
        tri-state (True|None, never False)."""
        from millm.api.schemas.circuit import CircuitIntensityResponse

        assert "superseded" in CircuitIntensityResponse.model_fields
        field = CircuitIntensityResponse.model_fields["superseded"]
        assert field.annotation is bool
        assert field.default is False


class TestSkipLogIsDiagnosable:
    def test_the_skip_log_carries_the_request_id_and_stranded_layers(self):
        """FR-16.3 requires the request id. R1 finding 8: skipping leaves the
        request's transient values live on layers the operator never touched,
        and nothing named them."""
        from unittest.mock import patch

        sae = make_sae({1: 99.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)
        saved = {
            "circuit": True, "epoch": 0, "request_id": "cmpl-abc",
            "layers": [{"sae_id": "sae-10", "layer": 10,
                        "values": {1: 40.0}, "enabled": True}],
        }
        AttachedSAEState().bump_steering_epoch("operator")

        import millm.services.inference_service as mod

        with patch.object(mod.logger, "info") as info:
            service()._restore_request_profile(saved)

        ev = [c for c in info.call_args_list
              if c.args and "superseded" in str(c.args[0])][0]
        assert ev.kwargs["request_id"] == "cmpl-abc"
        assert ev.kwargs["layers_left_dialled"] == [10]


class TestSetIntensityReturnIsTruthful:
    """MUT-A control: reverting FR-16.4 to the original defect
    (`"reapplied": reapplied` unconditional) must FAIL here. The ledger tests
    above exercise the primitives; this pins the RETURN VALUE a client reads,
    which is the behaviour FR-16.4 actually promises."""

    def _service(self):
        from unittest.mock import AsyncMock, MagicMock

        from millm.services.circuit_service import CircuitService

        svc = CircuitService.__new__(CircuitService)
        circuit = MagicMock(
            id="circ_1", rung=2, is_active=True, serving_mode="full",
            circuit_meta={}, intensity=1.0,
        )
        repo = MagicMock()
        repo.get = AsyncMock(return_value=circuit)
        repo.update = AsyncMock(return_value=circuit)
        svc.repository = repo
        svc._sae_service = MagicMock()
        svc.summarize = lambda c: {"id": "circ_1"}
        svc._parse_stored = lambda c: MagicMock(edges=[], members=[])
        svc._serving_members = staticmethod(lambda d: [])
        return svc, circuit

    async def test_reapplied_is_false_when_a_restore_reverted_it(self):
        import millm.services.circuit_service as mod

        svc, _ = self._service()
        state = AttachedSAEState()

        # Make the steering call land, then simulate an in-flight restore
        # writing over the epoch it produced.
        def _steer(*a, **k):
            e = state.steering_epoch
            state.note_restore_reverted(e)
            return MagicMock(hazards=[], clamp_warnings=[])

        svc._sae_service.set_circuit_steering = _steer
        out = await svc.set_intensity("circ_1", 1.5)

        assert out["reapplied"] is False, (
            "reapplied stayed true for a value a restore reverted — the exact "
            "falsehood FR-16.4 exists to correct"
        )
        assert out["superseded"] is True
        assert any("superseded" in w for w in out["warnings"])

    async def test_reapplied_is_true_for_a_clean_write(self):
        from unittest.mock import MagicMock

        svc, _ = self._service()
        svc._sae_service.set_circuit_steering = MagicMock(
            return_value=MagicMock(hazards=[], clamp_warnings=[])
        )
        out = await svc.set_intensity("circ_1", 1.5)
        assert out["reapplied"] is True
        assert out["superseded"] is False
