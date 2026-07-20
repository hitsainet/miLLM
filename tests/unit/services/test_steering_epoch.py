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
    yield
    state._entries.clear()
    state._steering_epoch = 0


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
