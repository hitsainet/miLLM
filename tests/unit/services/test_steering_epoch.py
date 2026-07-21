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
    state.reset_for_tests()
    state._steering_epoch = 0
    yield
    state.reset_for_tests()
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
        state.reset_for_tests()
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
    """R2: R1's "revert ledger" was structurally incapable of working — the
    restore only recorded when saved == current (nothing bumped), while
    set_intensity's applied_epoch is always post-bump, so the two conditions
    were mutually exclusive BY CONSTRUCTION. It also fired false positives on
    ordinary idle traffic. Removed.

    The guard is what makes `reapplied` truthful: our bump advances the epoch,
    an in-flight restore sees the mismatch and SKIPS, so it cannot revert us.
    The only way our write stops being live is another AUTHORITATIVE write —
    which the epoch comparison detects directly."""

    def test_an_in_flight_restore_cannot_revert_an_operator_write(self):
        """The guarantee that makes a ledger unnecessary."""
        sae = make_sae({1: 99.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)
        snapshot = AttachedSAEState().steering_epoch
        saved = {"values": {1: 40.0}, "enabled": True, "epoch": snapshot}

        applied = AttachedSAEState().bump_steering_epoch("operator")
        service()._restore_request_profile(saved)

        assert sae._applied == {1: 99.0}, "the restore reverted the operator"
        assert AttachedSAEState().steering_epoch == applied, (
            "a restore must never advance the epoch"
        )

    def test_a_later_authoritative_write_supersedes(self):
        state = AttachedSAEState()
        applied = state.bump_steering_epoch("operator_a")
        assert state.steering_epoch == applied
        state.bump_steering_epoch("operator_b")
        assert state.steering_epoch != applied, (
            "a later authoritative write must be detectable"
        )

    def test_superseded_is_a_plain_bool_on_the_response_model(self):
        """R1 findings 1 and 9: the field was computed by the service, omitted
        from the response model (so Pydantic silently DROPPED it), and
        tri-state (True|None, never False)."""
        from millm.api.schemas.circuit import CircuitIntensityResponse

        assert "superseded" in CircuitIntensityResponse.model_fields
        field = CircuitIntensityResponse.model_fields["superseded"]
        assert field.annotation is bool
        assert field.default is False

    def test_no_ledger_remains(self):
        """R2: the mechanism was removed, not patched. A partially-removed
        ledger would be worse than either."""
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
        for rel in ("millm/services/sae_service.py",
                    "millm/services/circuit_service.py",
                    "millm/services/inference_service.py"):
            src = (repo / rel).read_text()
            assert "was_reverted" not in src
            assert "note_restore_reverted" not in src


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
        # F18 R1-10: `_serving_members` no longer exists — `set_intensity`
        # calls `CircuitSteeringEngine.serving_members(definition)` directly —
        # so the stub that used to sit here was DEAD, and these tests silently
        # ran the real flattener against a MagicMock. That passed by luck of
        # the mock's shape rather than by intent, which is the difference
        # between a fixture and a coincidence.
        #
        # `members=[]` is now load-bearing: the real flattener iterates it and
        # correctly yields nothing, which is what these epoch tests want.
        svc._parse_stored = lambda c: MagicMock(edges=[], members=[])
        return svc, circuit

    async def test_reapplied_is_false_when_a_later_write_supersedes(self):
        """MUT-A control: reverting FR-16.4 to the unconditional
        `"reapplied": reapplied` must FAIL here. R2 replaced the ledger
        scenario with the one that can ACTUALLY happen — another authoritative
        write landing after ours."""
        from unittest.mock import MagicMock

        svc, _ = self._service()
        state = AttachedSAEState()

        def _steer(*a, **k):
            # Faithful to the real contract: bump, and report the epoch OUR
            # write produced on the OUTCOME — then a second operator lands.
            mine = state.bump_steering_epoch("our_write")
            state.bump_steering_epoch("another_operator")
            return MagicMock(hazards=[], clamp_warnings=[], applied_epoch=mine)

        svc._sae_service.set_circuit_steering = _steer
        out = await svc.set_intensity("circ_1", 1.5)

        assert out["reapplied"] is False, (
            "reapplied stayed true for a value another write superseded"
        )
        assert out["superseded"] is True
        assert any("superseded" in w for w in out["warnings"])

    async def test_reapplied_is_true_for_a_clean_write(self):
        """The stub must model the REAL contract — bump, and report the epoch
        the bump produced. R2 found the previous stub read the epoch without
        bumping, which is the fixture-agrees-by-construction pattern that let
        the ledger mutation survive."""
        from unittest.mock import MagicMock

        svc, _ = self._service()
        state = AttachedSAEState()

        def _steer(*a, **k):
            mine = state.bump_steering_epoch("our_write")
            return MagicMock(hazards=[], clamp_warnings=[], applied_epoch=mine)

        svc._sae_service.set_circuit_steering = _steer
        out = await svc.set_intensity("circ_1", 1.5)
        assert out["reapplied"] is True
        assert out["superseded"] is False


class TestR2SurvivingMutationGaps:
    """R2 ran mutations that SURVIVED — each is a test finding, not a code
    finding, so each gets a test that fails against the broken line."""

    def test_operator_clear_steering_bumps(self):
        """R2 finding 6: deleting this bump left 1621 green. R1 fixed six
        routes and pinned five."""
        from millm.services.sae_service import SAEService

        svc = SAEService.__new__(SAEService)
        svc._sae_state = AttachedSAEState()
        svc._last_write_epoch = 0
        AttachedSAEState().set(make_sae({1: 5.0}), "sae-10", 10, None)

        before = AttachedSAEState().steering_epoch
        svc.clear_steering()
        assert AttachedSAEState().steering_epoch == before + 1

    def test_the_internal_clear_does_not_double_bump(self):
        """R2 finding 10: _set_circuit_steering_locked calls
        clear_circuit_steering internally; bumping there advanced one logical
        action by 2 AND bumped through the dial's authoritative=False."""
        import inspect

        from millm.services.sae_service import SAEService

        src = inspect.getsource(SAEService._set_circuit_steering_locked)
        assert "clear_circuit_steering(authoritative=False)" in src

    def test_the_clear_bump_is_taken_under_the_lock(self):
        """R2 finding 9: set_circuit_steering bumps inside the lock with a
        comment explaining the race; this path had the identical race."""
        import inspect

        from millm.services.sae_service import SAEService

        src = inspect.getsource(SAEService.clear_circuit_steering)
        bump = src.index("bump_steering_epoch")
        lock = src.rindex("_ATTACHMENT_LOCK", 0, bump)
        assert lock < bump, "the clear bump is not taken under the lock"

    def test_set_intensity_reports_an_apply_failure(self):
        """R2 finding 11: the DB write commits BEFORE the steering call, so a
        raise left persisted λ diverging from live steering with the caller
        told nothing."""
        import inspect

        from millm.services.circuit_service import CircuitService

        src = inspect.getsource(CircuitService.set_intensity)
        assert "circuit_set_intensity_apply_failed" in src
        assert "Persisted and live steering now differ" in src


class TestRequestIdReachesTheLog:
    """R2 finding 7: removing `request_id=completion_id` from a call site left
    1621 green — the log test injected a saved dict that already had the key,
    so it tested rendering, never supply."""

    def test_both_generation_call_sites_supply_the_request_id(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parents[3]
               / "millm/services/inference_service.py").read_text()
        # R2 finding 7 (and its own follow-up): a COUNT is too weak — dropping
        # one site and leaving two elsewhere still passes. Assert that every
        # call to _apply_request_steering supplies it.
        import re

        calls = re.findall(
            r"_apply_request_steering\((.*?)\)", src, re.S
        )
        invoking = [c for c in calls if "request.profile" in c]
        assert invoking, "no generation call site found"
        for c in invoking:
            assert "request_id=" in c, (
                f"a generation path calls _apply_request_steering without a "
                f"request id: {c.strip()[:80]}"
            )

    def test_the_apply_functions_accept_and_store_it(self):
        import inspect

        sig = inspect.signature(
            InferenceService._apply_request_circuit_steering
        )
        assert "request_id" in sig.parameters
        sig2 = inspect.signature(InferenceService._apply_request_steering)
        assert "request_id" in sig2.parameters


class TestTheEpochSurvivesARealApply:
    """R3's root-cause finding: every restore-guard test hand-builds
    `{"epoch": 0, ...}` as a LITERAL, so the guard is exhaustively tested
    against dicts the tests themselves wrote — while the production code that
    POPULATES those dicts had no coverage at all.

    Six mutations survived the full 1627-test suite because of it (dropping the
    epoch key from the circuit dict, the profile dict and the λ=0 return, plus
    three request_id sites). These call the REAL apply and assert the key
    survives into the saved dict."""

    def _armed_dial(self):
        """A service whose circuit dial will actually run."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        svc = service()
        sae = make_sae({1: 40.0})
        AttachedSAEState().set(sae, "sae-10", 10, None)

        circuit = SimpleNamespace(
            id="circ_1", name="c", layers=[10], serving_mode="full",
            intensity=1.0, rung=2,
            circuit_meta={
                "kind": "mistudio.circuit-definition", "schema_version": "1",
                "name": "c",
                "saes": [{"layer": 10, "n_features": 8192,
                          "mistudio_sae_id": "sae-10"}],
                "members": [{"layer": 10, "feature": {
                    "feature_idx": 1, "strength": 40.0,
                    "max_activation": 10.0}}],
                "edges": [],
                "budget": {"layers": {}, "intensity": 1.0,
                           "intensity_range": [0.0, 2.0]},
            },
        )
        svc._active_full_circuit = AsyncMock(return_value=circuit)
        return svc, sae

    async def test_the_circuit_apply_records_its_epoch(self):
        """MUT-M18: dropping `"epoch"` here left 1627 green while US-16.1 was
        broken — the guard cannot compare what the apply never recorded."""
        svc, _ = self._armed_dial()
        saved = await svc._apply_request_circuit_steering(2.0, request_id="r-1")

        assert saved is not None
        assert "epoch" in saved, (
            "the apply produced a snapshot the guard can never evaluate"
        )
        assert isinstance(saved["epoch"], int)
        assert saved["request_id"] == "r-1"

    async def test_the_lambda_zero_path_records_its_epoch(self):
        """MUT-M41: the λ=0 fast path returns early, so it needs its own
        assertion — it is the one apply path with no steering call at all."""
        svc, _ = self._armed_dial()
        saved = await svc._apply_request_circuit_steering(0.0, request_id="r-0")

        assert saved is not None and "epoch" in saved
        assert saved["request_id"] == "r-0"

    async def test_the_captured_epoch_is_the_PRE_apply_value(self):
        """R1 finding 1, pinned end-to-end rather than by source inspection:
        the epoch must be the one in force when the SNAPSHOT was taken, so an
        operator write during the apply window is detected, not absorbed."""
        svc, _ = self._armed_dial()
        before = AttachedSAEState().steering_epoch
        saved = await svc._apply_request_circuit_steering(2.0, request_id="r-2")
        assert saved["epoch"] == before, (
            "the apply captured a post-apply epoch, absorbing any operator "
            "write that landed during the apply window"
        )

    async def test_a_saved_snapshot_round_trips_through_the_guard(self):
        """The full loop: a REAL apply, an operator write, then the REAL
        restore — no hand-built dict anywhere."""
        svc, sae = self._armed_dial()
        saved = await svc._apply_request_circuit_steering(2.0, request_id="r-3")
        dialled = dict(sae._applied)

        AttachedSAEState().bump_steering_epoch("operator")
        svc._restore_request_profile(saved)

        assert sae._applied == dialled, (
            "the restore reverted an operator write that landed mid-request"
        )
