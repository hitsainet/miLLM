"""Moving an imported probe's bar in place: what changes, what must not, and what a reader sees.

⚠ **THE DEFECT THIS WHOLE DESIGN EXISTS TO PREVENT IS A DATABASE WRITE THAT CHANGES NOTHING.**

`armed_probe_from_row` resolves a row into the runtime shape ONCE, at arm time
(`probe_arming.py:216`); `begin_request` snapshots the in-memory registry; the bar is read off the
frozen `ArmedProbe`. Meanwhile `GET /api/probes` serialises the ROW. So a route that updated the
database and stopped would show the new threshold in the UI while every verdict kept being judged
against the old one — invisible-but-visible, which is worse than a plain failure.

`TestADbOnlyUpdateChangesNothingServed` pins that defect itself and **must pass**. Its partner
proves the refresh is what fixes it. The pair is the point: one shows the write is insufficient,
the other shows the registry refresh is sufficient.

⚠ AND THE SECOND CLASS OF DEFECT IS A REFRESH THAT LOSES SOMETHING. `encoder` is built at arm
time and `windows` come from the arm REQUEST — neither is recoverable from the row, which is why
`status()` reads windows out of the registry. A refresh written as `armed_probe_from_row(probe)`
would turn a k-sparse probe into a dense one reading raw residuals through a narrow head, and
silently reset the operator's window choice to the default. `dataclasses.replace` on the live
object cannot: it carries every field it is not told to change, including fields added later.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_recalibration import (
    HISTORY_LIMIT,
    ProbeRecalibrationRefused,
    ProbeRecalibrationService,
    bar_is_calibrated,
    describes_the_same_probe,
    every_submitted_bar_survives_parsing,
    seeded_history,
    trim_history,
    window_delta,
)
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext, ProbeRuntimeState

D = 4
PROBE_ID = "pr_abc123"
MISTUDIO_ID = "pm_36d1a65f7953"
RUN_ID = "pmr_07e1c12e383a"


def decision(**over) -> dict:
    """A re-cut `decision` block, as the producer would send it."""
    base = {
        "threshold": 14.5201,
        "target_fpr": 0.005,
        "realised_fpr": 0.005,
        "threshold_source": "calibration_set",
        "calibration": None,
        "windows": None,
        "length_bands": None,
    }
    base.update(over)
    return base


def definition(**over) -> dict:
    """A stored definition: a detector half and a bar half."""
    base = {
        "schema": "mistudio.probe-definition/v1",
        # `norm_mean`/`norm_std` are REQUIRED by `head_from_definition` — a probe standardises
        # before it projects, and a fixture without them is not a probe this runtime can build.
        "head": {
            "weights": [1.0] * D,
            "bias": 0.0,
            "norm_mean": [0.0] * D,
            "norm_std": [1.0] * D,
        },
        "read": {"layer": 11, "hook_point": "resid_post"},
        "aggregation": {"rule": "mean", "params": {}},
        "decision": decision(threshold=11.9144, target_fpr=0.01, realised_fpr=0.01),
        "provenance": {"probe_id": MISTUDIO_ID, "run_id": RUN_ID},
    }
    base.update(over)
    return base


class _Row:
    """The probe row, with only what the service touches."""

    def __init__(self, **over):
        self.id = PROBE_ID
        self.name = "high-stakes"
        self.layer = 11
        self.rule = "mean"
        self.scope = "all"
        self.rung = 2
        self.armed = True
        self.threshold = 11.9144
        self.target_fpr = 0.01
        self.threshold_revision = 1
        self.threshold_calibration_id = None
        self.threshold_history = None
        self.armed_threshold_revision = 1
        self.created_at = None
        self.definition = definition()
        for key, value in over.items():
            setattr(self, key, value)


class _Repo:
    """Records every `update`, so the PAYLOAD and the CALL COUNT can both be asserted."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def update(self, row, **fields):
        self.calls.append(dict(fields))
        for key, value in fields.items():
            setattr(row, key, value)
        return row


def armed(**over) -> ArmedProbe:
    fields = dict(
        probe_id=PROBE_ID,
        name="high-stakes",
        head=ProbeHead(weight=torch.ones(D), bias=0.0, layer=11),
        rule="mean",
        scope="all",
        layer=11,
        rung=2,
        rung_language="detects on unseen tasks",
        threshold=11.9144,
        threshold_revision=1,
    )
    fields.update(over)
    return ArmedProbe(**fields)


@pytest.fixture(autouse=True)
def _clean_registry():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


def arm_into(state: ProbeRuntimeState, probe: ArmedProbe) -> ArmedProbe:
    """Put a probe in the registry without installing a torch hook.

    ⚠ `state.arm(probe, model=None)` CANNOT BE USED HERE. It installs a hook when the layer is
    absent, and `layer_resolution` needs a real module tree — so these tests would be about
    hooking rather than about the bar. Seeding `_handles` first makes `arm` take its
    already-hooked branch, which is the genuine state of a second probe on a layer that is
    already being read. The hook is what this feature deliberately does not touch: a re-cut
    changes the bar, never the layer.
    """
    # A stand-in with the one method the teardown path calls — `disarm_all` removes every handle,
    # and a bare `object()` raised there instead, which is the fixture being less forgiving than
    # the thing it stands in for in the other direction.
    class _Handle:
        def remove(self) -> None:
            return None

    state._handles[probe.layer] = _Handle()
    state.arm(probe, model=None)
    return probe


def observe(ctx, *, value: float, tokens: int = 8):
    ctx.set_prompt_length(tokens)
    ctx.observe(11, torch.full((1, tokens, D), value / D))


# ──────────────────────────────────────────────────────────────────────────────────────
# The headline control: a DB write alone changes nothing that is served.
# ──────────────────────────────────────────────────────────────────────────────────────


class TestADbOnlyUpdateChangesNothingServed:
    """⚠ THIS CLASS MUST PASS. IT PINS THE DEFECT, NOT THE FIX.

    It is the documented proof that writing a threshold to the database does not change what any
    verdict is judged against, so no future refactor can quietly assume the write is enough. The
    `GET /api/probes` route serialises the row, so without the registry refresh the UI and the
    monitor would disagree with nothing saying so.
    """

    def test_a_row_write_does_NOT_move_the_bar_a_verdict_is_judged_against(self):
        state = ProbeRuntimeState()
        probe = armed(threshold=5.0)
        arm_into(state, probe)
        row = _Row(threshold=5.0)

        # The row moves. The registry is untouched — exactly what a DB-only route would do.
        row.threshold = 20.0
        row.threshold_revision = 2

        ctx = ProbeRequestContext("r1", state.armed())
        observe(ctx, value=10.0)
        verdict = ctx.finish()[0]

        assert verdict.threshold == 5.0, (
            "the verdict was judged against the ROW's new bar, which means the registry is being "
            "re-read per request — if that is now true this whole design is unnecessary, and if "
            "it is not, something else is wrong"
        )
        assert verdict.fires is True, "a score of 10 must fire against the OLD bar of 5"
        assert verdict.threshold_revision == 1, (
            "the verdict carried the row's revision, so a failed registry refresh would be "
            "invisible on the event — the one place it must show"
        )

    @pytest.mark.asyncio
    async def test_the_recalibrate_PATH_does_move_it(self):
        """The partner. Same setup, through the service."""
        state = ProbeRuntimeState()
        arm_into(state, armed(threshold=5.0))
        row = _Row(threshold=5.0)
        repo = _Repo()

        outcome = await ProbeRecalibrationService(repo, state).recalibrate(
            row,
            decision=decision(threshold=20.0),
            mistudio_probe_id=MISTUDIO_ID,
            reason="tighter budget",
        )
        assert outcome["registry_updated"] is True

        ctx = ProbeRequestContext("r2", state.armed())
        observe(ctx, value=10.0)
        verdict = ctx.finish()[0]
        assert verdict.threshold == 20.0
        assert verdict.fires is False, "a score of 10 must not fire against the new bar of 20"
        assert verdict.threshold_revision == 2


# ──────────────────────────────────────────────────────────────────────────────────────
# The refresh must not lose what the row does not know.
# ──────────────────────────────────────────────────────────────────────────────────────


class TestTheRefreshKeepsWhatTheRowCannotTell:
    @pytest.mark.asyncio
    async def test_the_ENCODER_survives(self):
        """A k-sparse probe's encoder is built at arm time. Rebuilding from the row would set it
        to `None`, and the probe would read raw residuals through a narrow head."""
        sentinel = lambda t: t  # noqa: E731 - identity by object, which is the assertion
        state = ProbeRuntimeState()
        arm_into(state, armed(encoder=sentinel))
        await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(), decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert state.get(PROBE_ID).encoder is sentinel

    @pytest.mark.asyncio
    async def test_the_OPERATORS_WINDOW_CHOICE_survives(self):
        """`windows` comes from the arm REQUEST and is nowhere on the row. A rebuild would reset
        it to the default, and `status()` would then report windows nobody chose."""
        state = ProbeRuntimeState()
        arm_into(state, armed(windows=("prompt",)))
        await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(), decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert state.get(PROBE_ID).windows == ("prompt",)

    @pytest.mark.asyncio
    async def test_the_HEAD_and_every_identity_field_survive(self):
        state = ProbeRuntimeState()
        original = armed()
        arm_into(state, original)
        await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(), decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        refreshed = state.get(PROBE_ID)
        assert refreshed.head is original.head
        for field in ("rule", "scope", "layer", "rung", "rung_language", "rule_params"):
            assert getattr(refreshed, field) == getattr(original, field)

    @pytest.mark.asyncio
    async def test_the_WINDOW_and_LENGTH_bars_are_re_read_from_the_new_definition(self):
        """They live in the definition JSON, not the column. Moving only `threshold` would leave
        a stale per-length table judging every verdict — and it takes precedence."""
        state = ProbeRuntimeState()
        arm_into(state, armed())
        bands = [
            {"min_tokens": 0, "max_tokens": 100, "threshold": 1.0},
            {"min_tokens": 101, "max_tokens": None, "threshold": 2.0},
        ]
        await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(),
            decision=decision(windows={"prompt": {"threshold": 7.0}}, length_bands=bands),
            mistudio_probe_id=MISTUDIO_ID,
        )
        refreshed = state.get(PROBE_ID)
        assert refreshed.window_thresholds == {"prompt": 7.0}
        assert len(refreshed.length_bands) == 2

    def test_refresh_REFUSES_to_insert_a_probe_that_is_not_armed(self):
        """Inserting would create the half-armed state: a registry entry with no hook, reporting
        a bar while scoring nothing. Worse than a refusal, and invisible."""
        state = ProbeRuntimeState()
        assert state.refresh(armed()) is False
        assert state.get(PROBE_ID) is None

    @pytest.mark.asyncio
    async def test_a_probe_not_armed_here_still_gets_its_ROW_updated(self):
        state = ProbeRuntimeState()
        row = _Row(armed=False)
        repo = _Repo()
        outcome = await ProbeRecalibrationService(repo, state).recalibrate(
            row, decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert outcome["registry_updated"] is False
        assert outcome["stale_armed"] is False
        assert row.threshold == 14.5201, "the row must move even when nothing is armed here"

    @pytest.mark.asyncio
    async def test_a_row_claiming_armed_with_no_live_entry_is_REPORTED_not_reconciled(self):
        state = ProbeRuntimeState()
        row = _Row(armed=True)
        repo = _Repo()
        outcome = await ProbeRecalibrationService(repo, state).recalibrate(
            row, decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert outcome["stale_armed"] is True
        assert outcome["registry_updated"] is False
        # Reported, not fixed: `main.py` reconciles at startup, where the state becomes wrong.
        assert not any("armed" in call for call in repo.calls)

    def test_an_in_flight_request_keeps_the_bar_it_STARTED_with(self):
        """⚠ WHY `frozen=True` PLUS `replace` IS LOAD-BEARING, NOT STYLISTIC.

        `begin_request` snapshots the armed list into the context, holding references to the
        `ArmedProbe` objects. Replacing a dict entry builds a NEW instance, so a half-scored
        request is judged end to end against one bar. An in-place `object.__setattr__` would move
        the bar mid-request, and the event's own `threshold` field would then not describe the
        comparison that was actually made.
        """
        state = ProbeRuntimeState()
        arm_into(state, armed(threshold=5.0))
        ctx = ProbeRequestContext("r3", state.armed())
        observe(ctx, value=10.0)
        state.refresh(dataclasses.replace(state.get(PROBE_ID), threshold=20.0))
        verdict = ctx.finish()[0]
        assert verdict.threshold == 5.0
        assert verdict.fires is True


# ──────────────────────────────────────────────────────────────────────────────────────
# The write shape, which is where the doctrine is enforced.
# ──────────────────────────────────────────────────────────────────────────────────────


class TestTheWriteIsOneTransactionAndKeepsTheDetector:
    @pytest.mark.asyncio
    async def test_the_column_and_the_definition_move_in_ONE_update(self):
        """⚠ TWO CALLS WOULD OPEN A WINDOW IN WHICH THEY DISAGREE, and the model's own doctrine is
        that nothing writes a projected column without writing the definition in the same
        transaction. `ProbeRepository.update` is one `setattr` sweep and one commit, so one call
        is one transaction."""
        repo = _Repo()
        await ProbeRecalibrationService(repo, ProbeRuntimeState()).recalibrate(
            _Row(armed=False), decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert len(repo.calls) == 1, f"{len(repo.calls)} update calls, not one"
        payload = repo.calls[0]
        assert "definition" in payload and "threshold" in payload
        assert payload["threshold"] == 14.5201
        assert payload["target_fpr"] == 0.005
        assert payload["threshold_revision"] == 2

    @pytest.mark.asyncio
    async def test_the_definition_is_a_NEW_dict_carrying_the_detector_through(self):
        """⚠ THE `JSONVariant` TRAP. `Probe.definition` is a plain JSON column, NOT
        `MutableDict.as_mutable`, so `probe.definition["decision"] = ...` is never seen as dirty:
        SQLAlchemy would write the threshold column and leave the definition untouched — the
        row-vs-definition disagreement the doctrine forbids, silently, with a 200 response.
        """
        row = _Row(armed=False)
        original = row.definition
        repo = _Repo()
        await ProbeRecalibrationService(repo, ProbeRuntimeState()).recalibrate(
            row, decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        written = repo.calls[0]["definition"]
        assert written is not original, "the same dict object was handed back — not a new one"
        # The detector half travels through byte-identically: this is refusal 3 of the carve-out,
        # and it is what keeps "nothing replaces a definition in place" true of the weights.
        for key in ("head", "read", "aggregation", "provenance"):
            assert written[key] == original[key]
        assert written["decision"]["threshold"] == 14.5201

    @pytest.mark.asyncio
    async def test_the_history_SEEDS_revision_1_from_the_definitions_own_bar(self):
        """Without the seed, an event stamped `revision 1` is unanswerable once the probe reaches
        3: the row carries only the current bar."""
        repo = _Repo()
        await ProbeRecalibrationService(repo, ProbeRuntimeState()).recalibrate(
            _Row(armed=False), decision=decision(), mistudio_probe_id=MISTUDIO_ID, reason="why"
        )
        history = repo.calls[0]["threshold_history"]
        assert [e["revision"] for e in history] == [1, 2]
        assert history[0]["threshold"] == 11.9144
        assert history[0]["reason"] == "cut by the producer's training run"
        assert history[1]["threshold"] == 14.5201
        assert history[1]["reason"] == "why"

    @pytest.mark.asyncio
    async def test_a_second_recut_APPENDS(self):
        row = _Row(armed=False)
        repo = _Repo()
        service = ProbeRecalibrationService(repo, ProbeRuntimeState())
        await service.recalibrate(row, decision=decision(), mistudio_probe_id=MISTUDIO_ID)
        await service.recalibrate(
            row, decision=decision(threshold=3.0, target_fpr=0.05), mistudio_probe_id=MISTUDIO_ID
        )
        assert [e["revision"] for e in row.threshold_history] == [1, 2, 3]
        assert row.threshold_revision == 3

    def test_the_history_is_capped_and_SAYS_it_dropped_entries(self):
        """Capping an audit log can orphan an event's revision pointer, so the drop is recorded
        rather than silent — and 60 re-cuts of one probe is itself a signal worth not hiding."""
        long = [{"revision": n} for n in range(1, HISTORY_LIMIT + 6)]
        trimmed = trim_history(long)
        assert len(trimmed) == HISTORY_LIMIT
        assert trimmed[0]["earlier_entries_dropped"] == 5

    def test_a_short_history_is_untouched(self):
        short = [{"revision": 1}, {"revision": 2}]
        assert trim_history(short) is short

    def test_seeding_is_IDEMPOTENT_once_a_history_exists(self):
        row = _Row(threshold_history=[{"revision": 1}, {"revision": 2}])
        assert seeded_history(row) == [{"revision": 1}, {"revision": 2}]


class TestTheGatesRefuseBeforeAnythingIsWritten:
    """⚠ EVERY REFUSAL ASSERTS `update` WAS NEVER CALLED. A gate that runs after the write is not
    a gate. The gates are also NAMED FUNCTIONS, tested here by behaviour: a guard left inline in
    this repo was once defeated by `if False:` while a source-scraping test stayed green.
    """

    @pytest.mark.asyncio
    async def _refused(self, row, **kwargs):
        repo = _Repo()
        with pytest.raises(ProbeRecalibrationRefused) as caught:
            await ProbeRecalibrationService(repo, ProbeRuntimeState()).recalibrate(row, **kwargs)
        assert repo.calls == [], "a refusal still wrote to the row"
        return caught.value

    @pytest.mark.asyncio
    async def test_a_cut_naming_a_DIFFERENT_probe_is_refused(self):
        refusal = await self._refused(
            _Row(armed=False), decision=decision(), mistudio_probe_id="pm_somebody_else"
        )
        assert refusal.code == "probe_recalibration_mismatch"
        assert "pm_somebody_else" in refusal.detail and MISTUDIO_ID in refusal.detail

    @pytest.mark.asyncio
    async def test_a_cut_from_a_DIFFERENT_RUN_is_refused(self):
        """The same probe id from a different fit is a different detector."""
        refusal = await self._refused(
            _Row(armed=False),
            decision=decision(),
            mistudio_probe_id=MISTUDIO_ID,
            mistudio_run_id="pmr_someone_elses_run",
        )
        assert refusal.code == "probe_recalibration_mismatch"

    @pytest.mark.asyncio
    async def test_a_definition_with_NO_provenance_refuses_rather_than_defaulting_to_yes(self):
        """⚠ THE CARVE-OUT RESTS ON PROVING SAMENESS, and there is no head digest in the contract
        to fall back on — `weights_sha256` belongs to the `sae` block, the SAE FILE. An
        unverifiable definition therefore refuses; that probe goes disarm -> delete -> import."""
        row = _Row(armed=False, definition=definition(provenance={}))
        refusal = await self._refused(
            row, decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert refusal.code == "probe_recalibration_unverifiable"
        assert "disarm" in refusal.detail

    @pytest.mark.asyncio
    async def test_a_threshold_with_no_BUDGET_is_refused(self):
        refusal = await self._refused(
            _Row(armed=False),
            decision=decision(target_fpr=None),
            mistudio_probe_id=MISTUDIO_ID,
        )
        assert refusal.code == "probe_threshold_uncalibrated"
        assert "target_fpr" in refusal.detail

    @pytest.mark.asyncio
    async def test_a_threshold_with_no_SOURCE_is_refused(self):
        refusal = await self._refused(
            _Row(armed=False),
            decision=decision(threshold_source=None),
            mistudio_probe_id=MISTUDIO_ID,
        )
        assert refusal.code == "probe_threshold_uncalibrated"
        assert "threshold_source" in refusal.detail

    @pytest.mark.asyncio
    async def test_a_NULL_threshold_needs_neither_and_is_allowed(self):
        """`threshold=None` is a real operating point: the probe ranks but does not decide."""
        repo = _Repo()
        await ProbeRecalibrationService(repo, ProbeRuntimeState()).recalibrate(
            _Row(armed=False),
            decision=decision(threshold=None, target_fpr=None, threshold_source=None),
            mistudio_probe_id=MISTUDIO_ID,
        )
        assert repo.calls[0]["threshold"] is None

    @pytest.mark.asyncio
    async def test_a_window_entry_the_PARSER_WOULD_DROP_is_refused(self):
        """⚠ THE TOLERANCE THAT IS CORRECT AT IMPORT IS INVERTED HERE. The arming parsers drop a
        malformed window silently, because a bad block should cost the per-window bars rather than
        the arming. At recalibration that means a 200 that moved no bar for that window."""
        refusal = await self._refused(
            _Row(armed=False),
            decision=decision(windows={"prompt": {"threshold": "not a number"}}),
            mistudio_probe_id=MISTUDIO_ID,
        )
        assert refusal.code == "probe_threshold_uncalibrated"
        assert "prompt" in refusal.detail

    @pytest.mark.asyncio
    async def test_a_TORN_length_table_is_refused_WHOLE(self):
        refusal = await self._refused(
            _Row(armed=False),
            # A gap at 51-99, and no open-ended final band.
            decision=decision(length_bands=[
                {"min_tokens": 0, "max_tokens": 50, "threshold": 1.0},
                {"min_tokens": 100, "max_tokens": 200, "threshold": 2.0},
            ]),
            mistudio_probe_id=MISTUDIO_ID,
        )
        assert refusal.code == "probe_threshold_uncalibrated"
        assert "tile every length" in refusal.detail

    def test_the_predicates_refuse_by_BEHAVIOUR_not_by_source_shape(self):
        """Each gate is callable and testable on its own, which is the recorded remedy for a
        guard that `if False:` defeated while a source scrape stayed green."""
        with pytest.raises(ProbeRecalibrationRefused):
            describes_the_same_probe(definition(), "pm_other", None)
        with pytest.raises(ProbeRecalibrationRefused):
            bar_is_calibrated({"threshold": 1.0})
        with pytest.raises(ProbeRecalibrationRefused):
            every_submitted_bar_survives_parsing({"windows": {"prompt": {}}})
        # And each PASSES on the good case, so none is unconditionally raising.
        describes_the_same_probe(definition(), MISTUDIO_ID, RUN_ID)
        bar_is_calibrated(decision())
        every_submitted_bar_survives_parsing(decision())


class TestARecutReportsWhatItChangedAboutTHEHONESTYMARKERS:
    """⚠ A WINDOW THAT GAINS ITS OWN BAR RETIRES ITS `provisional` MARKER, correctly — the claim
    "judged against a bar cut for a different distribution" becomes false. But the marker then
    vanishes from the event list with no visible cause, and the REVERSE is worse: a window
    reported calibrated until now is not any more. An operator who moves a number and silently
    changes which verdicts carry an honesty marker should learn both in the same breath.
    """

    def test_a_newly_calibrated_window_is_named(self):
        assert window_delta({}, {"response": 1.0})["windows_newly_calibrated"] == ["response"]

    def test_a_DE_calibrated_window_is_named(self):
        assert window_delta({"response": 1.0}, {})["windows_no_longer_calibrated"] == ["response"]

    def test_no_change_names_nothing(self):
        delta = window_delta({"prompt": 1.0}, {"prompt": 2.0})
        assert delta == {"windows_newly_calibrated": [], "windows_no_longer_calibrated": []}

    @pytest.mark.asyncio
    async def test_the_outcome_carries_the_delta_so_the_route_can_report_it(self):
        state = ProbeRuntimeState()
        arm_into(state, armed())
        outcome = await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(), decision=decision(windows={"response": {"threshold": 9.0}}),
            mistudio_probe_id=MISTUDIO_ID,
        )
        assert outcome["windows_newly_calibrated"] == ["response"]

    @pytest.mark.asyncio
    async def test_the_outcome_states_both_ends_and_whether_the_registry_moved(self):
        state = ProbeRuntimeState()
        arm_into(state, armed(threshold=11.9144))
        outcome = await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(), decision=decision(), mistudio_probe_id=MISTUDIO_ID
        )
        assert outcome["previous_threshold"] == 11.9144
        assert outcome["previous_revision"] == 1
        assert outcome["threshold_revision"] == 2
        assert outcome["registry_updated"] is True
        assert outcome["armed"] is True


class TestTheREVISIONReachesEveryPlaceAReaderLooks:
    """⚠ WRITTEN BECAUSE A MUTATION SURVIVED. Hardcoding `threshold_revision=1` in
    `armed_probe_from_row` left all 47 tests green — so nothing asserted that a probe armed AFTER
    a re-cut judges against the revision it actually has.

    The consequence of that gap is the field's own purpose inverted: a probe re-cut to revision 3
    and then armed would judge correctly against the new bar while stamping every event
    `revision 1`, so a reader comparing events would conclude the bar had never moved. Four
    mutations in the probe-export arc were the same shape — a well-covered helper whose CALLER was
    covered by nothing.
    """

    def test_arming_reads_the_revision_OFF_THE_ROW(self):
        from millm.services.probe_arming import armed_probe_from_row

        row = _Row(threshold_revision=3, threshold=14.5201)
        resolved = armed_probe_from_row(row)
        assert resolved.threshold_revision == 3, (
            "the armed probe carries revision 1 while its row says 3 — every event it judges "
            "would claim a bar that is two cuts old"
        )

    def test_a_row_that_has_NEVER_been_recut_arms_at_revision_1(self):
        from millm.services.probe_arming import armed_probe_from_row

        assert armed_probe_from_row(_Row(threshold_revision=1)).threshold_revision == 1

    def test_the_verdict_carries_the_ARMED_probes_revision(self):
        state = ProbeRuntimeState()
        arm_into(state, armed(threshold_revision=4))
        ctx = ProbeRequestContext("r4", state.armed())
        observe(ctx, value=100.0)
        assert ctx.finish()[0].threshold_revision == 4

    def test_the_event_row_records_it_from_the_VERDICT(self):
        """⚠ FROM THE VERDICT, NOT THE PROBE ROW. A revision read from the row at write time would
        record the number that is WRONG in exactly the case this field exists for — a database
        write whose registry refresh failed. Same defect, one layer down."""
        import ast
        import inspect

        from millm.services import probe_event_service

        source = inspect.getsource(probe_event_service.ProbeEventService.record)
        tree = ast.parse(inspect.cleandoc(source))
        found = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "verdict"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "threshold_revision"
        ]
        assert found, (
            "the event row does not take threshold_revision from `verdict` — if it reads the "
            "probe row instead, a failed registry refresh becomes invisible on the one surface "
            "that must show it"
        )

    def test_event_summary_carries_it_to_every_reader(self):
        """One shape serves the REST list and the socket, so a field missing here is missing from
        both — and the UI event row is where two differently-judged verdicts are told apart."""
        from millm.services.probe_event_service import event_summary

        class _Event:
            id = 1
            probe_id = PROBE_ID
            request_id = "req"
            scored = True
            not_scored_reason = None
            score = 12.0
            threshold = 11.9144
            verdict = True
            rung = 2
            top_positions = [0]
            n_scored_tokens = 8
            summary = "score 12.0000 above threshold"
            window = "all"
            provisional = False
            threshold_revision = 3
            created_at = None

        assert event_summary(_Event())["threshold_revision"] == 3

    def test_status_reports_the_LIVE_bar_and_the_rows_SIDE_BY_SIDE(self):
        """⚠ THE ONLY SURFACE THAT CAN SEE A FAILED REFRESH. `GET /api/probes` serialises the row,
        so without this the UI would show the new number while the monitor used the old one."""
        import inspect

        from millm.services import probe_event_service

        source = inspect.getsource(probe_event_service.ProbeEventService.status)
        for field in (
            '"threshold": bars_by_id',
            '"row_threshold": p.threshold',
            '"threshold_disagreement": _bar_disagreement',
        ):
            assert field in source, f"status() does not report {field}"

    def test_the_disagreement_is_NOT_folded_into_paused_reason(self):
        """A probe judging against a previous bar IS still scoring. Conflating the two would
        dilute the one field whose whole job is "a probe never goes silently quiet"."""
        import inspect

        from millm.services import probe_event_service

        source = inspect.getsource(probe_event_service.ProbeEventService.status)
        # The disagreement helper must not be referenced inside `_paused_reason`'s body.
        paused = source.split("def _paused_reason", 1)[1].split("return {", 1)[0]
        assert "_bar_disagreement" not in paused
        assert "threshold" not in paused
