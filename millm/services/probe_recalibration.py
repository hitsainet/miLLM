"""Move an imported probe's decision bar in place, without touching its detector.

⚠ **WHY THIS DOES NOT REOPEN `on_conflict=replace`.**

`replace` is refused on import, normatively, in five places, and the reason is always the same:
overwriting a definition in place while its probe is ARMED would change the detector underneath a
running monitor while every event before and after kept the same `probe_id` — the history would
describe two different detectors as one.

That objection turns on a specific property of `probe_events`: the row records a `score` whose
MEANING comes from the detector, and nothing on the row records which detector produced it. Change
the weights and event #1's `score = 2.9` and event #900's `score = 2.9` are measurements of
different quantities under one id, with nothing to tell them apart.

A moved bar is not that, for one concrete reason: **the event row already records the bar it was
judged against, per verdict, at judgement time** — including the length-band override — and nothing
joins an event back to `probes.threshold`. After a re-cut, event #1 still says it was judged at
2.8786 and event #900 says 2.4011; both statements are true and both remain comparable, because the
score beneath each was produced by the same weights at the same layer under the same scope with the
same rule. The score is the measurement; the bar is the line drawn across it.

**THE RULE: a probe's identity is everything that determines its score; its bar is everything that
only determines the cut. The first may never change in place under a probe id. The second may.**

Four refusals keep that true, and this is only defensible with all four:

1. The request carries **nothing but the bar**. `ProbeRecalibrationRequest` is `extra="forbid"`,
   so it is structurally incapable of transporting a detector — not "accepts it and ignores it",
   which is one review away from honouring it.
2. The request must **prove it describes the same probe**, by `provenance.probe_id` and
   `provenance.run_id`. ⚠ There is no digest of a probe head's weights anywhere in
   `mistudio.probe-definition/v1` — `weights_sha256` belongs to the `sae` block, the SAE FILE —
   so identity here is a producer-asserted string, and a stored definition carrying neither field
   **cannot be verified and its bar does not move**. Such a probe goes disarm → delete → import
   like any other rebuild. A head digest would make this provable rather than attested; that is an
   additive change to the producer's contract and must not be improvised on this side, because a
   second interpretation of a canonical encoding across two repos is its own failure mode.
3. The incoming object is **never stored as the definition**. The stored definition becomes
   `{**stored, "decision": incoming}`, so the detector half travels through byte-identically and
   "nothing replaces a definition in place" stays literally true of the weights.
4. A `threshold` with no `target_fpr` and no `threshold_source` is **refused**: a number nobody
   calibrated is not a threshold, and a route that took one would let a caller set a bar no
   calibration produced.

⚠ AND THE GATES ARE NAMED FUNCTIONS, NOT INLINE CONDITIONS. A guard left inline in this repo was
once defeated by `if False:` while a source-scraping test stayed green; the recorded remedy is to
extract the decision, unit-test it by behaviour, and assert the CALL.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Capped so one probe's audit trail cannot grow without bound. The count of dropped entries is
#: retained, because "this probe has been re-cut 60 times" is itself worth not hiding.
HISTORY_LIMIT = 50


class ProbeRecalibrationRefused(Exception):
    """A refusal carrying the code and status every caller should use."""

    def __init__(self, code: str, detail: str, *, status: int = 409) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


def describes_the_same_probe(
    definition: dict[str, Any], probe_id: str, run_id: Optional[str]
) -> None:
    """Refuse unless the incoming cut names the probe this row actually holds.

    ⚠ AN UNVERIFIABLE DEFINITION REFUSES. IT DOES NOT DEFAULT TO YES. A document with no
    `provenance.probe_id` cannot be shown to describe this detector, and the whole carve-out above
    rests on being able to show that.
    """
    provenance = (definition or {}).get("provenance") or {}
    stored_probe = provenance.get("probe_id")
    if not stored_probe:
        raise ProbeRecalibrationRefused(
            "probe_recalibration_unverifiable",
            "this probe's definition records no provenance.probe_id, so a re-cut cannot be shown "
            "to describe the same detector. Moving its bar is refused; rebuild it through "
            "disarm -> delete -> import",
        )
    if stored_probe != probe_id:
        raise ProbeRecalibrationRefused(
            "probe_recalibration_mismatch",
            f"this definition was produced for probe {stored_probe!r} and the re-cut names "
            f"{probe_id!r} — a bar may only move on the probe it was cut for",
        )
    stored_run = provenance.get("run_id")
    if run_id and stored_run and stored_run != run_id:
        raise ProbeRecalibrationRefused(
            "probe_recalibration_mismatch",
            f"this definition came from run {stored_run!r} and the re-cut names {run_id!r} — the "
            f"same probe id from a different fit is a different detector",
        )


def bar_is_calibrated(decision: dict[str, Any]) -> None:
    """Refuse a threshold that no calibration produced.

    ⚠ THE BARE-FLOAT DEFENCE, AT FIELD LEVEL. `threshold=None` is legitimate — the probe ranks but
    does not decide. A number WITHOUT a budget and a source is not: it is somebody's opinion
    wearing a measurement's clothes, and every surface downstream would present it as the latter.
    """
    if decision.get("threshold") is None:
        return
    missing = [
        field for field in ("target_fpr", "threshold_source") if decision.get(field) in (None, "")
    ]
    if missing:
        raise ProbeRecalibrationRefused(
            "probe_threshold_uncalibrated",
            f"a threshold of {decision['threshold']} was sent without {' and '.join(missing)}. A "
            f"bar with no budget and no named source is not a calibration, and nothing downstream "
            f"could tell a reader what it spends",
        )


def every_submitted_bar_survives_parsing(decision: dict[str, Any]) -> None:
    """Refuse a submitted window or length table that the runtime parsers would DROP.

    ⚠ THE TOLERANCE THAT IS RIGHT AT IMPORT IS WRONG HERE, AND THIS REFUSAL EXISTS NOWHERE ELSE.
    `window_thresholds_from_definition` silently drops malformed entries and
    `length_bands_from_definition` returns `[]` on any — deliberately, because "a malformed block
    should cost the per-window thresholds, not the arming". At import that is right. At
    recalibration it inverts: the operator asked to MOVE a bar and would get a 200 with no bar
    moved. Run the same two parsers rather than a third copy of the arithmetic, and compare.
    """
    from millm.services.probe_arming import (
        length_bands_from_definition,
        window_thresholds_from_definition,
    )

    wrapped = {"decision": decision}
    submitted_windows = set((decision.get("windows") or {}).keys())
    parsed_windows = set(window_thresholds_from_definition(wrapped).keys())
    dropped = sorted(submitted_windows - parsed_windows)
    if dropped:
        raise ProbeRecalibrationRefused(
            "probe_threshold_uncalibrated",
            f"the per-window bars for {', '.join(dropped)} would be discarded by the runtime's "
            f"own parser, so committing this cut would move no bar for them while reporting "
            f"success. Every window entry needs a numeric threshold",
        )
    if decision.get("length_bands") and not length_bands_from_definition(wrapped):
        raise ProbeRecalibrationRefused(
            "probe_threshold_uncalibrated",
            "the per-length table would be discarded WHOLE by the runtime's own parser — it must "
            "tile every length contiguously from 0 with an open-ended final band, and a torn "
            "table is refused rather than half-applied",
        )
    # ⚠ A WINDOW NO ARMED PROBE CAN REPORT, OR A WINDOW'S OWN BANDS THE PARSER WOULD DROP
    # (2026-10-04) — both would be stored, reported as moved, and never applied.
    from millm.services.probe_arming import window_length_bands_from_definition
    from millm.services.probe_scope import WINDOWS

    unknown = sorted(w for w in submitted_windows if w not in WINDOWS)
    if unknown:
        raise ProbeRecalibrationRefused(
            "probe_threshold_uncalibrated",
            f"window(s) {', '.join(unknown)} are not windows this server can report "
            f"({', '.join(WINDOWS)}), so their bars would be stored and never applied",
        )
    with_bands = sorted(
        w for w, entry in (decision.get("windows") or {}).items()
        if isinstance(entry, dict) and entry.get("length_bands")
    )
    parsed_bands = window_length_bands_from_definition(wrapped)
    torn = [w for w in with_bands if w not in parsed_bands]
    if torn:
        raise ProbeRecalibrationRefused(
            "probe_threshold_uncalibrated",
            f"the per-length tables for window(s) {', '.join(torn)} would be discarded by the "
            f"runtime's own parser — each must tile every length from 0 with an open-ended final "
            f"band",
        )


def history_entry(probe: Any, decision: dict[str, Any], *, revision: int,
                  calibration_id: Optional[str], reason: str) -> dict[str, Any]:
    """One append to `threshold_history`, built from the row BEFORE it is overwritten."""
    return {
        "revision": revision,
        "at": datetime.now(timezone.utc).isoformat(),
        "threshold": decision.get("threshold"),
        "target_fpr": decision.get("target_fpr"),
        "realised_fpr": decision.get("realised_fpr"),
        "threshold_source": decision.get("threshold_source"),
        "calibration_id": calibration_id,
        "windows": sorted((decision.get("windows") or {}).keys()),
        "length_bands": len(decision.get("length_bands") or []),
        "reason": reason,
    }


def seeded_history(probe: Any) -> list[dict[str, Any]]:
    """The probe's history, seeding revision 1 from its own definition if it has none.

    ⚠ WITHOUT THE SEED, AN EVENT STAMPED `revision 1` IS UNANSWERABLE. The row carries only the
    current bar, so the bar the producer's run cut has to be recorded before it is replaced.
    """
    history = list(probe.threshold_history or [])
    if history:
        return history
    decision = (probe.definition or {}).get("decision") or {}
    return [{
        "revision": 1,
        "at": getattr(probe, "created_at", None).isoformat()
        if getattr(getattr(probe, "created_at", None), "isoformat", None)
        else None,
        "threshold": decision.get("threshold", probe.threshold),
        "target_fpr": decision.get("target_fpr", probe.target_fpr),
        "realised_fpr": decision.get("realised_fpr"),
        "threshold_source": decision.get("threshold_source"),
        "calibration_id": None,
        "windows": sorted((decision.get("windows") or {}).keys()),
        "length_bands": len(decision.get("length_bands") or []),
        "reason": "cut by the producer's training run",
    }]


def trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest `HISTORY_LIMIT` entries, recording how many were dropped."""
    if len(history) <= HISTORY_LIMIT:
        return history
    dropped = len(history) - HISTORY_LIMIT
    kept = history[-HISTORY_LIMIT:]
    kept[0] = {**kept[0], "earlier_entries_dropped": dropped}
    return kept


def window_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, list[str]]:
    """Which windows gained and which lost their own bar.

    ⚠ THIS IS WHY A RE-CUT MUST REPORT MORE THAN A NUMBER. A window that gains its own threshold
    retires ONE of its `provisional` reasons — "judged against a bar cut for a different
    distribution" — from the next verdict on. `response` keeps its marker regardless, because its
    weights never saw a reply (`window_weights_trained`), so "newly calibrated" here means it has
    a bar, not that it is measured. But the marker
    then vanishes from the event list with no visible cause, and the reverse direction is worse: a
    window reported calibrated now is not. An operator who moves a number and silently also
    changes which verdicts carry an honesty marker should learn both in the same breath.
    """
    return {
        "windows_newly_calibrated": sorted(set(after) - set(before)),
        "windows_no_longer_calibrated": sorted(set(before) - set(after)),
    }


class ProbeRecalibrationService:
    """Write the new bar, then refresh the live registry. In that order, deliberately."""

    def __init__(self, repository: Any, state: Any = None) -> None:
        from millm.services.probe_runtime import ProbeRuntimeState

        self.repository = repository
        self.state = state or ProbeRuntimeState()

    async def recalibrate(
        self,
        probe: Any,
        *,
        decision: dict[str, Any],
        mistudio_probe_id: str,
        mistudio_run_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Move this probe's bar. Every gate runs before any write."""
        from millm.services.probe_arming import (
            length_bands_from_definition,
            window_length_bands_from_definition,
            window_thresholds_from_definition,
        )

        # ── gates, cheapest first, and NOTHING has been written yet ──────────────────
        describes_the_same_probe(probe.definition, mistudio_probe_id, mistudio_run_id)
        bar_is_calibrated(decision)
        every_submitted_bar_survives_parsing(decision)

        previous_threshold = probe.threshold
        previous_revision = int(getattr(probe, "threshold_revision", 1) or 1)
        revision = previous_revision + 1
        before_windows = window_thresholds_from_definition(probe.definition)

        # ⚠ A NEW DICT, NOT AN IN-PLACE MUTATION. `Probe.definition` is a plain `JSONVariant`
        # column, NOT `MutableDict.as_mutable`, so `probe.definition["decision"] = ...` is not
        # seen as dirty: SQLAlchemy would write the projected `threshold` column and leave the
        # definition untouched, producing exactly the row-vs-definition disagreement the model's
        # own docstring forbids — silently, with a 200 response.
        new_definition = {**(probe.definition or {}), "decision": decision}
        history = trim_history(
            seeded_history(probe)
            + [history_entry(
                probe, decision, revision=revision,
                calibration_id=calibration_id, reason=reason,
            )]
        )

        # ⚠ ONE `update` CALL IS ONE TRANSACTION. Split in two, there is a window in which the
        # column and the definition disagree — and the model's doctrine is that nothing writes a
        # projected column without writing the definition in the same transaction.
        await self.repository.update(
            probe,
            definition=new_definition,
            threshold=decision.get("threshold"),
            target_fpr=decision.get("target_fpr"),
            threshold_revision=revision,
            threshold_calibration_id=calibration_id,
            threshold_history=history,
        )

        # ⚠ THE REGISTRY AFTER THE COMMIT, WHICH IS THE OPPOSITE ORDER FROM ARMING — and the
        # failure modes are not symmetric. Row-ahead-of-registry is a state `status()` reports,
        # whose worst consequence is verdicts judged at the previous revision, which the event
        # rows themselves state. Registry-ahead-of-row would be a NEW unattributable state:
        # events carrying a revision that exists in no row, against a bar nothing can look up.
        live = self.state.get(probe.id)
        registry_updated = False
        after_windows = before_windows
        if live is not None:
            # ⚠ `dataclasses.replace` ON THE LIVE OBJECT, NEVER A REBUILD FROM THE ROW. `encoder`
            # is built at arm time and `windows` come from the arm REQUEST; neither is recoverable
            # from the row, which is why `status()` reads windows out of this registry. A rebuild
            # would turn a k-sparse probe into a dense one reading raw residuals through a
            # narrow head, and reset the operator's window choice to the default.
            after_windows = window_thresholds_from_definition(new_definition)
            registry_updated = self.state.refresh(
                dataclasses.replace(
                    live,
                    threshold=decision.get("threshold"),
                    window_thresholds=after_windows,
                    length_bands=length_bands_from_definition(new_definition),
                    # A window's own bands move with its bar, or the live verdicts keep the old ones.
                    window_length_bands=window_length_bands_from_definition(new_definition),
                    threshold_revision=revision,
                )
            )

        delta = window_delta(before_windows, after_windows)
        if delta["windows_no_longer_calibrated"]:
            # The worse direction: a window reported calibrated until now is not any more, and
            # its verdicts become `provisional` again from the next one on.
            logger.warning(
                "probe_recalibrated_window_decalibrated probe_id=%s windows=%s revision=%s",
                probe.id, delta["windows_no_longer_calibrated"], revision,
            )
        logger.info(
            "probe_recalibrated probe_id=%s %s -> %s revision=%s->%s armed=%s "
            "registry_updated=%s reason=%s",
            probe.id, previous_threshold, decision.get("threshold"),
            previous_revision, revision, bool(probe.armed), registry_updated, reason or "-",
        )

        return {
            "id": probe.id,
            "threshold": decision.get("threshold"),
            "target_fpr": decision.get("target_fpr"),
            "threshold_revision": revision,
            "previous_threshold": previous_threshold,
            "previous_revision": previous_revision,
            "armed": bool(probe.armed),
            "registry_updated": registry_updated,
            # ⚠ A ROW SAYING ARMED WITH NO LIVE ENTRY. Reported, never reconciled — `main.py`
            # reconciles at startup, which is where the state actually becomes wrong.
            "stale_armed": bool(probe.armed) and live is None,
            "length_bands": len(decision.get("length_bands") or []),
            **delta,
        }
