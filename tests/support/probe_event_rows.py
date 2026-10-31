"""A `probe_events` row for tests that insert one directly (Feature 038, migration 032).

Since 032 every row stores its `mi.probe.result/v1` object and the contract columns are NOT NULL.
Production writes them in `ProbeEventService.record` from the finalized result; a test that seeds
rows directly gets them here FROM THE SAME BUILDER (`probe_result.build_probe_results`), over a
verdict made of the row's own legacy fields — never a hand-written result. The fields the test
passes win, so a seeded row keeps exactly the values its assertions read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

CONTRACT_COLUMNS = (
    "result", "result_id", "trace_id", "span_id", "event_time", "event_time_source", "outcome",
    "reason", "status", "persistence", "detector_digest", "detector_digest_source",
    "identity_algorithm", "decision_digest", "decision_revision", "migrated",
)


def event_row(*, behaviour: Any = None, **fields: Any) -> dict[str, Any]:
    """`fields` plus every contract column, built by the production builder.

    `behaviour` (`{key, version}`) is what the ARMED probe's descriptor would have put in the
    stored result — production records always carry it (039: the events list reads it from the
    stored result, never from today's definition)."""
    from millm.services.probe_result import build_probe_results, legacy_columns, stored_result
    from millm.services.probe_runtime import Verdict

    scored = bool(fields.get("scored", True))
    score = fields.get("score")
    verdict = Verdict(
        probe_id=fields["probe_id"], name=str(fields.get("name", "probe")),
        rung=int(fields.get("rung") or 2), rung_language="detects on unseen tasks",
        scored=scored, score=score, threshold=fields.get("threshold"),
        fires=fields.get("verdict"), not_scored_reason=fields.get("not_scored_reason"),
        n_scored_tokens=int(fields.get("n_scored_tokens") or (1 if scored else 0)),
        window=fields.get("window", "all"), provisional=bool(fields.get("provisional", False)),
        threshold_revision=int(fields.get("threshold_revision") or 1),
    )
    context = SimpleNamespace(request_id=fields.get("request_id"), probes=(), session=None,
                              trace=None, interrupted=None)
    moment = fields.get("created_at") or datetime.now(timezone.utc)
    result = build_probe_results([verdict], context, now=moment)[0]
    # The seeded row's correlation fields are the result's too, as `record` writes them.
    carried = {k: fields[k] for k in ("session_id", "turn_index", "batch_id", "batch_line",
                                      "trace_id", "span_id",
                                      "origin", "served_model") if fields.get(k) is not None}
    if behaviour is not None:
        from millm.api.schemas.probe_result import BehaviourOut

        carried["behaviour"] = BehaviourOut(**behaviour)
    if carried:
        result = result.model_copy(update=carried)
    columns = legacy_columns(result)
    columns["result"] = stored_result(result)
    return {**{k: columns[k] for k in CONTRACT_COLUMNS}, **fields}
