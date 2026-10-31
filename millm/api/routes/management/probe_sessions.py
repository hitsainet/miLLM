"""`/api/probes/sessions` — the session read API (Feature 36, FR-36.9, FR-36.11, FR-36.12).

⚠ INCLUDED BEFORE `probes_router` (`millm/api/routes/__init__.py`). That router declares
`GET /api/probes/{probe_id}`, which would otherwise match `sessions` as a probe id and answer 404
`PROBE_NOT_FOUND`. `test_probe_session_routes.py::test_probe_session_routes_are_served` asserts a
200 with an empty list, so moving the include after the probes router turns it red.

⚠ USER TEXT IS SERVED BY ONE ROUTE ONLY: `GET /api/probes/sessions/{id}/transcript`. The list and
the timeline say WHETHER a turn has text (`has_transcript`) and never what it is; a test asserts the
absence of the text keys on both.

The first consumer is miStudio's spot-check runner (miStudio 038), which pins these names from
`app.openapi()` as this repo does (`tests/unit/api/test_probe_session_api_names.py`).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from millm.api.dependencies import DbSession
from millm.api.schemas.common import ApiResponse, PageResponse
from millm.core.errors import ProbeSessionNotFoundError
from millm.core.probe_labels import behaviour_of, source_probe_id_of
from millm.db.repositories.probe_session_repository import ProbeSessionRepository
from millm.services import probe_sessions

from millm.services.probe_result import recorded_from_row  # noqa: E402  (038)

router = APIRouter(prefix="/api/probes/sessions", tags=["probe-sessions"])

SessionId = Annotated[str, Path(min_length=1, max_length=24)]


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _column_of(probe: Any) -> tuple[str, Optional[dict[str, Any]]]:
    """The timeline column a probe's verdicts go in: its behaviour key, else its own name (D18)."""
    behaviour = behaviour_of(probe.definition)
    if behaviour is not None:
        return behaviour["key"], behaviour
    return probe.name, None


def _session_row(row: Any, *, n_transcribed: int, behaviours: dict[str, Any]) -> dict[str, Any]:
    now = probe_sessions._now()
    return {
        "id": row.id,
        "client_session_id": row.client_session_id,
        "source": row.source,
        "predecessor_id": row.predecessor_id,
        "link_ambiguous": bool(row.link_ambiguous),
        "served_model": row.served_model,
        "opened_at": _iso(row.opened_at),
        "last_seen_at": _iso(row.last_seen_at),
        "open": probe_sessions.is_open(row.last_seen_at, now),
        "n_turns": int(row.n_turns or 0),
        "n_transcribed": int(n_transcribed),
        "behaviours": behaviours,
    }


def _summaries(pairs: list[tuple[Any, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """`{session_id: {column: {n_fired, n_scored, max_score}}}` from (event, probe) pairs."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for event, probe in pairs:
        column, _ = _column_of(probe)
        cell = out.setdefault(event.session_id, {}).setdefault(
            column, {"n_fired": 0, "n_scored": 0, "max_score": None}
        )
        if event.scored:
            cell["n_scored"] += 1
            if event.score is not None and (
                cell["max_score"] is None or event.score > cell["max_score"]
            ):
                cell["max_score"] = event.score
        if event.verdict is True:
            cell["n_fired"] += 1
    return out


@router.get("", response_model=PageResponse)
async def list_sessions(
    db: DbSession,
    since: Annotated[Optional[datetime], Query(description="ISO 8601")] = None,
    until: Annotated[Optional[datetime], Query(description="ISO 8601")] = None,
    served_model: Annotated[Optional[str], Query(max_length=255)] = None,
    behaviour: Annotated[Optional[str], Query(max_length=64)] = None,
    fired_only: Annotated[bool, Query()] = False,
    source: Annotated[Optional[Literal["header", "fingerprint", "single"]], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0, description="DEPRECATED (039): use cursor")] = 0,
    cursor: Annotated[Optional[str], Query(max_length=2048)] = None,
) -> PageResponse:
    """Sessions, most recently seen first, each with a per-behaviour summary. No text.

    039 (FR-39.15): a keyset `cursor` and `meta.next`; `offset` is kept for one contract version.
    """
    from millm.api.pagination import decode_cursor, encode_cursor, page_meta

    after = decode_cursor("sessions", cursor, arity=2)
    repo = ProbeSessionRepository(db)
    rows = await repo.list_sessions(
        since=since, until=until, served_model=served_model, behaviour=behaviour,
        fired_only=fired_only, source=source, limit=limit + 1,
        offset=offset if after is None else 0, after=after,
    )
    more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = (encode_cursor("sessions", [rows[-1].last_seen_at.isoformat(), rows[-1].id])
                   if more and rows else None)
    ids = [r.id for r in rows]
    transcribed = await repo.transcribed_counts(ids)
    summaries = _summaries(await repo.events_for_sessions(ids))
    return PageResponse.page([
        _session_row(r, n_transcribed=transcribed.get(r.id, 0), behaviours=summaries.get(r.id, {}))
        for r in rows
    ], page_meta(next_cursor, limit))


@router.get("/{session_id}", response_model=ApiResponse)
async def get_session(session_id: SessionId, db: DbSession) -> ApiResponse:
    """One session and its timeline: turns in order × behaviours, each cell the verdicts of that
    behaviour's probes in that turn, per window (FR-36.11). No text."""
    repo = ProbeSessionRepository(db)
    row = await repo.get(session_id)
    if row is None:
        raise ProbeSessionNotFoundError(f"No probe session {session_id}")
    pairs = await repo.events_for_sessions([session_id])
    turns = {t.turn_index: t for t in await repo.turns(session_id)}

    columns: dict[str, dict[str, Any]] = {}
    by_turn: dict[int, list[dict[str, Any]]] = {}
    for event, probe in pairs:
        column, behaviour = _column_of(probe)
        col = columns.setdefault(
            column, {"column": column, "behaviour": behaviour, "probe_ids": []}
        )
        if probe.id not in col["probe_ids"]:
            col["probe_ids"].append(probe.id)
        if event.turn_index is None:
            continue
        by_turn.setdefault(int(event.turn_index), []).append({
            "column": column,
            "event_id": event.id,
            "probe_id": probe.id,
            "probe_name": probe.name,
            "window": event.window,
            "scored": bool(event.scored),
            "not_scored_reason": event.not_scored_reason,
            "verdict": event.verdict,
            "score": event.score,
            "threshold": event.threshold,
            "provisional": bool(event.provisional),
            # 038 FR-38.3: the stored contract object (recorded copy) beside the legacy keys —
            # the session timeline is a sixth serializer no longer.
            **(recorded_from_row(event) or {}),
        })
    # 039 S2 (FR-39.23, 14.3): rule results join their behaviour's column as their OWN entries
    # (`kind: "rule"`, a count and never a score), after that turn's probe entries, by (kind, id).
    from millm.db.repositories.rule_set_repository import RuleSetRepository

    for rule in await RuleSetRepository(db).results_for_session(session_id):
        column = rule.behaviour_key or f"rule:{rule.rule_set_id}"
        col = columns.setdefault(column, {"column": column, "behaviour": {
            "key": rule.behaviour_key, "version": None} if rule.behaviour_key else None,
            "probe_ids": []})
        col.setdefault("rule_set_ids", [])
        if rule.rule_set_id not in col["rule_set_ids"]:
            col["rule_set_ids"].append(rule.rule_set_id)
        if rule.turn_index is None:
            continue
        by_turn.setdefault(int(rule.turn_index), []).append({
            "column": column,
            "kind": "rule",
            "rule_result_id": rule.id,
            "rule_set_id": rule.rule_set_id,
            "signal": rule.signal,
            "form": rule.form,
            "window": rule.window,
            "fires": rule.fires,
            "count": rule.count,
            "outcome": rule.outcome,
            "reason": rule.reason,
            **(recorded_from_row(rule) or {}),
        })
    timeline = []
    for index in sorted(set(turns) | set(by_turn)):
        turn = turns.get(index)
        timeline.append({
            "turn_index": index,
            "request_id": turn.request_id if turn is not None else None,
            "created_at": _iso(turn.created_at) if turn is not None else None,
            "has_transcript": bool(turn.has_transcript) if turn is not None else False,
            "transcript_incomplete": (
                bool(turn.transcript_incomplete) if turn is not None else False
            ),
            "verdicts": by_turn.get(index, []),
        })
    transcribed = sum(1 for t in turns.values() if t.has_transcript)
    detail = _session_row(
        row, n_transcribed=transcribed, behaviours=_summaries(pairs).get(session_id, {})
    )
    detail["columns"] = sorted(columns.values(), key=lambda c: c["column"])
    detail["turns"] = timeline
    return ApiResponse.ok(detail)


@router.get("/{session_id}/transcript", response_model=ApiResponse)
async def get_transcript(
    session_id: SessionId,
    db: DbSession,
    turn_index: Annotated[Optional[int], Query(ge=1)] = None,
) -> ApiResponse:
    """⚠ THE ONLY ROUTE THAT SERVES SESSION TEXT (FR-36.9, D5).

    The newest user message and the visible reply of each recorded turn, in turn order. Turns with
    no text (recording off, or the text expired) are absent; the timeline says which ones have it.
    `turn_index` (phase 4) returns that one turn, or 404 when it has no recorded text — miStudio's
    spot-check reads one sampled turn at a time through it.
    """
    repo = ProbeSessionRepository(db)
    if await repo.get(session_id) is None:
        raise ProbeSessionNotFoundError(f"No probe session {session_id}")
    rows = await repo.transcript(session_id, turn_index=turn_index)
    if turn_index is not None:
        if not rows:
            raise ProbeSessionNotFoundError(
                f"Probe session {session_id} has no recorded text for turn {turn_index}",
                details={"session_id": session_id, "turn_index": turn_index},
            )
        # ONE turn asked for, ONE turn answered — an object, not a list of one: miStudio 038's
        # spot-check reads `{turn_index, user_text, reply_text}` straight off `data`.
        return ApiResponse.ok(_transcript_row(*rows[-1]))
    return ApiResponse.ok([_transcript_row(turn, text) for turn, text in rows])


def _transcript_row(turn: Any, text: Any) -> dict[str, Any]:
    return {
        "turn_index": turn.turn_index,
        "request_id": turn.request_id,
        "user_text": text.user_text,
        "reply_text": text.reply_text,
        "n_choices": int(text.n_choices or 1),
        "incomplete": bool(turn.transcript_incomplete),
        "created_at": _iso(text.created_at),
    }


@router.delete("/{session_id}", response_model=ApiResponse)
async def delete_session(
    session_id: SessionId,
    db: DbSession,
    events: Annotated[Literal["keep", "delete"], Query()] = "keep",
) -> ApiResponse:
    """Delete a session, its turns and its text at once (FR-36.9).

    `events=keep` (the default, TC-8) keeps its verdicts — monitoring evidence — detached from the
    session with their `turn_index`; `events=delete` removes them too. The counts say what went.
    """
    repo = ProbeSessionRepository(db)
    if await repo.get(session_id) is None:
        raise ProbeSessionNotFoundError(f"No probe session {session_id}")
    return ApiResponse.ok(await repo.delete_session(session_id, delete_events=events == "delete"))


# ── the seeded, stratified sample (036 phase 4, FR-36.12, D15) ─────────────────────────────

STRATA = ("fired", "near_threshold", "not_fired")


class SampleStrata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fired: int = Field(0, ge=0, le=10_000)
    near_threshold: int = Field(0, ge=0, le=10_000)
    not_fired: int = Field(0, ge=0, le=10_000)


class SessionSampleRequest(BaseModel):
    """⚠ `extra="forbid"`: a misspelt field (`seeed`) must refuse, not silently sample unseeded."""

    model_config = ConfigDict(extra="forbid")

    behaviour: Optional[str] = Field(None, max_length=64)
    probe_id: Optional[str] = Field(None, max_length=24)
    window: Optional[str] = Field(None, max_length=16)
    strata: SampleStrata
    #: Half-width of the near-threshold band, in units of |threshold| (D54's 10% by default).
    near_band: float = Field(0.1, ge=0.0, le=10.0)
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    seed: int
    require_transcript: bool = True

    @model_validator(mode="after")
    def _one_target(self) -> "SessionSampleRequest":
        if (self.behaviour is None) == (self.probe_id is None):
            raise ValueError("name exactly one of behaviour or probe_id")
        return self


def stratum_of(event: Any, near_band: float) -> Optional[str]:
    """`fired` > `near_threshold` > `not_fired`: disjoint, in that priority (FTDD §5.1).

    A probe that ranks without a bar (`threshold` NULL) decided nothing and belongs to no stratum.

    ⚠ NO SCORE-VS-THRESHOLD ORDERING HERE. Which side of the bar a score is on is decided in ONE
    place, `ProbeRequestContext._verdict_for` (`test_verdict_boundary_is_one_place.py`), and recorded
    as `verdict`. This only measures DISTANCE from the bar: `not_fired` is a recorded `False` verdict
    farther than the band, which is exactly "below threshold − band".
    """
    if event.verdict is True:
        return "fired"
    if event.threshold is None or event.score is None:
        return None
    band = near_band * abs(event.threshold)
    distance = abs(event.score - event.threshold)
    if distance <= band:
        return "near_threshold"
    if event.verdict is False:
        return "not_fired"
    return None


def seeded_key(seed: int, event_id: int) -> str:
    """The order within a stratum: sha256(seed:id) — reproducible, independent of insert time."""
    return hashlib.sha256(f"{seed}:{event_id}".encode()).hexdigest()


@router.post("/sample", response_model=ApiResponse)
async def sample_turns(request: SessionSampleRequest, db: DbSession) -> ApiResponse:
    """A seeded, stratified sample of recorded turns for a judge to re-check (FR-36.12, D15).

    The same seed returns the same rows; a stratum with too few candidates is reported SHORT,
    never padded from another. miStudio's spot-check runner pulls this (D16: miLLM never calls a
    sibling) and reads each turn's text through `/{id}/transcript?turn_index=`.
    """
    repo = ProbeSessionRepository(db)
    candidates = await repo.sample_candidates(
        probe_ids=[request.probe_id] if request.probe_id else None,
        behaviour=request.behaviour, window=request.window, since=request.since,
        until=request.until, require_transcript=request.require_transcript,
    )
    buckets: dict[str, list[tuple[Any, Any, Any]]] = {name: [] for name in STRATA}
    for event, probe, turn in candidates:
        name = stratum_of(event, request.near_band)
        if name is not None:
            buckets[name].append((event, probe, turn))
    quotas = request.strata.model_dump()
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for name in STRATA:
        ordered = sorted(buckets[name], key=lambda r: seeded_key(request.seed, r[0].id))
        picked = ordered[: quotas[name]]
        summary[name] = {
            "requested": quotas[name],
            "returned": len(picked),
            "short": max(quotas[name] - len(picked), 0),
            "available": len(ordered),
        }
        for event, probe, turn in picked:
            rows.append({
                "stratum": name,
                "session_id": event.session_id,
                "request_id": event.request_id,
                "turn_index": event.turn_index,
                "event_id": event.id,
                "probe_id": probe.id,
                "source_probe_id": source_probe_id_of(probe.definition),
                "window": event.window,
                "score": event.score,
                "threshold": event.threshold,
                "verdict": event.verdict,
                "threshold_revision": int(getattr(event, "threshold_revision", 1) or 1),
                "has_transcript": bool(turn.has_transcript),
            })
    return ApiResponse.ok({"seed": request.seed, "near_band": request.near_band,
                           "strata": summary, "turns": rows})
