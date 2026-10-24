"""`/api/rule-sets` — rule sets as data (039 FR-39.17, FR-39.19, FR-39.25; FTDD §5.2).

The import and validate bodies ARE the producer's rule-set document, posted unchanged (a published
example set imports byte for byte); the import's own parameters (`version`, `digest`, `origin`)
ride the query string. That is what lets the size check run on the raw bytes BEFORE anything is
parsed (`RULE_SET_MAX_BYTES`).

`GET /validators` is declared above `GET /{rule_set_id}` so the literal path is not read as an id.

⚠ No route here returns user text except `POST /{id}/evaluate`, which returns matched spans of the
CALLER'S OWN text, and only with `return_rule_positions: true`. Nothing it evaluates is written.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from millm.api.dependencies import DbSession
from millm.api.schemas.common import ApiResponse, PageResponse
from millm.core.config import settings

router = APIRouter(prefix="/api/rule-sets", tags=["rule-sets"])

RULE_SET_ID = r"^rs_[0-9a-f]{12}$"
DIGEST = r"^[0-9a-f]{64}$"


class RuleSetEvaluateRequest(BaseModel):
    """The caller's own text. `messages` (chat) or `text` (a prompt), and optionally the reply."""

    model_config = ConfigDict(extra="forbid")

    messages: Optional[list[dict[str, Any]]] = Field(default=None, max_length=1000)
    text: Optional[str] = Field(default=None, max_length=1_000_000)
    response: Optional[str] = Field(default=None, max_length=1_000_000)
    windows: Optional[list[Literal["prompt", "last_user", "response", "all"]]] = None
    return_rule_positions: bool = False


async def _body(request: Request) -> bytes:
    limit = int(settings.RULE_SET_MAX_BYTES)
    raw = await request.body()
    if len(raw) > limit:
        from millm.core.errors import RuleSetInvalidError

        raise RuleSetInvalidError(
            f"the document is {len(raw)} bytes; the limit is {limit}",
            details={"rule": "rule_set_too_large", "bytes": len(raw), "limit": limit,
                     "setting": "RULE_SET_MAX_BYTES"})
    return raw


@router.get("/validators", response_model=ApiResponse)
async def list_validators() -> ApiResponse:
    """Every registered validator with its parameter schema, read from the LIVE registry, and
    the forms this miLLM evaluates (FR-39.19)."""
    from millm.rules import forms
    from millm.rules.validators import VALIDATORS, declaration

    return ApiResponse.ok({
        "validators": [declaration(VALIDATORS[k]) for k in sorted(VALIDATORS)],
        "forms": forms.listing(),
    })


@router.post("/validate", response_model=ApiResponse)
async def validate_rule_set(
    request: Request,
    digest: Annotated[Optional[str], Query(pattern=DIGEST)] = None,
    origin: Annotated[Optional[Literal["authored", "derived"]], Query()] = None,
) -> ApiResponse:
    """Every check import runs — schema, dialect, compile, the set's own cases — and WRITES
    NOTHING (the agent's dry run). The same refusal import would give."""
    from millm.services import rule_set_service

    prepared = rule_set_service.check(await _body(request), declared_digest=digest, origin=origin)
    return ApiResponse.ok({
        "valid": True,
        "digest": prepared.digest,
        "origin": prepared.origin,
        "behaviour_key": prepared.document["behaviour_key"],
        "cases_passed": prepared.cases.passed,
        "cases_unverified": prepared.cases.unverified,
        "pending_signals": sorted(prepared.compiled.pending_signals),
    })


@router.post("/import", response_model=ApiResponse)
async def import_rule_set(
    request: Request,
    session: DbSession,
    version: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    digest: Annotated[Optional[str], Query(pattern=DIGEST)] = None,
    origin: Annotated[Optional[Literal["authored", "derived"]], Query()] = None,
) -> ApiResponse:
    """Validate and store one rule set; an identical document returns the existing id."""
    from millm.services import rule_set_service

    row, created = await rule_set_service.import_rule_set(
        session, await _body(request), version=version, declared_digest=digest, origin=origin)
    return ApiResponse.ok({**rule_set_service.summary(row), "created": created})


@router.get("", response_model=PageResponse)
async def list_rule_sets(
    session: DbSession,
    behaviour: Annotated[Optional[str], Query(max_length=64)] = None,
    cursor: Annotated[Optional[str], Query(max_length=2048)] = None,
    limit: Annotated[Optional[int], Query(ge=1, le=1000)] = None,
) -> ApiResponse:
    """Keyset pages by `(behaviour_key, id)` (FR-39.15)."""
    from millm.api.pagination import decode_cursor, encode_cursor, page_meta
    from millm.db.repositories.rule_set_repository import RuleSetRepository
    from millm.services import rule_set_service

    size = int(limit or settings.PROBE_LIST_PAGE_SIZE)
    after = decode_cursor("rule-sets", cursor, arity=2)
    rows = await RuleSetRepository(session).page(behaviour=behaviour, after=after,
                                                limit=size + 1)
    more = len(rows) > size
    rows = rows[:size]
    next_cursor = (encode_cursor("rule-sets", [rows[-1].behaviour_key, rows[-1].id])
                   if more and rows else None)
    return PageResponse.page([rule_set_service.summary(r) for r in rows],
                             page_meta(next_cursor, size))


@router.get("/{rule_set_id}", response_model=ApiResponse)
async def get_rule_set(
    rule_set_id: Annotated[str, Path(pattern=RULE_SET_ID)], session: DbSession,
) -> ApiResponse:
    from millm.db.repositories.rule_set_repository import RuleSetRepository
    from millm.services import rule_set_service

    row = await rule_set_service.get_or_404(session, rule_set_id)
    attached = await RuleSetRepository(session).attachments_naming(rule_set_id)
    return ApiResponse.ok(rule_set_service.detail(row, sorted(a.behaviour_key for a in attached)))


@router.delete("/{rule_set_id}", response_model=ApiResponse)
async def delete_rule_set(
    rule_set_id: Annotated[str, Path(pattern=RULE_SET_ID)], session: DbSession,
    reason: Annotated[Optional[str], Query(max_length=500)] = None,
) -> ApiResponse:
    from millm.services import rule_set_service

    return ApiResponse.ok(await rule_set_service.delete_rule_set(
        session, rule_set_id, by="operator", reason=reason))


@router.post("/{rule_set_id}/evaluate", response_model=ApiResponse)
async def evaluate_rule_set(
    rule_set_id: Annotated[str, Path(pattern=RULE_SET_ID)],
    body: RuleSetEvaluateRequest,
    session: DbSession,
) -> ApiResponse:
    """Run one stored set on the caller's text (FR-39.25). Writes nothing. Operator code is never
    run (D41): its expressions report `rule_worker_unavailable`. Spans only on request."""
    import asyncio
    import time

    from millm.core.errors import ProbeScoreRequestError
    from millm.rules import registry as rule_registry
    from millm.rules.serving import evaluate_work
    from millm.rules.windows import window_texts
    from millm.services import rule_set_service

    if (body.messages is None) == (body.text is None):
        raise ProbeScoreRequestError("send exactly one of `messages` or `text`",
                                     details={"param": "messages"},
                                     next_tool="millm_evaluate_rule_set")
    row = await rule_set_service.get_or_404(session, rule_set_id)
    compiled, failed = rule_registry.compile_row(row)
    attached = rule_registry.AttachedSet(
        rule_set_id=row.id, digest=row.digest, version=int(row.version or 1),
        behaviour_key=row.behaviour_key, compiled=compiled, vectors_failed=failed, only=None,
        forms=rule_registry.document_forms(row.document))
    texts = window_texts(messages=body.messages, prompt=body.text, reply_text=body.response)
    windows = tuple(body.windows or [w for w in ("prompt", "last_user", "response", "all")
                                     if w in texts.texts])
    # Off the event loop (review R1): a long text through every validator must not stall the
    # loop that streams every other request's tokens. The live path does the same (`rules_finish`).
    items = await asyncio.to_thread(
        evaluate_work, [(row.behaviour_key, windows, attached)], texts,
        clock=time.monotonic, deadline=None, return_positions=body.return_rule_positions)
    return ApiResponse.ok({
        "rule_set_id": row.id,
        "rule_set_digest": row.digest,
        "rule_set_version": row.version,
        "behaviour": {"key": row.behaviour_key, "version": None},
        "results": [
            {"kind": "rule", "signal": i.signal, "form": i.form, "window": i.window,
             "fires": i.fires, "count": i.count, "outcome": i.outcome, "reason": i.reason,
             "status": i.status,
             **({"spans": [list(s) for s in (i.spans or ())]}
                if body.return_rule_positions and i.outcome == "completed" else {})}
            for i in items
        ],
    })
