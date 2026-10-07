"""Validate every line of a batch before any row runs (Feature 26, FR-26.2, FTID §7).

Validation is CPU work and takes NO request slot (FR-26.2.9). Parsing runs in a worker thread
(`asyncio.to_thread`); the only async step is resolving each distinct `body.model` name once.

Per line, in this order (FTID §7): size cap → JSON → line shape (`custom_id`, `method`, `url`,
`body`, duplicate `custom_id`) → `stream: true` → the endpoint's request schema → model
resolution → Feature 25's policy with `strict=True` whatever the create request said (FR-26.2.3)
→ the route's own `validate_<endpoint>()` → kind. Each refusal carries the SAME code and message
the synchronous endpoint gives (FR-26.2.4), because it is produced by the same functions.

The outcome is a list of row dicts ready for `bulk_insert_rows`, the single resolved model, and
the OpenAI `errors` list. Deciding the batch's next status is the caller's (the runner's).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from pydantic import ValidationError

from millm.core.batch_values import RowKind, RowState
from millm.core.config import settings
from millm.core.errors import MiLLMError
from millm.services.batch.state import error_line

#: The cancel flag is checked every this many lines (FTASKS 4.4).
CANCEL_CHECK_EVERY = 1000

PROBE_ENDPOINT = "/api/probes/score"


class ValidationCancelled(Exception):
    """The batch was cancelled while validating; it ends `cancelled` without running (FR-26.6.4)."""


@dataclass
class Candidate:
    """A line that passed parsing, shape and schema; model and policy are checked next."""

    line_no: int
    offset: int
    length: int
    custom_id: str
    request: Any
    model_name: Optional[str]


@dataclass
class ValidationOutcome:
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    invalid: int = 0
    #: model id -> (name, valid line count), for the single-model rule (FR-26.2.7).
    models: dict[int, tuple[str, int]] = field(default_factory=dict)

    @property
    def valid(self) -> int:
        return self.total - self.invalid


def request_model_for(endpoint: str) -> Any:
    """The endpoint's own request schema — the one the synchronous route validates against."""
    from millm.api.schemas.openai import (
        ChatCompletionRequest,
        EmbeddingRequest,
        TextCompletionRequest,
    )
    from millm.api.schemas.probe_scoring import ProbeScoreRequest

    return {
        "/v1/chat/completions": ChatCompletionRequest,
        "/v1/completions": TextCompletionRequest,
        "/v1/embeddings": EmbeddingRequest,
        PROBE_ENDPOINT: ProbeScoreRequest,
    }[endpoint]


def kind_of(endpoint: str, request: Any) -> RowKind:
    """FTID §7 step 9. Scoring on completions is `wants_scores()`; on chat, Feature 25's predicate."""
    if endpoint == PROBE_ENDPOINT:
        return RowKind.PROBE
    if endpoint == "/v1/embeddings":
        return RowKind.EMBEDDING
    return RowKind.SCORING if request.wants_scores() else RowKind.GENERATION


def _pydantic_reason(exc: ValidationError) -> tuple[str, Optional[str]]:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", ()))
    message = str(first.get("msg", "invalid request"))
    if message.startswith("Value error, "):
        message = message[len("Value error, "):]
    param = f"body.{loc}" if loc else "body"
    return f"Invalid value for '{param}': {message}", param


class _Collector:
    """Accumulates rows and errors in line order."""

    def __init__(self) -> None:
        self.outcome = ValidationOutcome()
        self.shown = int(settings.BATCH_ERRORS_SHOWN)

    def invalid(
        self, line_no: int, offset: int, length: int, custom_id: Optional[str], code: str,
        message: str, param: Optional[str] = None,
    ) -> None:
        self.outcome.invalid += 1
        self.outcome.rows.append({
            "line_no": line_no, "custom_id": custom_id, "kind": None, "byte_offset": offset,
            "byte_length": length, "state": RowState.INVALID.value,
            "result": error_line(custom_id, line_no, None, code, message),
        })
        self.outcome.errors.append(
            {"code": code, "line": line_no, "message": message, "param": param}
        )


def parse_lines(
    lines: Any, endpoint: str, cancelled: Callable[[], bool], collector: _Collector
) -> list[Candidate]:
    """Pass 1 (thread): size, JSON, shape, duplicates, `stream`, schema. Model names collected."""
    schema = request_model_for(endpoint)
    seen: set[str] = set()
    candidates: list[Candidate] = []
    limit = int(settings.BATCH_MAX_LINE_BYTES)
    for index, (offset, raw) in enumerate(lines):
        line_no = index + 1
        collector.outcome.total += 1
        if line_no % CANCEL_CHECK_EVERY == 0 and cancelled():
            raise ValidationCancelled()
        length = len(raw)
        if length > limit:
            collector.invalid(line_no, offset, length, None, "line_too_large",
                              f"Line {line_no} is {length} bytes; the limit is {limit} "
                              "(BATCH_MAX_LINE_BYTES).")
            continue
        try:
            obj = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            collector.invalid(line_no, offset, length, None, "invalid_json",
                              f"Line {line_no} is not valid JSON: {exc}")
            continue
        if not isinstance(obj, dict):
            collector.invalid(line_no, offset, length, None, "invalid_line",
                              f"Line {line_no} is not a JSON object.")
            continue
        custom_id = obj.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id or len(custom_id) > 512:
            collector.invalid(line_no, offset, length, None, "invalid_custom_id",
                              f"Line {line_no}: custom_id must be a non-empty string of at most "
                              "512 characters.", "custom_id")
            continue
        if obj.get("method") != "POST":
            collector.invalid(line_no, offset, length, custom_id, "invalid_method",
                              f"Line {line_no}: method must be 'POST', got {obj.get('method')!r}.",
                              "method")
            continue
        if obj.get("url") != endpoint:
            collector.invalid(line_no, offset, length, custom_id, "invalid_url",
                              f"Line {line_no}: url must equal the batch's endpoint {endpoint!r}, "
                              f"got {obj.get('url')!r}.", "url")
            continue
        body = obj.get("body")
        if not isinstance(body, dict):
            collector.invalid(line_no, offset, length, custom_id, "invalid_body",
                              f"Line {line_no}: body must be a JSON object.", "body")
            continue
        if custom_id in seen:
            collector.invalid(line_no, offset, length, custom_id, "duplicate_custom_id",
                              f"Line {line_no}: custom_id {custom_id!r} appears on an earlier "
                              "line; every custom_id must be unique.", "custom_id")
            continue
        seen.add(custom_id)
        if body.get("stream") is True:
            collector.invalid(line_no, offset, length, custom_id, "stream_not_supported",
                              f"Line {line_no}: stream=true is not supported in a batch.",
                              "body.stream")
            continue
        try:
            request = schema.model_validate(body)
        except ValidationError as exc:
            message, param = _pydantic_reason(exc)
            collector.invalid(line_no, offset, length, custom_id, "invalid_request",
                              f"Line {line_no}: {message}", param)
            continue
        candidates.append(Candidate(
            line_no, offset, length, custom_id, request, getattr(request, "model", None)
        ))
    return candidates


def check_candidates(
    candidates: list[Candidate],
    endpoint: str,
    rows_by_name: dict[Optional[str], Any],
    refusal_for_name: dict[Optional[str], tuple[str, str, int]],
    cbm_enabled: bool,
    cancelled: Callable[[], bool],
    collector: _Collector,
) -> None:
    """Pass 2 (thread): model, strict policy, the route's own checks, kind."""
    from millm.api.routes.openai.chat import validate_chat
    from millm.api.routes.openai.completions import validate_completions
    from millm.api.routes.openai.embeddings import validate_embeddings
    from millm.api.routes.openai.errors import OpenAIRefusal

    checks: dict[str, Callable[[Any, Any], Any]] = {
        "/v1/chat/completions": lambda r, m: validate_chat(r, m, strict=True, cbm_enabled=cbm_enabled),
        "/v1/completions": lambda r, m: validate_completions(r, m, strict=True),
        "/v1/embeddings": lambda r, m: validate_embeddings(r, m, strict=True),
    }
    counts: Counter[int] = Counter()
    for done, c in enumerate(candidates, start=1):
        if done % CANCEL_CHECK_EVERY == 0 and cancelled():
            raise ValidationCancelled()
        row = rows_by_name.get(c.model_name)
        if row is None:
            code, message, _status = refusal_for_name[c.model_name]
            collector.invalid(c.line_no, c.offset, c.length, c.custom_id, code,
                              f"Line {c.line_no}: {message}", "body.model")
            continue
        check = checks.get(endpoint)
        if check is not None:
            try:
                check(c.request, row)
            except OpenAIRefusal as refusal:
                error = refusal.body()["error"]
                collector.invalid(c.line_no, c.offset, c.length, c.custom_id, str(error["code"]),
                                  f"Line {c.line_no}: {error['message']}", error.get("param"))
                continue
            except MiLLMError as exc:
                param = exc.details.get("param") if isinstance(exc.details, dict) else None
                collector.invalid(c.line_no, c.offset, c.length, c.custom_id, exc.code.lower(),
                                  f"Line {c.line_no}: {exc.message}",
                                  param if isinstance(param, str) else None)
                continue
        counts[int(row.id)] += 1
        collector.outcome.models.setdefault(int(row.id), (row.name, 0))
        collector.outcome.rows.append({
            "line_no": c.line_no, "custom_id": c.custom_id,
            "kind": kind_of(endpoint, c.request).value, "byte_offset": c.offset,
            "byte_length": c.length, "state": RowState.PENDING.value, "result": None,
        })
    for model_id, n in counts.items():
        collector.outcome.models[model_id] = (collector.outcome.models[model_id][0], n)
    collector.outcome.rows.sort(key=lambda r: r["line_no"])


async def validate_file(
    *,
    lines: Any,
    endpoint: str,
    resolve_model: Callable[[str], Awaitable[Any]],
    resident_row: Callable[[], Awaitable[Any]],
    cbm_enabled: bool,
    cancelled: Callable[[], bool],
) -> ValidationOutcome:
    """Validate every line. `lines` yields `(byte_offset, line_bytes)` (FileStore.iter_lines).

    For `/api/probes/score` the line body names no model (027's schema is `extra="forbid"` and has
    no `model` field), so every line resolves to the RESIDENT model (recorded discrepancy with
    FR-26.2.7's "every line names `body.model`").
    """
    import asyncio

    collector = _Collector()
    candidates = await asyncio.to_thread(parse_lines, lines, endpoint, cancelled, collector)

    rows_by_name: dict[Optional[str], Any] = {}
    refusals: dict[Optional[str], tuple[str, str, int]] = {}
    for name in {c.model_name for c in candidates}:
        if endpoint == PROBE_ENDPOINT:
            row = await resident_row()
            if row is None:
                refusals[name] = ("model_not_loaded", "No model is loaded; probe scoring reads "
                                  "the resident model.", 503)
            rows_by_name[name] = row
            continue
        try:
            row = await resolve_model(name)
        except MiLLMError as exc:
            refusals[name] = (exc.code.lower(), exc.message, exc.status_code)
            rows_by_name[name] = None
            continue
        if row is None:
            refusals[name] = (
                "model_not_found",
                f"The model '{name}' does not exist or has not been downloaded. Download it "
                "first using the Management API.",
                404,
            )
        rows_by_name[name] = row

    await asyncio.to_thread(
        check_candidates, candidates, endpoint, rows_by_name, refusals, cbm_enabled, cancelled,
        collector,
    )
    # Pass 2's refusals arrive after pass 1's: order by line, then keep the first SHOWN
    # (`errors.data`); the error FILE carries every one.
    collector.outcome.errors.sort(key=lambda e: e["line"])
    del collector.outcome.errors[collector.shown:]
    return collector.outcome
