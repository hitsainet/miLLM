"""Run batch rows through the SYNCHRONOUS service code (Feature 26, FR-26.4.5, FTID §3, §7).

One path to the model: a generation row calls `create_chat_completion`/`create_text_completion`,
an embedding row `create_embeddings`, a probe row 027's `ProbeScoringService.score` — exactly what
the synchronous routes call, each of which takes `_admit()` and re-enters the chunk's slot
(FTDD §7 constraint 1). Packed scoring (task 6) calls `InferenceService._score_specs_packed`, the
same scorer the synchronous path uses with a pack of one. No executor touches the model.

A row that fails ON ITS OWN MERITS (`MiLLMError`) becomes an error line carrying the status code
and body the synchronous route would have returned — produced by the route's own exception
handler (`millm_error_handler`), not a copy of it (FR-26.6.7). Anything else is not a row failure;
it propagates and the runner fails the batch (FTID §12).

Each row runs with `BATCH_ROW` set to `(batch_id, line_no)`: `_use_cbm_for_request` keeps it off
the continuous batching manager and `_probe_record` marks its probe events `origin='batch'`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable, Optional, Sequence

from millm.core.batch_values import RowKind, RowState
from millm.core.errors import MiLLMError
from millm.db.repositories.batch_repository import RowResult
from millm.services.batch.state import BATCH_ROW, error_line

ENDPOINT_NAMES = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completions",
    "/v1/embeddings": "embeddings",
}


def output_line(
    custom_id: str, status_code: int, body: Any, *, packed: bool, headers: dict[str, str]
) -> dict[str, Any]:
    """OpenAI's output line (FR-26.6.6) plus `response.millm` (FR-26.5.6, FR-26.10.1)."""
    return {
        "id": f"batch_req_{uuid.uuid4().hex[:24]}",
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": f"req_{uuid.uuid4().hex[:24]}",
            "body": body,
            "millm": {"packed": packed, "headers": headers},
        },
        "error": None,
    }


async def error_response(endpoint: str, exc: MiLLMError) -> tuple[int, dict[str, Any]]:
    """The status and body the synchronous route answers `exc` with — from its own handler."""
    from starlette.requests import Request

    from millm.api.exception_handlers import millm_error_handler

    request = Request({"type": "http", "method": "POST", "path": endpoint, "headers": []})
    response = await millm_error_handler(request, exc)
    return int(response.status_code), json.loads(response.body)


async def failed_row(row: Any, endpoint: str, exc: MiLLMError, packed: bool) -> RowResult:
    status, body = await error_response(endpoint, exc)
    error = body.get("error") or {}
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    line = error_line(
        row.custom_id, row.line_no, status, str(code or exc.code.lower()), str(message or exc.message),
        body=body,
    )
    return RowResult(row.line_no, RowState.FAILED, line, packed)


def parse_row_request(endpoint: str, raw: bytes) -> Any:
    """Rebuild the request object from the stored line (validated once already)."""
    from millm.services.batch.validator import request_model_for

    return request_model_for(endpoint).model_validate(json.loads(raw)["body"])


class RowExecutor:
    """Runs rows of one batch. Stateless across chunks; built per chunk by the runner."""

    def __init__(
        self,
        *,
        batch_id: str,
        endpoint: str,
        inference: Any,
        model_row: Any,
        read_line: Callable[[Any], Awaitable[bytes]],
        session_factory: Any,
    ) -> None:
        self.batch_id = batch_id
        self.endpoint = endpoint
        self.inference = inference
        self.model_row = model_row
        self.read_line = read_line
        self.session_factory = session_factory

    async def run(
        self, rows: Sequence[Any], *, pack: bool, cancelled: Callable[[], bool]
    ) -> list[RowResult]:
        """Run `rows` in order. Between rows of an unpacked chunk the cancel flag is checked, and
        a cancelled batch starts no further row (FR-26.6.4) — the rows already run are returned
        and recorded."""
        if pack and len(rows) > 1 and rows[0].kind == RowKind.SCORING.value:
            packed = await self._run_packed(rows)
            if packed is not None:
                return packed
        results: list[RowResult] = []
        for row in rows:
            # ⚠ The row-boundary cancel check (mutation control M6).
            if cancelled():
                break
            results.append(await self.run_one(row))
        return results

    async def run_one(self, row: Any) -> RowResult:
        raw = await self.read_line(row)
        request = parse_row_request(self.endpoint, raw)
        token = BATCH_ROW.set((self.batch_id, int(row.line_no)))
        try:
            try:
                status, body, headers = await self._call(row.kind, request)
            except MiLLMError as exc:
                return await failed_row(row, self.endpoint, exc, False)
        finally:
            BATCH_ROW.reset(token)
        return RowResult(
            row.line_no, RowState.DONE,
            output_line(row.custom_id, status, body, packed=False, headers=headers), False,
        )

    async def _call(self, kind: str, request: Any) -> tuple[int, Any, dict[str, str]]:
        """The synchronous service call for one row, and the provenance the route would set."""
        from millm.api.provenance import (
            PreGeneration,
            finish_body,
            post_generation,
            pre_generation,
        )

        if kind == RowKind.PROBE.value:
            return 200, await self._probe(request), {}
        name = ENDPOINT_NAMES[self.endpoint]
        if name == "embeddings":
            result = await self.inference.create_embeddings(request)
            headers = post_generation(
                request, self.inference, endpoint=name, pre=PreGeneration(), ignored_header=None
            )
            return 200, result.model_dump(mode="json"), headers
        pre = await pre_generation(request, self.inference, chat=name == "chat")
        if name == "chat":
            result = await self.inference.create_chat_completion(request)
        else:
            result = await self.inference.create_text_completion(request)
        headers = post_generation(
            request, self.inference, endpoint=name, pre=pre, ignored_header=None
        )
        finish_body(result, self.model_row, self.inference)
        return 200, result.model_dump(mode="json"), headers

    async def _probe(self, request: Any) -> Any:
        """027's scoring service — one input at a time, its own slot per input (FR-26.5.3)."""
        from millm.api.schemas.common import ApiResponse
        from millm.db.repositories.probe_repository import ProbeRepository
        from millm.services.probe_scoring import ProbeScoringService

        async with self.session_factory() as session:
            data = await ProbeScoringService(ProbeRepository(session), self.inference).score(
                request, session
            )
        return ApiResponse.ok(data).model_dump(mode="json")

    async def _run_packed(self, rows: Sequence[Any]) -> Optional[list[RowResult]]:
        """Packed scoring (task 6). None means "run these rows singly"."""
        from millm.services.batch.packing import run_packed_scoring

        return await run_packed_scoring(self, rows)
