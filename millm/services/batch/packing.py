"""Packed scoring for batch rows (Feature 26, FR-26.5, FTASKS 6.2-6.5).

A chunk of SCORING rows with `pack: true` is scored in right-padded packs by
`InferenceService._score_specs_packed` — the same scorer the synchronous path reaches with a pack
of one — and each row's body is built by the SAME response builder the synchronous endpoint uses
(`_text_scoring_response` / `_chat_scoring_response`). Its `X-miLLM-*` headers come from
`millm.api.provenance`, as the route's do.

Rows that cannot be packed run singly (`packed: false`), never approximated:
* the llama.cpp engine (FR-26.5.7) — there is no distribution to pack;
* a row asking for `return_sae_activations` — one request's positions cannot be attributed
  inside a pack (027 FR-27.1g's reasoning for the batching manager);
* a row carrying several prompts or conversations — it is already its own multi-prompt request.
Generation and probe rows never reach here (`RowExecutor.run` packs scoring rows only, and the
runner's chunk for them is one row — FR-26.5.2, FR-26.5.3). Embedding rows stay single until
Feature 30's mask-aware pooling exists (FTASKS 6.6).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from millm.core.batch_values import RowState
from millm.core.errors import MiLLMError
from millm.db.repositories.batch_repository import RowResult
from millm.services.batch.state import BATCH_ROW


def packable(endpoint: str, request: Any) -> bool:
    if getattr(request, "return_sae_activations", None) is not None:
        return False
    if endpoint == "/v1/completions":
        prompt = request.prompt
        return isinstance(prompt, str) or (isinstance(prompt, list) and len(prompt) == 1)
    if endpoint == "/v1/chat/completions":
        return not request.extra_messages
    return False


async def run_packed_scoring(executor: Any, rows: Sequence[Any]) -> Optional[list[RowResult]]:
    """Score `rows`, packing those that can be packed. None: run them all singly."""
    from millm.api.provenance import finish_body, post_generation, pre_generation
    from millm.core.config import settings
    from millm.services.batch.executors import (
        ENDPOINT_NAMES,
        failed_row,
        output_line,
        parse_row_request,
    )
    from millm.services.inference_service import ScoreSpec

    inference = executor.inference
    if inference._engine_is_llamacpp():
        return None
    endpoint = executor.endpoint
    chat = endpoint == "/v1/chat/completions"
    parsed = [(row, parse_row_request(endpoint, await executor.read_line(row))) for row in rows]
    pack = [(row, req) for row, req in parsed if packable(endpoint, req)]
    if len(pack) < 2:
        return None
    single = [row for row, req in parsed if not packable(endpoint, req)]
    results: dict[int, RowResult] = {}

    async with inference._admit():  # re-enters the chunk's slot (FTDD §7)
        specs: list[Any] = []
        texts: list[str] = []
        for row, req in pack:
            try:
                if chat:
                    text = inference._chat_scoring_texts(req, [req.messages])[0]
                    specs.append(ScoreSpec(text, False, req.allowed_token_ids, req.temperature,
                                           req.top_logprobs or 0))
                else:
                    text = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
                    specs.append(ScoreSpec(text, req.add_special_tokens, req.allowed_token_ids,
                                           req.temperature, req.logprobs or 0))
                texts.append(text)
            except MiLLMError as exc:
                specs.append(exc)
                texts.append("")
        runnable = [(i, s) for i, s in enumerate(specs) if not isinstance(s, MiLLMError)]
        outcomes = await inference._score_specs_packed(
            [s for _, s in runnable], max_rows=max(int(settings.BATCH_PACK_MAX_ROWS), 1)
        )
        by_index: dict[int, Any] = {i: s for i, s in enumerate(specs) if isinstance(s, MiLLMError)}
        by_index.update({i: o for (i, _), o in zip(runnable, outcomes, strict=True)})

        for i, (row, req) in enumerate(pack):
            outcome = by_index[i]
            token = BATCH_ROW.set((executor.batch_id, int(row.line_no)))
            try:
                if isinstance(outcome, MiLLMError):
                    results[row.line_no] = await failed_row(row, endpoint, outcome, True)
                    continue
                pre = await pre_generation(req, inference, chat=chat)
                scored = [(outcome.scores, outcome.prompt_tokens)]
                body = (
                    inference._chat_scoring_response(req, scored, [])
                    if chat else inference._text_scoring_response(req, [texts[i]], scored, [])
                )
                headers = post_generation(
                    req, inference, endpoint=ENDPOINT_NAMES[endpoint], pre=pre, ignored_header=None
                )
                finish_body(body, executor.model_row, inference)
                results[row.line_no] = RowResult(
                    row.line_no, RowState.DONE,
                    output_line(row.custom_id, 200, body.model_dump(mode="json"), packed=True,
                                headers=headers),
                    True,
                )
            finally:
                BATCH_ROW.reset(token)
    for row in single:
        results[row.line_no] = await executor.run_one(row)
    return [results[row.line_no] for row in rows]
