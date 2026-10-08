"""Stateless probe scoring — `POST /api/probes/score` (Feature 27, FR-27.4 – FR-27.7).

Scores stored inputs with any imported probe, armed or not, and persists NOTHING: no
`probe_events` row, no `begin_request`, no change to the armed set or to a stored parity report.

⚠ **REUSE, NEVER COPY (BRD-04 RSK-07).** Offline must equal live, and the way to guarantee that is
to run the same code: the runtime probe is built by `armed_probe_from_row`, the forward is
`build_probe_forward` (which parity also runs), and the decision is `ProbeRequestContext.finish()`
→ `_verdict_for`, the one place `score >= threshold` lives (P-03). A copy of any of them would let
offline and live scores drift apart quietly — the defect the stateless path exists to rule out.

⚠ **ONE INPUT PER SLOT, ONE INPUT PER FORWARD, NEVER PACKED.** bfloat16 is not batch-invariant
(miStudio measured 0.177 batched vs one-at-a-time), and miStudio scores test vectors one at a time.
Each input takes its own admission slot through `InferenceService.run_model_work`, which also runs
it with every attached SAE suppressed (T-73, X-09) — so interactive traffic waits at most one
input's forward (FR-27.6a).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import torch

from millm.core.config import settings
from millm.core.errors import (
    ModelBusyError,
    ProbeNoModelLoadedError,
    ProbeNotFoundError,
    ProbeScoreRequestError,
)
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: ⚠ T-49: `text` is rendered as ONE USER TURN, as miStudio built its training corpus — and that is
#: an assumption until it has reproduced one miStudio-evaluated set's AUROC on the probe's model.
#: That check needs the GPU node (027 FTASKS 0.2). Until it passes, `text` inputs are REFUSED,
#: naming the pending verification, rather than scored under an unverified render. Flip this in
#: the same commit that records 0.2's pass, and delete the refusal test with it.
TEXT_INPUT_VERIFIED = False

TEXT_PENDING_MESSAGE = (
    "`text` inputs are not enabled yet: the one-user-turn render (T-49) has not yet reproduced a "
    "miStudio-reported AUROC on this probe's model (027 FTASKS 0.2, a hardware check). Send "
    "`messages` with one user turn, or the recorded `token_ids`, instead."
)

#: Per-input error codes. Data, not exceptions: one bad input does not discard the others.
MODEL_CHANGED = "MODEL_CHANGED"
TOKENIZATION_FAILED = "TOKENIZATION_FAILED"


@dataclass
class PreparedInput:
    """One input, ready to score: the ids, where the prompt ends, and the `last_user` span."""

    index: int
    input_kind: str
    ids: list[int] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    last_user_span: Optional[tuple[int, int]] = None
    last_user_reason: Optional[str] = None
    error: Optional[dict[str, str]] = None


class _ModelChanged(Exception):
    """Raised inside the slot when the loaded model is no longer the one pinned at input 0."""


# ── shape ─────────────────────────────────────────────────────────────────────────


def check_shape(request: Any) -> None:
    """Refuse a request as a whole, before any work (FTDD §5.1). Raises ProbeScoreRequestError."""
    inputs = request.inputs
    if not inputs:
        raise ProbeScoreRequestError("A scoring request needs at least one input",
                                     details={"param": "inputs"})
    if len(inputs) > settings.PROBE_SCORE_MAX_INPUTS:
        raise ProbeScoreRequestError(
            f"{len(inputs)} inputs exceeds the limit of {settings.PROBE_SCORE_MAX_INPUTS} per "
            "request; split the batch",
            details={"param": "inputs", "max_inputs": settings.PROBE_SCORE_MAX_INPUTS},
        )
    if request.probe_ids is not None and len(request.probe_ids) > settings.PROBE_SCORE_MAX_PROBES:
        raise ProbeScoreRequestError(
            f"{len(request.probe_ids)} probes exceeds the limit of "
            f"{settings.PROBE_SCORE_MAX_PROBES} per request",
            details={"param": "probe_ids", "max_probes": settings.PROBE_SCORE_MAX_PROBES},
        )
    for i, item in enumerate(inputs):
        kinds = item.kinds()
        if len(kinds) != 1:
            raise ProbeScoreRequestError(
                f"input {i} must be exactly one of token_ids, messages or text; "
                f"got {', '.join(kinds) or 'none'}",
                details={"param": f"inputs[{i}]", "index": i},
            )
        if item.prompt_tokens is not None:
            if item.token_ids is None:
                raise ProbeScoreRequestError(
                    f"input {i}: prompt_tokens is only accepted with token_ids (it is derived "
                    "for messages)",
                    details={"param": f"inputs[{i}].prompt_tokens", "index": i},
                )
            if item.prompt_tokens > len(item.token_ids):
                raise ProbeScoreRequestError(
                    f"input {i}: prompt_tokens ({item.prompt_tokens}) exceeds the input's length "
                    f"({len(item.token_ids)})",
                    details={"param": f"inputs[{i}].prompt_tokens", "index": i},
                )
        if item.token_ids is not None and any(t < 0 for t in item.token_ids):
            raise ProbeScoreRequestError(
                f"input {i}: token_ids must be non-negative",
                details={"param": f"inputs[{i}].token_ids", "index": i},
            )
        if item.text is not None and not TEXT_INPUT_VERIFIED:
            raise ProbeScoreRequestError(
                TEXT_PENDING_MESSAGE,
                details={"param": f"inputs[{i}].text", "index": i,
                         "pending_verification": "T-49 (027 FTASKS 0.2)"},
            )


# ── preparing an input ────────────────────────────────────────────────────────────


class ServedRender(NamedTuple):
    """A conversation as miLLM serves it: the ids, where the prompt ends, and which render made them."""

    ids: list[int]
    prompt_tokens: Optional[int]
    #: Whether the render ended with the generation prompt. Everything positional computed over
    #: `ids` afterwards — the `last_user` span above all — must be computed over THIS render.
    generation_prompt: bool


def served_render(
    tokenizer: Any, messages: list[dict[str, str]], render: Callable[[list[dict[str, str]], bool], str]
) -> ServedRender:
    """THE served-render rule — the one place it lives in miLLM (miStudio: `probe_monitor_render.
    served_render`, which mirrors it branch for branch and must stay identical).

    * last turn is `assistant` → rendered WITHOUT the generation prompt; the prompt ends at the
      render of the preceding turns WITH it — accepted only when those ids are a prefix of the
      full render, `None` otherwise, never guessed (T-72);
    * anything else → rendered WITH the generation prompt; the whole input is prompt (TD6);
    * tokenized by `prompt_encoding.rendered_chat_ids` — exactly one BOS (2026-10-08).

    Used by `/api/probes/score` (`ProbeInputPreparer.prepare`) and by parity's informational
    `messages` round-trip (`ProbeParityEngine._drift`), so the two cannot disagree about what a
    served-render definition's `messages` should reproduce.
    """
    from millm.services.prompt_encoding import rendered_chat_ids
    from millm.services.probe_turns import served_generation_prompt

    generation_prompt = served_generation_prompt(messages)
    if not generation_prompt:
        ids = rendered_chat_ids(tokenizer, render(messages, False))
        head = rendered_chat_ids(tokenizer, render(messages[:-1], True)) if messages[:-1] else []
        prompt_tokens = len(head) if head and ids[: len(head)] == head else None
    else:
        ids = rendered_chat_ids(tokenizer, render(messages, True))
        prompt_tokens = len(ids)
    return ServedRender(ids, prompt_tokens, generation_prompt)


def template_renderer(tokenizer: Any) -> Callable[[list[dict[str, str]], bool], str]:
    """The model's chat template, verbatim — what live serving's renderer reduces to whenever the
    tokenizer HAS a template (`InferenceService._format_chat_messages`), for a caller with no
    inference service (parity)."""

    def render(messages: list[dict[str, str]], generation_prompt: bool) -> str:
        return str(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        ))

    return render


class ProbeInputPreparer:
    """Turns one input into ids + boundaries. Pure: the tokenizer and the renderer are arguments,
    so it needs no model.

    `render(messages, generation_prompt)` must be live serving's renderer
    (`InferenceService._format_chat_messages` with the generation prompt on), and ids are produced
    exactly as live serving produces them (`prompt_encoding.encode_rendered_chat` — the SAME
    function, never a copy), so a `messages` input scored here is the sequence a live chat request
    would have read (FR-27.5). Before 2026-10-08 both used `tokenizer(prompt)`, which put a
    duplicate BOS on every Llama 3 / gemma / LFM2.5 render; they were identical, and both wrong.
    """

    def __init__(self, tokenizer: Any, render: Callable[[list[dict[str, str]], bool], str]) -> None:
        self.tokenizer = tokenizer
        self.render = render

    def prepare(self, index: int, item: Any) -> PreparedInput:
        from millm.services.probe_turns import last_user_token_span

        kind = item.kinds()[0]
        if kind == "token_ids":
            return PreparedInput(
                index=index, input_kind=kind, ids=list(item.token_ids),
                prompt_tokens=item.prompt_tokens,
                last_user_span=None, last_user_reason="token_ids_have_no_turns",
            )
        if kind == "text":
            # T-49: one user turn, as miStudio built its corpus.
            messages = [{"role": "user", "content": item.text}]
        else:
            messages = [{"role": m.role, "content": m.content} for m in item.messages]
        try:
            served = served_render(self.tokenizer, messages, self.render)
        except Exception as exc:  # noqa: BLE001 - a per-input failure is data, not a 500
            return PreparedInput(
                index=index, input_kind=kind,
                error={"code": TOKENIZATION_FAILED, "message": f"rendering failed: {exc}"},
            )
        try:
            span, reason = last_user_token_span(self.tokenizer, messages, served.ids, None)
        except Exception:  # noqa: BLE001
            span, reason = None, "last_user_span_unresolved"
        return PreparedInput(
            index=index, input_kind=kind, ids=served.ids, prompt_tokens=served.prompt_tokens,
            last_user_span=span, last_user_reason=reason,
        )


# ── the wire mapping ─────────────────────────────────────────────────────────────


def verdict_payload(v: Any) -> dict[str, Any]:
    """THE wire mapping of one `Verdict` for scoring results (FTID §11; reused by Feature 26).

    `verdict` is `Verdict.fires` renamed: `True`, `False` or `None` — `None` means the probe said
    nothing and is never coerced to `False` (FR-27.4c). `rung_language` is passed verbatim, so no
    consumer composes a phrase from a rung number (FR-27.4d). `provisional` and
    `threshold_revision` are carried as recorded (P-20).
    """
    return {
        "probe_id": v.probe_id,
        "name": v.name,
        "window": v.window,
        "score": v.score,
        "threshold": v.threshold,
        "verdict": v.fires,
        "rung": v.rung,
        "rung_language": v.rung_language,
        "provisional": v.provisional,
        "threshold_revision": v.threshold_revision,
        "n_scored_tokens": v.n_scored_tokens,
        "not_scored_reason": v.not_scored_reason,
    }


def parity_status(row: Any) -> dict[str, Any]:
    """A probe's STORED parity, reported and never required (T-74).

    `checked_against` is the model block the report recorded; `"unknown"` for a report written
    before Feature 27 added the key — never a guess at what it was checked against.
    """
    parity = getattr(row, "parity", None)
    if not parity:
        return {"status": "never_run", "checked_against": None, "checked_at": None}
    return {
        "status": "passed" if parity.get("passed") is True else "failed",
        "checked_against": parity.get("model") or "unknown",
        "checked_at": parity.get("checked_at"),
    }


# ── the service ──────────────────────────────────────────────────────────────────


class ProbeScoringService:
    """Resolves probes, prepares inputs, and scores each input in its own slot."""

    def __init__(self, repository: Any, inference: Any) -> None:
        self.repository = repository
        self.inference = inference

    async def _resolve(
        self, request: Any, loaded: Any
    ) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
        """`(rows, encoders, skipped)`. Given ids refuse on any mismatch; omitted ids skip."""
        from millm.services.probe_arm_bridge import build_probe_encoder
        from millm.services.probe_arming import identity_refusal, scope_refusal

        given = request.probe_ids is not None
        if given:
            rows = []
            for probe_id in dict.fromkeys(request.probe_ids):
                row = await self.repository.get(probe_id)
                if row is None:
                    raise ProbeNotFoundError(f"No probe {probe_id}",
                                             details={"probe_id": probe_id})
                rows.append(row)
        else:
            rows = list(await self.repository.list())

        kept, encoders, skipped = [], [], []
        for row in rows:
            try:
                refusal = identity_refusal(row, loaded) or scope_refusal(row)
                if refusal is not None:
                    raise refusal
                encoder = await build_probe_encoder(row)
            except Exception as exc:
                code = getattr(exc, "code", None)
                if given or code is None:
                    raise
                skipped.append({"probe_id": row.id, "code": code, "reason": str(exc)})
                continue
            kept.append(row)
            encoders.append(encoder)

        if not kept:
            raise ProbeScoreRequestError(
                "No imported probe can score on the loaded model"
                + (": every probe was skipped" if skipped else ": no probe is imported"),
                details={"skipped": skipped},
            )
        if len(kept) > settings.PROBE_SCORE_MAX_PROBES:
            raise ProbeScoreRequestError(
                f"{len(kept)} imported probes match the loaded model, over the limit of "
                f"{settings.PROBE_SCORE_MAX_PROBES} per request; name them with probe_ids",
                details={"param": "probe_ids", "max_probes": settings.PROBE_SCORE_MAX_PROBES},
            )
        return kept, encoders, skipped

    async def score(self, request: Any, session: Any) -> dict[str, Any]:
        from millm.ml.model_loader import LoadedModelState
        from millm.services.probe_arm_bridge import build_probe_forward, loaded_identity
        from millm.services.probe_arming import armed_probe_from_row
        from millm.services.probe_parity import model_summary

        started = time.perf_counter()
        check_shape(request)
        identity, model, tokenizer = await loaded_identity(session)
        rows, encoders, skipped = await self._resolve(request, identity)
        try:
            probes = [
                armed_probe_from_row(row, encoder=enc, windows=request.windows)
                for row, enc in zip(rows, encoders, strict=True)
            ]
        except ValueError as exc:
            raise ProbeScoreRequestError(str(exc), details={"param": "windows"}) from exc

        current = LoadedModelState().current
        if current is None:
            raise ProbeNoModelLoadedError("The model was unloaded before scoring began")
        pin = (current.model_id, current.loaded_at)

        # Prepared OUTSIDE any slot: tokenisation needs none, and the slot is held only for the
        # forward (FTID §13).
        preparer = ProbeInputPreparer(tokenizer, self._renderer(tokenizer))
        prepared = [preparer.prepare(i, item) for i, item in enumerate(request.inputs)]

        # Whole-request refusals that need the ids: out-of-vocabulary ids (an out-of-range id is
        # a device-side assert, not a 400, if it reaches the embedding table) and inputs over the
        # context window — refused BEFORE any forward, so a request is never half-run (FTDD §8).
        vocab = _vocab_size(model)
        for p in prepared:
            if p.error:
                continue
            if vocab is not None and p.ids and max(p.ids) >= vocab:
                raise ProbeScoreRequestError(
                    f"input {p.index}: token id {max(p.ids)} is outside the loaded model's "
                    f"vocabulary of {vocab}",
                    details={"param": f"inputs[{p.index}].token_ids", "index": p.index,
                             "vocab_size": vocab},
                )
            self.inference._check_context_length(len(p.ids), 0)

        layers = sorted({probe.layer for probe in probes})
        forward = build_probe_forward(model, layers)
        results: list[dict[str, Any]] = []
        changed = False
        for p in prepared:
            entry: dict[str, Any] = {
                "index": p.index, "input_kind": p.input_kind, "n_tokens": len(p.ids),
                "prompt_tokens": p.prompt_tokens,
                "token_ids": list(p.ids) if request.return_token_ids else None,
                "verdicts": [], "error": p.error,
            }
            if p.error is None and changed:
                entry["error"] = _model_changed_error()
            elif p.error is None:
                try:
                    verdicts = await self.inference.run_model_work(
                        lambda p=p: self._score_one(p, probes, forward, pin)
                    )
                    entry["verdicts"] = [verdict_payload(v) for v in verdicts]
                except (_ModelChanged, ModelBusyError):
                    # FR-27.6d: no input is scored on a different model from the first. The
                    # remaining inputs fail with the reason; the earlier results are kept.
                    changed = True
                    entry["error"] = _model_changed_error()
            results.append(entry)

        logger.info(
            "probe_score",
            probes=len(probes), skipped=len(skipped), inputs=len(prepared),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        return {
            "model": model_summary(identity),
            "probes": [
                {"probe_id": row.id, "name": row.name, "layer": row.layer,
                 "parity": parity_status(row)}
                for row in rows
            ],
            "skipped": skipped,
            "results": results,
        }

    def _score_one(
        self, p: PreparedInput, probes: list[Any], forward: Any, pin: tuple[Any, Any]
    ) -> list[Any]:
        """One input's forward, inside the slot. Never opens the runtime's request context: the
        context is THIS function's own, so an armed hook on the same layer sees no request and
        records nothing (FR-27.4f, FR-27.7)."""
        from millm.ml.model_loader import LoadedModelState
        from millm.services.probe_runtime import ProbeRequestContext

        current = LoadedModelState().current
        if current is None or (current.model_id, current.loaded_at) != pin:
            raise _ModelChanged()
        context = ProbeRequestContext(f"score:{p.index}", probes)
        if p.prompt_tokens is not None:
            context.set_prompt_length(p.prompt_tokens)
        context.set_last_user_span(p.last_user_span, p.last_user_reason)
        forward(torch.tensor([p.ids], dtype=torch.long), context)
        return context.finish()

    def _renderer(self, tokenizer: Any) -> Callable[[list[dict[str, str]], bool], str]:
        """Live serving's renderer for the generation-prompt case; the template itself without
        it (the assistant-ended case has no live counterpart to share)."""
        from millm.api.schemas.openai import ChatMessage

        def render(messages: list[dict[str, str]], generation_prompt: bool) -> str:
            if generation_prompt:
                return str(self.inference._format_chat_messages(
                    [ChatMessage.model_validate(m) for m in messages]
                ))
            return str(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))

        return render


def _model_changed_error() -> dict[str, str]:
    return {
        "code": MODEL_CHANGED,
        "message": "The loaded model changed during this request, so this input was not scored "
                   "(no input is scored on a different model from the first).",
    }


def _vocab_size(model: Any) -> Optional[int]:
    try:
        return int(model.get_input_embeddings().num_embeddings)
    except Exception:  # noqa: BLE001
        size = getattr(getattr(model, "config", None), "vocab_size", None)
        return int(size) if isinstance(size, int) else None
