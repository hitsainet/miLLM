"""The decoded token window a probe event carries, for one operator question:
*which prompt did this verdict apply to?*

⚠ **EVERY OTHER PART OF THIS FEATURE WAS BUILT AND NOTHING PRODUCED THE DATA.** The
`probe_events.context_text` / `context_token_ids` columns exist (migration 016), the
`GET /api/probes/events/{id}` route serves them and is documented as the only route that does, the
admin UI renders them in a modal, `ProbeEventService.record()` takes a `contexts=` argument, the
broadcast strips the keys so they never reach a socket, the privacy tests assert that stripping —
and **`PROBE_EVENT_CONTEXT_TOKENS: int = 24` had no reader anywhere in the repo.** `_probe_record`
never passed `contexts`, so `context_text` was `None` on every event ever recorded and the modal
opened empty.

Reported 2026-09-28 as *"the window to see the associated prompt still does not open."* The comment
immediately above the offending call records the identical defect being fixed for the neighbouring
argument: *"THE OVERHEAD IS PASSED. `note_request_overhead` existed, was unit-tested by direct call,
and had NO production caller."* One argument was wired, the one beside it was not, and the same
review round touched both.

The window is deliberately a separate pure function rather than inlined at the call site: it is the
part with edge cases (a decode-phase position beyond prompt-only ids, a 2-D batch tensor, a
tokenizer that is not loaded), and inlining it would leave those testable only through a generation.
"""

from __future__ import annotations

from typing import Any, Optional


def context_window(
    full_ids: Any,
    position: int,
    k: int,
    tokenizer: Any,
    span: tuple[int, Optional[int]] = (0, None),
) -> tuple[Optional[str], Optional[list[int]]]:
    """The ±`k` token window around `position`, decoded.

    Returns `(text, window_ids)`, or `(None, None)` when no honest window exists — which is a
    deliberate outcome and not a failure. An empty or wrong window is worse than none: the modal's
    whole job is to tell an operator which text a verdict was about, and a window that silently
    starts at position 0 would answer that question incorrectly.

    `full_ids` may be a 1-D or 2-D tensor or a plain list of ints; the CBM path holds a list.
    """
    if k <= 0 or full_ids is None or tokenizer is None:
        return None, None
    try:
        ids = full_ids
        # A 2-D tensor is (batch, seq). Probes only ever score a single row — `observe` marks a
        # batched pass `not_scored` — so row 0 is the request's own tokens.
        if hasattr(ids, "dim"):
            ids = ids[0] if ids.dim() == 2 else ids
            total = int(ids.shape[-1])
            take = lambda lo, hi: ids[lo:hi].tolist()  # noqa: E731
        else:
            ids = list(ids)
            if ids and isinstance(ids[0], (list, tuple)):
                ids = list(ids[0])
            total = len(ids)
            take = lambda lo, hi: [int(x) for x in ids[lo:hi]]  # noqa: E731

        if total <= 0 or position < 0 or position >= total:
            # ⚠ A decode-phase position against prompt-only ids lands here. The streaming paths
            # fall back to `inputs["input_ids"]` when no step captured ids, so `position` can
            # legitimately exceed what is available. Returning a window from the wrong end would
            # attribute the verdict to text it never scored.
            return None, None

        # ⚠ CLIPPED TO THE SPAN THE VERDICT'S WINDOW READ (2026-10-04). A ±k window cut from the
        # whole sequence let a prompt verdict whose peak sat near the end of the prompt display
        # the start of the model's REPLY — text that window never scored — and a response verdict
        # display the end of the prompt. The context is evidence for one window's verdict, so it
        # shows that window's tokens only.
        span_lo = max(0, int(span[0] or 0))
        span_hi = total if span[1] is None else min(total, int(span[1]))
        if not (span_lo <= position < span_hi):
            return None, None
        lo = max(span_lo, position - k)
        hi = min(span_hi, position + 1 + k)
        window = take(lo, hi)
        if not window:
            return None, None
        text = tokenizer.decode(window, skip_special_tokens=True)
        if not text:
            # A window of nothing but specials decodes to "". Storing an empty string would render
            # as an empty box that looks like a bug rather than an absence.
            return None, window
        return text, window
    except Exception:
        # Context is a convenience on an event that must be written regardless. A tokenizer that
        # raises must not cost the verdict.
        return None, None


def window_span(window: str, prompt_length: Optional[int]) -> tuple[int, Optional[int]]:
    """The token span a contract window reads: prompt `[0, n_prompt)`, response `[n_prompt, end)`.

    Unknown prompt length leaves the span unbounded rather than guessing a boundary — the same
    position the runtime takes, which then marks such a window `prompt_boundary_unknown`.
    """
    if prompt_length is None or window == "all":
        return (0, None)
    if window == "prompt":
        return (0, int(prompt_length))
    if window == "response":
        return (int(prompt_length), None)
    return (0, None)


def contexts_for(
    verdicts: Any,
    full_ids: Any,
    k: int,
    tokenizer: Any,
    prompt_length: Optional[int] = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """`{(probe_id, window): {context_text, context_token_ids}}` for verdicts with a top position.

    A verdict with no `top_positions` (not scored, or scored with no firing position) contributes
    no entry rather than an empty one, so `context_text IS NULL` keeps meaning "no window", not
    "a window we could not fill".

    ⚠ KEYED BY (probe_id, window), NOT probe_id. One probe now reports several windows, and each
    one's top position is somewhere different — the prompt window's peak is in the person's
    words, the response window's is in the model's. Keyed by probe id alone the second verdict
    would SILENTLY OVERWRITE the first, and both events would open on the same text while showing
    different scores. Nothing would have raised; the reader would simply have been shown the
    wrong evidence for one of them.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for verdict in verdicts or []:
        positions = getattr(verdict, "top_positions", None) or []
        if not positions:
            continue
        text, window = context_window(
            full_ids, int(positions[0]), k, tokenizer,
            span=window_span(getattr(verdict, "window", "all"), prompt_length),
        )
        if text is None and window is None:
            continue
        out[(verdict.probe_id, verdict.window)] = {
            "context_text": text,
            "context_token_ids": window,
        }
    return out
