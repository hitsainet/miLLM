"""Where the newest user message sits in a served prompt — the `last_user` window's span.

⚠ COMPUTED EXACTLY AS miSTUDIO CALIBRATES IT, OR THE WINDOW'S BAR DOES NOT APPLY. miStudio labels
each message's tokens by rendering the conversation's prefixes (`messages[:i]`) with the chat
template, `add_generation_prompt=False`, encoding with `add_special_tokens=False`, and giving
message i the tokens `[len(prefix_{i-1}), len(prefix_i))` — header, content and end-of-turn
included (`probe_monitor_render.render_messages`). The `last_user` bar is a quantile of
aggregates over exactly that span, so this computes the same span from the same template, then
places it inside the ids actually served (which may carry a BOS the template render does not).

Every way it can fail is a REASON, never a guess: a window scored over a span that is not the
calibrated one would be a rate of one distribution applied to another.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

#: The reasons a request has no resolvable `last_user` span. Stated on the verdict.
NO_USER_TURN = "no_user_turn"
NO_USER_HEADER = "no_user_header"
NO_CHAT_TEMPLATE = "no_chat_template"
SPAN_UNRESOLVED = "last_user_span_unresolved"


#: Two tiny conversations whose final user turns differ only in content; the common start of those
#: turns' token spans is the template's user header. The SAME construction as miStudio's
#: `probe_monitor_render.user_header_ids`, pinned by `docs/schemas/last-user-span-cases.json`.
_HEADER_PROBES = (
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "x"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "y"}],
)


def user_header_ids(tokenizer: Any, template_kwargs: Optional[dict] = None) -> Optional[list[int]]:
    """The token ids of a user turn's role header, or `None` when they cannot be isolated."""
    kwargs = dict(template_kwargs or {})
    spans: list[list[int]] = []
    try:
        for conversation in _HEADER_PROBES:
            before = list(tokenizer(tokenizer.apply_chat_template(
                conversation[:2], tokenize=False, add_generation_prompt=False, **kwargs
            ), add_special_tokens=False)["input_ids"])
            through = list(tokenizer(tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False, **kwargs
            ), add_special_tokens=False)["input_ids"])
            if through[: len(before)] != before:
                return None
            spans.append(through[len(before):])
    except Exception:  # a template that refuses the probe has no isolable header
        return None
    common: list[int] = []
    for a, b in zip(*spans):
        if a != b:
            break
        common.append(a)
    return common or None


def last_user_token_span(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    served_ids: Sequence[int],
    template_kwargs: Optional[dict] = None,
) -> tuple[Optional[tuple[int, int]], Optional[str]]:
    """`((start, end), None)` in served-id positions, or `(None, reason)`."""
    roles = [str(m.get("role", "")) for m in messages]
    last = max((i for i, role in enumerate(roles) if role == "user"), default=None)
    if last is None:
        return None, NO_USER_TURN
    if not getattr(tokenizer, "chat_template", None):
        return None, NO_CHAT_TEMPLATE
    kwargs = dict(template_kwargs or {})
    plain = [{"role": m.get("role"), "content": m.get("content")} for m in messages]

    def encode(conversation: list, generation_prompt: bool) -> list[int]:
        if not conversation:
            return []
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=generation_prompt, **kwargs
        )
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    try:
        before = encode(plain[:last], False)
        through = encode(plain[: last + 1], False)
        full = encode(plain, True)
    except Exception:  # a template may refuse a partial conversation
        return None, SPAN_UNRESOLVED
    # THE PREFIX PROPERTY, CHECKED — the same check miStudio makes before trusting a role mask.
    if through[: len(before)] != before or full[: len(through)] != through:
        return None, SPAN_UNRESOLVED
    served = list(served_ids)
    offset = len(served) - len(full)
    # The served ids are the same render, possibly with a BOS the tokenizer prepended.
    if offset < 0 or served[offset:] != full:
        return None, SPAN_UNRESOLVED
    start, end = offset + len(before), offset + len(through)
    if last == 0:
        # ⚠ MESSAGE 0 ALSO HOLDS THE BOS AND ANY PREAMBLE THE TEMPLATE INJECTS — on Llama-3.1 a
        # ~25-token system block even when no system message was sent (review round 1, H1). The
        # window starts at the message's own role header, its LAST occurrence in that span, as
        # miStudio's calibration does.
        header = user_header_ids(tokenizer, kwargs)
        if not header:
            return None, NO_USER_HEADER
        found = None
        for candidate in range(end - len(header), start - 1, -1):
            if served[candidate: candidate + len(header)] == header:
                found = candidate
                break
        if found is None:
            return None, NO_USER_HEADER
        start = found
    if end <= start:
        return None, SPAN_UNRESOLVED
    return (start, end), None
