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

import json
import weakref
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


#: Single-message renders whose contents differ in one character: their common prefix is the
#: template's preamble plus the first user header and holds NO content (review round 2, H-A).
_FIRST_PROBES = ([{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}])
#: Weakly keyed by tokenizer, then by template and kwargs, so the four extra renders are paid once
#: per tokenizer rather than per request (review round 2, L-B).
_cache: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def _cached(tokenizer: Any, name: str, kwargs: dict, compute: Any) -> Any:
    try:
        key = f"{name}:{getattr(tokenizer, 'chat_template', None)!r}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        per_tokenizer = _cache.setdefault(tokenizer, {})
    except TypeError:  # not weakly referenceable: computed every time, never wrongly shared
        return compute()
    if key not in per_tokenizer:
        per_tokenizer[key] = compute()
    return per_tokenizer[key]


def _encode_render(tokenizer: Any, conversation: list, kwargs: dict, generation_prompt: bool = False) -> list[int]:
    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=generation_prompt, **kwargs
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _normalised_text(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False).replace("\u2581", " ").strip()


def first_user_header(
    tokenizer: Any, template_kwargs: Optional[dict] = None
) -> Optional[tuple[list[int], int]]:
    """`(prefix, start)` for a conversation whose message 0 is a user turn, or `None`.

    `prefix` is what every such render begins with (preamble and header, never content) and
    `start` is where the header begins inside it. ⚠ NEVER LOCATED IN THE CONTENT: searching
    message 0 for the header let a user who typed a role header into a single-turn message move
    the window past everything before it (review round 2, H-A). The header is matched by its TEXT
    at its own position, because a SentencePiece tokenizer encodes a header at the start of a
    string with a leading `▁` that the same header after a turn lacks (M-A). Identical to
    miStudio's `probe_monitor_render.first_user_header`.
    """
    kwargs = dict(template_kwargs or {})
    return _cached(tokenizer, "first", kwargs, lambda: _first_user_header(tokenizer, kwargs))


def _first_user_header(tokenizer: Any, kwargs: dict) -> Optional[tuple[list[int], int]]:
    header = user_header_ids(tokenizer, kwargs)
    if not header:
        return None
    try:
        a, b = (_encode_render(tokenizer, c, kwargs) for c in _FIRST_PROBES)
    except Exception:  # a template that refuses a lone user turn has no answer here
        return None
    prefix: list[int] = []
    for x, y in zip(a, b):
        if x != y:
            break
        prefix.append(x)
    shared = 0
    while shared < min(len(header), len(prefix)) and prefix[-1 - shared] == header[-1 - shared]:
        shared += 1
    start = len(prefix) - shared - (0 if shared == len(header) else 1)
    if start < 0 or _normalised_text(tokenizer, prefix[start:]) != _normalised_text(tokenizer, header):
        return None
    return prefix, start


def user_header_ids(tokenizer: Any, template_kwargs: Optional[dict] = None) -> Optional[list[int]]:
    """The token ids of a user turn's role header, or `None` when they cannot be isolated."""
    kwargs = dict(template_kwargs or {})
    return _cached(tokenizer, "header", kwargs, lambda: _user_header_ids(tokenizer, kwargs))


def _user_header_ids(tokenizer: Any, kwargs: dict) -> Optional[list[int]]:
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
        # header start comes from content-free renders, never from the content (round 2, H-A),
        # and the served ids must begin with exactly that prefix.
        found = first_user_header(tokenizer, kwargs)
        if found is None:
            return None, NO_USER_HEADER
        prefix, header_start = found
        if served[offset: offset + len(prefix)] != prefix or offset + len(prefix) > end:
            return None, NO_USER_HEADER
        start = offset + header_start
    if end <= start:
        return None, SPAN_UNRESOLVED
    return (start, end), None
