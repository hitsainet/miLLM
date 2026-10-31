"""THE text each window gives a rule (039 TD39-21 – TD39-23; FTDD §6.6).

The one builder: `_rules_finish`, `POST /api/probes/score` and `POST /api/rule-sets/{id}/evaluate`
all call `window_texts`, and `tests/unit/rules/test_rule_text_reduction_cases.py` recomputes every
case of `docs/schemas/rule-text-reduction-cases.json` — a file miLLM OWNS and miStudio 040's
`reduce.py` vendors byte-identically. A change lands here, in that file, first.

The caller's own text, never template markup:

- `prompt`: every message's `content` that is a string, in order, joined by `"\\n"` (a message whose
  `content` is null contributes nothing; `reasoning_content` is never read). On `/v1/completions`,
  the prompt string as given.
- `last_user`: the last `user` message's `content`. A conversation with no user message, or a text
  completion, has no `last_user` window (`missing` names why: `no_user_turn`,
  `text_completion_has_no_user_turn`).
- `response`: the first choice's VISIBLE reply — exactly the `reply_text` the request records
  (reasoning excluded, TD39-22; with `n > 1`, the first choice only, TD39-23). No reply → missing
  (`no_scored_tokens`).
- `all`: `prompt + "\\n" + response` when both exist, else whichever exists.
- `token_ids` (the score route only): decoded with special tokens skipped; the caller marks each
  result `decoded_from_token_ids: true`.

Nothing is normalised (F39-0e): no case folding, no Unicode normalisation, no whitespace change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

WINDOWS = ("prompt", "last_user", "response", "all")


@dataclass(frozen=True)
class WindowTexts:
    texts: Mapping[str, str]
    #: `{window: reason}` for a window this request has no text for.
    missing: Mapping[str, str] = field(default_factory=dict)


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)


def _content(message: Any) -> Optional[str]:
    value = (message.get("content") if isinstance(message, Mapping)
             else getattr(message, "content", None))
    return value if isinstance(value, str) else None


def window_texts(*, messages: Optional[Iterable[Any]] = None, prompt: Optional[Any] = None,
                 reply_text: Optional[str] = None) -> WindowTexts:
    """The four windows for a chat (`messages`) or a text completion (`prompt`)."""
    texts: dict[str, str] = {}
    missing: dict[str, str] = {}
    if messages is not None:
        items = list(messages)
        parts = [c for c in (_content(m) for m in items) if c is not None]
        texts["prompt"] = "\n".join(parts)
        users = [_content(m) for m in items if _role(m) == "user"]
        users = [u for u in users if u is not None]
        if users:
            texts["last_user"] = users[-1]
        else:
            missing["last_user"] = "no_user_turn"
    else:
        if isinstance(prompt, str):
            texts["prompt"] = prompt
        elif isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], str):
            texts["prompt"] = prompt[0]  # the first prompt, as the first choice (TD39-23)
        missing["last_user"] = "text_completion_has_no_user_turn"
    if isinstance(reply_text, str):
        texts["response"] = reply_text
    else:
        missing["response"] = "no_scored_tokens"
    if "prompt" in texts and "response" in texts:
        texts["all"] = texts["prompt"] + "\n" + texts["response"]
    elif "prompt" in texts:
        texts["all"] = texts["prompt"]
    elif "response" in texts:
        texts["all"] = texts["response"]
    if "prompt" not in texts:
        missing.setdefault("prompt", "no_scored_tokens")
    return WindowTexts(texts, missing)
