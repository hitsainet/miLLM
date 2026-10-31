"""How a prompt becomes token ids — ONE rule, used by every path that serves, scores or probes one.

⚠ THE DEFECT THIS EXISTS FOR (operator-approved fix, 2026-10-08). A chat is rendered by the
tokenizer's chat template, and the Llama 3, gemma and LFM2.5 templates all BEGIN with the BOS
token's text (`{{ bos_token }}`). Tokenizing that render with the default
`add_special_tokens=True` prepends ANOTHER BOS, so every live chat on those models started
`128000, 128000` (Llama 3) / `2, 2` (gemma) / `1, 1` (LFM2.5). Nothing raised and the answers
stayed fluent; what moved was every score read off those positions — an armed probe calibrated in
miStudio on single-BOS ids (`probe_monitor_render`, `add_special_tokens=False`) was being judged
on a sequence it never saw. Measured on production 2026-10-07: removing the duplicate moved one
probe's AUROC on 540 rows from 0.9604 to 0.9558.

THE RULE, for text rendered by a chat template (`encode_rendered_chat`):

* the render already begins with the tokenizer's `bos_token` text → add NO special tokens
  (Llama 3, gemma 2/3/4, LFM2.5 — the template carries the one BOS);
* otherwise → let the tokenizer add its special tokens exactly as it would by default, so a
  model whose template carries no BOS but whose tokenizer adds one still gets ONE (TinyLlama /
  Zephyr-style Llama 2 templates), and a tokenizer that adds nothing still adds nothing (Qwen2.5:
  no `bos_token` at all; granite 4.x and Phi-4: a `bos_token` the tokenizer never adds and the
  template never writes).

Measured on the real tokenizers (`0xcc/reviews/chat_double_bos_2026-10-08.md` §2), never assumed:
the rule yields exactly one BOS at position 0 on every family that uses one and zero duplicates
on all of them.

Raw text (a `/v1/completions` prompt) is NOT rendered by a template, so it keeps honouring the
request's own `add_special_tokens` (default True) — `encode_prompt(rendered_chat=False, ...)`.

Every tokenize call in `millm/` that sees a rendered chat goes through this module;
`tests/unit/services/test_prompt_encoding_guard.py` walks the AST and fails on one that does not.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


def bos_text(tokenizer: Any) -> Optional[str]:
    """The tokenizer's BOS token as text, or None when it has none (Qwen2.5) or it is unreadable.

    `isinstance` rather than truthiness: a test double's attribute is not a string, and treating
    it as one would make the rule depend on whatever the double returns.
    """
    value = getattr(tokenizer, "bos_token", None)
    return value if isinstance(value, str) and value else None


def rendered_chat_adds_special_tokens(tokenizer: Any, text: str) -> bool:
    """`add_special_tokens` for one template render: False exactly when the render already
    begins with the BOS the tokenizer would add — the one decision this module exists to make."""
    bos = bos_text(tokenizer)
    return not (bos is not None and text.startswith(bos))


def encode_rendered_chat(tokenizer: Any, text: str, **kwargs: Any) -> Any:
    """Tokenize text produced by a chat template, with exactly one BOS when the model uses one.

    Returns what `tokenizer(...)` returns (a `BatchEncoding`); `kwargs` pass through
    (`return_tensors`, ...). `add_special_tokens` is refused: deciding it is this function's job,
    and a caller passing one is a caller about to bypass the rule.
    """
    if "add_special_tokens" in kwargs:
        raise TypeError("encode_rendered_chat decides add_special_tokens itself")
    return tokenizer(
        text, add_special_tokens=rendered_chat_adds_special_tokens(tokenizer, text), **kwargs
    )


def rendered_chat_ids(tokenizer: Any, text: str) -> list[int]:
    """`encode_rendered_chat(...)["input_ids"]` as a plain list."""
    return list(encode_rendered_chat(tokenizer, text)["input_ids"])


def encode_rendered_chats(tokenizer: Any, texts: Sequence[str], **kwargs: Any) -> Any:
    """Tokenize several renders as ONE padded batch (`padding`, `padding_side`, `return_tensors`
    pass through), each row under the rule above.

    One tokenizer call when every row needs the same decision — always, in practice, since every
    row is rendered by the same template. If rows ever disagree, each is encoded on its own and
    the rows are padded together with `tokenizer.pad`, so no row inherits another's decision.
    """
    if "add_special_tokens" in kwargs:
        raise TypeError("encode_rendered_chats decides add_special_tokens itself")
    texts = list(texts)
    decisions = {rendered_chat_adds_special_tokens(tokenizer, t) for t in texts}
    if len(decisions) <= 1:
        special = decisions.pop() if decisions else True
        return tokenizer(texts, add_special_tokens=special, **kwargs)
    rows = [{"input_ids": rendered_chat_ids(tokenizer, t)} for t in texts]
    pad_kwargs = {k: kwargs[k] for k in ("padding", "padding_side", "return_tensors") if k in kwargs}
    return tokenizer.pad(rows, **pad_kwargs)


def encode_prompt(
    tokenizer: Any, text: str, *, rendered_chat: bool, add_special_tokens: bool = True,
    **kwargs: Any,
) -> Any:
    """THE entry point for a prompt of either kind.

    `rendered_chat=True`: `encode_rendered_chat` (the request's `add_special_tokens` is not a
    chat field and is ignored). `rendered_chat=False`: raw text, tokenized with the caller's
    `add_special_tokens` — a `/v1/completions` prompt's own setting, default True.
    """
    if rendered_chat:
        return encode_rendered_chat(tokenizer, text, **kwargs)
    return tokenizer(text, add_special_tokens=add_special_tokens, **kwargs)


def llamacpp_completion_prompt(model: Any, text: str, bos: Optional[str]) -> str:
    """A rendered chat for llama.cpp's `create_completion`, which ADDS the vocabulary's BOS itself.

    `create_completion` tokenizes its prompt with `add_bos=True, special=True`, so a render that
    already begins with the BOS text would start with two (the same defect, on the GGUF engine's
    continuation path; `create_chat_completion` avoids it by passing `add_bos=False` for a
    template it rendered). The leading BOS text is removed ONLY when llama.cpp is shown to add
    one: tokenizing the empty string with `add_bos=True` returns exactly `[token_bos()]`. A
    vocabulary that adds nothing keeps the template's BOS. Anything unreadable leaves the text
    unchanged — this can cost the fix, never the request.
    """
    if not bos or not text.startswith(bos):
        return text
    try:
        added = list(model.tokenize(b"", add_bos=True, special=True))
        if added == [int(model.token_bos())]:
            return text[len(bos):]
    except Exception:  # noqa: BLE001 - never fail a request on this
        pass
    return text
