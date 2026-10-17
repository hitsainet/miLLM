"""Switch a llama.cpp context out of embedding mode for the length of a generation.

⚠ THE DEFECT THIS EXISTS FOR (suite tracker 1.9, 2026-10-16). miLLM loads every GGUF model with
`embedding=True` and MEAN pooling (`GGUF_ENABLE_EMBEDDINGS`, default on) so one instance can answer
`/v1/embeddings` as well as chat. A context in embedding mode with pooling does NOT produce usable
next-token logits: on `LFM2.5-1.2B-Instruct-GGUF:Q4_K_M` every chat answered 200 with EMPTY
`content` and non-zero `completion_tokens` (reported from miForge at three different
`max_tokens`; a judge pointed at it read `judge_reply_unparseable` forever). Reproduced on CPU with
llama-cpp-python 0.3.36 and the same file: the same request with `embedding=False` answers
normally, and with `embedding=True` + `pooling_type=MEAN` the content is `''`.

llama.cpp lets a context change mode at run time (`llama_set_embeddings`). Turning embeddings OFF
around a generation and back ON afterwards gave correct chat output, and embeddings computed after
a chat were bit-identical to a fresh embedding-only instance (cosine 1.0, measured in the same
reproduction). The pooling type is fixed at construction and stays MEAN, so `/v1/embeddings` is
unchanged.

Every llama.cpp generation runs inside one admission slot (max_concurrent = 1, `Llama` is not
thread-safe), as do embeddings, so the flag never flips under another caller's feet.

The decision is read from the INSTANCE (`context_params.embeddings`), never from the setting: a
load that fell back to a context without embeddings must not have them switched on afterwards.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import structlog

logger = structlog.get_logger()


def _embedding_context(model: Any) -> Optional[Any]:
    """The raw `llama_context` pointer when `model` was built with embeddings on, else None."""
    params = getattr(model, "context_params", None)
    # `is True`, not truthiness: llama-cpp-python gives a real bool; a test double must opt in.
    if params is None or getattr(params, "embeddings", False) is not True:
        return None
    holder = getattr(model, "_ctx", None)
    return getattr(holder, "ctx", None) if holder is not None else None


def _set_embeddings(ctx: Any, enabled: bool) -> None:
    import llama_cpp

    llama_cpp.llama_set_embeddings(ctx, enabled)


@contextmanager
def generation_mode(model: Any) -> Iterator[None]:
    """Run the block with `model`'s context generating tokens, not pooling embeddings.

    A no-op for an instance built without embeddings. The flag is restored in `finally`, so an
    exception (or a streaming client hanging up) cannot leave `/v1/embeddings` broken.
    """
    ctx = _embedding_context(model)
    if ctx is None:
        yield
        return
    _set_embeddings(ctx, False)
    try:
        yield
    finally:
        try:
            _set_embeddings(ctx, True)
        except Exception as exc:  # noqa: BLE001 - never mask the generation's own outcome
            logger.error(
                "llamacpp_embeddings_restore_failed",
                error=str(exc)[:200],
                detail="/v1/embeddings on this GGUF model will fail until it is reloaded",
            )
