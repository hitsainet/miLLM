"""Pooling and normalisation for `/v1/embeddings` (Feature 30, FR-30.2).

ONE pooling path. The transformers embedding service and Feature 26's batch executor both call
these functions; nothing is copied, so a padded batch row and a single request pool identically.

Pooling is defined over REAL tokens, read from the attention mask, never over fixed indices:
under left padding index 0 is padding, under right padding the last index is. A fixture whose
pad positions hold zeros agrees with a mask-blind mean by construction, so the tests put large
values there.

The default (`mean`, no normalisation) returns today's floats element for element: an unpadded
row takes `hidden.mean(dim=1)`, the exact expression the service used before this feature, and
the vector is converted with the same `.cpu().tolist()`. Stored retrieval indexes built from
earlier vectors keep matching new queries (FR-30.2.2).

Imports only `torch` and the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

PoolingMode = Literal["mean", "last", "cls"]


class NonFiniteEmbeddingError(ValueError):
    """A pooled vector with a non-finite value, or a zero / non-finite norm under `normalize`.

    The service turns this into a 500 naming the input's index: the request was valid, and the
    output must never carry NaN or infinity (FR-30.2.6)."""


@dataclass(frozen=True)
class EmbeddingOptions:
    """What a request asks of the pooled vector. Built from the request body by the service and
    by Feature 26's executor from a batch row, the same way."""

    pooling: PoolingMode = "mean"
    normalize: bool = False


def pool_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor, mode: PoolingMode
) -> torch.Tensor:
    """Pool `[B, T, D]` hidden states to `[B, D]` over the positions whose mask is 1.

    * `mean`: the average over real positions;
    * `last`: the vector at the last real position;
    * `cls`: the vector at the first real position. On a causal decoder that position attends
      only to itself, so where the tokenizer prepends a fixed BOS every input gets the same
      `cls` vector (T-92: served and documented, not refused).

    Returns `[B, D]` and never squeezes: a width-1 model must still return a one-element vector.
    """
    if hidden.dim() != 3 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError(
            f"hidden {tuple(hidden.shape)} and mask {tuple(attention_mask.shape)} disagree"
        )
    mask = attention_mask.to(hidden.device).bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("a row has no real tokens")
    if mode == "mean":
        if bool(mask.all()):
            # Unpadded: today's exact expression, so the default vector does not move.
            return hidden.mean(dim=1)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    # argmax over an int tensor returns the FIRST maximum; that is what makes both indices right.
    if mode == "cls":
        idx = mask.int().argmax(dim=1)
    elif mode == "last":
        idx = hidden.shape[1] - 1 - mask.flip(1).int().argmax(dim=1)
    else:
        raise ValueError(f"unknown pooling mode {mode!r}")
    return hidden[rows, idx]


def finalize_vector(
    vec: torch.Tensor | Sequence[float], normalize: bool
) -> list[float]:
    """The returned floats for one pooled vector.

    `normalize=False` returns the vector's own values unchanged (a list from llama.cpp comes back
    as the same floats). `normalize=True` L2-normalises in float32, so the norm is 1 within 1e-5
    whatever the model's dtype. Non-finite values, and a zero or non-finite norm, raise
    NonFiniteEmbeddingError rather than emit NaN.
    """
    if not isinstance(vec, torch.Tensor):
        values = list(vec)
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise NonFiniteEmbeddingError("pooled vector has non-finite values")
        if not normalize:
            return [float(v) for v in values]
        vec = tensor
    if not bool(torch.isfinite(vec).all()):
        raise NonFiniteEmbeddingError("pooled vector has non-finite values")
    if not normalize:
        return vec.cpu().tolist()
    v = vec.to(torch.float32)
    norm = torch.linalg.vector_norm(v)
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        raise NonFiniteEmbeddingError("pooled vector has zero or non-finite norm")
    return (v / norm).cpu().tolist()  # type: ignore[no-any-return]
