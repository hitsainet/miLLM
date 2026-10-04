"""The next token's log-probabilities, for scoring-mode completions (2026-10-04).

A typed-decision judge (a Jev-style classifier such as autotrust/JEV-9B, or jevify) does not generate
an answer: it reads the probability of a handful of answer tokens at the next position. This module
is that arithmetic and nothing else, so it can be tested without a model.

Semantics match vLLM's `processed_logprobs` with `allowed_token_ids`:

* the logits are divided by the temperature when it is positive (0 means greedy: no scaling);
* with `allowed` ids, the distribution is renormalised over THAT SET ONLY — a judge's answer tokens
  are then always present, whatever the model would rank in its unrestricted top 20;
* the chosen token is the most probable allowed token. Scoring mode is for reading probabilities,
  so the token returned is deterministic rather than sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class NextTokenScores:
    chosen_id: int
    chosen_logprob: float
    #: (token id, log-probability), most probable first. Always contains the chosen token.
    top: list[tuple[int, float]]


def next_token_scores(
    logits: torch.Tensor,
    *,
    allowed: Optional[Sequence[int]],
    temperature: float,
    top_k: int,
) -> NextTokenScores:
    """Score the next token from one position's logits (shape `[vocab]`)."""
    if logits.dim() != 1:
        raise ValueError(f"expected one position's logits, got shape {tuple(logits.shape)}")
    z = logits.float()
    if temperature > 0:
        z = z / temperature
    if allowed is not None:
        ids = torch.tensor(list(dict.fromkeys(int(i) for i in allowed)), dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("allowed token ids are empty")
        if int(ids.max()) >= z.numel() or int(ids.min()) < 0:
            raise ValueError(f"allowed token ids fall outside the vocabulary of {z.numel()}")
        logprobs = torch.log_softmax(z[ids], dim=0)
    else:
        ids = torch.arange(z.numel())
        logprobs = torch.log_softmax(z, dim=0)
    order = torch.argsort(logprobs, descending=True)
    keep = order[: max(1, min(int(top_k), ids.numel()))]
    top = [(int(ids[j]), float(logprobs[j])) for j in keep]
    best = int(order[0])
    return NextTokenScores(chosen_id=int(ids[best]), chosen_logprob=float(logprobs[best]), top=top)
