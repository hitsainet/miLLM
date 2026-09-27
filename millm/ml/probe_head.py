"""The probe readout: per-token scores, and the six rules that combine them.

⚠ THIS IS A SECOND IMPLEMENTATION OF A CONTRACT MISTUDIO OWNS, AND THAT IS THE RISK.
miStudio's `ml/probe_monitor_model.py` fitted the weights; this module has to reproduce its
arithmetic exactly or the probe detects something subtly different from what was measured. Two
implementations of one definition is precisely the shape that drifts silently.

The mitigation is not care, it is the **parity gate** (FR-24.4): before a probe can be armed,
the definition's test vectors are scored through this code and compared with the scores miStudio
recorded, within 1e-3. miStudio's own acceptance measured those vectors reproducing at 0.000e+00
from the recorded `token_ids`, so the tolerance is absorbing nothing on the reference path — any
real divergence here shows up as a refusal to arm, not as a wrong verdict.

Every deviation from the obvious implementation below is deliberate and is carried over from a
defect miStudio already paid for. They are marked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

#: Every combining rule, in contract order.
RULES: Tuple[str, ...] = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")

#: Rules whose value over the tokens seen SO FAR is their value, so a serving runtime can answer
#: "what is the verdict now" at every token.
#:
#: `last` is absent BY DEFINITION, not by omission. Keeping the newest score would be trivial, but
#: `last` is not defined until generation stops: the score at the current token is a guess that the
#: row is about to end, not the row's score. The online form refuses rather than returning the
#: newest value and letting a caller believe it is monitoring.
STREAMABLE: frozenset = frozenset(RULES) - {"last"}

DEFAULT_TAU: float = 1.0
DEFAULT_WINDOW: int = 16


def is_streamable(rule: str) -> bool:
    """Whether `rule` can be computed while tokens stream.

    Raises on an unknown rule rather than returning False: a typo reading as "not streamable"
    would silently disable streaming for a rule that supports it, and the caller would see a
    plausible answer.
    """
    if rule not in RULES:
        raise ValueError(f"unknown combining rule {rule!r}; known: {', '.join(RULES)}")
    return rule in STREAMABLE


@dataclass(frozen=True)
class ProbeHead:
    """One probe's weights, standardisation statistics and optional attention query.

    `weight` is 1-D. Its length is `d_model` for a dense probe and `k` for a k-sparse SAE probe —
    the head does not care which, because by the time activations reach it an SAE probe's input has
    already been encoded to its k features.
    """

    weight: torch.Tensor
    bias: float = 0.0
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    attention_query: Optional[torch.Tensor] = None
    eps: float = 1e-6
    layer: Optional[int] = None

    def __post_init__(self) -> None:
        if self.weight.ndim != 1:
            raise ValueError(f"weight must be 1-D, got {tuple(self.weight.shape)}")
        for name in ("mean", "std", "attention_query"):
            vector = getattr(self, name)
            if vector is not None and vector.shape != self.weight.shape:
                raise ValueError(
                    f"{name} has shape {tuple(vector.shape)} but weight is "
                    f"{tuple(self.weight.shape)} — a probe's normalisation must match its weights"
                )

    def standardise(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., d) -> (..., d), using the TRAINING statistics.

        ⚠ A channel whose training std is at or below `eps` is ZEROED, not divided by a floor.
        Clamping a 0 std to 1e-6 turns a 0.001 drift into 1000.0 — miStudio records that
        amplification, and the probe-definition contract refuses a `norm_std` of 0 for the same
        reason. On training data the two agree exactly (a centred constant is 0); they differ only
        off it, which is exactly where a monitor operates.

        ⚠ **THREE THINGS HERE ARE BOOKKEEPING AND NOT ARITHMETIC**, and the distinction is the
        whole point: every value this returns is bit-identical to the obvious form, because the
        head is frozen and `torch.where` is a selection.

        1. `degenerate` and `divisor` depend only on `std` and `eps`, so they are MEMOISED
           instead of rebuilt on every call (4 kernel launches).
        2. The zero is 0-dim. `torch.where` broadcasts it to exactly the values a full
           `zeros_like(out)` would have selected, without a (T, d) memset — 33.6 MB at
           4k x 2048.
        3. When NO channel is degenerate the `where` is SKIPPED, because
           `where(all_false, z, x)` is `x`. That is the case for every probe miStudio can
           export, since the probe-definition contract refuses a `norm_std` of 0 — so the
           branch that exists for the degenerate case stops being paid for by the probes that
           do not have one.

        Why any of this is worth writing down: this runs once per forward pass per probe, so a
        request with a 4k prompt and a 192-token completion ran it 386 times per probe, and at
        one token per pass the cost is kernel LAUNCHES, not FLOPs. Measured on a 3090, one
        decode pass for one probe went 0.117 ms -> 0.052 ms -> 0.0xx ms as these came off.
        """
        out = activations
        if self.mean is not None:
            out = out - self.mean
        if self.std is not None:
            degenerate, divisor, any_degenerate = self._standardisation()
            out = out / divisor
            if any_degenerate:
                out = torch.where(degenerate, out.new_zeros(()), out)
        return out

    def _standardisation(self) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """`(degenerate, divisor, any_degenerate)` for `std`, computed once per head.

        Cached on the instance rather than recomputed, because this head is frozen: `std` and
        `eps` cannot change, so the answer cannot either. Stored outside the dataclass fields so
        equality and the field list are untouched — a memo, not state.

        `any_degenerate` is resolved to a Python bool HERE, once, so the hot path branches on it
        without a device synchronisation.
        """
        cached = self.__dict__.get("_std_cache")
        if cached is None:
            assert self.std is not None  # only reached from the `std is not None` branch
            degenerate = self.std.abs() <= self.eps
            divisor = torch.where(degenerate, torch.ones_like(self.std), self.std)
            cached = (degenerate, divisor, bool(degenerate.any()))
            object.__setattr__(self, "_std_cache", cached)
        return cached

    def token_scores(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., T, d) -> (..., T). The per-token score, before any rule.

        ⚠ **THE ARITHMETIC IS DELIBERATELY NOT OPTIMISED, AND THAT IS A MEASURED DECISION.**

        `standardise(z) @ w` materialises a (T, d) intermediate — 33.6 MB at 4k tokens and
        d_model 2048 — for a (T,) result, and it folds away exactly:

            ((z - mean) / std) . w  ==  z . (w / std)  -  (mean / std) . w

        one matvec, no intermediate, **14.1 ms -> 5.4 ms** measured at 4k x 2048. It is not
        used, because the fold moves the score by **1.9e-06** and
        `test_probe_head_matches_mistudio.py` requires **bit-exact** agreement with
        miStudio's scorer — the property 033's acceptance recorded as "0.000e+00, all
        sixteen vectors". Exactness is a far stronger guarantee than "inside the 1e-3
        tolerance", and it is what makes a parity pass mean something. A 2.6x saving on
        arithmetic that is no longer the bottleneck (see `ProbeRequestContext.observe` —
        the copy was) does not buy that back.

        Also rejected, and worth recording so nobody re-derives it: doing the matvec in the
        activations' native fp16 is another 4x and moves the score by up to **8.5e-02**,
        which is 85x the parity tolerance. It would make every armed probe fail parity, or
        pass it by luck while scoring differently from what was measured.

        What DID change is where this runs: the weights follow the activations' device, so a
        GPU tensor is scored on the GPU and only the (T,) result crosses to the host.
        """
        if activations.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"activations are d={activations.shape[-1]} but this probe is "
                f"d={self.weight.shape[0]}"
            )
        head = self.to_device(activations.device)
        return head.standardise(activations.to(torch.float32)) @ head.weight + head.bias

    def to_device(self, device: torch.device) -> "ProbeHead":
        """This head with its tensors on `device`, in float32. Returns self when already there.

        float32 regardless of the model's dtype: an fp16 matvec moves the score by up to
        8.5e-02 against a 1e-3 parity tolerance, so precision here is a correctness
        requirement and not a tuning knob.
        """
        if self.weight.device == device and self.weight.dtype == torch.float32:
            return self

        def move(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.to(device=device, dtype=torch.float32)

        return ProbeHead(
            weight=self.weight.to(device=device, dtype=torch.float32),
            bias=self.bias,
            mean=move(self.mean),
            std=move(self.std),
            attention_query=move(self.attention_query),
            eps=self.eps,
            layer=self.layer,
        )

    def attention_logits(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., T, d) -> (..., T). `q · standardise(z_t)`, for the `attention` rule.

        Standardised with the SAME statistics as the score, because the query was trained in that
        space. Refuses when no query is set rather than falling back to the scores — that fallback
        would silently turn `attention` into `softmax` at tau=1, a different detector under the
        same name.
        """
        if self.attention_query is None:
            raise ValueError(
                "this probe has no attention_query, so the `attention` rule cannot be computed; "
                "using the scores as the weighting would silently make it `softmax` at tau=1"
            )
        return self.standardise(activations) @ self.attention_query


# ── masking ───────────────────────────────────────────────────────────────────
# A mask is 1 for a scored position and 0 otherwise. It is required reasoning, not an
# optimisation: a rule that folds padding or out-of-scope positions into a mean or a max
# reports a number about the padding.

def _require_mask(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match scores {tuple(scores.shape)}")
    return mask.to(torch.bool)


def _require_any_token(valid: torch.Tensor) -> None:
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every row must have at least one scored position")


def _last_valid_index(valid: torch.Tensor) -> torch.Tensor:
    """Index of the last unmasked token per row, for ANY padding side.

    ⚠ Searching from the right, not counting real tokens. Counting and taking `count - 1` is only
    the last real token under RIGHT padding; under left padding (`[0,0,1,1]`) it returns index 1,
    a PAD position. miStudio shipped that version with a docstring claiming left-padding
    correctness while every test used right padding.
    """
    length = valid.shape[-1]
    reversed_first_valid = valid.flip(-1).to(torch.int8).argmax(dim=-1)
    return (length - 1) - reversed_first_valid


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over unmasked positions only, max-shifted for stability."""
    filled = logits.masked_fill(~valid, float("-inf"))
    shifted = filled - filled.amax(dim=-1, keepdim=True)
    weights = shifted.exp() * valid
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def _rolling_mean_max(scores: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    """Maximum, over windows of `window` tokens, of the windowed mean.

    A sequence SHORTER than the window is scored as one window over all its real tokens, which
    makes the rule equal `mean` there. Refusing, or padding the window, would make short rows
    unscoreable or score them against padding.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    out = torch.empty(flat_scores.shape[0], dtype=scores.dtype, device=scores.device)

    for row in range(flat_scores.shape[0]):
        real = flat_scores[row][flat_valid[row]]
        n = real.numel()
        if n <= window:
            out[row] = real.mean()
            continue
        cumulative = torch.cat(
            [torch.zeros(1, dtype=real.dtype, device=real.device), real.cumsum(0)]
        )
        window_sums = cumulative[window:] - cumulative[:-window]
        out[row] = (window_sums / window).max()

    return out.reshape(scores.shape[:-1])


def combine(
    rule: str,
    scores: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    attention_logits: Optional[torch.Tensor] = None,
    tau: float = DEFAULT_TAU,
    window: int = DEFAULT_WINDOW,
) -> torch.Tensor:
    """Combine per-token scores into one score per row. `scores` is (..., T); returns (...,).

    `attention_logits` is (..., T) and is REQUIRED by `attention` only: that rule weights tokens by
    `softmax(q · z_t)` while the value stays `w · z_t`, so the weighting is a learned function of
    the activation and NOT of the score. Passing the scores as the logits silently turns
    `attention` into `softmax` at tau=1.
    """
    valid = _require_mask(scores, mask)
    _require_any_token(valid)

    if rule == "mean":
        return (scores * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)

    if rule == "max":
        return scores.masked_fill(~valid, float("-inf")).amax(dim=-1)

    if rule == "last":
        index = _last_valid_index(valid)
        return scores.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    if rule == "softmax":
        if tau <= 0:
            raise ValueError(f"softmax temperature must be > 0, got {tau}")
        weights = _masked_softmax(scores / tau, valid)
        return (weights * scores).sum(dim=-1)

    if rule == "attention":
        if attention_logits is None:
            raise ValueError(
                "the `attention` rule needs `attention_logits` (q · z_t); passing the scores "
                "instead would make it `softmax` at tau=1 under another name"
            )
        if attention_logits.shape != scores.shape:
            raise ValueError(
                f"attention_logits {tuple(attention_logits.shape)} must match scores "
                f"{tuple(scores.shape)}"
            )
        weights = _masked_softmax(attention_logits, valid)
        return (weights * scores).sum(dim=-1)

    if rule == "rolling_mean_max":
        return _rolling_mean_max(scores, valid, window)

    raise ValueError(f"unknown combining rule {rule!r}; known: {', '.join(RULES)}")


class OnlineRule:
    """Incremental combiner: `update(score)` per token, then read `value`.

    One instance is ONE request. It exists so the runtime can answer "what is the verdict now"
    while tokens stream, and so that ability is testable against `combine` rather than assumed —
    `tests/unit/ml/test_probe_head.py` runs the two forms against each other on every streamable
    rule.

    The softmax/attention accumulator uses running-max rescaling rather than a naive sum of
    exponentials, because that is where the online and batch forms part company first.
    """

    def __init__(self, rule: str, *, tau: float = DEFAULT_TAU, window: int = DEFAULT_WINDOW):
        if not is_streamable(rule):  # raises on an unknown rule
            raise ValueError(
                f"the {rule!r} rule cannot be computed while tokens stream: its value is not "
                f"defined until the sequence ends, so a partial answer would be a guess that the "
                f"row is about to finish, not a verdict"
            )
        if rule == "softmax" and tau <= 0:
            raise ValueError(f"softmax temperature must be > 0, got {tau}")
        if rule == "rolling_mean_max" and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        self.rule = rule
        self.tau = tau
        self.window = window
        self.count = 0
        self._sum = 0.0
        self._max = -math.inf
        self._shift = -math.inf
        self._num = 0.0
        self._den = 0.0
        self._recent: List[float] = []
        self._window_sum = 0.0
        self._window_best = -math.inf

    def update(self, score: float, *, attention_logit: Optional[float] = None) -> float:
        """Consume one token's score and return the combined value SO FAR."""
        score = float(score)
        self.count += 1

        if self.rule == "mean":
            self._sum += score
        elif self.rule == "max":
            self._max = max(self._max, score)
        elif self.rule in ("softmax", "attention"):
            if self.rule == "attention":
                if attention_logit is None:
                    raise ValueError(
                        "the `attention` rule needs `attention_logit` (q · z_t) per token"
                    )
                logit = float(attention_logit)
            else:
                logit = score / self.tau
            if logit > self._shift:
                factor = math.exp(self._shift - logit) if self._shift > -math.inf else 0.0
                self._num *= factor
                self._den *= factor
                self._shift = logit
            weight = math.exp(logit - self._shift)
            self._num += weight * score
            self._den += weight
        elif self.rule == "rolling_mean_max":
            self._recent.append(score)
            self._window_sum += score
            if len(self._recent) > self.window:
                self._window_sum -= self._recent.pop(0)
            if len(self._recent) == self.window:
                self._window_best = max(self._window_best, self._window_sum / self.window)

        return self.value

    @property
    def value(self) -> float:
        if self.count == 0:
            raise ValueError("no tokens consumed yet, so there is no score")
        if self.rule == "mean":
            return self._sum / self.count
        if self.rule == "max":
            return self._max
        if self.rule in ("softmax", "attention"):
            return self._num / self._den
        if self.rule == "rolling_mean_max":
            if self._window_best > -math.inf:
                return self._window_best
            # Fewer tokens than the window: one window over everything seen, matching what the
            # batch form does for a short sequence.
            return self._window_sum / len(self._recent)
        raise AssertionError(f"unhandled streamable rule {self.rule!r}")


def rule_parameters(
    rule: str, *, tau: float = DEFAULT_TAU, window: int = DEFAULT_WINDOW
) -> Dict[str, float]:
    """The parameters that must travel with a probe for this rule.

    A rule whose parameters are absent from a definition is not a rule with defaults — it is an
    under-specified detector, because `softmax` at tau=0.1 and tau=10 are different functions.
    """
    if rule not in RULES:
        raise ValueError(f"unknown combining rule {rule!r}; known: {', '.join(RULES)}")
    if rule == "softmax":
        return {"tau": tau}
    if rule == "rolling_mean_max":
        return {"window": window}
    return {}
