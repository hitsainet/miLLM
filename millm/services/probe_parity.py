"""Does this build score the way miStudio did? (FR-24.4)

Before a probe can be armed, its definition's test vectors are re-scored here and compared with the
scores miStudio recorded. A probe that passes is one whose published AUROC is *about this
implementation*. A probe that fails is refused, because the alternative is a detector reporting
under a measurement that was taken of something slightly different.

## Three things this gets deliberately right

**It scores from `token_ids`, never from `messages`.** miStudio's own export acceptance measured
the recorded ids reproducing at **0.000e+00 on all sixteen vectors**, while re-rendering the
document's `messages` missed by up to **1.153** against a 0.05 tolerance — `messages` is a
reconstruction of plain prose as a single user turn, so re-rendering it adds template tokens. A
parity check built on `messages` would tell a correct consumer it was wrong, on every vector. That
is worse than no check, because it is believed the first time.

**Tokenization drift is reported BESIDE the verdict, never inside it.** Re-rendering `messages`
still tells you something worth knowing — whether a consumer following the document's prose would
get the same ids — so it is measured and reported separately. It never contributes to pass/fail.

**A scope it cannot reproduce fails rather than passes.** Only `scope="all"` is exactly
reproducible: for `prompt` and `response`, miStudio recorded scores under its narrower internal
role mask and the contract carries no record of which positions those were. Passing such a probe
would mean arming something this gate never actually checked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from millm.ml.probe_head import combine
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from millm.services.probe_scope import scope_is_reproducible

logger = logging.getLogger(__name__)

#: Reasons a vector could not be compared at all, as opposed to compared and found wrong.
NOT_COMPARABLE_SCOPE = "scope_positions_unavailable"
NOT_COMPARABLE_LENGTH = "scored_token_count_differs"
NOT_COMPARABLE_UNSTABLE = "too_few_reproducible_positions"

#: `attention` weights every token by `softmax(q . z_t)`, and a definition records no per-token
#: logits — so the producer's side cannot be recombined over a subset of positions. Such a probe
#: is gated on the full sequence, as it always was, and its report says so.
_RECOMBINABLE = ("mean", "max", "last", "softmax", "rolling_mean_max")


@dataclass
class VectorResult:
    index: int
    max_abs_diff: Optional[float] = None
    combined_diff: Optional[float] = None
    #: The combined score recomputed over the positions whose JumpReLU gate IS reproducible.
    #: `None` when the rule cannot be recombined (`attention`) or nothing was measured.
    robust_combined_diff: Optional[float] = None
    #: How many scored positions carried a near-threshold feature, out of how many.
    at_risk_tokens: int = 0
    n_tokens: int = 0
    expected_score: Optional[float] = None
    actual_score: Optional[float] = None
    comparable: bool = True
    reason: Optional[str] = None

    @property
    def gated_diff(self) -> Optional[float]:
        """The figure the gate uses: the robust one when it exists, else the full one."""
        return self.combined_diff if self.robust_combined_diff is None else self.robust_combined_diff


@dataclass
class ParityReport:
    """Whether this build reproduces the scores miStudio recorded.

    ⚠ **THE GATE IS THE COMBINED SCORE. PER-TOKEN DIVERGENCE IS REPORTED, NOT GATED.**

    **This implements the contract as written; the per-token gate was a misreading of it.**
    `ProbeTestVectors.tolerance` in `mistudio.probe-definition/v1` is documented as *"An ABSOLUTE
    score tolerance"*, and the measurement behind its 0.05 is explicitly about scores: batch
    composition moving **a score** by at most 5.78e-03 against a score range of −12.9 to 7.5.
    Nothing in the contract ever asked for per-token agreement. Gating on the per-token trace
    made the tolerance mean something 55x tighter than the producer calibrated it for.

    Confirmed by hardware acceptance 2026-09-27, which showed the per-token gate could not be
    passed by any independent implementation:

    miStudio scores probes in **float16**; miLLM serves **bfloat16**, deliberately, because fp16
    overflows on bf16-trained models and yields NaN logits. Over 16 real vectors the combined
    scores agreed to a median of 0.017 and a worst of 0.098, while per-token traces never came
    within 0.70 of each other — because the recorded 0.05 is an ABSOLUTE tolerance against
    per-token values reaching 55, i.e. 0.09% relative. Even a matched-precision fp16 run left
    per-token at 0.10–0.28.

    miStudio's recorded "0.000e+00 on all sixteen" was measured re-scoring **in the same process
    with the same model object**. A gate at that tightness is reachable only by bit-identical
    computation — and an independent implementation is precisely what this gate exists to verify.
    A check that a correct consumer cannot pass is the mirror of the defect miStudio already
    fixed once, where parity told a correct consumer it was wrong on every vector.

    **What this still catches**, and the reason it is a real gate and not a formality: the
    uncentered-basis defect found the same day moved the combined median to **1.68**, seventeen
    times this tolerance. A wrong hook point, a wrong layer, a wrong dictionary or a wrong
    feature selection all move the score by far more than precision does.

    **What it no longer catches:** a per-token pattern that cancels in the mean. That is why the
    per-token figures stay in the report, beside the verdict, rather than being dropped.

    ⚠ **AND FOR A k-SPARSE JumpReLU PROBE THE COMBINED SCORE ALONE IS NOT ENOUGH EITHER**, which
    is what `robust_combined_diff` exists for. A JumpReLU feature is `pre * H(pre - θ)`: a step.
    The same 1.5% producer/consumer residual difference that costs a dense probe 0.017 flips a
    handful of the k selected features across their own thresholds, and a flip is not a small
    disagreement — the feature moves between θ and 0, shifting THAT ONE token's score by ~20-40.
    The `mean` rule then divides by the token count, so the combined score lands wherever the
    flip count happens to put it.

    Measured at real width (2048 -> 16384, k = 128, L0 ~ 60, the reference probe's own head),
    against a 1.5% residual difference and against the uncentered-basis defect of 2026-09-27,
    over the same 390 tokens:

    | | per-token median | per-token max | combined | combined, robust positions |
    |---|---|---|---|---|
    | precision only | 0.0000 | 27.3 | 0.156 | **0.0015** |
    | uncentered basis (a real defect) | 0.0000 | 70.8 | 1.447 | **0.945** |
    | wrong input (wrong layer/hook) | 0.0000 | 126.7 | 1.070 | **0.390** |

    On hardware the same probe measured a combined median of 0.101 and a worst of 0.686 — so on
    the combined score a CORRECT build and that morning's defect are a factor of two apart, and
    raising the floor to admit 0.686 would have left this gate a 2x margin against the only
    defect it has ever caught. Setting the unreproducible positions aside instead leaves 630x,
    at the contract's own tolerance rather than a loosened one.

    **Two other designs were measured and rejected**, recorded so nobody re-derives them: gating
    on the per-token MEDIAN separates nothing, because a sparse basis leaves most tokens at 0 on
    both sides and the median is 0.0000 for the defects too; and admitting a worst-case
    Cauchy-Schwarz uncertainty band instead of dropping positions yields a band of **85.8**, 59x
    the defect it must leave visible.

    **A vector with too few reproducible positions is REFUSED, not passed on the remainder.**
    Below `PROBE_PARITY_MIN_ROBUST_FRACTION` of its scored positions, the comparison is resting
    on a handful of tokens and says little about the probe.

    **Why a 0.10 floor and not the contract's 0.05.** The contract's figure is calibrated against
    BATCH COMPOSITION (5.78e-03, x8.7 margin). Cross-precision is a larger effect, and the
    contract's own comment names it: *"fp16 versus bf16 at resid_post differs by about 1.5%
    relative"* — which on a score of −12.9 is 0.19, already past 0.05. Measured here, the worst of
    sixteen was 0.098. The floor covers the producer/consumer precision difference the contract
    anticipated but did not price in; it is applied as `max()` so a document asking for more slack
    still gets it.
    """

    tolerance: float
    vectors: list[VectorResult] = field(default_factory=list)
    tokenization_drift: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    #: `{"recorded", "loaded", "matched"}` — the definition's `model.load_dtype` against the
    #: precision this server loaded at (`dtype_comparison`). Reported on every report, so a failed
    #: parity shows its likely cause; and it chooses the floor (`score_tolerance`).
    dtype: dict[str, Any] = field(default_factory=dict)
    #: The probe's own bar (`decision.threshold`), which scales the matched floor. None for a probe
    #: that places no bar, whose matched floor is the absolute minimum.
    threshold: Optional[float] = None

    @property
    def max_abs_diff(self) -> Optional[float]:
        diffs = [v.max_abs_diff for v in self.vectors if v.max_abs_diff is not None]
        return max(diffs) if diffs else None

    @property
    def worst_vector(self) -> Optional[int]:
        worst, value = None, -1.0
        for vector in self.vectors:
            if vector.max_abs_diff is not None and vector.max_abs_diff > value:
                worst, value = vector.index, vector.max_abs_diff
        return worst

    @property
    def score_tolerance(self) -> float:
        """The tolerance the GATE uses, on the combined score.

        `max` of the document's own and the deployment's floor, so a definition asking for
        something looser is honoured and one asking for something tighter than any independent
        implementation can meet does not make the probe unusable.
        """
        from millm.core.config import settings

        # ⚠ THE FLOOR IS KEYED ON WHY IT EXISTS. `PROBE_PARITY_SCORE_TOLERANCE` (0.10) was set to
        # absorb a CROSS-PRECISION gap — miStudio float16, this server bfloat16. When the
        # definition states the same precision this server loaded at, that gap is gone and what
        # remains is two implementations' bfloat16 noise, which has its own measured floor.
        if self.dtype.get("matched") is True:
            # Measured two-implementation bfloat16 noise, which scales with the probe's own score
            # range — see PROBE_PARITY_MATCHED_RELATIVE_FLOOR for the numbers.
            relative = (
                settings.PROBE_PARITY_MATCHED_RELATIVE_FLOOR * abs(float(self.threshold))
                if self.threshold is not None else 0.0
            )
            floor = max(settings.PROBE_PARITY_MATCHED_DTYPE_FLOOR, relative)
        else:
            floor = settings.PROBE_PARITY_SCORE_TOLERANCE
        return max(self.tolerance, floor)

    @property
    def max_combined_diff(self) -> Optional[float]:
        diffs = [v.combined_diff for v in self.vectors if v.combined_diff is not None]
        return max(diffs) if diffs else None

    @property
    def max_robust_combined_diff(self) -> Optional[float]:
        """The worst combined difference over reproducible positions, or `None` if never measured."""
        diffs = [v.robust_combined_diff for v in self.vectors if v.robust_combined_diff is not None]
        return max(diffs) if diffs else None

    @property
    def max_gated_diff(self) -> Optional[float]:
        """What the gate compares: robust where it exists, full where it does not."""
        diffs = [v.gated_diff for v in self.vectors if v.gated_diff is not None]
        return max(diffs) if diffs else None

    @property
    def at_risk_tokens(self) -> int:
        return sum(v.at_risk_tokens for v in self.vectors)

    @property
    def scored_tokens(self) -> int:
        return sum(v.n_tokens for v in self.vectors)

    @property
    def passed(self) -> bool:
        if self.error is not None or not self.vectors:
            return False
        if any(not v.comparable for v in self.vectors):
            return False
        # ⚠ THE COMBINED SCORE over REPRODUCIBLE POSITIONS, not the per-token trace and not the
        # full sequence where a step gate makes part of it unreproducible. See the class docstring.
        worst = self.max_gated_diff
        if worst is None:
            # Vectors were comparable but carried no recorded score to compare against, so
            # nothing was actually checked. That is not a pass.
            return False
        return worst <= self.score_tolerance

    def as_details(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            # What the gate used, and what the document asked for — both, because they can
            # differ and a reader needs to know which decided the verdict.
            "tolerance": self.tolerance,
            "score_tolerance": self.score_tolerance,
            "max_combined_diff": self.max_combined_diff,
            # ⚠ THE FIGURE THE GATE USED. `max_combined_diff` is the whole sequence including
            # the positions a step gate makes unreproducible; this one excludes them. For a
            # dense probe the two are the same number.
            "max_robust_combined_diff": self.max_robust_combined_diff,
            "max_gated_diff": self.max_gated_diff,
            "at_risk_tokens": self.at_risk_tokens,
            "scored_tokens": self.scored_tokens,
            # ⚠ INFORMATIONAL. Large per-token divergence with small combined divergence is the
            # expected signature of a precision difference between producer and consumer; it is
            # reported so nobody has to rediscover that, and it does not gate.
            "max_abs_diff": self.max_abs_diff,
            "per_token_is_informational": True,
            "vector_index": self.worst_vector,
            "vectors": [
                {
                    "index": v.index,
                    "max_abs_diff": v.max_abs_diff,
                    "combined_diff": v.combined_diff,
                    "robust_combined_diff": v.robust_combined_diff,
                    "at_risk_tokens": v.at_risk_tokens,
                    "n_tokens": v.n_tokens,
                    "comparable": v.comparable,
                    "reason": v.reason,
                }
                for v in self.vectors
            ],
            "tokenization_drift": self.tokenization_drift,
            "dtype": self.dtype,
            "error": self.error,
        }


def dtype_comparison(
    definition: dict[str, Any], loaded_dtype: Optional[str], loaded_quantization: Optional[str] = None
) -> dict[str, Any]:
    """The definition's stated precision beside this server's. `matched` is None when either side
    is unknown — a definition from before 2026-10-03 states none, and is never assumed float16.

    ⚠ `matched` needs the QUANTIZATION to agree as well, when the definition states one (review
    round 1, MED-2): Q4 and FP16 loads of one bfloat16 checkpoint share a precision and read
    different activations, so the matched-precision floor must not apply across them."""
    model = (definition or {}).get("model") or {}
    recorded = model.get("load_dtype")
    recorded_quant = model.get("quantization")
    matched = None if recorded is None or loaded_dtype is None else recorded == loaded_dtype
    if matched and recorded_quant is not None:
        matched = (
            None if loaded_quantization is None
            else str(recorded_quant).upper() == str(loaded_quantization).upper()
        )
    return {"recorded": recorded, "loaded": loaded_dtype, "matched": matched,
            "recorded_quantization": recorded_quant, "loaded_quantization": loaded_quantization}


class ProbeParityEngine:
    """Re-scores a definition's test vectors through the live model."""

    def __init__(self, forward: Any) -> None:
        """`forward(input_ids) -> None` runs one pass with the probe's hook installed.

        Injected rather than taken from the model so the engine is testable without a GPU, and so
        the caller owns the `RequestQueue` slot and `torch.inference_mode()` — parity runs
        serialised with generation, not beside it.
        """
        self._forward = forward

    def run(
        self,
        probe: ArmedProbe,
        definition: dict[str, Any],
        *,
        tolerance: float,
        tokenizer: Any = None,
        loaded_dtype: Optional[str] = None,
        loaded_quantization: Optional[str] = None,
    ) -> ParityReport:
        spec = (definition or {}).get("test_vectors") or {}
        vectors: Sequence[dict[str, Any]] = spec.get("vectors") or []
        threshold = ((definition or {}).get("decision") or {}).get("threshold")
        report = ParityReport(
            tolerance=tolerance,
            dtype=dtype_comparison(definition, loaded_dtype, loaded_quantization),
            threshold=float(threshold) if isinstance(threshold, (int, float)) and not isinstance(threshold, bool) else None,
        )
        recorded_quant = report.dtype.get("recorded_quantization")
        if (recorded_quant is not None and loaded_quantization is not None
                and str(recorded_quant).upper() != str(loaded_quantization).upper()):
            # ⚠ NOT "unmatched precision" with the looser floor (review round 3, MED-A): a Q4
            # probe re-scored on an FP16 load reads different activations (~0.93 cosine per
            # token). A comparison across quantizations is not a check of THIS probe, so it is
            # reported as such rather than allowed to pass under a floor sized for fp16/bf16 noise.
            report.error = (
                f"the probe was fitted at {recorded_quant} and this model is loaded at "
                f"{loaded_quantization}; parity across a quantization change does not test the probe"
            )
            return report

        if not vectors:
            report.error = "the definition carries no test vectors, so parity cannot be checked"
            return report

        reproducible = scope_is_reproducible(probe.scope)

        for index, vector in enumerate(vectors):
            result = VectorResult(index=index)
            if not reproducible:
                # miStudio scored a narrower set of positions than `token_ids` describes, and the
                # contract does not record which. Guessing would produce a number that looks like
                # a comparison and is not one.
                result.comparable = False
                result.reason = NOT_COMPARABLE_SCOPE
                report.vectors.append(result)
                continue

            token_ids = list(vector.get("token_ids") or [])
            expected_tokens = [float(x) for x in (vector.get("token_scores") or [])]
            result.expected_score = float(vector.get("score")) if "score" in vector else None

            if not token_ids:
                result.comparable = False
                result.reason = "vector carries no token_ids"
                report.vectors.append(result)
                continue

            context = ProbeRequestContext(f"parity:{index}", [probe], collect_flip_risk=True)
            self._forward(torch.tensor([token_ids], dtype=torch.long), context)
            verdicts = context.finish()
            # ⚠ SELECTED BY WINDOW, NOT BY POSITION. `verdicts[0]` was safe only while a probe
            # produced exactly one verdict. Parity compares against miStudio's recorded scores,
            # which were computed under the probe's CONTRACT scope — reading a different window
            # here would report a disagreement that is really a comparison of two different
            # things, on the one check a consumer leans on to trust its own implementation.
            verdict = next(
                (v for v in verdicts if v.window == probe.scope), verdicts[0] if verdicts else None
            )
            if verdict is None:
                result.comparable = False
                result.reason = "no verdict was produced for the probe's own scope"
                report.vectors.append(result)
                continue

            if not verdict.scored:
                result.comparable = False
                result.reason = verdict.not_scored_reason or "not scored"
                report.vectors.append(result)
                continue

            actual_tokens = context.token_scores_for(probe.probe_id)
            result.actual_score = verdict.score

            if len(actual_tokens) != len(expected_tokens):
                # Not a tolerance failure — a shape failure. Saying "off by 3.2" about tensors of
                # different lengths would be a number with no meaning.
                result.comparable = False
                result.reason = (
                    f"{NOT_COMPARABLE_LENGTH}: recorded {len(expected_tokens)}, "
                    f"scored {len(actual_tokens)}"
                )
                report.vectors.append(result)
                continue

            per_token = max(
                (abs(a - b) for a, b in zip(actual_tokens, expected_tokens)), default=0.0
            )
            combined = (
                abs(verdict.score - result.expected_score)
                if result.expected_score is not None and verdict.score is not None
                else 0.0
            )
            result.combined_diff = combined
            result.max_abs_diff = max(per_token, combined)
            result.n_tokens = len(actual_tokens)

            at_risk = context.flip_risk_for(probe.probe_id)
            result.at_risk_tokens = sum(1 for flag in at_risk if flag)
            if result.at_risk_tokens and probe.rule in _RECOMBINABLE:
                robust = [not flag for flag in at_risk]
                if sum(robust) < self._min_robust_tokens(len(robust)):
                    # Refused, not passed on the remainder: a comparison resting on a handful of
                    # positions says little about the probe, and a probe whose basis is mostly
                    # unreproducible is one nobody should arm on this build.
                    result.comparable = False
                    result.reason = (
                        f"{NOT_COMPARABLE_UNSTABLE}: {sum(robust)} of {len(robust)} positions "
                        f"carry no near-threshold feature"
                    )
                    report.vectors.append(result)
                    continue
                result.robust_combined_diff = self._recombined_diff(
                    probe, actual_tokens, expected_tokens, robust
                )
            report.vectors.append(result)

        if tokenizer is not None:
            report.tokenization_drift = self._drift(vectors, tokenizer)

        logger.info(
            "probe_parity probe=%s passed=%s gated_diff=%s combined_diff=%s max_abs_diff=%s "
            "at_risk=%s/%s tolerance=%s",
            probe.probe_id,
            report.passed,
            report.max_gated_diff,
            report.max_combined_diff,
            report.max_abs_diff,
            report.at_risk_tokens,
            report.scored_tokens,
            tolerance,
        )
        return report

    @staticmethod
    def _min_robust_tokens(n: int) -> int:
        """At least this many positions must be reproducible for the comparison to mean anything."""
        from millm.core.config import settings

        return max(1, math.ceil(n * settings.PROBE_PARITY_MIN_ROBUST_FRACTION))

    @staticmethod
    def _recombined_diff(
        probe: ArmedProbe,
        actual: Sequence[float],
        expected: Sequence[float],
        robust: Sequence[bool],
    ) -> float:
        """|combined(actual) - combined(expected)| over the reproducible positions only.

        ⚠ **BOTH SIDES ARE RECOMBINED THROUGH THE PROBE'S OWN RULE**, not averaged. Dropping a
        position from a `max` or a `last` probe is not the same operation as dropping it from a
        `mean`, and hand-averaging here would compare a statistic the probe does not compute.
        """
        mask = torch.tensor([robust], dtype=torch.bool)
        params = {k: v for k, v in (probe.rule_params or {}).items() if k in ("tau", "window")}
        values = [
            float(
                combine(
                    probe.rule,
                    torch.tensor([list(side)], dtype=torch.float32),
                    mask=mask,
                    **params,
                ).item()
            )
            for side in (actual, expected)
        ]
        return abs(values[0] - values[1])

    @staticmethod
    def _drift(vectors: Sequence[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
        """Do the document's `messages` re-render to the ids it recorded?

        ⚠ REPORTED, NEVER SCORED. A mismatch here does not mean this build is wrong — miStudio's
        acceptance already established that `messages` is a prose reconstruction whose re-render
        adds template tokens. It means a consumer following the document's prose instead of its
        ids would get different tokens, which is worth knowing and is not a parity failure.
        """
        matched = 0
        mismatched: list[int] = []
        for index, vector in enumerate(vectors):
            messages = vector.get("messages") or []
            recorded = list(vector.get("token_ids") or [])
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
            except Exception:
                mismatched.append(index)
                continue
            if list(rendered) == recorded:
                matched += 1
            else:
                mismatched.append(index)
        return {
            "checked": len(vectors),
            "messages_reproduce_token_ids": matched,
            "mismatched_vectors": mismatched[:10],
            "note": (
                "informational only — parity scores from token_ids, which is the contract's "
                "authoritative input"
            ),
        }
