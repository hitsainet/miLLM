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
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from millm.services.probe_scope import scope_is_reproducible

logger = logging.getLogger(__name__)

#: Reasons a vector could not be compared at all, as opposed to compared and found wrong.
NOT_COMPARABLE_SCOPE = "scope_positions_unavailable"
NOT_COMPARABLE_LENGTH = "scored_token_count_differs"


@dataclass
class VectorResult:
    index: int
    max_abs_diff: Optional[float] = None
    combined_diff: Optional[float] = None
    expected_score: Optional[float] = None
    actual_score: Optional[float] = None
    comparable: bool = True
    reason: Optional[str] = None


@dataclass
class ParityReport:
    """Whether this build reproduces the scores miStudio recorded.

    ⚠ **THE GATE IS THE COMBINED SCORE. PER-TOKEN DIVERGENCE IS REPORTED, NOT GATED.**
    Decided 2026-09-27 by the product owner, after hardware acceptance showed the original
    per-token gate could not be passed by any independent implementation:

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
    """

    tolerance: float
    vectors: list[VectorResult] = field(default_factory=list)
    tokenization_drift: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

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

        return max(self.tolerance, settings.PROBE_PARITY_SCORE_TOLERANCE)

    @property
    def max_combined_diff(self) -> Optional[float]:
        diffs = [v.combined_diff for v in self.vectors if v.combined_diff is not None]
        return max(diffs) if diffs else None

    @property
    def passed(self) -> bool:
        if self.error is not None or not self.vectors:
            return False
        if any(not v.comparable for v in self.vectors):
            return False
        # ⚠ THE COMBINED SCORE, not the per-token trace. See the class docstring.
        worst = self.max_combined_diff
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
                    "comparable": v.comparable,
                    "reason": v.reason,
                }
                for v in self.vectors
            ],
            "tokenization_drift": self.tokenization_drift,
            "error": self.error,
        }


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
    ) -> ParityReport:
        spec = (definition or {}).get("test_vectors") or {}
        vectors: Sequence[dict[str, Any]] = spec.get("vectors") or []
        report = ParityReport(tolerance=tolerance)

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

            context = ProbeRequestContext(f"parity:{index}", [probe])
            self._forward(torch.tensor([token_ids], dtype=torch.long), context)
            verdicts = context.finish()
            verdict = verdicts[0]

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
            report.vectors.append(result)

        if tokenizer is not None:
            report.tokenization_drift = self._drift(vectors, tokenizer)

        logger.info(
            "probe_parity probe=%s passed=%s max_abs_diff=%s tolerance=%s",
            probe.probe_id,
            report.passed,
            report.max_abs_diff,
            tolerance,
        )
        return report

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
