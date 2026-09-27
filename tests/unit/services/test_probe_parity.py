"""The gate that refuses to arm a probe this build does not reproduce.

The three properties worth holding onto:

* parity scores from **`token_ids`**, never from `messages`;
* **tokenization drift is reported beside the verdict, never inside it**;
* a scope it **cannot** reproduce **fails** rather than passing — because passing would mean
  arming something this gate never checked.
"""

from __future__ import annotations

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_parity import (
    NOT_COMPARABLE_LENGTH,
    ParityReport,
    VectorResult,
    NOT_COMPARABLE_SCOPE,
    ProbeParityEngine,
)
from millm.services.probe_runtime import ArmedProbe


def make_probe(scope="all", threshold=1.0, d=4):
    return ArmedProbe(
        probe_id="pr_1",
        name="high-stakes",
        head=ProbeHead(weight=torch.ones(d), bias=0.0, layer=1),
        rule="mean",
        scope=scope,
        layer=1,
        rung=2,
        rung_language="detects on unseen tasks",
        threshold=threshold,
    )


def forward_with(value: float, d: int = 4):
    """A fake forward pass: every token's activation is `value` across d dims.

    With weight=ones and bias=0 that gives a per-token score of `value * d`.
    """

    def _forward(input_ids: torch.Tensor, context):
        n = input_ids.shape[1]
        context.observe(1, torch.full((1, n, d), value))

    return _forward


def definition_with(token_ids, token_scores, score, **over):
    doc = {
        "test_vectors": {
            "tolerance": 0.001,
            "vectors": [
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "token_ids": token_ids,
                    "token_scores": token_scores,
                    "score": score,
                }
            ],
        }
    }
    doc["test_vectors"].update(over)
    return doc


class TestPassing:
    def test_exact_reproduction_passes(self):
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(),
            definition_with([5, 6, 7], [4.0, 4.0, 4.0], 4.0),
            tolerance=0.001,
        )
        assert report.passed is True
        assert report.max_abs_diff == pytest.approx(0.0)
        assert report.worst_vector == 0

    def test_a_difference_inside_tolerance_passes(self):
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(),
            definition_with([5, 6], [4.0005, 4.0005], 4.0005),
            tolerance=0.001,
        )
        assert report.passed is True
        assert 0 < report.max_abs_diff <= 0.001


class TestFailing:
    def test_a_difference_beyond_tolerance_fails_and_names_the_vector(self):
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(),
            definition_with([5, 6], [9.0, 9.0], 9.0),
            tolerance=0.001,
        )
        assert report.passed is False
        assert report.max_abs_diff == pytest.approx(5.0)
        assert report.worst_vector == 0

    def test_the_report_carries_what_an_operator_needs(self):
        engine = ProbeParityEngine(forward_with(1.0))
        details = engine.run(
            make_probe(), definition_with([5], [9.0], 9.0), tolerance=0.001
        ).as_details()
        assert details["passed"] is False
        assert details["tolerance"] == 0.001
        assert details["max_abs_diff"] == pytest.approx(5.0)
        assert details["vector_index"] == 0

    def test_no_vectors_fails_rather_than_vacuously_passing(self):
        """⚠ An empty check is not a passed check.

        A definition with no vectors could otherwise arm having proven nothing.
        """
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(make_probe(), {"test_vectors": {"vectors": []}}, tolerance=0.001)
        assert report.passed is False
        assert "no test vectors" in report.error

    def test_a_missing_test_vectors_block_fails(self):
        engine = ProbeParityEngine(forward_with(1.0))
        assert engine.run(make_probe(), {}, tolerance=0.001).passed is False

    def test_an_empty_report_with_no_error_still_fails(self):
        """⚠ ADDED AFTER A MUTATION SURVIVED. `run()` always sets `error` when there are no
        vectors, so the `not self.vectors` clause is unreachable through `run()` and deleting it
        changed nothing there.

        It is still the clause that matters: a report constructed any other way — a future caller,
        a deserialised record — must not read as passed just because nobody recorded an error.
        An empty check is not a passed check.
        """
        assert ParityReport(tolerance=0.001).passed is False

    def test_a_length_mismatch_is_a_SHAPE_failure_not_a_tolerance_one(self):
        """Saying "off by 3.2" about tensors of different lengths is a number with no meaning."""
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(),
            definition_with([5, 6, 7], [4.0, 4.0], 4.0),
            tolerance=0.001,
        )
        assert report.passed is False
        assert NOT_COMPARABLE_LENGTH in report.vectors[0].reason
        assert report.vectors[0].max_abs_diff is None

    def test_a_vector_that_cannot_be_scored_fails(self):
        def refuse(input_ids, context):
            context.mark_not_scored("batched_request")

        report = ProbeParityEngine(refuse).run(
            make_probe(), definition_with([5], [4.0], 4.0), tolerance=0.001
        )
        assert report.passed is False
        assert report.vectors[0].reason == "batched_request"


class TestScopeReproducibility:
    @pytest.mark.parametrize("scope", ["prompt", "response"])
    def test_an_unreproducible_scope_FAILS_rather_than_passing(self, scope):
        """⚠ miStudio recorded these under its narrower internal role mask, and the contract
        carries no record of which positions those were.

        Passing would mean arming a probe this gate never actually checked.
        """
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(scope=scope), definition_with([5, 6], [4.0, 4.0], 4.0), tolerance=0.001
        )
        assert report.passed is False
        assert report.vectors[0].reason == NOT_COMPARABLE_SCOPE
        assert report.vectors[0].comparable is False

    def test_ONE_uncomparable_vector_fails_the_whole_report(self):
        """⚠ ADDED AFTER A MUTATION SURVIVED. My first test used a single vector, so removing the
        `comparable` check changed nothing: with one uncomparable vector `max_abs_diff` is None and
        the report failed anyway, by a different route.

        The check only bites when SOME vectors compare cleanly and others cannot — which is the
        realistic shape, and the one where a missing check would report a confident pass over a
        probe that was only partly verified.
        """
        report = ParityReport(tolerance=0.001)
        report.vectors = [
            VectorResult(index=0, max_abs_diff=0.0, combined_diff=0.0),
            VectorResult(index=1, comparable=False, reason=NOT_COMPARABLE_SCOPE),
        ]
        assert report.max_abs_diff == pytest.approx(0.0), "the comparable vector passed cleanly"
        assert report.passed is False, "a partly-verified probe must not report as verified"

    def test_scope_all_is_checked_normally(self):
        engine = ProbeParityEngine(forward_with(1.0))
        assert engine.run(
            make_probe(scope="all"), definition_with([5], [4.0], 4.0), tolerance=0.001
        ).passed is True


class TestTokenizationDrift:
    class Tok:
        def __init__(self, ids):
            self.ids = ids

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            return self.ids

    def test_drift_is_reported_but_does_NOT_fail_parity(self):
        """⚠ The defect this avoids: miStudio measured re-rendered `messages` missing by up to
        1.153 while the recorded ids reproduced exactly.

        A parity check built on `messages` would tell a correct consumer it was wrong on every
        vector — worse than no check, because it is believed the first time.
        """
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(),
            definition_with([5, 6], [4.0, 4.0], 4.0),
            tolerance=0.001,
            tokenizer=self.Tok([99, 99, 99]),  # renders to DIFFERENT ids
        )
        assert report.passed is True, "tokenization drift must not fail parity"
        assert report.tokenization_drift["messages_reproduce_token_ids"] == 0
        assert report.tokenization_drift["mismatched_vectors"] == [0]

    def test_agreement_is_reported_too(self):
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(),
            definition_with([5, 6], [4.0, 4.0], 4.0),
            tolerance=0.001,
            tokenizer=self.Tok([5, 6]),
        )
        assert report.tokenization_drift["messages_reproduce_token_ids"] == 1
        assert report.tokenization_drift["mismatched_vectors"] == []

    def test_a_template_that_raises_counts_as_drift_not_a_crash(self):
        class Boom:
            def apply_chat_template(self, *a, **k):
                raise RuntimeError("nope")

        report = ProbeParityEngine(forward_with(1.0)).run(
            make_probe(), definition_with([5], [4.0], 4.0), tolerance=0.001, tokenizer=Boom()
        )
        assert report.passed is True
        assert report.tokenization_drift["mismatched_vectors"] == [0]

    def test_no_tokenizer_means_no_drift_section(self):
        report = ProbeParityEngine(forward_with(1.0)).run(
            make_probe(), definition_with([5], [4.0], 4.0), tolerance=0.001
        )
        assert report.tokenization_drift == {}


class TestTheGateIsTheCombinedScore:
    """The 2026-09-27 decision, pinned.

    ⚠ Every one of the seventeen tests above passed unchanged when the gate moved from the
    per-token trace to the combined score — so none of them was asserting which one decided.
    A change to what a safety gate MEASURES that no test notices is the thing to fix first.
    """

    @staticmethod
    def _report(pairs, tolerance=0.05):
        """`pairs` is [(per_token_diff, combined_diff), ...]."""
        report = ParityReport(tolerance=tolerance)
        for i, (pt, cb) in enumerate(pairs):
            report.vectors.append(
                VectorResult(
                    index=i, max_abs_diff=pt, combined_diff=cb,
                    expected_score=1.0, actual_score=1.0 + cb, comparable=True,
                )
            )
        return report

    def test_a_huge_per_token_divergence_with_a_small_score_PASSES(self):
        """The production case: fp16 producer, bf16 consumer.

        Per-token 6.875 against a recorded 0.05, combined 0.098 — the real numbers measured on
        the node. This must pass, because the alternative is a gate no independent
        implementation can clear.
        """
        report = self._report([(6.875, 0.098), (0.706, 0.0014), (0.958, 0.0168)])
        assert report.passed
        assert report.max_abs_diff == 6.875, "the per-token figure must still be REPORTED"
        assert report.as_details()["per_token_is_informational"] is True

    def test_a_small_per_token_divergence_with_a_large_score_FAILS(self):
        """The inverse, and the one that matters: the score is what the probe decides with."""
        assert not self._report([(0.001, 0.5)]).passed

    def test_the_uncentered_basis_defect_would_STILL_be_caught(self):
        """⚠ The gate must remain a real gate, not a formality.

        The uncentered-basis defect found the same day moved the combined median to 1.68 —
        seventeen times this tolerance. If a change to the gate ever lets these numbers through,
        it has stopped catching the class of defect it exists for.
        """
        assert not self._report([(58.26, 1.683), (71.20, 9.987)]).passed

    def test_the_floor_applies_when_the_document_asks_for_something_tighter(self):
        report = self._report([(6.875, 0.098)], tolerance=0.001)
        assert report.score_tolerance == 0.10
        assert report.passed

    def test_a_document_asking_for_something_LOOSER_is_honoured(self):
        """`max`, not a replacement: a definition that knows it needs slack gets it."""
        report = self._report([(1.0, 0.4)], tolerance=0.5)
        assert report.score_tolerance == 0.5
        assert report.passed

    def test_comparable_vectors_with_no_recorded_SCORE_are_not_a_pass(self):
        """Nothing was checked. Silence must not read as agreement."""
        report = ParityReport(tolerance=0.05)
        report.vectors.append(
            VectorResult(index=0, max_abs_diff=0.001, combined_diff=None, comparable=True)
        )
        assert not report.passed

    def test_an_incomparable_vector_still_fails_regardless_of_score(self):
        report = self._report([(0.0, 0.0)])
        report.vectors.append(VectorResult(index=1, comparable=False, reason="scope"))
        assert not report.passed

    def test_the_details_name_BOTH_tolerances(self):
        """A reader must be able to tell which number decided the verdict."""
        details = self._report([(6.875, 0.098)]).as_details()
        assert details["tolerance"] == 0.05
        assert details["score_tolerance"] == 0.10
        assert details["max_combined_diff"] == 0.098
