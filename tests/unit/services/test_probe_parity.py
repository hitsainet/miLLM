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


class TestTheContractSaysScore:
    """⚠ The per-token gate was a MISREADING of the contract, not a stricter reading of it.

    `ProbeTestVectors.tolerance` is documented in `mistudio.probe-definition/v1` as *"An ABSOLUTE
    score tolerance"*, and the measurement behind its 0.05 is explicitly about scores — batch
    composition moving a SCORE by at most 5.78e-03. Nothing ever asked for per-token agreement.

    Asserted against miStudio's own source, so that if the producer ever redefines the field, this
    goes red rather than miLLM quietly continuing to mean something else by it.
    """

    @staticmethod
    def _contract_source() -> str:
        import os
        from pathlib import Path

        root = Path(
            os.environ.get("MISTUDIO_REPO", str(Path(__file__).resolve().parents[3].parent / "miStudio"))
        )
        path = root / "backend" / "src" / "schemas" / "probe_definition.py"
        if not path.exists():
            pytest.skip(f"miStudio checkout not found at {root} — set MISTUDIO_REPO")
        return path.read_text(encoding="utf-8")

    def test_the_producer_calls_it_a_SCORE_tolerance(self):
        source = self._contract_source()
        assert "ABSOLUTE score tolerance" in source, (
            "miStudio no longer documents `tolerance` as a score tolerance — re-read the "
            "contract before trusting this gate's interpretation of it"
        )

    def test_the_producer_anticipated_the_precision_difference(self):
        """The floor is not an invention; the contract names the effect it covers."""
        source = self._contract_source()
        assert "fp16 versus bf16" in source

# ── k-sparse JumpReLU: the positions no independent implementation can reproduce ──────────

D_MODEL_K, K_FEATURES, N_TOKENS = 32, 8, 40
#: The one feature whose threshold is placed ON a token's pre-activation. Not arbitrary: it is
#: the mechanism, made deterministic instead of waited for.
FLIP_FEATURE, HEAD_WEIGHTS = 2, [0.4, -0.7, 1.5, 0.9, -0.3, 0.6, -1.1, 0.2]


def _ksparse_parts():
    """A JumpReLU slice, the consumer's residual, and the producer's (1.5% apart).

    ⚠ **`b_dec` IS NON-ZERO, DELIBERATELY.** With `b_dec = 0` a centred encode and the
    uncentered defect of 2026-09-27 are the SAME function, so a fixture built that way passes
    against the very bug these tests exist to keep out. Every number below moves if the
    centring is dropped, which is what `test_the_uncentered_basis_defect_still_fails` checks.
    """
    from dataclasses import replace

    from millm.services.probe_sae_slice import (
        PRODUCER_PRECISION_GAP,
        SaeFeatureSlice,
    )

    g = torch.Generator().manual_seed(4)
    weight = torch.randn(D_MODEL_K, K_FEATURES, generator=g) / D_MODEL_K**0.5
    b_dec = torch.randn(D_MODEL_K, generator=g) * 0.6 + 0.4
    consumer = torch.randn(N_TOKENS, D_MODEL_K, generator=g) * 0.05 + 0.02
    producer = consumer * (
        1.0 + PRODUCER_PRECISION_GAP * torch.randn(consumer.shape, generator=g)
    )

    scaffold = SaeFeatureSlice(
        weight=weight.contiguous(),
        bias=torch.zeros(K_FEATURES),
        architecture="jumprelu",
        normalization_mode="constant_norm_rescale",
        thresholds=torch.zeros(K_FEATURES),
        feature_indices=tuple(range(K_FEATURES)),
        decoder_bias=b_dec.contiguous(),
    )
    pre_c = scaffold.pre_activations(consumer)
    pre_p = scaffold.pre_activations(producer)
    theta = torch.quantile(pre_c, 0.95, dim=0).clamp_min(1e-4)
    token = int(pre_c[:, FLIP_FEATURE].argmax())
    # BETWEEN the two precisions' pre-activations: active for one side, inactive for the other.
    theta[FLIP_FEATURE] = (pre_c[token, FLIP_FEATURE] + pre_p[token, FLIP_FEATURE]) / 2.0
    slice_ = replace(scaffold, thresholds=theta.contiguous())

    features = slice_.encode(consumer)
    head = ProbeHead(
        weight=torch.tensor(HEAD_WEIGHTS),
        bias=-0.2,
        mean=features.mean(0),
        std=features.std(0).clamp_min(0.2),
        layer=1,
    )
    return slice_, head, consumer, producer


def _ksparse_case(*, producer_slice=None):
    """`(probe, definition, forward)` for one k-sparse vector.

    `producer_slice` lets a test have miStudio's recorded scores come from a DIFFERENT encode —
    which is how a real basis defect is simulated.
    """
    slice_, head, consumer, producer = _ksparse_parts()
    recording = producer_slice or slice_
    expected = head.token_scores(recording.encode(producer if producer_slice is None else consumer))
    probe = ArmedProbe(
        probe_id="pr_sae",
        name="k-sparse",
        head=head,
        rule="mean",
        scope="all",
        layer=1,
        rung=2,
        rung_language="detects on unseen tasks",
        threshold=1.0,
        encoder=slice_,
    )
    definition = definition_with(
        list(range(100, 100 + N_TOKENS)),
        [float(v) for v in expected],
        float(expected.mean()),
    )

    def forward(input_ids: torch.Tensor, context):
        context.observe(1, consumer.unsqueeze(0))

    return probe, definition, forward


class TestKSparseJumpReLUPositions:
    """A JumpReLU basis has positions whose gate is not reproducible, and they must be set
    aside rather than tolerated.

    The producer scores in float16, this build serves bfloat16, and the residual they see
    differs by about 1.5% relative. A JumpReLU feature is `pre * H(pre - theta)`: for a feature
    sitting inside that band of its own threshold the two sides gate it differently, and the
    feature then moves between theta and 0 — which after the head's `norm_std` shifts THAT ONE
    token's score by tens of units. Diluted by the `mean` rule it lands in the same range as a
    real basis defect, so the combined score over the whole sequence cannot separate them.
    """

    def test_the_full_sequence_figure_would_have_failed(self):
        """The fixture is in the regime this change is about — asserted, not assumed.

        Without setting the unreproducible positions aside there is nothing to discuss: the
        gate refuses a build that is correct.
        """
        probe, definition, forward = _ksparse_case()
        report = ProbeParityEngine(forward).run(probe, definition, tolerance=0.001)
        assert report.max_combined_diff > report.score_tolerance, (
            "this fixture no longer exercises the gate-flip regime; the test below proves "
            "nothing until it does"
        )

    def test_a_correct_build_passes_on_the_reproducible_positions(self):
        probe, definition, forward = _ksparse_case()
        report = ProbeParityEngine(forward).run(probe, definition, tolerance=0.001)
        assert report.max_robust_combined_diff is not None
        assert report.max_robust_combined_diff < report.max_combined_diff / 10, (
            "setting the unreproducible positions aside must remove most of the divergence, "
            "or the divergence was not the gate"
        )
        assert report.passed is True

    def test_the_positions_set_aside_are_reported_not_hidden(self):
        probe, definition, forward = _ksparse_case()
        report = ProbeParityEngine(forward).run(probe, definition, tolerance=0.001)
        details = report.as_details()
        assert 0 < details["at_risk_tokens"] < details["scored_tokens"]
        assert details["max_combined_diff"] > details["max_gated_diff"]
        assert details["vectors"][0]["n_tokens"] == N_TOKENS

    def test_the_uncentered_basis_defect_still_fails(self):
        """The defect of 2026-09-27, over the SAME positions. It must not be admitted.

        `b_dec` is non-zero in the fixture, so dropping the centring is a real change. On the
        full sequence this defect and a correct build are a factor of four apart; over the
        reproducible positions they are two orders of magnitude apart.
        """
        from dataclasses import replace

        slice_, _, _, _ = _ksparse_parts()
        probe, definition, forward = _ksparse_case(
            producer_slice=replace(slice_, decoder_bias=None)
        )
        report = ProbeParityEngine(forward).run(probe, definition, tolerance=0.001)
        assert report.max_robust_combined_diff > report.score_tolerance
        assert report.passed is False

    def test_a_vector_with_almost_nothing_reproducible_is_refused(self):
        """Not passed on the remainder. A comparison resting on a handful of positions says
        little about the probe, and a basis that sits on its own thresholds is one to refuse."""
        from dataclasses import replace

        from millm.services.probe_parity import NOT_COMPARABLE_UNSTABLE

        from millm.services.probe_sae_slice import PRODUCER_PRECISION_GAP

        slice_, head, _, _ = _ksparse_parts()
        # A near-constant residual: every position has the SAME pre-activations, so a threshold
        # placed on them is on the edge for all of them at once.
        g = torch.Generator().manual_seed(12)
        consumer = torch.full((N_TOKENS, D_MODEL_K), 0.05) + 1e-4 * torch.randn(
            (N_TOKENS, D_MODEL_K), generator=g
        )
        producer = consumer * (
            1.0 + PRODUCER_PRECISION_GAP * torch.randn(consumer.shape, generator=g)
        )
        everywhere = replace(
            slice_, thresholds=slice_.pre_activations(consumer).mean(0).contiguous()
        )
        expected = head.token_scores(everywhere.encode(producer))
        probe = ArmedProbe(
            probe_id="pr_sae",
            name="k-sparse",
            head=head,
            rule="mean",
            scope="all",
            layer=1,
            rung=2,
            rung_language="detects on unseen tasks",
            encoder=everywhere,
        )

        def forward(input_ids, context):
            context.observe(1, consumer.unsqueeze(0))

        report = ProbeParityEngine(forward).run(
            probe,
            definition_with(
                list(range(100, 100 + N_TOKENS)),
                [float(v) for v in expected],
                float(expected.mean()),
            ),
            tolerance=0.001,
        )
        assert report.passed is False
        assert NOT_COMPARABLE_UNSTABLE in (report.vectors[0].reason or "")

    def test_a_dense_probe_is_unchanged(self):
        """No encoder, so no position is set aside and the gate is exactly what it was."""
        engine = ProbeParityEngine(forward_with(1.0))
        report = engine.run(
            make_probe(), definition_with([5, 6, 7], [4.0, 4.0, 4.0], 4.0), tolerance=0.001
        )
        assert report.passed is True
        assert report.at_risk_tokens == 0
        assert report.max_robust_combined_diff is None
        assert report.max_gated_diff == report.max_combined_diff

    def test_the_robust_figure_goes_through_the_PROBE_S_OWN_RULE(self):
        """⚠ Not an average. Dropping a position from a `max` probe is not the same operation as
        dropping it from a `mean`, and hand-averaging here would compare a statistic the probe
        does not compute — the `attention`-as-`softmax` failure in another costume."""
        from millm.ml.probe_head import combine

        slice_, head, consumer, producer = _ksparse_parts()
        expected = head.token_scores(slice_.encode(producer))
        actual = head.token_scores(slice_.encode(consumer))
        probe = ArmedProbe(
            probe_id="pr_max",
            name="k-sparse-max",
            head=head,
            rule="max",
            scope="all",
            layer=1,
            rung=2,
            rung_language="detects on unseen tasks",
            encoder=slice_,
        )

        def forward(input_ids, context):
            context.observe(1, consumer.unsqueeze(0))

        report = ProbeParityEngine(forward).run(
            probe,
            definition_with(
                list(range(100, 100 + N_TOKENS)),
                [float(v) for v in expected],
                float(expected.max()),
            ),
            tolerance=0.001,
        )
        mask = (~slice_.flip_risk(consumer)).unsqueeze(0)
        by_rule = abs(
            float(combine("max", actual.unsqueeze(0), mask=mask).item())
            - float(combine("max", expected.unsqueeze(0), mask=mask).item())
        )
        by_average = abs(
            float(combine("mean", actual.unsqueeze(0), mask=mask).item())
            - float(combine("mean", expected.unsqueeze(0), mask=mask).item())
        )
        assert by_rule != pytest.approx(by_average), (
            "this fixture cannot tell the two apart, so it cannot pin the rule"
        )
        assert report.vectors[0].robust_combined_diff == pytest.approx(by_rule, abs=1e-6)
