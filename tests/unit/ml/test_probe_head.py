"""The probe readout, and the drift this module exists to prevent.

`millm/ml/probe_head.py` is a SECOND implementation of arithmetic miStudio owns. The parity gate
catches divergence at arm time, but only for the vectors a definition happens to carry. These tests
catch it here, on every rule, including the cases a test-vector sample is unlikely to contain:
left padding, a degenerate standardisation channel, a sequence shorter than the rolling window.

The load-bearing test is `TestOnlineEqualsBatch`. A streaming verdict is produced by `OnlineRule`
and a parity verdict by `combine`; if those two disagree, a probe passes parity at arm time and
then reports something else on live traffic — which is worse than failing, because it is invisible.
"""

from __future__ import annotations

import math

import pytest
import torch

from millm.ml.probe_head import (
    DEFAULT_TAU,
    DEFAULT_WINDOW,
    RULES,
    STREAMABLE,
    OnlineRule,
    ProbeHead,
    combine,
    is_streamable,
    rule_parameters,
)


def head(d: int = 4, *, query: bool = False) -> ProbeHead:
    torch.manual_seed(0)
    return ProbeHead(
        weight=torch.randn(d),
        bias=0.25,
        mean=torch.randn(d),
        std=torch.rand(d) + 0.5,
        attention_query=torch.randn(d) if query else None,
        layer=11,
    )


class TestOnlineEqualsBatch:
    """The streaming form must equal the batch form, on every rule that has one."""

    @pytest.mark.parametrize("rule", sorted(STREAMABLE))
    def test_they_agree_on_a_realistic_sequence(self, rule):
        torch.manual_seed(7)
        scores = torch.randn(23) * 3.0
        logits = torch.randn(23) * 2.0 if rule == "attention" else None

        batch = combine(
            rule,
            scores.unsqueeze(0),
            mask=torch.ones(1, 23, dtype=torch.bool),
            attention_logits=None if logits is None else logits.unsqueeze(0),
            tau=DEFAULT_TAU,
            window=DEFAULT_WINDOW,
        ).item()

        online = OnlineRule(rule, tau=DEFAULT_TAU, window=DEFAULT_WINDOW)
        for i, s in enumerate(scores.tolist()):
            online.update(s, attention_logit=None if logits is None else logits[i].item())

        assert online.value == pytest.approx(batch, abs=1e-6), (
            f"{rule}: online {online.value} != batch {batch}"
        )

    @pytest.mark.parametrize("rule", ["softmax", "attention"])
    def test_they_agree_when_the_largest_logit_arrives_last(self, rule):
        """The running-max rebase is the part most likely to be wrong.

        A naive sum of exponentials agrees with the batch form until a late token carries the
        largest logit and forces a rescale, so a sequence whose maximum is at the end is the
        case that separates the two.
        """
        scores = torch.tensor([0.5, -1.0, 2.0, 0.1, 40.0])
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 40.0]) if rule == "attention" else None

        batch = combine(
            rule,
            scores.unsqueeze(0),
            mask=torch.ones(1, 5, dtype=torch.bool),
            attention_logits=None if logits is None else logits.unsqueeze(0),
        ).item()
        online = OnlineRule(rule)
        for i, s in enumerate(scores.tolist()):
            online.update(s, attention_logit=None if logits is None else logits[i].item())

        assert math.isfinite(online.value)
        assert online.value == pytest.approx(batch, abs=1e-6)

    def test_last_refuses_to_stream_rather_than_guessing(self):
        """`last` is undefined until the sequence ends; a partial answer would be a guess."""
        assert is_streamable("last") is False
        with pytest.raises(ValueError, match="not defined until the sequence ends"):
            OnlineRule("last")

    def test_an_unknown_rule_raises_rather_than_reading_as_not_streamable(self):
        with pytest.raises(ValueError, match="unknown combining rule"):
            is_streamable("meen")


class TestTheRulesAreDistinctDetectors:
    def test_attention_refuses_without_its_query(self):
        with pytest.raises(ValueError, match="attention_logits"):
            combine("attention", torch.randn(1, 5), mask=torch.ones(1, 5, dtype=torch.bool))

    def test_a_head_without_a_query_refuses_to_produce_logits(self):
        with pytest.raises(ValueError, match="no attention_query"):
            head(query=False).attention_logits(torch.randn(1, 5, 4))

    def test_attention_is_not_softmax_under_another_name(self):
        """Passing the scores as the logits is the silent failure this guards.

        If someone wires `attention_logits=scores`, `attention` becomes `softmax` at tau=1 — a
        different detector reporting under the probe's name. These must not agree on real input.
        """
        torch.manual_seed(3)
        scores = torch.randn(1, 12)
        logits = torch.randn(1, 12)
        mask = torch.ones(1, 12, dtype=torch.bool)

        real = combine("attention", scores, mask=mask, attention_logits=logits).item()
        as_softmax = combine("softmax", scores, mask=mask, tau=1.0).item()
        assert abs(real - as_softmax) > 1e-3

    def test_softmax_temperature_must_be_positive(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            combine("softmax", torch.randn(1, 4), mask=torch.ones(1, 4, dtype=torch.bool), tau=0.0)

    def test_every_rule_is_reachable(self):
        """A rule in the contract that `combine` cannot compute is an unservable definition."""
        torch.manual_seed(11)
        scores, mask = torch.randn(1, 9), torch.ones(1, 9, dtype=torch.bool)
        logits = torch.randn(1, 9)
        for rule in RULES:
            value = combine(rule, scores, mask=mask, attention_logits=logits)
            assert torch.isfinite(value).all(), rule


class TestMaskingIsNotOptional:
    def test_last_is_correct_under_LEFT_padding(self):
        """Counting real tokens and taking count-1 is only right under right padding.

        With mask [0,0,1,1] that arithmetic returns index 1 — a pad position — while the answer
        is index 3. miStudio shipped the counting version with a docstring claiming left-padding
        correctness.
        """
        scores = torch.tensor([[99.0, 98.0, 1.0, 7.0]])
        mask = torch.tensor([[False, False, True, True]])
        assert combine("last", scores, mask=mask).item() == pytest.approx(7.0)

    def test_masked_positions_do_not_enter_a_mean(self):
        scores = torch.tensor([[1.0, 1.0, 1000.0]])
        mask = torch.tensor([[True, True, False]])
        assert combine("mean", scores, mask=mask).item() == pytest.approx(1.0)

    def test_masked_positions_do_not_win_a_max(self):
        scores = torch.tensor([[1.0, 1.0, 1000.0]])
        mask = torch.tensor([[True, True, False]])
        assert combine("max", scores, mask=mask).item() == pytest.approx(1.0)

    def test_a_row_with_nothing_scored_is_refused_not_returned_as_zero(self):
        with pytest.raises(ValueError, match="at least one scored position"):
            combine("mean", torch.randn(1, 4), mask=torch.zeros(1, 4, dtype=torch.bool))

    def test_a_mismatched_mask_is_refused(self):
        with pytest.raises(ValueError, match="does not match scores"):
            combine("mean", torch.randn(1, 4), mask=torch.ones(1, 5, dtype=torch.bool))


class TestStandardisation:
    def test_a_degenerate_channel_is_zeroed_not_clamped(self):
        """⚠ Clamping a 0 std to eps turns a 0.001 drift into 1000.

        The contract refuses a `norm_std` of 0 for this reason, but a std that is merely tiny
        arrives through validation, so the head must handle it rather than divide by a floor.
        """
        h = ProbeHead(
            weight=torch.tensor([1.0, 1.0]),
            mean=torch.tensor([0.0, 0.0]),
            std=torch.tensor([1.0, 0.0]),
        )
        out = h.standardise(torch.tensor([[2.0, 0.001]]))
        assert out[0, 0].item() == pytest.approx(2.0)
        assert out[0, 1].item() == 0.0, "a degenerate channel must contribute nothing"

    def test_the_wrong_width_is_refused_with_both_widths_named(self):
        with pytest.raises(ValueError, match="d=7 but this probe is d=4"):
            head().token_scores(torch.randn(1, 3, 7))

    def test_normalisation_vectors_must_match_the_weights(self):
        with pytest.raises(ValueError, match="normalisation must match its weights"):
            ProbeHead(weight=torch.randn(4), std=torch.randn(5))


class TestRollingMeanMax:
    def test_a_sequence_shorter_than_the_window_equals_mean(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        assert combine("rolling_mean_max", scores, mask=mask, window=16).item() == pytest.approx(2.0)

    def test_it_finds_a_sustained_run_over_an_isolated_spike(self):
        """This is the whole reason the rule exists: `max` fires on one token, this needs a run.

        ⚠ My first fixture here was [0,0,9,0,3,3,3] expecting 3.0, and the real answer is 4.0 —
        the window [9,0,3] averages 4. The fixture did not separate a spike from a run at all;
        the code was right and the expectation was wrong. This one does separate them: the
        isolated 9 can only ever reach 3.0 in any window, while the run of 5s reaches 5.0.
        """
        scores = torch.tensor([[0.0, 0.0, 9.0, 0.0, 0.0, 5.0, 5.0, 5.0]])
        mask = torch.ones(1, 8, dtype=torch.bool)

        assert combine("max", scores, mask=mask).item() == pytest.approx(9.0)
        assert combine("rolling_mean_max", scores, mask=mask, window=3).item() == pytest.approx(5.0)

    def test_window_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            combine("rolling_mean_max", torch.randn(1, 4), mask=torch.ones(1, 4, dtype=torch.bool), window=0)


class TestRuleParameters:
    def test_parameterised_rules_declare_what_must_travel(self):
        assert rule_parameters("softmax", tau=0.3) == {"tau": 0.3}
        assert rule_parameters("rolling_mean_max", window=8) == {"window": 8}

    def test_unparameterised_rules_declare_nothing(self):
        for rule in ("mean", "max", "last", "attention"):
            assert rule_parameters(rule) == {}
