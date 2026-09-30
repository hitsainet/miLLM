"""One probe, several windows: the user's half and the model's half, read separately.

⚠ WHY THIS EXISTS, MEASURED ON THE LIVE SYSTEM 2026-09-30.

A probe scores every token in its window and takes the MEAN. The window was the whole request —
the person's prompt AND the model's reply. The reply is the model being calm and helpful, which
scores low, so a long answer drags the average down. Same sentence, only `max_tokens` varied:

    "my brother ... forced out of his flat"   8-token reply  +9.86 FIRES
                                              full reply     -0.32 silent

Nothing about the situation changed. The monitor's verdict depended on how much the model said.
The bigger the crisis the less it matters, so what gets lost is precisely the borderline case a
monitor exists for.

The fix is not a different threshold. It is to stop averaging two different questions together:
the prompt says something about the USER, the response says something about the MODEL.

⚠ `windows` IS NOT `scope`. `scope` is the probe's identity — what it was trained on, what its
threshold was cut under, and the only thing parity can verify. A probe whose CONTRACT scope is
`prompt` or `response` still cannot be armed here at all. `windows` is a reporting choice over the
same weights, and any window that is not the probe's own scope is marked `provisional`.
"""

from __future__ import annotations

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

D = 4


def probe(*, scope="all", windows=(), threshold=1.0, probe_id="pr_1"):
    return ArmedProbe(
        probe_id=probe_id,
        name=probe_id,
        head=ProbeHead(weight=torch.ones(D), bias=0.0, layer=1),
        rule="mean",
        scope=scope,
        layer=1,
        rung=2,
        rung_language="detects on unseen tasks",
        threshold=threshold,
        windows=windows,
    )


def run(ctx, *, prompt_tokens, generated, prompt_value=3.0, reply_value=-3.0):
    """Prefill of `prompt_tokens` then `generated` single-token decode passes.

    The prompt activations are deliberately HIGH and the reply's LOW, which is the real
    asymmetry: the person states a crisis, the model answers calmly.
    """
    ctx.set_prompt_length(prompt_tokens)
    ctx.observe(1, torch.full((1, prompt_tokens, D), prompt_value / D))
    for _ in range(generated):
        ctx.observe(1, torch.full((1, 1, D), reply_value / D))


def by_window(verdicts):
    return {v.window: v for v in verdicts}


class TestOneVerdictPerWindow:
    def test_three_windows_give_three_labelled_verdicts(self):
        ctx = ProbeRequestContext("r", [probe(windows=("all", "prompt", "response"))])
        run(ctx, prompt_tokens=8, generated=4)
        got = by_window(ctx.finish())
        assert set(got) == {"all", "prompt", "response"}
        assert got["prompt"].n_scored_tokens == 8
        assert got["response"].n_scored_tokens == 4
        assert got["all"].n_scored_tokens == 12

    def test_no_windows_is_the_old_behaviour_exactly(self):
        """A probe armed without a window choice must be unchanged — one verdict, its own scope."""
        ctx = ProbeRequestContext("r", [probe()])
        run(ctx, prompt_tokens=8, generated=4)
        verdicts = ctx.finish()
        assert len(verdicts) == 1
        assert verdicts[0].window == "all"
        assert verdicts[0].provisional is False

    def test_the_windows_disagree_which_is_the_entire_point(self):
        ctx = ProbeRequestContext("r", [probe(windows=("all", "prompt", "response"))])
        run(ctx, prompt_tokens=8, generated=4)
        got = by_window(ctx.finish())
        assert got["prompt"].score > got["all"].score > got["response"].score, (
            "the three windows returned the same reading, so nothing was actually re-masked"
        )


class TestThePromptReadingDoesNotDependOnHowMuchTheModelSaid:
    """⚠ THE ACCEPTANCE CRITERION. This is the defect, stated as a property.

    Both halves are required. Without the second test the first passes against an implementation
    that ignores generated tokens everywhere — including in the `all` window, which would mean the
    dilution is not being exercised and the first test proved nothing.
    """

    def test_identical_under_a_short_and_a_long_reply(self):
        short = ProbeRequestContext("s", [probe(windows=("prompt",))])
        run(short, prompt_tokens=8, generated=2)
        long = ProbeRequestContext("l", [probe(windows=("prompt",))])
        run(long, prompt_tokens=8, generated=200)

        a = by_window(short.finish())["prompt"]
        b = by_window(long.finish())["prompt"]
        assert a.score == b.score, (
            f"the prompt reading moved from {a.score} to {b.score} because the model talked "
            f"longer — which is the defect this feature exists to remove"
        )
        assert a.n_scored_tokens == b.n_scored_tokens == 8
        assert a.fires == b.fires

    def test_and_the_ALL_window_still_moves(self):
        short = ProbeRequestContext("s", [probe(windows=("all",))])
        run(short, prompt_tokens=8, generated=2)
        long = ProbeRequestContext("l", [probe(windows=("all",))])
        run(long, prompt_tokens=8, generated=200)
        assert by_window(short.finish())["all"].score != by_window(long.finish())["all"].score, (
            "the whole-request window did not move with reply length, so the fixture cannot "
            "exhibit dilution and the test above passed for the wrong reason"
        )

    def test_the_dilution_has_the_sign_we_measured(self):
        """On real traffic the long reply always scored LOWER. Pin the direction, not just change."""
        short = ProbeRequestContext("s", [probe(windows=("all",))])
        run(short, prompt_tokens=8, generated=2)
        long = ProbeRequestContext("l", [probe(windows=("all",))])
        run(long, prompt_tokens=8, generated=200)
        assert by_window(long.finish())["all"].score < by_window(short.finish())["all"].score


class TestProvisionalIsMarkedNotInferred:
    def test_the_probes_own_scope_is_calibrated(self):
        ctx = ProbeRequestContext("r", [probe(scope="all", windows=("all", "prompt"))])
        run(ctx, prompt_tokens=6, generated=3)
        got = by_window(ctx.finish())
        assert got["all"].provisional is False

    def test_every_other_window_is_provisional(self):
        ctx = ProbeRequestContext("r", [probe(scope="all", windows=("all", "prompt", "response"))])
        run(ctx, prompt_tokens=6, generated=3)
        got = by_window(ctx.finish())
        assert got["prompt"].provisional is True
        assert got["response"].provisional is True

    def test_a_provisional_window_STILL_FIRES(self):
        """Operator decision 2026-09-30: alerts now, flagged, over silence. The flag is the
        condition that decision was made under, so it is asserted beside the firing."""
        ctx = ProbeRequestContext("r", [probe(scope="all", windows=("prompt",), threshold=1.0)])
        run(ctx, prompt_tokens=6, generated=3, prompt_value=9.0)
        got = by_window(ctx.finish())["prompt"]
        assert got.fires is True
        assert got.provisional is True

    def test_no_threshold_still_means_no_verdict(self):
        ctx = ProbeRequestContext("r", [probe(windows=("prompt",), threshold=None)])
        run(ctx, prompt_tokens=6, generated=3)
        assert by_window(ctx.finish())["prompt"].fires is None


class TestTheAwkwardWindows:
    def test_a_response_window_with_nothing_generated_says_so(self):
        """Not a score of zero, and not a crash: the probe never looked."""
        ctx = ProbeRequestContext("r", [probe(windows=("prompt", "response"))])
        run(ctx, prompt_tokens=8, generated=0)
        got = by_window(ctx.finish())
        assert got["response"].scored is False
        assert got["response"].not_scored_reason == "no_scored_tokens"
        assert got["prompt"].scored is True, "a dead response window must not silence the prompt"

    def test_an_unknown_boundary_costs_ONE_window_not_the_request(self):
        """⚠ THE POISONING BUG, FIXED HERE. `window_for` returning None used to call
        `mark_not_scored` and abort the WHOLE request, so one probe's unresolvable window
        silenced every other probe's verdict for that request."""
        ctx = ProbeRequestContext("r", [probe(windows=("all", "prompt"))])
        # No `set_prompt_length` — the boundary is unknown.
        ctx.observe(1, torch.full((1, 8, D), 0.5))
        got = by_window(ctx.finish())
        assert got["prompt"].scored is False
        assert got["prompt"].not_scored_reason == "prompt_boundary_unknown"
        assert got["all"].scored is True, (
            "an unresolvable prompt window silenced the all window too — one missing verdict "
            "must not become a blind request"
        )

    def test_two_probes_two_window_sets(self):
        a = probe(probe_id="pr_a", windows=("prompt",))
        b = probe(probe_id="pr_b", windows=("all", "response"))
        ctx = ProbeRequestContext("r", [a, b])
        run(ctx, prompt_tokens=6, generated=4)
        verdicts = ctx.finish()
        assert {(v.probe_id, v.window) for v in verdicts} == {
            ("pr_a", "prompt"), ("pr_b", "all"), ("pr_b", "response"),
        }


class TestTheExtraWindowsAreCheap:
    def test_scoring_three_windows_does_not_rescore_the_tokens(self):
        """The per-token scores are computed in `observe` and stored UNMASKED, so extra windows
        cost a mask and a combine. If a window were re-running the head, `observe`'s recorded
        overhead would have to grow with the window count — it must not."""
        one = ProbeRequestContext("a", [probe(windows=("all",))])
        run(one, prompt_tokens=64, generated=16)
        before_one = one.overhead_ms
        one.finish()

        three = ProbeRequestContext("b", [probe(windows=("all", "prompt", "response"))])
        run(three, prompt_tokens=64, generated=16)
        before_three = three.overhead_ms
        three.finish()

        # The OBSERVE half — the part that touches the model's activations — is identical work.
        assert before_three == pytest.approx(before_one, rel=2.0), (
            "observing cost materially more with three windows, which means a window is doing "
            "work during the forward pass rather than at finish()"
        )
