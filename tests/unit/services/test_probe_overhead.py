"""SC-4: what the probe costs per request, and what a CPU test can honestly say about it.

⚠ **THIS FILE DOES NOT ASSERT THE 5 ms BUDGET, AND THE REASON MATTERS.**

The first version did, and it failed at **37 ms** — on the reasoning that CPU arithmetic is a
floor under the GPU figure. That reasoning is backwards: for an elementwise pass and a matvec
over 8.4M elements the CPU is *slower* than the card by orders of magnitude, so a CPU timing
bounds the production number from neither side. Asserting 5 ms here would either fail forever
on correct code or, if loosened to pass, become a number nobody can interpret.

The failure was still worth having, because it exposed a real defect:
`ProbeRequestContext.observe` copied the WHOLE residual to the host and scored it there —
33.6 MB per forward pass at 4k x 2048, then single-threaded float32 arithmetic. That is now
scored on the tensor's own device with only the (T,) result crossing. Both facts are pinned
below, because a future edit that reintroduces either is invisible in a suite that only
checks the numbers.

**The absolute 5 ms figure is a hardware measurement** (task 10.4, second half), recorded in
`0xcc/reviews/`. What is automated here is everything a CPU can decide:

* the score path does not copy the activations to the host (the defect, by assertion);
* cost is linear in probe count, so `PROBE_MAX_ARMED = 8` is affordable;
* the two tempting optimisations are pinned as REJECTED, with their measured error, so nobody
  re-derives them and ships a probe that scores differently from what miStudio measured.
"""

from __future__ import annotations

import time

import pytest
import torch

from millm.core.config import settings
from millm.ml.probe_head import ProbeHead, combine

#: The criterion's own shapes (FR-24.14, SC-4).
N_TOKENS = 4096
D_MODEL = 2048
N_PROBES = 2

WARMUP_RUNS = 2
MEASURED_RUNS = 5


def _head(d_model: int = D_MODEL) -> ProbeHead:
    """A head with standardisation ON, because that is the costly shape.

    `mean`/`std` are optional and omitting them skips a subtract and a divide over the whole
    (tokens x d_model) tensor, so a measurement without them measures a path no real probe
    takes — every exported definition carries both.
    """
    torch.manual_seed(0)
    return ProbeHead(
        weight=torch.randn(d_model, dtype=torch.float32),
        bias=0.25,
        mean=torch.zeros(d_model),
        std=torch.ones(d_model),
    )


def _score_once(heads: list[ProbeHead], hidden: torch.Tensor, rule: str) -> list:
    """One request's worth of probe work, through the REAL functions.

    `combine()` and not the online form: the live path calls `combine()` precisely so that
    parity verifies the implementation that serves, and timing the other one would measure
    code production never runs.
    """
    return [combine(rule, head.token_scores(hidden)) for head in heads]


def _median_ms(heads: list[ProbeHead], hidden: torch.Tensor, rule: str = "mean") -> float:
    """The MEDIAN of several runs. One wall-clock sample measures the machine.

    miStudio shipped a scaling test that flaked on CI and passed against the quadratic
    algorithm it was written to catch, for exactly this reason.
    """
    for _ in range(WARMUP_RUNS):
        _score_once(heads, hidden, rule)
    samples = []
    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        _score_once(heads, hidden, rule)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


@pytest.fixture(scope="module")
def hidden() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, N_TOKENS, D_MODEL, dtype=torch.float16)


class TestTheShapesAreTheCriterions:
    def test_the_constants_match_the_stated_criterion(self):
        """A performance test at the wrong size is a number about nothing."""
        assert (N_TOKENS, D_MODEL, N_PROBES) == (4096, 2048, 2)

    def test_the_budget_setting_exists_and_is_the_stated_one(self):
        """The threshold an operator sees in `GET /api/probes/status` is this number.

        Not asserted as a timing here — see the module docstring — but pinned, because the
        hardware measurement is taken against it and a silent change would invalidate the
        recorded result.
        """
        assert settings.PROBE_MAX_OVERHEAD_MS == 5.0


class TestTheActivationsAreNotCopiedToTheHost:
    """⚠ THE DEFECT, PINNED. This is the assertion that would have caught it.

    `observe` copied `hidden[0].detach().to(torch.float32).cpu()` — the entire residual,
    upcast, to the host — and scored it there. The comment above it said "THE ONE
    DEVICE-TO-HOST COPY", which was true and beside the point: one copy of 33.6 MB is the
    cost. No timing assertion caught it because there was no timing assertion, and no
    correctness test could, because the scores were right.
    """

    def test_observe_does_not_move_the_activations(self):
        """Asserted by RECORDING the calls made on the tensor, not by reading the source.

        A source scrape for `.cpu()` fails open on any rewrite, and this estate has shipped
        that mistake in three separate arcs.
        """
        from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

        moved: list[str] = []

        class WatchedTensor(torch.Tensor):
            """A tensor that records `.cpu()` / `.to(...)` calls on ITSELF."""

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, False)

            def cpu(self, *a, **k):  # type: ignore[override]
                moved.append("cpu")
                return super().cpu(*a, **k)

            def to(self, *a, **k):  # type: ignore[override]
                # Only a DEVICE move is the defect; a dtype cast stays on the card.
                for arg in a:
                    if isinstance(arg, (str, torch.device)):
                        moved.append(f"to:{arg}")
                if "device" in k and k["device"] is not None:
                    moved.append(f"to:{k['device']}")
                return super().to(*a, **k)

        head = _head(d_model=8)
        probe = ArmedProbe(
            probe_id="pr_1",
            name="t",
            head=head,
            rule="mean",
            scope="all",
            layer=1,
            rung=2,
            rung_language="detects on unseen tasks",
        )
        context = ProbeRequestContext("req", [probe])
        watched = WatchedTensor(torch.randn(1, 4, 8))
        context.observe(1, watched)

        assert not moved, (
            f"the score path moved the activations off their device ({moved}) — at 4k tokens "
            "and d_model 2048 that is a 33.6 MB copy per forward pass, which is the whole "
            "overhead budget"
        )
        # And it must actually have scored, or the assertion above passes vacuously.
        assert context.token_scores_for("pr_1"), "nothing was scored"

    def test_the_head_follows_the_activations_device(self):
        """The mechanism that makes the above possible, asserted directly."""
        head = _head(d_model=16)
        assert head.to_device(torch.device("cpu")) is head, (
            "to_device rebuilt the head for the device it was already on — that is a copy "
            "of every weight on every forward pass"
        )

    def test_the_sae_slice_follows_the_activations_device_too(self):
        """A k-sparse probe encodes before it scores, so its slice has the same rule.

        Moving `x` to the slice instead would undo the fix silently: the numbers would be
        right and the copy would be back.
        """
        from millm.services.probe_sae_slice import SaeFeatureSlice

        slice_ = SaeFeatureSlice(
            weight=torch.randn(8, 4),
            bias=torch.zeros(4),
            architecture="standard",
            feature_indices=(0, 1, 2, 3),
        )
        assert slice_.to_device(torch.device("cpu")) is slice_


class TestTheRejectedOptimisationsStayRejected:
    """⚠ Two faster forms exist. Both are wrong, and both look right.

    Recorded as tests rather than comments because a comment does not fail. Each carries its
    measured error against the 1e-3 parity tolerance.
    """

    def test_folding_the_standardisation_changes_the_score(self):
        """`z @ (w/std) - (mean/std).w` is algebraically identical and 2.6x faster.

        It is not used because it perturbs the score at float32 rounding, and
        `test_probe_head_matches_mistudio.py` requires BIT-EXACT agreement with miStudio's
        scorer — the property 033's acceptance recorded as "0.000e+00, all sixteen vectors".
        Exactness is a stronger guarantee than "within tolerance", and it is what makes a
        parity pass mean something.

        This test asserts the two forms DIFFER. If they ever agree bit-for-bit, the fold is
        safe and this test should be deleted in the same commit that adopts it.
        """
        torch.manual_seed(3)
        d = 512
        head = ProbeHead(
            weight=torch.randn(d),
            bias=0.25,
            mean=torch.randn(d) * 0.1,
            std=torch.rand(d) + 0.5,
        )
        z = torch.randn(64, d)
        exact = head.standardise(z) @ head.weight + head.bias
        w_eff = head.weight / head.std
        folded = z @ w_eff + (head.bias - float((head.mean / head.std * head.weight).sum()))
        diff = (exact - folded).abs().max().item()
        assert diff > 0.0, (
            "the folded form now agrees bit-for-bit — adopt it and delete this test"
        )
        assert diff < 1e-3, (
            f"the folded form differs by {diff:.3e}, outside the parity tolerance — the "
            "algebra is wrong, not merely inexact"
        )

    def test_an_fp16_matvec_would_break_parity(self):
        """The other tempting 4x, and the dangerous one.

        Scoring in the activations' native fp16 moves the result far OUTSIDE the 1e-3 parity
        tolerance, so every armed probe would fail parity — or, worse, pass it on a lucky
        vector set and then score differently from what miStudio measured, with no symptom.
        `token_scores` upcasts to float32 for this reason, and this test is why it must.
        """
        torch.manual_seed(4)
        d = 2048
        head = _head(d_model=d)
        z16 = torch.randn(256, d, dtype=torch.float16)

        exact = head.token_scores(z16)
        cheap = (z16 @ head.weight.to(torch.float16)).to(torch.float32) + head.bias
        diff = (exact - cheap).abs().max().item()
        assert diff > settings.PROBE_PARITY_TOLERANCE, (
            f"an fp16 matvec differs by only {diff:.3e}, inside the "
            f"{settings.PROBE_PARITY_TOLERANCE} tolerance — if that is genuinely true on "
            "this hardware, the upcast in token_scores can go, but measure it on the GPU "
            "first: this test ran on CPU"
        )


class TestScalingIsAffordable:
    def test_the_harness_measures_per_probe_work(self):
        """⚠ Specificity first. A timing assertion that cannot fail is a comment."""
        eight = _median_ms([_head() for _ in range(8)], self._hidden())
        two = _median_ms([_head() for _ in range(N_PROBES)], self._hidden())
        assert eight > two, (
            f"8 probes ({eight:.3f} ms) did not cost more than 2 ({two:.3f} ms) — the "
            "harness is not measuring per-probe work, so every timing below is vacuous"
        )

    def test_the_cost_is_not_superlinear_in_probe_count(self):
        """One hook per layer is shared; the per-probe cost is the arithmetic.

        Asserted as a factor rather than a ratio, because this runs on whatever machine CI
        gives it. What it rules out is a superlinear per-probe path, which would make
        `PROBE_MAX_ARMED = 8` a limit nobody can afford.
        """
        one = _median_ms([_head()], self._hidden())
        eight = _median_ms([_head() for _ in range(8)], self._hidden())
        assert eight < one * 16, (
            f"8 probes cost {eight:.3f} ms against 1 probe's {one:.3f} ms — more than 16x "
            "for 8x the work is superlinear, and the armed limit of 8 is unaffordable"
        )

    @pytest.mark.parametrize("rule", ["mean", "max", "last"])
    def test_every_streamable_rule_costs_about_the_same(self, rule):
        """A rule that was individually slow would hide behind a `mean`-only measurement.

        Compared against `mean` rather than against a wall-clock budget, so this stays a
        statement about the rules and not about the runner.
        """
        heads = [_head() for _ in range(N_PROBES)]
        baseline = _median_ms(heads, self._hidden(), "mean")
        measured = _median_ms(heads, self._hidden(), rule)
        assert measured < baseline * 3, (
            f"rule {rule!r} cost {measured:.3f} ms against mean's {baseline:.3f} ms — the "
            "rules should differ by a reduction, not by a factor of three"
        )

    _CACHED: torch.Tensor | None = None

    @classmethod
    def _hidden(cls) -> torch.Tensor:
        if cls._CACHED is None:
            torch.manual_seed(1)
            cls._CACHED = torch.randn(1, N_TOKENS, D_MODEL, dtype=torch.float16)
        return cls._CACHED
