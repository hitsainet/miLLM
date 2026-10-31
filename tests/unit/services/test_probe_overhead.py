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

    def test_the_budget_is_PER_PASS_because_that_is_where_the_cost_IS(self):
        """⚠ THIS ASSERTION USED TO READ `PROBE_MAX_OVERHEAD_MS == 5.0`, AND THE CRITERION IT
        PINNED NAMED THE WRONG VARIABLE.

        FR-24.14 said "under 5 ms at 4k-token contexts with 2 armed probes on LFM2". Measured on
        Llama-3.1-8B 2026-09-30, one probe, varying ONLY `max_tokens` on an identical prompt:

            4 generated -> 0.837 ms      30 -> 3.425 ms      120 -> 11.223 ms

        while **2820 prompt tokens with 8 generated cost 1.94 ms**. Twenty-one times the tokens
        for a third of the overhead. Prefill scores the whole prompt in ONE call; decode scores one
        token per call, so a prompt token is ~300x cheaper than a generated one.

        So the criterion was measurable on its cheapest case, and the per-request threshold was
        crossed by any completion over ~50 tokens on ANY model — including the LFM2 it was set
        against. A warning that fires on normal use is noise, not signal.

        Per pass, one number describes every request shape. The measured rate is ~0.09 ms; 0.25
        leaves room for variation while still catching the regression this file exists for — the
        whole-residual host copy measured 25-37 ms for two probes, which is ~140x the budget.
        """
        assert settings.PROBE_MAX_OVERHEAD_MS_PER_PASS == 0.25

    def test_the_absolute_backstop_is_pathology_only(self):
        """It is kept LIVE rather than reported-and-unused, and set where it means "something is
        wrong" rather than "that was a long answer": at ~0.09 ms/pass a 2000-token completion is
        ~180 ms, so 500 ms is not reachable by length alone.
        """
        assert settings.PROBE_MAX_OVERHEAD_MS == 500.0
        assert settings.PROBE_MAX_OVERHEAD_MS > 180.0, (
            "the backstop is inside the range a long completion reaches legitimately, so it will "
            "fire on normal use — which is what made the old per-request threshold noise"
        )


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


class TestNothingProportionalToTheTokensHappensOnTheHostPerPass:
    """⚠ THE SECOND HALF OF THE SAME DEFECT. The activations stopped crossing to the host; the
    SCORES did not.

    Per forward pass, per probe, `observe` used to build a Python list of one float per token, a
    Python list of one bool per token, and `finish` then rebuilt tensors from both. Measured on
    the 3090 at 4096 tokens, per probe: 0.051 ms for the `.tolist()`, 0.006 for the mask,
    0.133 + 0.192 to rebuild the tensors, and **0.431 ms for a Python `sorted()` over every
    scored position to keep five of them** — 0.813 ms of bookkeeping around 0.42 ms of
    arithmetic, against a 5 ms budget for the whole request.

    These assert the SHAPE of what is accumulated rather than a timing, because a timing at
    4096 x 2048 on CI measures the runner. The measured figures are in the commit and in
    `0xcc/reviews/`.
    """

    @staticmethod
    def _probe(rule: str = "mean", d_model: int = 16, attention: bool = False):
        from millm.services.probe_runtime import ArmedProbe

        torch.manual_seed(2)
        return ArmedProbe(
            probe_id="pr_1",
            name="t",
            head=ProbeHead(
                weight=torch.randn(d_model),
                bias=0.25,
                mean=torch.zeros(d_model),
                std=torch.ones(d_model),
                attention_query=torch.randn(d_model) if attention else None,
            ),
            rule=rule,
            scope="all",
            layer=1,
            rung=2,
            rung_language="detects on unseen tasks",
            threshold=0.0,
        )

    def test_the_accumulated_scores_are_TENSORS_one_per_pass(self):
        from millm.services.probe_runtime import ProbeRequestContext

        context = ProbeRequestContext("req", [self._probe()])
        context.observe(1, torch.randn(1, 64, 16))
        context.observe(1, torch.randn(1, 1, 16))
        parts = context._scores["pr_1"]
        assert len(parts) == 2, "one tensor per forward pass, not one entry per token"
        assert all(isinstance(p, torch.Tensor) for p in parts), (
            f"the score accumulator holds {[type(p).__name__ for p in parts]} — a Python float "
            "per token is 0.051 ms of `.tolist()` per probe per pass at 4k, and `finish` then "
            "rebuilds the tensor it needed all along"
        )
        assert [tuple(p.shape) for p in parts] == [(64,), (1,)]

    def test_the_mask_is_a_TENSOR_too_and_is_shared_across_probes_on_a_layer(self):
        """The window depends on the position and the length, not on the probe."""
        from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

        a = self._probe()
        b = ArmedProbe(**{**a.__dict__, "probe_id": "pr_2"})
        context = ProbeRequestContext("req", [a, b])
        context.observe(1, torch.randn(1, 32, 16))
        windows = (context._mask["pr_1"][0], context._mask["pr_2"][0])
        assert all(isinstance(w, torch.Tensor) and w.dtype == torch.bool for w in windows)
        assert windows[0] is windows[1], (
            "each probe built its own mask — that is a Python list per probe per pass"
        )

    def test_the_attention_logits_are_tensors_too(self):
        """The `attention` rule has its own accumulator, and it had its own `.tolist()`."""
        from millm.services.probe_runtime import ProbeRequestContext

        context = ProbeRequestContext("req", [self._probe(rule="attention", attention=True)])
        context.observe(1, torch.randn(1, 48, 16))
        parts = context._logits["pr_1"]
        assert parts and all(isinstance(p, torch.Tensor) for p in parts)
        assert context.finish()[0].scored is True

    def test_the_head_is_moved_to_the_device_ONCE_not_once_per_pass(self):
        """⚠ `to_device` COPIES every weight it is given. It was called from inside
        `token_scores`, so a 192-token completion re-uploaded `weight`, `mean` and `std` 193
        times per probe — measured at 0.035 ms per call on the 3090, which is 13 ms for two
        probes, more than twice the whole 5 ms budget, for weights that never change.

        Counted by spying on the head, not by reading the source — and counting the calls that
        actually COPIED, since `to_device` returns `self` when there is nothing to move and being
        called is not the same as costing anything. The head's weights are float64 here so that a
        copy is required on CPU at all; on the card the same copy is a host-to-device upload.
        """
        from millm.ml import probe_head as head_module
        from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

        copies: list[object] = []
        real = head_module.ProbeHead.to_device

        def counted(self, device):
            out = real(self, device)
            if out is not self:
                copies.append(device)
            return out

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(head_module.ProbeHead, "to_device", counted)
        try:
            probe = ArmedProbe(
                probe_id="pr_1",
                name="t",
                head=ProbeHead(weight=torch.randn(16, dtype=torch.float64)),
                rule="mean",
                scope="all",
                layer=1,
                rung=2,
                rung_language="detects on unseen tasks",
                threshold=0.0,
            )
            context = ProbeRequestContext("req", [probe])
            for _ in range(12):
                context.observe(1, torch.randn(1, 4, 16))
        finally:
            monkeypatch.undo()

        assert len(copies) == 1, (
            f"the head's weights were copied {len(copies)} times over 12 forward passes — that "
            "is a re-upload of weight, mean and std per pass, measured at 0.035 ms each"
        )
        assert context.finish()[0].n_scored_tokens == 48

    def test_finish_brings_the_row_to_the_HOST_so_combine_runs_where_parity_runs(self):
        """⚠ NOT AN OVERSIGHT THAT THE ROW LEAVES THE CARD.

        `combine()` is the function the parity gate calls, and the parity gate scores its test
        vectors on the host. A reduction over 4k float32 values does not associate the same way
        on both: measured on the 3090, the `mean` over a 4160-token row differed by **4.8e-07**
        between a device reduction and a host one, deterministically. Four orders inside the
        1e-3 tolerance, and still two numbers where the design says there is one.

        Asserted by recording the call, because there is no CUDA on CI.
        """
        from millm.services import probe_runtime

        moved: list[str] = []

        class ElsewhereTensor(torch.Tensor):
            """A CPU tensor that reports a non-CPU device, so the branch runs without a GPU."""

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, False)

            @property
            def device(self):  # type: ignore[override]
                return torch.device("cuda", 0)

            def cpu(self, *a, **k):  # type: ignore[override]
                moved.append("cpu")
                return torch.Tensor(self)

        row = probe_runtime._row([ElsewhereTensor(torch.randn(8))])
        assert moved == ["cpu"], "the row stayed on the card, so combine() reduces there"
        assert row.device.type == "cpu"

    def test_a_row_already_on_the_host_is_not_copied(self):
        """Specificity: an unconditional `.cpu()` is a copy of every score on every request."""
        from millm.services import probe_runtime

        part = torch.randn(8)
        assert probe_runtime._row([part]) is part

    def test_top_positions_break_ties_by_ASCENDING_POSITION(self):
        """⚠ The Python `sorted()` this replaced is stable, so equal scores came back in position
        order. `topk` and an unstable sort do not promise that, and the difference only shows on
        a row with repeated scores — where it would make the reported positions vary between
        runs on identical input, which is unreproducible rather than wrong.

        ⚠ **128 TIED POSITIONS, NOT 9, AND THE NUMBER IS THE WHOLE TEST.** Written first with 9,
        it was mutation-controlled by flipping `stable=True` to `stable=False` — and the control
        SURVIVED. torch's CPU sort uses insertion sort below a size threshold, which is stable
        whatever the flag says, so at n=9 the two forms return identical indices and the fixture
        agreed with the mutation by construction. Measured on this build: identical at n=9,
        divergent from n=100 upward (at 128 the unstable sort's first index is not 0). A tie-break
        test below the threshold asserts nothing at all.
        """
        from millm.services.probe_runtime import ProbeRequestContext

        n_tied = 128
        probe = self._probe(d_model=1)
        # weight is a single value, so a constant activation gives every position one score.
        context = ProbeRequestContext("req", [probe])
        context.observe(1, torch.ones(1, n_tied, 1))
        verdict = context.finish()[0]
        assert verdict.top_positions == [0, 1, 2, 3, 4], (
            f"{n_tied} tied scores reported {verdict.top_positions} — ties must resolve to the "
            "lowest positions, as the Python sort this replaced did"
        )


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

class TestTheWarningFiresOnTheRateAndNotOnTheLENGTH:
    """⚠ THE WHOLE POINT OF THE RESHAPE, and the only tests here that would fail against the old
    per-request threshold.

    Under `PROBE_MAX_OVERHEAD_MS == 5.0` per request, a perfectly healthy 400-token answer cost
    ~36 ms and warned; a 4k-token prompt with a 4-token answer cost ~0.8 ms and passed. The
    operator therefore learned to ignore the warning, which is the failure mode — a guard nobody
    reads is worse than no guard, because it still looks like coverage.
    """

    @staticmethod
    def _service():
        from unittest.mock import MagicMock

        from millm.services.probe_event_service import ProbeEventService

        return ProbeEventService(MagicMock(), MagicMock())

    def test_a_long_healthy_completion_does_NOT_warn(self, caplog):
        """401 passes at the measured ~0.09 ms/pass = 36 ms. Healthy, and over seven times the
        old per-request threshold."""
        service = self._service()
        with caplog.at_level("WARNING"):
            service.note_request_overhead(36.0, n_passes=401)
        assert "probe_overhead_above_threshold" not in caplog.text, (
            "a normal long answer at the measured rate warned — this is the noise the reshape "
            "exists to remove"
        )
        assert "probe_overhead_request_backstop" not in caplog.text

    def test_a_slow_PASS_warns_even_on_a_short_request(self, caplog):
        """Specificity: the test above must not pass by never warning at all.

        Two passes at 5 ms each is 10 ms total — a small number, and 25x the per-pass budget. The
        old threshold would have warned here too, but for the wrong reason (the total), and would
        have said nothing if the same rate arrived over one pass.
        """
        service = self._service()
        with caplog.at_level("WARNING"):
            service.note_request_overhead(10.0, n_passes=2)
        assert "probe_overhead_above_threshold" in caplog.text
        assert "overhead_ms_per_pass=5.0000" in caplog.text, (
            "the warning must report the RATE it judged, or an operator cannot tell which guard "
            "fired or what to compare it against"
        )

    def test_one_slow_pass_warns(self, caplog):
        """The regression this file exists for arrives as ONE expensive pass: the whole-residual
        host copy measured 25-37 ms for two probes. A per-request threshold of 500 ms would miss
        it entirely on a short request."""
        service = self._service()
        with caplog.at_level("WARNING"):
            service.note_request_overhead(30.0, n_passes=1)
        assert "probe_overhead_above_threshold" in caplog.text

    def test_the_backstop_still_catches_pathology(self, caplog):
        """Both guards are live. A total that no completion length explains is reported as its own
        thing, so it is not mistaken for a rate problem."""
        service = self._service()
        with caplog.at_level("WARNING"):
            service.note_request_overhead(900.0, n_passes=40000)   # rate is fine, total is not
        assert "probe_overhead_request_backstop" in caplog.text
        assert "probe_overhead_above_threshold" not in caplog.text

    def test_nothing_scored_is_not_a_rate_of_zero(self):
        """`n_passes=0` has no rate. Reporting 0.0 would read as "free", which is a different
        claim from "not measured" — the same distinction `last_request_overhead_ms` already makes
        with `None`.
        """
        service = self._service()
        service.note_request_overhead(4.0, n_passes=0)
        assert service._last_overhead_ms_per_pass is None

    def test_the_rate_is_reported_not_just_judged(self):
        service = self._service()
        service.note_request_overhead(12.0, n_passes=120)
        assert service._last_overhead_ms_per_pass == pytest.approx(0.1)
        assert service._last_request_n_passes == 120
