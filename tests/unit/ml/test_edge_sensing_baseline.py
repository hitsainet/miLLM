"""Feature 17 task 1.4 — PARITY BASELINES for the extraction.

Three benchmark shapes measured against the CURRENT code. After F17 moves the
matcher into `millm/ml/edge_sensing.py`, these are the numbers the refactor
must not regress. They are expressed as assertions rather than a printed
report so a regression fails the suite instead of needing someone to notice.

Since 2026-10-20 every cost is a RATIO against a reference measured in the same
run, never a millisecond ceiling (see the note below the imports). Each bound
sits about 2x above the highest ratio measured under heavy load and below what
a planted 5x slowdown produces; this is a parity net for a refactor, not a
performance gate.
"""

import bisect

import pytest

import torch

from millm.ml.edge_sensing import EdgeSensingRequestContext
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import (
    CircuitSensingConfig,
    EdgeFireRing,
    EdgeSpec,
    LoadedSAE,
)
from tests.support.timing import paired_min_ms, single_torch_thread

#: NO BARE WALL-CLOCK CEILINGS. Every cost here is a ratio against a reference
#: measured in the same run, interleaved, best of several rounds, in thread CPU
#: time, on one torch thread (tests/support/timing.py says why each matters).
#:
#: History, so nobody puts a millisecond bound back: this file asserted 30 / 50
#: / 60 ms ceilings, then the same ceilings times an 8x `MILLM_PERF_SLACK` on CI.
#: It failed on shared runners at 51.5 / 57.5 / 52.0 / 254.2 ms against 50 ms,
#: at 457.3 ms against the slackened 400 ms (2026-09-28, which held three probe
#: fixes out of production behind the image gate), and at 160 ms against 60 ms
#: at load average 19. None of those was a regression.
#:
#: The reference for a sensing pass is the ENCODE it cannot avoid: the same
#: hidden states through the same member columns, thresholded and copied to the
#: host — the pass's own first three lines, without the matcher. Anything the
#: pass adds on top of that is what a regression would grow.


@pytest.fixture(autouse=True)
def _one_thread():
    with single_torch_thread():
        yield


D_IN = 128
D_SAE = 512
#: The contract maximum (CIRCUIT_MAX_EDGES) — the worst shape a real circuit
#: can present to the matcher.
CONTRACT_MAX_EDGES = 200


def _armed(threshold: float, n_edges: int = CONTRACT_MAX_EDGES) -> LoadedSAE:
    torch.manual_seed(7)
    W_enc = torch.randn(D_IN, D_SAE) * 0.1
    sae = LoadedSAE(
        W_enc=W_enc,
        b_enc=torch.zeros(D_SAE),
        W_dec=torch.zeros(D_SAE, D_IN),
        b_dec=torch.zeros(D_IN),
        config=SAEConfig(d_in=D_IN, d_sae=D_SAE, model_name="t",
                         hook_name="t", hook_layer=1),
        device="cpu",
    )
    members = list(range(40))
    specs = [
        EdgeSpec(
            edge_key=f"e{i}", up_layer=10, up_feature_idx=i % 40,
            up_col=i % 40, down_layer=10, down_feature_idx=(i * 7 + 3) % 40,
            down_col=(i * 7 + 3) % 40, rung=2,
            rung_language="causally validated (edge)", edge_type="computed",
        )
        for i in range(n_edges)
    ]
    cfg = CircuitSensingConfig(
        circuit_id="c", layer=10, member_indices=members,
        thresholds=torch.full((40,), threshold), threshold_mode="epsilon_max",
        edges=specs, max_token_lag=8, context_tokens=16,
        max_events_per_request=20,
    )
    sae.arm_edge_sensing(cfg)
    sae.bind_context(EdgeSensingRequestContext(
        request_id="baseline",
        circuit_ids=frozenset({cfg.circuit_id}),
        cap=cfg.max_events_per_request,
    ))
    sae.begin_edge_sensing_request("baseline")
    return sae


def _hidden(seq: int) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(seq, D_IN)


def _pass(threshold: float, seq: int):
    """A fresh armed SAE's sensing pass, ready to time (a pass mutates its ring and offsets)."""
    sae, hs = _armed(threshold), _hidden(seq)
    return lambda: sae._sense_edges(hs)


def _encode(threshold: float, seq: int):
    """The reference: the encode a sensing pass cannot avoid, on the same shape, without matching."""
    sae, hs = _armed(threshold), _hidden(seq)
    W, b = sae._W_enc_e.clone(), sae._b_enc_e.clone()
    thresholds = sae._edge_sensing.thresholds.clone()

    def run():
        acts = torch.relu(hs @ W + b)
        fired = acts > thresholds
        bool(fired.any())
        fired.to("cpu")
        acts.to("cpu", torch.float32)

    return run


def _pass_over_encode(threshold: float, seq: int) -> tuple[float, float, float]:
    sense_ms, encode_ms = paired_min_ms(
        _pass(threshold, seq), _encode(threshold, seq),
        setup=lambda: (_pass(threshold, seq), _encode(threshold, seq)),
    )
    return sense_ms / max(encode_ms, 1e-9), sense_ms, encode_ms


#: Bounds on sensing-pass cost / encode cost (thread CPU time, one torch thread,
#: best of 7). Measured 2026-10-20, 8 trials each:
#:
#:   shape        load ~19 (shared host)   + busy loop on all 16 cores (load 31)
#:   512 tokens   9.9x - 12.8x             11.0x - 13.7x
#:   4096 tokens  16.8x - 21.8x            16.7x - 19.3x
#:
#: `_sense_edges` planted to run its body 5 times: 512 -> 57.2x-71.8x,
#: 4096 -> 59.8x-66.2x under the burner (49.1x / 55.1x minima at load ~18 in
#: wall time). The bounds fail that plant with ~1.4x margin to spare and pass
#: the loaded maxima with ~2x.
PASS_OVER_ENCODE_512 = 30.0
PASS_OVER_ENCODE_4096 = 40.0


class TestParityBaselines:
    """Measured 2026-07-20 against the pre-extraction implementation; turned
    into same-run ratios 2026-10-20."""

    def test_a_saturated_long_prefill_sheds_and_stays_cheap(self):
        """A miscalibrated threshold on a 4096-token prefill. Load shedding
        (F15 R1) must keep this off the critical path: it was 1430 ms before
        shedding existed, ~1 ms after."""
        # The BEHAVIOURAL half, asserted unconditionally: shedding must happen
        # and must be visible. This is the part that actually protects the
        # feature, and it is machine-independent.
        sae = _armed(threshold=0.5)
        sae._sense_edges(_hidden(4096))
        assert sae._edge_truncated is True, "saturation must be visible"

        # The cost half is a ratio against the CALIBRATED pass: same shape,
        # same token count, a threshold that does not saturate. It was
        # `ms < _budget(50.0)` until a correct build measured 457.3 ms on a
        # contended runner (2026-09-28).
        #
        # The regression this exists for is shedding not happening at all:
        # 1430 ms against an ~11 ms calibrated pass, about 130x. Measured
        # 2026-10-20 (thread CPU time, one thread): 2.2x-2.5x under a busy loop
        # on every core. 8x still fails an order of magnitude below the unshed
        # cost.
        saturated_ms, calibrated_ms = paired_min_ms(
            _pass(0.5, 4096), _pass(3.0, 4096),
            setup=lambda: (_pass(0.5, 4096), _pass(3.0, 4096)),
        )
        ratio = saturated_ms / max(calibrated_ms, 1e-9)
        assert ratio < 8.0, (
            f"saturated 4096-token pass cost {saturated_ms:.1f}ms against a "
            f"calibrated {calibrated_ms:.1f}ms on the same machine "
            f"({ratio:.2f}x, bound 8x) — shedding is not keeping saturation "
            f"off the critical path"
        )

    def test_a_realistic_long_prefill_stays_within_the_per_layer_budget(self):
        """A calibrated threshold at the contract's 200-edge maximum. This is
        the shape a real circuit presents. Was `ms < 60` (x8 on CI)."""
        ratio, sense_ms, encode_ms = _pass_over_encode(3.0, 4096)
        assert ratio < PASS_OVER_ENCODE_4096, (
            f"realistic 4096-token pass {sense_ms:.2f}ms against its own encode "
            f"{encode_ms:.2f}ms ({ratio:.1f}x, bound {PASS_OVER_ENCODE_4096}x) — "
            "the matcher's cost over the unavoidable encode has grown"
        )

    def test_a_typical_prefill_is_cheap(self):
        """Was `ms < 30` (x8 on CI)."""
        ratio, sense_ms, encode_ms = _pass_over_encode(3.0, 512)
        assert ratio < PASS_OVER_ENCODE_512, (
            f"512-token pass {sense_ms:.2f}ms against its own encode "
            f"{encode_ms:.2f}ms ({ratio:.1f}x, bound {PASS_OVER_ENCODE_512}x)"
        )

    def test_the_matcher_scales_with_FIRES_not_positions_times_edges(self):
        """F15 R1's vectorisation and R2's bisect made cost track the number
        of fires rather than seq_len x edge_count. A move that reintroduced
        the nested scan would show up as superlinear growth here."""
        # R1-10: this asserted `long_ < max(short * 8.0, 60.0)`, and the
        # `short * 8` term NEVER BOUND (the 60 ms floor dominated every run),
        # so a test named for GROWTH asserted only an absolute ceiling, and the
        # 13x match_down regression (R1-08) sailed straight through it.
        #
        # Growth here is LINEAR, not sublinear — the cost of this shape is
        # dominated by the per-position work, not by matching — so the bound is
        # set to catch the QUADRATIC blowup a restored positions x edges scan
        # would produce (~64x for 8x the tokens), not to assert sublinearity
        # the code never had. Measured multi-threaded 7.69x-8.45x (2026-07);
        # 7.7x-9.4x on one thread in CPU time under a busy loop on every core
        # (2026-10-20).
        #
        # KNOWN LIMIT, recorded rather than hidden: at threshold=3.0 few
        # positions fire, so this shape does not exercise the matcher hard
        # enough to catch a quadratic-in-FIRES regression — verified by
        # mutation (an O(fires^2) loop passes here and is caught instead by
        # test_a_saturated_long_prefill_sheds_and_stays_cheap, where the fire
        # count is the point). What this test does protect is growth in
        # SEQUENCE LENGTH. The two are complementary.
        #
        # The absolute backstop that sat here (`long_ < 60 ms`) is gone: a
        # uniformly slow pass is test_a_realistic_long_prefill's job, measured
        # against the encode rather than against the clock.
        long_, short = paired_min_ms(
            _pass(3.0, 4096), _pass(3.0, 512),
            setup=lambda: (_pass(3.0, 4096), _pass(3.0, 512)),
        )
        ratio = long_ / max(short, 1e-9)
        assert ratio < 16.0, (
            f"512-tok={short:.2f}ms 4096-tok={long_:.2f}ms ({ratio:.1f}x for "
            "8x the tokens) — superlinear growth suggests the positions x "
            "edges scan has returned"
        )


class TestMatchDownIsLogarithmicOnTheCrossLayerPath:
    """R1-08. The F17 extraction silently replaced `bisect_left` with a linear
    backward walk — reverting F15 R3's O(n)->O(log n) fix — while leaving the
    docstring describing the bisect. Reading the function told you the opposite
    of what it ran.

    The existing bound test could not catch it: it probes `last + 1`, the one
    position where a backward walk terminates on iteration ONE. This probes
    ASCENDING, which is the order the docstring itself names as normal — hooks
    run in layer order, so the upstream layer records its whole prefill before
    the downstream layer matches ascending.

    Measured on a full 512-fire ring with a window wide enough that the walk
    cannot break early:

        linear walk : 7.38 ms / 2000   <- the regression, suite green
        bisect      : 0.55 ms / 2000   <- restored
    """

    N = 2000

    def _full_ring(self):
        # max_lag deliberately huge: with a small window the linear walk breaks
        # out early and looks fast, which is how this hid.
        ring = EdgeFireRing(100_000)
        for pos in range(EdgeFireRing._MAX_FIRES_PER_EDGE):
            ring.record_up("e", pos, 1.0)
        return ring

    def _ascending(self, ring):
        return lambda: [ring.match_down("e", down) for down in range(self.N)]

    def test_ascending_matching_over_a_full_ring_stays_cheap(self):
        """Against the bare bisect it is built on, over the same list and the
        same probes. Was `elapsed_ms < 3.0`."""
        ring = self._full_ring()
        fires = ring._fires["e"]
        lo = float("-inf")

        def bare_bisect():
            return [bisect.bisect_left(fires, (down, lo)) for down in range(self.N)]

        ascending, reference = paired_min_ms(self._ascending(ring), bare_bisect)
        ratio = ascending / max(reference, 1e-9)
        assert ratio < MATCH_OVER_BISECT, (
            f"{ascending:.2f} ms / {self.N} ascending matches against "
            f"{reference:.2f} ms of bare bisect ({ratio:.1f}x, bound "
            f"{MATCH_OVER_BISECT}x) — the lookup is doing more than a bisect "
            "and a short walk"
        )

    def test_ascending_is_not_dramatically_worse_than_a_tail_probe(self):
        """The shape-independent form: a logarithmic lookup costs about the
        same wherever it lands, a linear walk does not. Was
        `ascending < max(tail * 8, 1.5 ms)`, whose floor was a bare ceiling."""
        ring = self._full_ring()
        tail = ring._fires["e"][-1][0] + 1
        ascending, tail_ms = paired_min_ms(
            self._ascending(ring),
            lambda: [ring.match_down("e", tail) for _ in range(self.N)],
        )
        ratio = ascending / max(tail_ms, 1e-9)
        assert ratio < 8.0, (
            f"ascending {ascending:.2f} ms vs tail {tail_ms:.2f} ms — a "
            f"{ratio:.1f}x gap means the lookup is walking, not bisecting"
        )


#: match_down / bare bisect over the same 2000 probes. Measured 2026-10-20:
#: 1.5x-2.2x, with or without a busy loop on every core; `match_down` planted to
#: do its work 5 times: 6.3x-8.3x (7.1x-7.9x under the burner). The linear walk
#: R1-08 restored measured 13x-26x.
MATCH_OVER_BISECT = 4.0
