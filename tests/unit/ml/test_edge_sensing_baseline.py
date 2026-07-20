"""Feature 17 task 1.4 — PARITY BASELINES for the extraction.

Three benchmark shapes measured against the CURRENT code. After F17 moves the
matcher into `millm/ml/edge_sensing.py`, these are the numbers the refactor
must not regress. They are expressed as assertions rather than a printed
report so a regression fails the suite instead of needing someone to notice.

The thresholds are deliberately loose (roughly 3x the measured value) — this
is a parity net for a refactor, not a performance gate. A 3x regression means
something structural changed, which is exactly what should stop a "pure move".
"""

import time

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


def _time_pass(sae: LoadedSAE, seq: int) -> float:
    torch.manual_seed(11)
    hs = torch.randn(seq, D_IN)
    started = time.perf_counter()
    sae._sense_edges(hs)
    return (time.perf_counter() - started) * 1000.0


class TestParityBaselines:
    """Measured 2026-07-20 against the pre-extraction implementation."""

    def test_a_saturated_long_prefill_sheds_and_stays_cheap(self):
        """A miscalibrated threshold on a 4096-token prefill. Load shedding
        (F15 R1) must keep this off the critical path: it was 1430 ms before
        shedding existed, ~1 ms after."""
        sae = _armed(threshold=0.5)
        ms = _time_pass(sae, 4096)
        assert sae._edge_truncated is True, "saturation must be visible"
        assert ms < 50.0, f"saturated 4096-token pass cost {ms:.1f}ms"

    def test_a_realistic_long_prefill_stays_within_the_per_layer_budget(self):
        """A calibrated threshold at the contract's 200-edge maximum. This is
        the shape a real circuit presents."""
        sae = _armed(threshold=3.0)
        ms = _time_pass(sae, 4096)
        assert ms < 60.0, f"realistic 4096-token pass cost {ms:.1f}ms"

    def test_a_typical_prefill_is_cheap(self):
        sae = _armed(threshold=3.0)
        ms = _time_pass(sae, 512)
        assert ms < 30.0, f"512-token pass cost {ms:.1f}ms"

    def test_the_matcher_scales_with_FIRES_not_positions_times_edges(self):
        """F15 R1's vectorisation and R2's bisect made cost track the number
        of fires rather than seq_len x edge_count. A move that reintroduced
        the nested scan would show up as superlinear growth here."""
        short = _time_pass(_armed(threshold=3.0), 512)
        long_ = _time_pass(_armed(threshold=3.0), 4096)
        # 8x the tokens must not cost anywhere near 8x, let alone 64x.
        assert long_ < max(short * 8.0, 60.0), (
            f"512-tok={short:.1f}ms 4096-tok={long_:.1f}ms — growth suggests "
            "the positions x edges scan has returned"
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

    def _full_ring(self):
        from millm.ml.edge_sensing import EdgeFireRing

        # max_lag deliberately huge: with a small window the linear walk breaks
        # out early and looks fast, which is how this hid.
        ring = EdgeFireRing(100_000)
        for pos in range(EdgeFireRing._MAX_FIRES_PER_EDGE):
            ring.record_up("e", pos, 1.0)
        return ring

    def test_ascending_matching_over_a_full_ring_stays_cheap(self):
        ring = self._full_ring()
        ring.match_down("e", 1)                      # warm
        started = time.perf_counter()
        for down in range(2000):
            ring.match_down("e", down)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert elapsed_ms < 3.0, (
            f"{elapsed_ms:.2f} ms / 2000 ascending matches — the linear walk "
            "is back; measured 7.38 ms for it and 0.55 ms for the bisect"
        )

    def test_ascending_is_not_dramatically_worse_than_a_tail_probe(self):
        """The shape-independent form: a logarithmic lookup costs about the
        same wherever it lands, a linear walk does not. This survives machines
        faster or slower than the one the absolute bound was measured on."""
        ring = self._full_ring()
        ring.match_down("e", 1)

        started = time.perf_counter()
        for down in range(2000):
            ring.match_down("e", down)
        ascending = (time.perf_counter() - started) * 1000.0

        tail = EdgeFireRing_last = ring._fires["e"][-1][0] + 1
        started = time.perf_counter()
        for _ in range(2000):
            ring.match_down("e", tail)
        tail_ms = (time.perf_counter() - started) * 1000.0

        assert ascending < max(tail_ms * 8.0, 1.5), (
            f"ascending {ascending:.2f} ms vs tail {tail_ms:.2f} ms — a "
            f"{ascending / max(tail_ms, 1e-9):.1f}x gap means the lookup is "
            "walking, not bisecting"
        )
