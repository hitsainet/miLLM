"""Feature 17 task 1.0 — CHARACTERIZATION GATE.

These tests pin the edge matcher's CURRENT behaviour, written and green BEFORE
any code moves. They are the parity contract for the extraction: after F17
moves this machinery into `millm/ml/edge_sensing.py`, every one of them must
still pass unchanged.

Why this gate exists (BRD-002 locked decision 3): this is the most defect-dense
code in the arc — eight criticals across three review rounds of Feature 15, plus
four more in Feature 16. Refactoring it without a behavioural net would be
moving code whose behaviour nobody has written down.

**From here on, editing a test in this file is a BEHAVIOUR CHANGE and requires
justification in the review record (CTX-V2).** A failure here after the move is
a regression, not a stale fixture.
"""

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.edge_sensing import EdgeSensingRequestContext
from millm.ml.sae_wrapper import (
    CircuitSensingConfig,
    EdgeFireRing,
    EdgeSpec,
    LoadedSAE,
)

D_IN = 8
D_SAE = 32


def real_sae() -> LoadedSAE:
    """Deterministic SAE: encoder column j responds 1:1 to input dim j % d_in."""
    W_enc = torch.zeros(D_IN, D_SAE)
    for j in range(D_SAE):
        W_enc[j % D_IN, j] = 1.0
    return LoadedSAE(
        W_enc=W_enc,
        b_enc=torch.zeros(D_SAE),
        W_dec=torch.zeros(D_SAE, D_IN),
        b_dec=torch.zeros(D_IN),
        config=SAEConfig(d_in=D_IN, d_sae=D_SAE, model_name="t",
                         hook_name="t", hook_layer=1),
        device="cpu",
    )


def hidden(*rows) -> torch.Tensor:
    out = torch.zeros(len(rows), D_IN)
    for i, row in enumerate(rows):
        for dim, val in row.items():
            out[i, dim] = val
    return out


def spec(up_col=0, down_col=1, up_layer=10, down_layer=10, rung=2,
         key="1@10->2@10"):
    return EdgeSpec(
        edge_key=key, up_layer=up_layer, up_feature_idx=1, up_col=up_col,
        down_layer=down_layer, down_feature_idx=2, down_col=down_col,
        rung=rung, rung_language="causally validated (edge)",
        edge_type="computed",
    )


def config(edges=None, max_lag=4, cap=20, layer=10):
    return CircuitSensingConfig(
        circuit_id="circ_1", layer=layer, member_indices=[1, 2],
        thresholds=torch.tensor([0.5, 0.5]), threshold_mode="epsilon_max",
        edges=edges if edges is not None else [spec()],
        max_token_lag=max_lag, context_tokens=8, max_events_per_request=cap,
    )


def ctx_for(cfg, request_id="char"):
    """A request context for one circuit — the F17 replacement for a bare ring.

    Test BODIES are unchanged (CTX-V2); only these helpers were retargeted, per
    task 3.8. Passing the same ctx to two `armed()` calls is now how two SAEs
    cooperate on one circuit, exactly as passing the same ring used to be.
    """
    return EdgeSensingRequestContext(
        request_id=request_id,
        circuit_ids=frozenset({cfg.circuit_id}),
        cap=cfg.max_events_per_request,
    )


def armed(cfg=None, ring=None, ctx=None):
    sae = real_sae()
    cfg = cfg or config()
    sae.arm_edge_sensing(cfg)
    # `ring=` is still accepted so the pre-extraction test bodies read
    # unchanged; a bare ring is wrapped in a context that hands it back.
    if ctx is None:
        ctx = ctx_for(cfg)
        if ring is not None:
            ctx._rings[cfg.circuit_id] = ring
    sae.bind_context(ctx)
    sae.begin_edge_sensing_request("char")
    return sae


# ─────────────────────────────────────────────────────────────────────────
# 1.1 — the matcher's core contract
# ─────────────────────────────────────────────────────────────────────────

class TestOrderingContract:
    def test_strict_up_then_down_matches(self):
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos, ev.token_lag) == (0, 1, 1)

    def test_a_same_position_co_fire_does_NOT_match(self):
        """Simultaneous firing is co-activation, not a sequence. Reporting it
        as up→down would assert an ordering that was never observed."""
        sae = armed()
        sae._sense_edges(hidden({1: 2.0, 2: 2.0}))
        assert sae._sensed_edges == []

    def test_reversed_order_does_NOT_match(self):
        sae = armed()
        sae._sense_edges(hidden({2: 2.0}, {1: 2.0}))
        assert sae._sensed_edges == []

    def test_a_lone_upstream_fire_does_NOT_match(self):
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {1: 2.0}))
        assert sae._sensed_edges == []


class TestLagBoundary:
    def test_exactly_at_the_window_matches(self):
        cfg = config(max_lag=3)
        sae = armed(cfg, EdgeFireRing(3))
        sae._sense_edges(hidden({1: 2.0}, {}, {}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        assert sae._sensed_edges[0].token_lag == 3

    def test_one_past_the_window_does_NOT_match(self):
        cfg = config(max_lag=3)
        sae = armed(cfg, EdgeFireRing(3))
        sae._sense_edges(hidden({1: 2.0}, {}, {}, {}, {2: 2.0}))
        assert sae._sensed_edges == []


class TestAntecedentSelection:
    def test_the_newest_antecedent_wins(self):
        """The closest antecedent is the most defensible attribution."""
        sae = armed(config(max_lag=8), EdgeFireRing(8))
        sae._sense_edges(hidden({1: 2.0}, {1: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        assert sae._sensed_edges[0].up_pos == 1

    def test_the_read_is_NON_DESTRUCTIVE(self):
        """One upstream fire can father several downstream events. The FTDD
        originally specified a 'pop'; the implementation reads without
        removing, and that is the better evidence model — two downstream
        partners responding to one upstream fire is two real observations."""
        sae = armed(config(max_lag=8), EdgeFireRing(8))
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 2
        assert {e.up_pos for e in sae._sensed_edges} == {0}


class TestRingRetention:
    def test_the_ring_evicts_the_OLDEST_when_full(self):
        ring = EdgeFireRing(4)
        for pos in range(EdgeFireRing._MAX_FIRES_PER_EDGE + 50):
            ring.record_up("e", pos, 1.0)
        assert len(ring._fires["e"]) <= EdgeFireRing._MAX_FIRES_PER_EDGE
        newest = EdgeFireRing._MAX_FIRES_PER_EDGE + 49
        assert ring.match_down("e", newest + 1) == (newest, 1.0)

    def test_match_is_strictly_before(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 5, 1.0)
        assert ring.match_down("e", 5) is None
        assert ring.match_down("e", 6) == (5, 1.0)


class TestPhaseAccounting:
    def test_prefill_flips_to_decode_exactly_once(self):
        sae = armed(config(max_lag=16), EdgeFireRing(16))
        assert sae._edge_phase == "prefill"
        sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_phase == "decode"
        sae._sense_edges(hidden({}))
        assert sae._edge_phase == "decode"

    def test_positions_are_absolute_across_passes(self):
        sae = armed(config(max_lag=16), EdgeFireRing(16))
        sae._sense_edges(hidden({}, {1: 2.0}))
        sae._sense_edges(hidden({2: 2.0}))
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos) == (1, 2)


# ─────────────────────────────────────────────────────────────────────────
# 1.2 — the twelve fixed criticals, characterized BY BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────

class TestFixedCriticalsStayFixed:
    """Each of these pins a defect that a review round found and fixed. The
    extraction must not resurrect any of them."""

    def test_F15R1_01_cross_layer_survives_a_noisy_upstream(self):
        """R1-01: pruning inside the per-position loop let the upstream layer
        wipe the ring before the downstream hook ran — cross-layer sensing
        went dark on any real traffic."""
        s = spec(up_layer=10, down_layer=13, key="1@10->2@13")
        ring = EdgeFireRing(3)

        up = real_sae()
        up_cfg = config(edges=[EdgeSpec(**{**s.__dict__, "up_col": 0, "down_col": -1})],
                        max_lag=3, layer=10)
        shared = ctx_for(up_cfg, "c")
        shared._rings[up_cfg.circuit_id] = ring
        up.arm_edge_sensing(up_cfg)
        up.bind_context(shared)
        up.begin_edge_sensing_request("c")
        rows = [{}] * 12
        rows[2] = {1: 2.0}
        for p in range(6, 12):
            rows[p] = {1: 2.0}
        up._sense_edges(hidden(*rows))

        down = real_sae()
        down_cfg = config(edges=[EdgeSpec(**{**s.__dict__, "up_col": -1, "down_col": 1})],
                          max_lag=3, layer=13)
        down.arm_edge_sensing(down_cfg)
        down.bind_context(shared)
        down.begin_edge_sensing_request("c")
        drows = [{}] * 12
        drows[4] = {2: 2.0}
        down._sense_edges(hidden(*drows))

        assert len(down._sensed_edges) == 1
        assert down._sensed_edges[0].up_pos == 2

    def test_F15R1_03_offset_advances_on_EVERY_return_path(self):
        """R1-03: early returns skipped the offset advance, so one SAE fell
        behind its siblings and the SHARED ring's absolute-position key
        silently diverged."""
        # suppressed
        sae = armed()
        with sae.suppressed():
            sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_token_offset == 3

        # batched
        sae2 = armed()
        sae2._sense_edges(hidden({}, {}).unsqueeze(0).repeat(3, 1, 1))
        assert sae2._edge_token_offset == 2

        # raising
        sae3 = armed()
        sae3._match_edges = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        sae3._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae3._edge_token_offset == 2

    def test_F15R1_04_out_of_range_column_is_a_clean_arm_time_error(self):
        """R1-04: an IndexError inside the matcher was swallowed by the broad
        except, abandoning the ENTIRE pass rather than one bad spec."""
        sae = real_sae()
        bad = EdgeSpec(**{**spec().__dict__, "down_col": 99})
        with pytest.raises(ValueError, match="column out of range"):
            sae.arm_edge_sensing(config(edges=[bad]))

    def test_F15R2_07_minus_one_is_the_only_valid_sentinel(self):
        """R2-07: validation checked only the upper bound, so -2 passed and the
        matcher silently skipped that half — armed, sensable, never firing."""
        sae = real_sae()
        with pytest.raises(ValueError, match="column out of range"):
            sae.arm_edge_sensing(
                config(edges=[EdgeSpec(**{**spec().__dict__, "down_col": -2})]))
        # -1 IS legitimate: "not my half"
        ok = real_sae()
        ok.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**spec().__dict__, "up_col": -1})]))
        assert ok.is_edge_sensing_armed is True

    def test_F15R2_03_a_shedding_layer_still_feeds_its_siblings(self):
        """R2-03: load shedding returned before recording upstream fires, so a
        saturated upstream layer starved a quiet downstream sibling — and the
        truncated flag landed on the wrong layer."""
        s = spec(up_layer=10, down_layer=13, key="1@10->2@13")
        ring = EdgeFireRing(64)

        up = real_sae()
        up_cfg = config(
            edges=[EdgeSpec(**{**s.__dict__, "up_col": 0, "down_col": -1})],
            max_lag=64, layer=10)
        up_cfg.thresholds = torch.tensor([0.0001, 0.0001])
        up.arm_edge_sensing(up_cfg)
        shared = ctx_for(up_cfg, "c")
        shared._rings[up_cfg.circuit_id] = ring
        up.bind_context(shared)
        up.begin_edge_sensing_request("c")
        up._sense_edges(torch.rand(4096, D_IN) + 1.0)
        assert up._edge_truncated is True

        down = real_sae()
        down.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**s.__dict__, "up_col": -1, "down_col": 1})],
                   max_lag=64, layer=13))
        down.bind_context(shared)
        down.begin_edge_sensing_request("c")
        down._sense_edges(hidden(*([{}] * 4095 + [{2: 5.0}])))
        assert down._sensed_edges, "the shedding layer starved its sibling"

    def test_F15R3_02_a_capped_layer_still_feeds_its_siblings(self):
        """R3-02: the cap returned from the whole pass, so a capped layer
        stopped feeding the shared ring — R2-03's bug via the cap path."""
        ring = EdgeFireRing(64)
        sae = real_sae()
        sae.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**spec().__dict__, "up_col": 0, "down_col": 1})],
                   max_lag=64, cap=1, layer=10))
        capped_ctx = ctx_for(config(max_lag=64, cap=1, layer=10), "c")
        capped_ctx._rings["circ_1"] = ring
        sae.bind_context(capped_ctx)
        sae.begin_edge_sensing_request("c")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}, {1: 2.0}, {2: 2.0}))
        assert sae._edge_done is True

        before = len(ring._fires.get("1@10->2@10", []))
        sae._sense_edges(hidden({1: 2.0}))
        assert len(ring._fires.get("1@10->2@10", [])) > before, (
            "a capped layer stopped feeding the shared ring"
        )

    def test_F15R3_01_pruning_is_wired_and_respects_the_slowest_layer(self):
        """R3-01: pruning was declared 'request-level' in two consecutive
        rounds and wired in neither. The ring now tracks layer progress
        itself and prunes to the SLOWEST."""
        ring = EdgeFireRing(8)
        ring.record_up("e", 40, 1.0)
        ring.note_layer_progress(10, 5000)
        ring.note_layer_progress(13, 42)
        assert ring.match_down("e", 44) == (40, 1.0), (
            "pruned past a fire the lagging layer still needed"
        )

    def test_a_single_layer_alone_never_prunes(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 0, 1.0)
        ring.note_layer_progress(10, 1000)
        assert ring.match_down("e", 1) == (0, 1.0)

    def test_saturation_sheds_visibly_rather_than_silently(self):
        sae = real_sae()
        cfg = config()
        cfg.thresholds = torch.tensor([0.0001, 0.0001])
        sae.arm_edge_sensing(cfg)
        sae.bind_context(ctx_for(cfg, "c"))
        sae.begin_edge_sensing_request("c")
        sae._sense_edges(torch.rand(4096, D_IN) + 1.0)
        assert sae._edge_truncated is True
        assert sae._edge_saturation_warned is True

    def test_a_pass_without_begin_senses_nothing(self):
        """Omitting the began-guard would buffer stale cross-request edges."""
        sae = real_sae()
        sae.arm_edge_sensing(config())
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._sensed_edges == []

    def test_collect_without_begin_returns_nothing(self):
        sae = real_sae()
        sae.arm_edge_sensing(config())
        assert sae.collect_sensed_edges() == ("", [], False)

    def test_the_rung_phrase_is_carried_verbatim(self):
        """The evidence guarantee: an observation never upgrades a rung."""
        s = spec(rung=0)
        s.rung_language = "associated"
        sae = armed(config(edges=[s]))
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        ev = sae._sensed_edges[0]
        assert ev.rung == 0 and ev.rung_language == "associated"
        assert "causal" not in ev.rung_language.lower()


# ─────────────────────────────────────────────────────────────────────────
# 3.2 — the advance/report split, pinned directly
#
# Task 3.2 collapsed a triplicated offset advance into one call above every
# guard. The first attempt also hoisted the ring PROGRESS report to the same
# place, which resurrected F15 R1-01: this layer's own advance pruned the ring
# before the downstream layer had read it, and cross-layer sensing went dark.
# test_F15R1_01 caught it, but only end-to-end. These pin the ordering itself,
# so the next refactor fails here — at the cause, not three layers away.
# ─────────────────────────────────────────────────────────────────────────


class TestPositionAdvancesBeforeGuardsProgressReportsAfterWork:

    def test_progress_is_NOT_reported_before_the_match_runs(self):
        """The load-bearing order. If progress were reported at advance time,
        the ring would prune to this layer's new position while an unread
        sibling still needed the older entries."""
        seen: list[str] = []
        sae = armed()
        ring = sae._edge_ring
        real_note = ring.note_layer_progress

        def spy(layer, pos):
            seen.append("progress")
            return real_note(layer, pos)

        ring.note_layer_progress = spy
        real_match = sae._match_edges

        def match_spy(*a, **kw):
            seen.append("match")
            return real_match(*a, **kw)

        sae._match_edges = match_spy
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))

        assert "match" in seen, "the matcher never ran; this test proves nothing"
        assert seen.index("match") < seen.index("progress"), (
            "progress was reported before matching — this is F15 R1-01, the "
            "defect that made cross-layer sensing go dark"
        )

    def test_a_suppressed_pass_still_reports_progress(self):
        """A suppressed layer that never reports leaves `_progress` under the
        ring's len<2 guard, so pruning never runs at all and the ring grows
        unbounded (EC-17.1)."""
        sae = armed()
        with sae.suppressed():
            sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_ring._progress.get(sae._edge_sensing.layer) == 3

    def test_position_advances_even_when_the_matcher_raises(self):
        """Position must survive a matcher failure, or this SAE silently
        desynchronises from its siblings on the SHARED absolute-position key."""
        sae = armed()

        def boom(*a, **kw):
            raise RuntimeError("matcher exploded")

        sae._match_edges = boom
        # R1-19: this used to wrap the call in `try/except RuntimeError`, so it
        # passed whether `_sense_edges` SWALLOWED the failure or PROPAGATED it.
        # Propagating would break generation itself — an observation path must
        # never do that — and the escape hatch made the contract untestable.
        # Calling it bare asserts the swallow.
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._edge_token_offset == 2
        # And the ring is left consistent: a pass that died mid-match must not
        # leave half-recorded upstream fires for positions it never finished,
        # or a sibling matches against an antecedent that was never fully
        # observed.
        assert sae._edge_ring._fires == {}, (
            f"partial state survived a failed pass: {dict(sae._edge_ring._fires)}"
        )

    def test_a_batched_pass_still_reports_progress(self):
        """The batched-pass bail returns from ABOVE the try, so it skips the
        `finally`. Found by execution during the task-3.2 extraction: the
        offset advanced to 5 while `_progress` stayed empty — the same EC-17.1
        stall as the suppressed path, surviving on a second path."""
        import torch
        sae = armed()
        layer = sae._edge_sensing.layer
        sae._sense_edges(torch.zeros(2, 5, sae._W_enc_e.shape[0]))
        assert sae._edge_token_offset == 5
        assert sae._edge_ring._progress.get(layer) == 5

    def test_a_pass_with_no_fires_still_reports_progress(self):
        """The quiet path is the common one in production — most passes fire
        nothing. If it did not report, pruning would stall on ordinary traffic
        rather than on an edge case."""
        sae = armed()
        layer = sae._edge_sensing.layer
        sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_ring._progress.get(layer) == 3


# ─────────────────────────────────────────────────────────────────────────
# R1-01 — the per-circuit budget, asserted THROUGH THE MATCHER
#
# EventBudget was written, unit-tested, and never wired: `try_spend` had no
# production supplier, so an N-layer circuit still emitted N x its cap. Its
# tests passed forever because they drove EventBudget directly instead of
# through a real LoadedSAE — asserting the mechanism EXISTS rather than that
# it is CALLED, which is the anti-pattern BR-005 forbids by name.
#
# These drive real SAEs through the real entry point. A test that touches
# EventBudget directly cannot catch the wiring coming undone; these can.
# ─────────────────────────────────────────────────────────────────────────


class TestThePerCircuitBudgetIsWiredNotJustDeclared:

    def _circuit(self, cap, layers=(10, 11, 12)):
        cfgs = {}
        for L in layers:
            c = config(
                edges=[spec(up_layer=L, down_layer=L, key=f"1@{L}->2@{L}")],
                max_lag=8, cap=cap, layer=L,
            )
            c.circuit_id = "circ_1"
            cfgs[L] = c
        ctx = ctx_for(cfgs[layers[0]], "r")
        saes = []
        for L, c in cfgs.items():
            s = real_sae()
            s.arm_edge_sensing(c)
            s.bind_context(ctx)
            s.begin_edge_sensing_request("r")
            saes.append(s)
        return ctx, saes

    def _fire(self, sae, pairs=10):
        rows = []
        for _ in range(pairs):
            rows += [{1: 2.0}, {2: 2.0}]
        sae._sense_edges(hidden(*rows))

    def test_an_N_layer_circuit_emits_at_most_the_CIRCUIT_cap(self):
        """The defect: cap 3 over three layers emitted NINE events, because the
        live cap was per-SAE. Measured before the fix; this pins the after."""
        ctx, saes = self._circuit(cap=3)
        for s in saes:
            self._fire(s)
        total = sum(len(s._sensed_edges) for s in saes)
        # R1-11: this asserted `total <= 3`, which `total == 0` satisfies. Break
        # the matcher completely — return early, null the ring, emit nothing —
        # and the test named for the budget stayed green while the headline
        # feature was dark. A cap test must bound BOTH sides, or it is a
        # liveness test that has been inverted into a silence test.
        assert total == 3, (
            f"{total} events against a per-circuit cap of 3 — "
            + ("the budget is not wired and each layer is spending its own"
               if total > 3 else
               "sensing produced less than the cap allows; the matcher is "
               "dark, not merely bounded")
        )
        # ...and prove the cap is what bound it: the same traffic uncapped
        # produces strictly more, so `== 3` is a limit and not a coincidence.
        loose_ctx, loose_saes = self._circuit(cap=1000)
        for s in loose_saes:
            self._fire(s)
        loose_total = sum(len(s._sensed_edges) for s in loose_saes)
        assert loose_total > 3, (
            f"uncapped traffic produced only {loose_total} events, so capping "
            "at 3 proves nothing about the budget"
        )

    def test_the_budget_actually_RECORDS_the_spend(self):
        """`spent` stayed 0 while nine events were emitted — the tell that
        nothing was calling try_spend."""
        ctx, saes = self._circuit(cap=3)
        for s in saes:
            self._fire(s)
        assert ctx.budget.spent("circ_1") == 3

    def test_a_budget_refusal_is_REPORTED_as_truncation(self):
        """Wiring the budget initially dropped events without setting
        `_edge_truncated`, because try_spend refuses before the per-SAE latch
        is reached. The drain then reported a clean, complete result while
        events were being discarded — silent-dark, reintroduced by the fix
        for silent-dark."""
        ctx, saes = self._circuit(cap=1)
        for s in saes:
            self._fire(s)
        starved = [s for s in saes if not s._sensed_edges]
        assert starved, "precondition: some layer must have been refused"
        assert all(s._edge_truncated for s in starved), (
            "a layer whose events were dropped reported itself complete"
        )

    def test_a_refused_layer_still_feeds_its_siblings(self):
        """R2-03/R3-02 through the budget path: refusal must CONTINUE, never
        return, or a starved layer blinds every layer downstream of it."""
        ctx, saes = self._circuit(cap=1)
        for s in saes:
            self._fire(s)
        ring = ctx.ring("circ_1", 8)
        assert any(ring._fires.get(f"1@{L}->2@{L}") for L in (10, 11, 12)), (
            "no upstream fires reached the ring — a refused layer returned "
            "instead of continuing"
        )

    def test_one_circuit_saturating_does_not_spend_another_s_budget(self):
        """FPRD §9 criterion 3 verbatim: a saturating circuit must not reduce
        another circuit's recorded observations."""
        ctx, saes = self._circuit(cap=2)
        for s in saes:
            self._fire(s)
        assert ctx.budget.spent("circ_1") == 2
        assert ctx.budget.spent("circ_OTHER") == 0, (
            "a second circuit's budget was consumed by the first"
        )
        assert ctx.budget.try_spend("circ_OTHER", 10) is True


class TestTruncationHasONESourceOfTruth:
    """R1-07/R1-09. Two mechanisms record truncation — the per-SAE
    `_edge_truncated` flag and `EventBudget` — and wiring the budget made them
    genuinely divergent: a SHED pass set the SAE flag while the budget stayed
    empty, because shedding drops events before the budget is ever consulted.

    `truncated_layers` happens to read the SAE flag today, so the API was
    right — but two sources of truth for one operator-facing honesty signal is
    a trap, and F19 (where the budget becomes the per-circuit authority) is
    where it would bite."""

    def _shed(self, layer=10):
        import torch
        c = config(edges=[spec()], max_lag=8, cap=20, layer=layer)
        c.circuit_id = "circ_1"
        # A pathologically low threshold fires on nearly every (pos, member):
        # 8192 fires against a budget of max(20*8, 2048) = 2048.
        c.thresholds = torch.tensor([0.0001, 0.0001])
        ctx = ctx_for(c, "r")
        sae = real_sae()
        sae.arm_edge_sensing(c)
        sae.bind_context(ctx)
        sae.begin_edge_sensing_request("r")
        sae._sense_edges(torch.rand(4096, D_IN) + 1.0)
        return ctx, sae

    def test_a_shed_reaches_the_circuit_budget_too(self):
        ctx, sae = self._shed()
        assert sae._edge_saturation_warned is True, "precondition: must shed"
        assert sae._edge_truncated is True
        assert ctx.budget.truncated_layers("circ_1") == [10], (
            "the SAE recorded truncation and the circuit budget did not — two "
            "sources of truth disagreeing about whether data was lost"
        )

    def test_truncation_names_the_SHEDDING_layer_not_an_edge_endpoint(self):
        """A cross-layer edge's `down_layer` can name a layer this SAE does not
        own and that may not be armed at all — the R1-04 defect (status naming
        an uncontained layer) reached through the budget."""
        s = spec(up_layer=42, down_layer=99, key="1@42->2@99")
        c = config(edges=[s], max_lag=8, cap=1, layer=42)
        c.circuit_id = "circ_1"
        ctx = ctx_for(c, "r")
        sae = real_sae()
        sae.arm_edge_sensing(c)
        sae.bind_context(ctx)
        sae.begin_edge_sensing_request("r")
        rows = []
        for _ in range(6):
            rows += [{1: 2.0}, {2: 2.0}]
        sae._sense_edges(hidden(*rows))
        named = ctx.budget.truncated_layers("circ_1")
        assert named == [42], (
            f"named {named} — 99 is the edge's downstream endpoint, not the "
            "layer that shed; this SAE is armed on 42"
        )


class TestUncoveredPathsThatBehaveCorrectly:
    """R1-17/18. Three paths with correct behaviour and NO coverage. Nothing is
    fixed here — these pin behaviour a refactor could silently break, which is
    the whole reason the characterization gate exists."""

    def test_a_batch_of_exactly_one_senses_normally(self):
        """`dim()==3, shape[0]==1` is the intended SERIAL path — it must fall
        through to `x = hidden_states[0]`, not trip the batched-pass bail. Only
        `shape[0] > 1` was ever tested, so a guard changed to `>= 1` would take
        every request dark while looking like a tightening."""
        import torch
        sae = armed()
        h = torch.zeros(1, 2, D_IN)
        h[0, 0, 1] = 2.0
        h[0, 1, 2] = 2.0
        sae._sense_edges(h)
        assert sae._edge_batch_warned is False, "the serial path was refused"
        assert len(sae._sensed_edges) == 1
        assert sae._edge_token_offset == 2

    def test_to_device_during_an_OPEN_request_keeps_the_boundary(self):
        """A migration mid-request must not detach the context or rewind the
        position — either would silently desynchronise this SAE from its
        siblings on the shared absolute-position key."""
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        ctx_before = sae._edge_ctx
        offset_before = sae._edge_token_offset

        sae.to_device("cpu")

        assert sae._edge_ctx is ctx_before, "the migration dropped the context"
        assert sae._edge_token_offset == offset_before, "the position rewound"
        assert sae._edge_began is True, "the boundary was closed by a migration"

    def test_sensing_continues_across_a_migration(self):
        """The stronger claim: it still WORKS afterwards, not merely that the
        fields survived."""
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        first = len(sae._sensed_edges)
        sae.to_device("cpu")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) > first, "sensing went dark after a move"
        assert sae._edge_token_offset == 4, "positions did not stay absolute"

    def test_a_re_run_pass_double_counts_positions_which_is_why_speculative_is_excluded(self):
        """R1-20. Documents WHY speculative decoding must be excluded, rather
        than leaving the reason in a comment nobody can test.

        `_advance_edge_position` is monotonic: it has no notion of a rejected
        token, so re-running a pass over the same tokens advances the offset
        AGAIN. Under speculative decoding a verification pass advances by a
        whole candidate block and rejected tokens re-run, so two layers that
        see different acceptance counts would silently disagree about absolute
        position — the shared-by-construction invariant that replaced
        `ctx.advance()` depends on every layer counting the same tokens once.

        If this ever becomes false — if the offset learns to roll back — the
        exclusion in `_circuit_sensing_begin` can be revisited. Until then this
        pins the reason."""
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._edge_token_offset == 2
        # The SAME two tokens again, as a rejected-then-re-run block would be.
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._edge_token_offset == 4, (
            "the offset rolled back — if re-runs are now handled, revisit the "
            "speculative-decoding exclusion in _circuit_sensing_begin"
        )


class TestR2ASiblingsSpendingNeverStarvesAQuietLayer:
    """F17 R2-03. R1-05 routed budget refusals through `on_cap`, which LATCHES
    `_edge_done` — correct for a layer's own cap, wrong for the shared circuit
    budget. Measured with a circuit cap of 4 across two layers:

        pass 1 (only L10 busy): L10 4 events | L11 0 events, done=False
        pass 2 (L11 now fires): L11 0 events, done=TRUE   <- latched
        pass 3:                 L11 0 events              <- dark all request

    Layer 11 recorded NOTHING because a SIBLING spent the budget — the
    R2-03/R3-02 starvation this codebase has already fixed twice, arriving a
    third time through the budget path.

    R1's `test_a_refused_layer_still_feeds_its_siblings` missed it because it
    fires every layer once in a single even pass; the latch needs a SECOND pass
    to bite. The interaction surface is where these defects live."""

    def _circuit(self, cap=4, layers=(10, 11)):
        cfgs = {}
        for L in layers:
            c = config(
                edges=[spec(up_layer=L, down_layer=L, key=f"1@{L}->2@{L}")],
                max_lag=8, cap=cap, layer=L,
            )
            c.circuit_id = "circ_1"
            cfgs[L] = c
        ctx = ctx_for(cfgs[layers[0]], "r")
        saes = {}
        for L, c in cfgs.items():
            s = real_sae()
            s.arm_edge_sensing(c)
            s.bind_context(ctx)
            s.begin_edge_sensing_request("r")
            saes[L] = s
        return ctx, saes

    def _busy(self, pairs=6):
        rows = []
        for _ in range(pairs):
            rows += [{1: 2.0}, {2: 2.0}]
        return hidden(*rows)

    def test_a_quiet_layer_is_not_latched_by_a_busy_sibling(self):
        ctx, saes = self._circuit()
        saes[10]._sense_edges(self._busy())          # spends the circuit budget
        saes[11]._sense_edges(hidden(*[{}] * 12))    # quiet pass
        saes[11]._sense_edges(self._busy())          # NOW it fires
        assert saes[11]._edge_done is False, (
            "a sibling's spending latched this layer for the rest of the "
            "request — R2-03 starvation through the budget path"
        )

    def test_a_starved_layer_still_reports_truncation(self):
        """It must not go quietly dark either: refused events are lost data."""
        ctx, saes = self._circuit()
        saes[10]._sense_edges(self._busy())
        saes[11]._sense_edges(self._busy())
        assert saes[11]._edge_truncated is True

    def test_a_layer_hitting_its_OWN_cap_still_latches(self):
        """The per-SAE latch is a real optimisation and must survive: reaching
        your own cap means you genuinely are done for this request."""
        ctx, saes = self._circuit(cap=2, layers=(10,))
        ctx.budget.cap = 10_000                      # the circuit is not binding
        saes[10]._sense_edges(self._busy())
        assert saes[10]._edge_done is True
        assert len(saes[10]._sensed_edges) == 2

    def test_both_cap_paths_tell_the_circuit_budget(self):
        """R1-07's one-source-of-truth rule survives the reordering: whichever
        cap fires, `truncated_layers` learns about it."""
        ctx, saes = self._circuit(cap=2, layers=(10,))
        saes[10]._sense_edges(self._busy())
        assert ctx.budget.truncated_layers("circ_1") == [10]
