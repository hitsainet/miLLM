"""
Feature 11 Task 2.5: LoadedSAE sensing-core unit tests — predicate matrix,
epsilon-fallback mode, debounced spans (incl. cross-pass tail merge), the
per-request cap, offset/phase accounting for real pass shapes, suppressed()
respect, arm idempotence, and buffer hygiene.
"""

import torch
import pytest

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE, SensedHit, SensingConfig

D_IN = 8
D_SAE = 32


def make_sae() -> LoadedSAE:
    """Deterministic SAE: encoder column j responds 1:1 to input dim j%d_in.

    With b_enc=0 and W_enc one-hot, feature j's activation at a position is
    exactly the input value at dim j % D_IN — tests can dial activations
    precisely by constructing hidden states.
    """
    W_enc = torch.zeros(D_IN, D_SAE)
    for j in range(D_SAE):
        W_enc[j % D_IN, j] = 1.0
    config = SAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        model_name="test",
        hook_name="test",
        hook_layer=1,
    )
    return LoadedSAE(
        W_enc=W_enc,
        b_enc=torch.zeros(D_SAE),
        W_dec=torch.zeros(D_SAE, D_IN),
        b_dec=torch.zeros(D_IN),
        config=config,
        device="cpu",
    )


def make_config(members=(0, 1, 2), thresholds=(1.0, 1.0, 1.0), min_k=2,
                mode="epsilon_max", cap=20, context_tokens=16):
    return SensingConfig(
        profile_id="prof_sense01",
        member_indices=list(members),
        thresholds=torch.tensor(list(thresholds)),
        threshold_mode=mode,
        min_k=min_k,
        context_tokens=context_tokens,
        max_events_per_request=cap,
    )


def hidden(rows: list[list[float]]) -> torch.Tensor:
    """(1, seq, d_in) hidden states from per-position d_in rows."""
    return torch.tensor(rows).unsqueeze(0)


def pos_row(active: dict[int, float]) -> list[float]:
    """One position where input dim d carries value v (feature j==d fires
    at v for members < D_IN)."""
    row = [0.0] * D_IN
    for d, v in active.items():
        row[d] = v
    return row


@pytest.fixture
def sae():
    return make_sae()


class TestPredicate:
    def test_quorum_fires_event(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        # members 0,1 above theta=1.0 at position 1; member 2 quiet
        sae._sense(hidden([pos_row({}), pos_row({0: 2.0, 1: 3.0})]))
        _, hits, truncated = sae.collect_sensing_hits()
        assert truncated is False
        assert len(hits) == 1
        assert hits[0].pos_start == hits[0].pos_end == 1
        assert hits[0].fired_count == 2
        assert dict(hits[0].fired) == {0: 2.0, 1: 3.0}
        assert hits[0].score == pytest.approx(3.0)  # max(act/theta)

    def test_below_quorum_no_event(self, sae):
        sae.arm_sensing(make_config(min_k=3))
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 2.0, 1: 3.0})]))  # only 2 fire
        _, hits, _ = sae.collect_sensing_hits()
        assert hits == []

    def test_at_threshold_does_not_fire(self, sae):
        """Fired means act > theta, strictly."""
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 1.0, 1: 1.0})]))
        _, hits, _ = sae.collect_sensing_hits()
        assert hits == []

    def test_per_member_thresholds(self, sae):
        sae.arm_sensing(make_config(thresholds=(0.5, 4.0, 1.0)))
        sae.begin_sensing_request("req-1")
        # member 0 (theta .5) fires at 1.0; member 1 (theta 4) does NOT at
        # 3.0; member 2 (theta 1) fires at 1.5
        sae._sense(hidden([pos_row({0: 1.0, 1: 3.0, 2: 1.5})]))
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1
        assert sorted(dict(hits[0].fired)) == [0, 2]

    def test_zero_floor_score_guard(self, sae):
        """floor_only mode with theta=0: score must not divide by zero."""
        sae.arm_sensing(make_config(thresholds=(0.0, 0.0, 0.0),
                                    mode="floor_only"))
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 2.0, 1: 1.0})]))
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(2.0)  # falls back to raw act


class TestSpans:
    def test_consecutive_positions_debounce_to_one_span(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot, hot, hot, pos_row({}), hot]))
        _, hits, _ = sae.collect_sensing_hits()
        assert [(h.pos_start, h.pos_end) for h in hits] == [(0, 2), (4, 4)]

    def test_cross_pass_tail_merge(self, sae):
        """FTID pitfall 3: a span continuing into the next decode pass must
        extend the buffer tail, not open a new event."""
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([pos_row({}), pos_row({}), hot]))  # prefill, pos 2
        sae._sense(hidden([hot]))                            # decode, pos 3
        sae._sense(hidden([hot]))                            # decode, pos 4
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1
        assert (hits[0].pos_start, hits[0].pos_end) == (2, 4)
        assert hits[0].phase == "prefill"  # span started during prefill

    def test_span_merge_keeps_peak_activations(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 2.0, 1: 2.0})]))
        sae._sense(hidden([pos_row({0: 5.0, 1: 1.5})]))  # pos 1 continues
        _, hits, _ = sae.collect_sensing_hits()
        assert dict(hits[0].fired)[0] == 5.0  # running peak
        assert hits[0].score == pytest.approx(5.0)

    def test_cap_truncates_and_stops(self, sae):
        sae.arm_sensing(make_config(cap=2))
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        cold = pos_row({})
        sae._sense(hidden([hot, cold, hot, cold, hot, cold, hot]))
        _, hits, truncated = sae.collect_sensing_hits()
        assert len(hits) == 2
        assert truncated is True

    def test_offsets_across_pass_shapes(self, sae):
        """Prefill (seq=N), decode (seq=1), speculative verify (seq=k+1)
        all advance the absolute offset by their own length."""
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([pos_row({}), pos_row({}), pos_row({})]))  # 0-2
        sae._sense(hidden([pos_row({})]))                            # 3
        sae._sense(hidden([pos_row({}), hot, pos_row({})]))          # 4-6
        _, hits, _ = sae.collect_sensing_hits()
        assert [(h.pos_start, h.pos_end) for h in hits] == [(5, 5)]
        assert hits[0].phase == "decode"

    def test_phase_prefill_then_decode(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot, pos_row({})]))  # prefill
        sae._sense(hidden([pos_row({}), hot]))  # decode (non-adjacent)
        _, hits, _ = sae.collect_sensing_hits()
        assert [h.phase for h in hits] == ["prefill", "decode"]


class TestHygieneAndLifecycle:
    def test_no_begin_yields_empty_collect(self, sae):
        """FTID pitfall 1: a missed begin must never leak prior hits."""
        sae.arm_sensing(make_config())
        sae._sense(hidden([pos_row({0: 2.0, 1: 2.0})]))
        assert sae.collect_sensing_hits() == ("", [], False)

    def test_begin_resets_prior_request_state(self, sae):
        sae.arm_sensing(make_config(cap=1))
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot, pos_row({}), hot]))  # hits cap
        sae.begin_sensing_request("req-2")           # NO collect in between
        sae._sense(hidden([hot]))
        request_id, hits, truncated = sae.collect_sensing_hits()
        assert request_id == "req-2"
        assert len(hits) == 1 and hits[0].pos_start == 0
        assert truncated is False

    def test_collect_closes_the_boundary(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot]))
        rid, hits, _ = sae.collect_sensing_hits()
        assert rid == "req-1" and len(hits) == 1
        # a stray pass after collect must not resurrect the request
        sae._sense(hidden([hot]))
        assert sae.collect_sensing_hits() == ("", [], False)

    def test_suppressed_passes_do_not_sense_or_advance(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        with sae.suppressed():
            sae._sense(hidden([hot, hot, hot]))  # e.g. embeddings pass
        sae._sense(hidden([hot]))
        _, hits, _ = sae.collect_sensing_hits()
        assert [(h.pos_start, h.pos_end) for h in hits] == [(0, 0)]

    def test_arm_is_idempotent_and_replaces(self, sae):
        sae.arm_sensing(make_config(members=(0, 1, 2)))
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 2.0, 1: 2.0})]))
        sae.arm_sensing(make_config(members=(3, 4, 5)))  # re-arm clears
        assert sae.collect_sensing_hits() == ("", [], False)
        assert sae.is_sensing_armed

    def test_disarm_clears_everything(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae.disarm_sensing()
        assert not sae.is_sensing_armed
        sae._sense(hidden([pos_row({0: 2.0, 1: 2.0})]))  # no-op, no raise
        assert sae.collect_sensing_hits() == ("", [], False)

    def test_unarmed_sense_is_inert(self, sae):
        sae._sense(hidden([pos_row({0: 2.0})]))  # must not raise
        assert not sae.is_sensing_armed

    def test_overhead_accumulates(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({})]))
        assert sae._sensing_overhead_ms > 0.0

    def test_dtype_cast_matches_encode(self, sae):
        """fp16 hidden states against fp32 SAE weights must cast, not raise
        (FTID pitfall 2)."""
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        x = hidden([pos_row({0: 2.0, 1: 2.0})]).to(torch.float16)
        sae._sense(x)
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1


class TestHookIntegration:
    def test_hook_senses_pre_steer_values(self, sae):
        """The REAL hook_fn (sae_hooker) senses the pre-steer hidden states
        even with steering active, and returns the steered output."""
        from millm.ml.sae_hooker import SAEHooker

        # W_dec row 0 must be nonzero for steering to change the output
        sae.W_dec[0, :] = 1.0
        sae.set_steering_batch({0: 100.0})
        sae.enable_steering(True)
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")

        hooker = SAEHooker()
        hook_fn = hooker._create_hook_fn(sae)
        x = hidden([pos_row({0: 2.0, 1: 2.0})])
        steered = hook_fn(None, None, x)

        assert not torch.equal(steered, x)  # steering applied
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1
        # sensed activation is the PRE-steer value
        assert dict(hits[0].fired)[0] == pytest.approx(2.0)

    def test_unarmed_hook_zero_extra_work(self, sae):
        """Un-armed cost is one boolean — no sensing state is touched."""
        from millm.ml.sae_hooker import SAEHooker

        hooker = SAEHooker()
        hook_fn = hooker._create_hook_fn(sae)
        x = hidden([pos_row({0: 2.0})])
        hook_fn(None, None, x)
        assert sae._sensing_overhead_ms == 0.0


class TestRound3MutationPins:
    def test_peak_keeps_max_when_later_position_is_lower(self, sae):
        """R3 mutation pin: merged[idx] must be the running MAX — a plain
        assignment survives when only the increasing member is asserted."""
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 5.0, 1: 2.0})]))
        sae._sense(hidden([pos_row({0: 2.0, 1: 2.0})]))  # pos 1: member 0 LOWER
        _, hits, _ = sae.collect_sensing_hits()
        assert dict(hits[0].fired)[0] == 5.0  # peak retained

    def test_post_cap_adjacent_positions_do_not_resurrect(self, sae):
        """R3 mutation pin: _sensing_done must stop sensing even for
        positions adjacent to the buffer tail."""
        sae.arm_sensing(make_config(cap=1))
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot, pos_row({}), hot]))  # span 0, cap at pos 2
        sae._sense(hidden([hot]))  # pos 3 — would tail-merge if not done
        _, hits, truncated = sae.collect_sensing_hits()
        assert truncated is True
        assert [(h.pos_start, h.pos_end) for h in hits] == [(0, 0)]

    def test_merged_fired_count_is_union_size(self, sae):
        """R3 mutation pin: fired_count == len(fired_members) after merges
        across positions firing DIFFERENT members."""
        sae.arm_sensing(make_config(members=(0, 1, 2, 3), min_k=2,
                                    thresholds=(1.0, 1.0, 1.0, 1.0)))
        sae.begin_sensing_request("req-1")
        sae._sense(hidden([pos_row({0: 2.0, 1: 2.0})]))
        sae._sense(hidden([pos_row({2: 2.0, 3: 2.0})]))  # pos 1, other members
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1
        assert hits[0].fired_count == 4 == len(hits[0].fired)


class TestHistoryDedup:
    """Goal item 2: positions below the report-from boundary are re-read
    history and must not create events."""

    def test_boundary_suppresses_history_positions(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-2")
        sae.set_sensing_report_from(3)
        hot = pos_row({0: 2.0, 1: 2.0})
        # positions 0-2 are re-read history; 3-4 are new
        sae._sense(hidden([hot, hot, hot, hot, pos_row({})]))
        _, hits, _ = sae.collect_sensing_hits()
        assert [(h.pos_start, h.pos_end) for h in hits] == [(3, 3)]

    def test_boundary_resets_on_begin(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae.set_sensing_report_from(100)
        sae.begin_sensing_request("req-2")  # fresh request: boundary 0
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot]))
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1

    def test_boundary_ignored_without_begin(self, sae):
        sae.arm_sensing(make_config())
        sae.set_sensing_report_from(5)  # no open boundary — no-op
        sae.begin_sensing_request("req-1")
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot]))
        _, hits, _ = sae.collect_sensing_hits()
        assert len(hits) == 1

    def test_offsets_still_advance_through_suppressed_passes(self, sae):
        sae.arm_sensing(make_config())
        sae.begin_sensing_request("req-1")
        sae.set_sensing_report_from(2)
        hot = pos_row({0: 2.0, 1: 2.0})
        sae._sense(hidden([hot, hot]))   # both suppressed (0,1)
        sae._sense(hidden([hot]))        # pos 2: new -> event
        _, hits, _ = sae.collect_sensing_hits()
        assert [(h.pos_start, h.pos_end) for h in hits] == [(2, 2)]
