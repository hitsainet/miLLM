"""Feature 15 Task 3.8: CircuitSensingService.

The interesting behaviour is REFUSAL. An edge is sensable only when both
endpoints resolve to an attached SAE with a usable threshold; every edge that
fails must be reported with a reason rather than silently dropped, because a
missing event is indistinguishable from "the edge never fired" — and that
silence would read as evidence of absence.
"""

from types import SimpleNamespace

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.circuit_sensing_service import (
    CircuitSensingService,
    edge_key_for,
)

D_IN = 8
D_SAE = 32


def make_sae() -> LoadedSAE:
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


def node(layer, feature_idx, kind="feature"):
    return SimpleNamespace(layer=layer, feature_idx=feature_idx, kind=kind)


def edge(up_layer=10, up_idx=1, down_layer=13, down_idx=2, rung=2, type_="computed"):
    return SimpleNamespace(
        up=node(up_layer, up_idx),
        down=node(down_layer, down_idx),
        rung=rung,
        type=type_,
    )


def member(layer, feature_idx, max_activation=10.0):
    return SimpleNamespace(
        layer=layer,
        feature=SimpleNamespace(feature_idx=feature_idx, max_activation=max_activation),
        expanded_members=None,
    )


def definition(edges=None, members=None):
    return SimpleNamespace(
        edges=edges if edges is not None else [edge()],
        members=members
        if members is not None
        else [member(10, 1), member(13, 2)],
    )


def circuit(**overrides):
    base = dict(id="circ_1", name="fear→threat", circuit_meta={})
    base.update(overrides)
    return SimpleNamespace(**base)


def two_saes():
    return {10: make_sae(), 13: make_sae()}


class TestSensableEdgeResolution:
    def test_a_fully_resolved_edge_arms_both_layers(self):
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(circuit(), definition(), two_saes())
        assert unsensable == []
        assert svc.is_armed and svc._armed_layers == [10, 13]

    def test_each_sae_gets_only_its_own_half_of_the_edge(self):
        """A cross-layer edge is sensed cooperatively; a column offset that
        belonged to the other layer's slice would read the wrong feature."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)

        up_spec = svc._configs[10].edges[0]
        assert up_spec.up_col >= 0 and up_spec.down_col == -1
        down_spec = svc._configs[13].edges[0]
        assert down_spec.up_col == -1 and down_spec.down_col >= 0

    def test_the_rung_phrase_comes_from_the_ladder(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(), definition(edges=[edge(rung=0)]), two_saes()
        )
        spec = svc._configs[10].edges[0]
        assert spec.rung == 0
        assert spec.rung_language == "associated"
        assert "causal" not in spec.rung_language.lower()


class TestUnsensableEdges:
    def test_an_unattached_layer_is_reported_not_dropped(self):
        """EC-15.4: slice-fallback serves ONE layer, so nearly every
        cross-layer edge lands here."""
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(
            circuit(), definition(), {10: make_sae()}  # layer 13 missing
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "layer_not_attached"
        assert "13" in unsensable[0].detail
        assert svc.is_armed is False, "nothing sensable ⇒ not armed"

    def test_a_cluster_supernode_endpoint_is_unsensable(self):
        """feature_idx is nullable and kind may be 'cluster' — there is no
        single activation to threshold. Neither EC-15.4 nor EC-15.6 covers it."""
        svc = CircuitSensingService()
        e = edge()
        e.up = node(10, None, kind="cluster")
        unsensable = svc.arm_for_circuit(circuit(), definition(edges=[e]), two_saes())
        assert len(unsensable) == 1
        assert unsensable[0].reason == "endpoint_not_a_feature"

    def test_a_member_without_max_activation_is_unsensable(self):
        """EC-15.6: no activation scale means theta is either 0 (fires on
        anything) or inf (never fires). Both make the edge unobservable."""
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(
            circuit(),
            definition(members=[member(10, 1, None), member(13, 2)]),
            two_saes(),
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "no_activation_threshold"

    def test_a_zero_max_activation_is_treated_as_missing(self):
        """theta = eps*0 = 0 would fire on any positive activation — as
        degenerate as a missing stat (011 R3 #2)."""
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(
            circuit(),
            definition(members=[member(10, 1, 0.0), member(13, 2)]),
            two_saes(),
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "no_activation_threshold"

    def test_sensable_and_unsensable_edges_coexist(self):
        svc = CircuitSensingService()
        good = edge(up_idx=1, down_idx=2)
        bad = edge(up_idx=3, down_idx=4, down_layer=99)
        unsensable = svc.arm_for_circuit(
            circuit(),
            definition(
                edges=[good, bad],
                members=[member(10, 1), member(13, 2), member(10, 3), member(99, 4)],
            ),
            two_saes(),
        )
        assert len(unsensable) == 1
        assert svc.is_armed is True
        assert svc.status()["sensable_edges"] == 1

    def test_unsensable_edges_are_surfaced_in_status(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(circuit(), definition(), {10: make_sae()})
        rows = svc.status()["unsensable_edges"]
        assert rows and rows[0]["reason"] == "layer_not_attached"
        assert "edge_key" in rows[0] and "detail" in rows[0]


class TestThresholds:
    def test_theta_is_epsilon_times_max_activation(self):
        svc = CircuitSensingService()
        assert svc._theta(10.0, 0.1, 0.0) == pytest.approx(1.0)

    def test_the_floor_wins_when_higher(self):
        svc = CircuitSensingService()
        assert svc._theta(1.0, 0.1, 0.5) == pytest.approx(0.5)

    def test_a_missing_stat_with_no_floor_is_infinite_not_zero(self):
        """Infinity means 'never fires'. Zero would mean 'always fires' —
        the wrong direction to fail in."""
        svc = CircuitSensingService()
        assert svc._theta(None, 0.1, 0.0) == float("inf")

    def test_a_missing_stat_with_a_positive_floor_uses_the_floor(self):
        svc = CircuitSensingService()
        assert svc._theta(None, 0.1, 0.25) == pytest.approx(0.25)


class TestOverrides:
    def test_an_authored_lag_override_is_honoured(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 3}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 3

    def test_a_negative_override_degrades_to_the_default(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": -5}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 8

    def test_an_absurd_lag_is_capped(self):
        """A window wide enough to catch anything is a coincidence detector,
        not an attribution."""
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 100_000}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 64

    def test_local_overrides_beat_document_overrides(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(
                circuit_meta={
                    "sensing": {"max_token_lag": 3},
                    "sensing_overrides": {"max_token_lag": 5},
                }
            ),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 5


class TestLifecycle:
    def test_begin_gives_the_request_a_ring_nothing_else_can_reach(self):
        """F17 (CTX-V2 behaviour change, justified): the old contract was
        "begin CLEARS the shared ring, once, for the whole circuit" — a rule
        that had to be obeyed by every participant and silently corrupted the
        request when one forgot. Clearing is now structurally unnecessary: the
        ring belongs to the request context, so a previous request's fires are
        unreachable rather than merely cleared.

        The assertion is strictly stronger. Stale state cannot survive begin
        because there is no shared object for it to survive in."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)

        assert svc.begin_request("req-1", saes) is True
        first = svc._ctx
        first.ring("circ_1", 8).record_up("stale", 1, 1.0)
        svc.close_request()

        # A second request must not see the first's fires.
        assert svc.begin_request("req-2", saes) is True
        assert svc._ctx is not first, "each request gets its own context"
        assert svc._ctx.ring("circ_1", 8).match_down("stale", 2) is None
        assert all(s._edge_began for s in saes.values())

    def test_collect_merges_every_layer_and_sums_overhead(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("req-1", saes)
        for s in saes.values():
            s._edge_overhead_ms = 2.0

        request_id, edges, truncated = svc.collect_edges(saes)
        assert request_id == "req-1"
        assert svc._last_request_overhead_ms == pytest.approx(4.0), "summed, not assigned"

    def test_disarm_releases_every_layer(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.disarm(saes)
        assert svc.is_armed is False
        assert not any(s.is_edge_sensing_armed for s in saes.values())

    def test_status_reconciles_a_swallowed_disarm(self):
        """Reporting armed forever after a disarm that failed is the exact
        failure F11's reconciliation exists to prevent."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        saes[13].disarm_edge_sensing()  # one layer lost behind the service's back

        assert svc.status(saes)["armed"] is False

    def test_begin_on_an_unarmed_service_is_a_noop(self):
        svc = CircuitSensingService()
        assert svc.begin_request("req-1", two_saes()) is False


class TestSummary:
    def test_the_summary_carries_the_rung_phrase_verbatim(self):
        svc = CircuitSensingService()
        ev = SimpleNamespace(
            up_feature_idx=1, up_layer=10, down_feature_idx=2, down_layer=13,
            token_lag=2, rung_language="associated",
        )
        text = svc.summarize(ev)
        assert "associated" in text
        assert "causal" not in text.lower()

    def test_a_rung_two_summary_names_the_ladder_phrase(self):
        svc = CircuitSensingService()
        ev = SimpleNamespace(
            up_feature_idx=1, up_layer=10, down_feature_idx=2, down_layer=13,
            token_lag=1, rung_language="causally validated (edge)",
        )
        assert "causally validated (edge)" in svc.summarize(ev)

    def test_the_summary_fits_the_column(self):
        svc = CircuitSensingService()
        ev = SimpleNamespace(
            up_feature_idx=1, up_layer=10, down_feature_idx=2, down_layer=13,
            token_lag=1, rung_language="x" * 500,
        )
        assert len(svc.summarize(ev)) <= 300


class TestEdgeKey:
    def test_the_key_is_stable_and_directional(self):
        assert edge_key_for(10, 1, 13, 2) == "1@10->2@13"
        assert edge_key_for(13, 2, 10, 1) != edge_key_for(10, 1, 13, 2)


class TestRequestIdentityIsSnapshotted:
    """R2 CRITICAL (R1 deferred C): record() read self._circuit_id at DRAIN
    time, so a re-arm between begin and flush persisted circuit A's
    observations under circuit B's id — confidently wrong data, not merely
    lost sensing."""

    def test_the_boundary_remembers_which_circuit_opened_it(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(id="circ_A"), definition(), saes)
        svc.begin_request("req-1", saes)
        assert svc._request_circuit_id == "circ_A"

        # An operator re-arms a different circuit mid-request. R3: arming now
        # disarms the prior set first, which also RELEASES the boundary — so
        # the correct outcome is that the stale snapshot is gone, not that it
        # survives. A drain arriving after this must attribute nothing rather
        # than attribute circuit A's edges to circuit B.
        svc.arm_for_circuit(circuit(id="circ_B"), definition(), two_saes())
        assert svc._circuit_id == "circ_B"
        assert svc._request_circuit_id is None, (
            "a released boundary must not keep attributing to a stale circuit"
        )

    def test_a_drain_after_disarm_attributes_nothing(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(id="circ_A"), definition(), saes)
        svc.begin_request("req-1", saes)
        svc.disarm(saes)
        assert svc._request_circuit_id is None

    def test_close_request_releases_the_snapshot(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(id="circ_A"), definition(), saes)
        svc.begin_request("req-1", saes)
        svc.collect_edges(saes)
        svc.close_request()
        assert svc._request_circuit_id is None


class TestLagWindowDoesNotLeakBetweenCircuits:
    def test_a_failed_arm_does_not_change_the_reported_lag(self):
        """R2: build_configs assigned _max_token_lag BEFORE arming could fail,
        so a circuit that never armed still changed the reported value — and
        the next EdgeFireRing was built from it."""
        svc = CircuitSensingService()
        before = svc._max_token_lag
        # No layer attached ⇒ nothing sensable ⇒ arming bails.
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 55}}),
            definition(),
            {},
        )
        assert svc.is_armed is False
        assert svc._max_token_lag == before

    def test_disarm_restores_the_configured_default(self):
        """R2: every other field was cleared but this one, so a circuit with
        no override silently inherited the previous circuit's window."""
        svc = CircuitSensingService()
        default = svc._max_token_lag
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 3}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 3
        svc.disarm(two_saes())
        assert svc._max_token_lag == default


class TestTheDeadPruneTrioIsGone:
    """F17 task 3.5. `prune_ring`, `safe_prune_boundary` and
    `prune_between_passes` were R2's request-level pruning design and had ZERO
    production callers — R2 fixed R1's "declared a mechanism and never wired
    it" finding by declaring a mechanism and never wiring it.

    The tests that used to live here were named `TestRingPruningIsWired` while
    asserting only that the entry points EXISTED, which is the precise
    anti-pattern BR-005 now forbids. R3 superseded the whole design with the
    ring tracking layer progress itself, so the trio is deleted rather than
    carried alongside the live mechanism."""

    def test_the_superseded_entry_points_are_removed(self):
        svc = CircuitSensingService()
        for name in ("prune_ring", "safe_prune_boundary", "prune_between_passes"):
            assert not hasattr(svc, name), (
                f"{name} is R2's superseded design; carrying two pruning "
                "mechanisms is how the next reader picks the wrong one"
            )

    def test_pruning_still_happens_via_layer_progress(self):
        """The live mechanism: the ring prunes to the SLOWEST layer itself, so
        no caller has to know about siblings — which is exactly why this design
        was wireable when the previous two were not."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        ring = svc._ctx.ring("circ_1", 8)
        ring.record_up("e", 0, 1.0)

        ring.note_layer_progress(10, 1000)
        ring.note_layer_progress(13, 900)
        assert ring.match_down("e", 1001) is None

    def test_a_lagging_layer_still_holds_the_boundary_back(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 4}}),
            definition(),
            saes,
        )
        svc.begin_request("r", saes)
        ring = svc._ctx.ring("circ_1", 4)
        ring.record_up("e", 38, 1.0)
        ring.note_layer_progress(10, 5000)
        ring.note_layer_progress(13, 40)
        assert ring.match_down("e", 41) == (38, 1.0), (
            "pruned past a fire the lagging layer still needed"
        )


class TestEmitKeepsTheMostRecent:
    def test_the_flush_cap_keeps_the_newest_events(self):
        """R2: this kept the FIRST 5, and collect_edges sorts by down_pos — so
        a live panel always showed a request's EARLIEST edges and never its
        most recent."""
        svc = CircuitSensingService()
        sent = []
        import millm.sockets.progress as progress

        original = progress.progress_emitter.emit_circuit_sensing_event
        progress.progress_emitter.emit_circuit_sensing_event = sent.append
        try:
            svc._emit([{"id": i} for i in range(12)])
        finally:
            progress.progress_emitter.emit_circuit_sensing_event = original

        assert [p["id"] for p in sent] == [7, 8, 9, 10, 11]
        # R1-14: this asserted `_ws_dropped == 7`, PINNING the defect — it
        # required a healthy flush to report the 7 events the per-flush cap
        # declined as "dropped", which is a delivery-failure alarm on a system
        # that is working. The cap is a throttle; those events are persisted
        # and readable through the events API. Nothing was lost, so nothing is
        # counted.
        assert svc._ws_dropped == 0


class TestWebSocketPayloadCarriesNoPromptText:
    """R3 mutation finding: flipping the WS broadcast to include_context=True
    was caught by NO test — 135/135 stayed green. R1 recorded "privacy holds"
    under *verified clean*, but verified it by READING, not by pinning. One
    word breaks the manual's entire Privacy promise undetectably."""

    def test_the_broadcast_omits_every_context_field(self):
        from millm.db.models.circuit_edge_sensing_event import (
            CircuitEdgeSensingEvent,
        )

        row = CircuitEdgeSensingEvent(
            id=1, circuit_id="c", request_id="r", phase="decode",
            edge_key="1@10->2@13", up_layer=10, up_feature_idx=1, up_pos=1,
            up_act=1.0, down_layer=13, down_feature_idx=2, down_pos=2,
            down_act=1.0, token_lag=1, edge_rung=2,
            edge_rung_language="causally validated (edge)", summary="s",
            truncated=False,
            context_text="the user's private prompt",
            context_token_ids=[1, 2, 3],
            context_parts={"before": "a", "span": "b", "after": "c"},
        )
        svc = CircuitSensingService()
        sent = []
        import millm.sockets.progress as progress

        original = progress.progress_emitter.emit_circuit_sensing_event
        progress.progress_emitter.emit_circuit_sensing_event = sent.append
        try:
            svc._emit([row.to_dict(include_context=False)])
        finally:
            progress.progress_emitter.emit_circuit_sensing_event = original

        assert sent, "expected a broadcast"
        payload = sent[0]
        for key in ("context_text", "context_token_ids", "context_parts"):
            assert key not in payload, f"WS payload leaked {key}"
        blob = repr(payload)
        assert "private prompt" not in blob

    def test_record_broadcasts_the_context_free_shape(self):
        """Pins the call site itself, not just the serialiser: R3's mutation
        changed record()'s argument, which the serialiser test would miss.

        R1-16: this is a SOURCE GREP and was the ONLY coverage of the privacy
        guarantee — it passes if the substring appears in a comment and fails
        if someone writes `include_context = False` with spaces. It is kept
        because it is cheap and catches a call site removed entirely, but the
        real protection is now
        `TestR1RecordIsActuallyExercised::test_the_broadcast_carries_NO_prompt_text`,
        which drives `record()` and asserts the kwarg on the ACTUAL call —
        verified by mutation, where flipping it to True fails that test and
        this one too."""
        import inspect

        src = inspect.getsource(CircuitSensingService.record)
        assert "include_context=False" in src


class TestAmbientFiredCountHonoursTheContract:
    """`ambient_fired_count` is the alone-vs-within signal, defined by Feature
    11 and the millm_sensing_events MCP contract as the WHOLE-SAE fired count,
    populated ONLY when un-compacted monitoring co-ran and NULL otherwise —
    "never estimated".

    R3 populated it with the circuit's own armed-member fires, always non-null.
    That is a different quantity under the same field name: a reader comparing
    an F15 row against an F11 row would be comparing incompatible numbers, and
    a never-null value silently claims a denominator nobody measured."""

    class _Probe:
        """Minimal stand-in: is_monitoring_enabled is a read-only property on
        the real LoadedSAE, so the monitoring states are modelled here."""

        def __init__(self, enabled=False, subset=None, acts=None, raises=False):
            self.is_monitoring_enabled = enabled
            self._monitored_features = subset
            self._acts = acts
            self._raises = raises

        def get_feature_activations_for_item(self, _i):
            if self._raises:
                raise RuntimeError("boom")
            return self._acts

    def _armed(self, probe=None):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        if probe is not None:
            # R1-15: this assigned the SAME probe to every layer, so the
            # fixture agreed with itself and could never observe that
            # `_ambient_fired_count` returned whichever layer came first.
            # Exactly ONE layer answers here; the multi-answer case is tested
            # explicitly below.
            svc._armed_saes = {svc._armed_layers[0]: probe}
            for layer in svc._armed_layers[1:]:
                svc._armed_saes[layer] = self._Probe(enabled=False)
        return svc, saes

    def test_it_is_none_when_monitoring_is_not_running(self):
        svc, _ = self._armed(self._Probe(enabled=False))
        assert svc._ambient_fired_count() is None

    def test_it_is_none_when_monitoring_is_compacted(self):
        """A monitored-feature subset cannot answer "how many fired across the
        WHOLE SAE", so it must decline rather than report the subset."""
        svc, _ = self._armed(self._Probe(enabled=True, subset=[1, 2, 3]))
        assert svc._ambient_fired_count() is None

    def test_it_reports_the_whole_sae_count_when_monitoring_co_ran(self):
        import torch

        # 3 of 5 features positive on the last captured position.
        probe = self._Probe(
            enabled=True, acts=torch.tensor([[0.0, 1.0, 0.0, 2.0, 3.0]])
        )
        svc, _ = self._armed(probe)
        assert svc._ambient_fired_count() == 3

    def test_a_failing_probe_declines_rather_than_guessing(self):
        svc, _ = self._armed(self._Probe(enabled=True, raises=True))
        assert svc._ambient_fired_count() is None


class TestTheRequestBoundaryIsActuallyReleased:
    """F17 mutation findings. Three load-bearing lines in the close path were
    each unprotected: deleting `ctx.close()`, deleting the `bind_context(None)`
    unbind, or weakening the auto-bind guard to ignore `is_closed` all left the
    full suite green.

    Every one of them fails the same way — a context outliving its request, so
    the NEXT request inherits the previous one's rings and budget. That is the
    precise defect F17 exists to remove, and nothing was pinning it."""

    def test_close_request_closes_the_context(self):
        """A closed context refuses late writes (CTX-L2/EC-17.5). If close is
        skipped, a hung generate thread's write lands in the next request."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        ctx = svc._ctx
        svc.close_request()
        assert ctx.is_closed is True, "close_request left the context OPEN"
        # A late write from a hung generate thread is refused rather than
        # silently landing in the next request's accounting (CTX-L2).
        ctx.report_progress(10, 500, circuit_id="circ_1", max_lag=8)
        assert ctx._rings == {}, "a post-close write rebuilt a ring"

    def test_close_request_unbinds_every_sae(self):
        """Closing without unbinding leaves each SAE holding a closed context.
        Sensing would go quietly dark rather than raise."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        svc.close_request()
        for layer, sae in saes.items():
            assert sae._edge_ctx is None, f"layer {layer} still bound after close"

    def test_a_closed_context_is_replaced_not_reused(self):
        """The auto-bind guard must treat a CLOSED context as absent. Checking
        only `is None` would rebind the dead context from the previous request,
        whose rings still hold its fires."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", saes)
        first = svc._ctx
        first.ring("circ_1", 8).record_up("carryover", 1, 1.0)
        svc.close_request()

        # Bind the CLOSED context back on, simulating an SAE that holds one
        # without having gone through close_request (a partially-armed layer,
        # or any caller that binds directly). close_request's own unbind
        # already sets _edge_ctx to None, so routing through it cannot
        # exercise this guard — the first version of this test did, and the
        # mutation survived it.
        sae = next(iter(saes.values()))
        sae.bind_context(first)
        sae.begin_edge_sensing_request("r2")
        assert sae._edge_ctx is not first, "reused the CLOSED context"
        assert sae._edge_ctx.is_closed is False
        assert sae._edge_ring.match_down("carryover", 2) is None, (
            "the previous request's fires are visible to this one"
        )


class TestTruncatedLayersNamesTheLayer:
    """BR-006 / F17 task 4.3. A request-wide `truncated` boolean tells an
    operator their view is incomplete without telling them WHERE, so a layer
    that observed everything is indistinguishable from one that dropped
    events — and the honest reading of any empty result becomes "maybe"."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        return svc, saes

    def test_the_truncating_layer_is_named(self):
        svc, saes = self._armed()
        saes[13]._edge_truncated = True
        _, _, truncated = svc.collect_edges(saes)
        assert truncated is True
        assert svc.last_request_truncated_layers == [13], (
            "named the wrong layer, or none — the operator cannot tell whether "
            "the gap is where they are looking"
        )

    def test_a_complete_request_names_no_layers(self):
        """Empty is a POSITIVE claim: every armed layer reported completely."""
        svc, saes = self._armed()
        _, _, truncated = svc.collect_edges(saes)
        assert truncated is False
        assert svc.last_request_truncated_layers == []

    def test_every_truncating_layer_is_named_and_sorted(self):
        svc, saes = self._armed()
        for sae in saes.values():
            sae._edge_truncated = True
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [10, 13]

    def test_the_list_does_not_leak_across_requests(self):
        """A stale layer list is worse than none: it accuses a layer that
        reported completely this time."""
        svc, saes = self._armed()
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [13]

        svc.close_request()
        svc.begin_request("r2", saes)
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [], (
            "the previous request's truncation is still being reported"
        )

    def test_the_property_returns_a_COPY(self):
        """A caller mutating the returned list must not edit service state."""
        svc, saes = self._armed()
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        got = svc.last_request_truncated_layers
        got.append(999)
        assert svc.last_request_truncated_layers == [13]

    def test_truncated_layers_reaches_the_STATUS_PAYLOAD(self):
        """The F16 R1 failure mode: a field the service computes, the response
        model does not declare, and Pydantic silently drops. Asserting the
        service alone would not have caught it."""
        from millm.api.schemas.circuit_sensing import CircuitSensingStatusResponse

        svc, saes = self._armed()
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)

        payload = svc.status(saes)
        assert payload["truncated_layers"] == [13]
        model = CircuitSensingStatusResponse(**payload)
        assert model.truncated_layers == [13]
        assert model.model_dump()["truncated_layers"] == [13], (
            "declared on the model but dropped on serialization"
        )


class TestR1ConcurrentRequestsAreRefusedNotInterleaved:
    """F17 R1-03. `MAX_CONCURRENT_REQUESTS` must be 1 for this service to
    attribute observations correctly, and config.py enforced that with a
    COMMENT. Setting it to 2 was measured to corrupt sensing silently: the first
    request's context was orphaned (rings leaked, nothing would ever close
    them), its edges were never drained, and the drain reported BOTH requests'
    events under the second request's id.

    Fabricated attribution on an evidence surface is categorically worse than
    lost observations, so the second boundary is refused."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_a_second_boundary_is_refused_while_one_is_open(self):
        svc, saes = self._armed()
        assert svc.begin_request("A", saes) is True
        assert svc.begin_request("B", saes) is False

    def test_a_refused_request_never_borrows_the_open_boundary(self):
        """The refusal must protect the OPEN request's attribution, not merely
        reject B.

        R2-01 (CTX-V2 behaviour change, justified): this used to assert
        `svc._ctx is ctx_a` — that A's context object SURVIVES the refusal.
        That assertion encoded a deadlock: leaving the stale boundary open made
        one hung request disable sensing permanently, and disarm+re-arm did not
        clear it. B's boundary is now reclaimed on refusal.

        What actually matters is unchanged and asserted here: B is refused, and
        nothing B might have observed is ever drained under A's identity. The
        old test pinned the mechanism; this pins the guarantee."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        assert svc.begin_request("B", saes) is False
        # Whatever the service does with the stale boundary, it must not admit
        # B into A's — the two requests' observations can never be merged.
        assert svc._request_circuit_id is None or svc._ctx is None or (
            svc._ctx.request_id != "B"
        ), "B was admitted into the open boundary"

    def test_the_refusal_says_why(self):
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.begin_request("B", saes)
        assert svc.status(saes)["paused_reason"] == "concurrent_request"

    def test_the_guard_does_not_latch(self):
        """A guard that latched would refuse every later request — turning a
        race fix into a total sensing outage."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.collect_edges(saes)
        svc.close_request()
        assert svc.begin_request("B", saes) is True


class TestR1ADarkLayerIsNeverReportedAsComplete:
    """F17 R1-02. `begin_request` returned True if ANY layer began, so a layer
    absent from `layer_saes` was skipped, never bound, never begun — and the
    drain reported `truncated_layers: []`, which the status contract defines as
    'every armed layer reported completely'. Half the circuit was blind and the
    operator was told it was quiet."""

    def test_a_layer_with_no_sae_is_named_not_hidden(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", {10: saes[10]})     # layer 13 absent
        svc.collect_edges(saes)
        assert 13 in svc.last_request_truncated_layers, (
            "a dark layer reported as complete — a false completeness claim"
        )

    def test_the_operator_is_told_why(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", {10: saes[10]})
        assert svc.status(saes)["paused_reason"] == "layer_unavailable"

    def test_a_complete_request_still_claims_completeness(self):
        """The positive claim must survive: this must not flag every request."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", saes)
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == []


class TestR1StaleTruncationDoesNotSurviveDisarm:
    """F17 R1-04. Measured after disarm: `layers: []` with
    `truncated_layers: [13]` — accusing a layer the armed circuit does not
    contain."""

    def test_disarm_clears_the_truncation_report(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", saes)
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [13]
        svc.close_request()
        svc.disarm(saes)
        assert svc.last_request_truncated_layers == []

    def test_status_never_names_an_unarmed_layer(self):
        """The invariant behind the contract's 'positive claim' language."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", saes)
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        svc.close_request()
        svc.disarm(saes)
        st = svc.status(saes)
        assert set(st["truncated_layers"]) <= set(st["layers"]), (
            f"named {st['truncated_layers']} while armed on {st['layers']}"
        )


class TestR1SilentSkipsAreVisibleToTheOperator:
    """F17 R1-06. `_circuit_sensing_begin` had three silent `return None`
    paths. A deployment with `speculative_model` set senses NOTHING, FOREVER,
    while status reported:

        armed: true | paused_reason: null | events_recorded: 0

    — indistinguishable from quiet traffic. That is the "armed but silently
    dark" mode F15 R1-01 existed to kill, surviving on the skip path because
    the skip lives in inference_service and the status lives here."""

    def _wire(self, monkeypatch, speculative=None, saes_for_begin=None):
        import millm.api.dependencies as deps
        from millm.services.inference_service import InferenceService

        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        monkeypatch.setattr(deps, "_circuit_sensing_service", svc, raising=False)

        chosen = saes if saes_for_begin is None else saes_for_begin

        class Stub(InferenceService):
            _tokenizer = None

            def __init__(self):
                self._speculative_model_id = speculative

            def is_model_loaded(self):
                return False

            def _circuit_sensing_layer_saes(self):
                return chosen

        return svc, saes, Stub()

    def test_speculative_decoding_is_named(self, monkeypatch):
        svc, saes, stub = self._wire(monkeypatch, speculative="draft-model")
        assert stub._circuit_sensing_begin("r1") is None
        assert svc.status(saes)["paused_reason"] == "speculative_decoding"

    def test_no_attached_saes_is_named(self, monkeypatch):
        svc, saes, stub = self._wire(monkeypatch, saes_for_begin={})
        assert stub._circuit_sensing_begin("r1") is None
        assert svc.status(saes)["paused_reason"] == "no_attached_saes"

    def test_a_resumed_request_CLEARS_the_stale_reason(self, monkeypatch):
        """A reason that outlives its cause is its own lie — the operator
        keeps seeing why sensing was paused after it has resumed."""
        svc, saes, stub = self._wire(monkeypatch)
        svc.note_paused("speculative_decoding")
        assert stub._circuit_sensing_begin("r1") is not None
        assert svc.status(saes)["paused_reason"] is None

    def test_begin_requests_own_reason_is_not_overwritten(self, monkeypatch):
        """`begin_request` records the more specific reason
        (concurrent_request / layer_unavailable); the caller must not clobber
        it with a generic one."""
        svc, saes, stub = self._wire(monkeypatch)
        svc.begin_request("A", saes)             # holds the boundary open
        assert stub._circuit_sensing_begin("B") is None
        assert svc.status(saes)["paused_reason"] == "concurrent_request"


class TestR1RequestsSensedDistinguishesQuietFromDark:
    """F17 R1-06 second half. `armed + zero events` had two very different
    meanings — 'traffic was quiet' and 'no request ever reached sensing' — and
    the status could not tell them apart. The second is a wiring failure."""

    def test_zero_while_armed_means_sensing_never_ran(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        assert svc.status(saes)["requests_sensed"] == 0

    def test_it_counts_boundaries_actually_opened(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        for rid in ("r1", "r2"):
            svc.begin_request(rid, saes)
            svc.collect_edges(saes)
            svc.close_request()
        assert svc.status(saes)["requests_sensed"] == 2

    def test_a_refused_boundary_is_not_counted(self):
        """Counting a refused request would restore the ambiguity: the
        operator would see activity that never observed anything."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("A", saes)
        svc.begin_request("B", saes)          # refused
        assert svc.status(saes)["requests_sensed"] == 1

    def test_the_count_resets_on_disarm(self):
        """The count belongs to the CURRENT arming; carrying it over would
        report the previous circuit's activity against this one."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", saes)
        svc.collect_edges(saes)
        svc.close_request()
        svc.disarm(saes)
        assert svc.status(saes)["requests_sensed"] == 0

    def test_it_reaches_the_STATUS_PAYLOAD(self):
        """The F16 R1 failure mode: computed by the service, undeclared on the
        response model, silently dropped by Pydantic."""
        from millm.api.schemas.circuit_sensing import CircuitSensingStatusResponse

        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r1", saes)
        svc.collect_edges(saes)
        payload = svc.status(saes)
        model = CircuitSensingStatusResponse(**payload)
        assert model.model_dump()["requests_sensed"] == 1


class TestR1WsDroppedCountsLossNotThrottling:
    """F17 R1-14. `undelivered` was `len(payloads) - sent`, but the loop only
    ATTEMPTS the last `_WS_MAX_PER_FLUSH`. A healthy 20-event flush reported
    `ws_dropped: 15` — measured — conflating the intentional per-flush cap with
    delivery failure and raising a dropped-events alarm on a working system.

    The counter exists so a real discrepancy is observable; inflating it on the
    happy path destroys that signal. Also the first coverage of the emit
    FAILURE branch, which the R1 comment says it was written for."""

    def _emitter(self, monkeypatch, fail_on=None):
        import millm.sockets.progress as sp

        sent: list[dict] = []
        calls = {"n": 0}

        def emit(payload):
            calls["n"] += 1
            if fail_on is not None and calls["n"] == fail_on:
                raise RuntimeError("socket died")
            sent.append(payload)

        monkeypatch.setattr(
            sp.progress_emitter, "emit_circuit_sensing_event", emit, raising=False
        )
        return sent

    def test_a_healthy_flush_over_the_cap_drops_NOTHING(self, monkeypatch):
        sent = self._emitter(monkeypatch)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(20)])
        assert len(sent) == svc._WS_MAX_PER_FLUSH
        assert svc._ws_dropped == 0, (
            f"ws_dropped={svc._ws_dropped} on a healthy flush — the per-flush "
            "cap is a throttle, not a loss; those events are persisted and "
            "readable through the events API"
        )

    def test_a_partial_failure_counts_only_what_was_attempted(self, monkeypatch):
        """2 sent, then a raise: 3 of the 5 attempted did not land."""
        self._emitter(monkeypatch, fail_on=3)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(5)])
        assert svc._ws_dropped == 3

    def test_a_total_failure_counts_the_whole_attempt(self, monkeypatch):
        self._emitter(monkeypatch, fail_on=1)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(5)])
        assert svc._ws_dropped == 5

    def test_an_emit_failure_never_raises_into_the_caller(self, monkeypatch):
        """`_emit` runs after persistence; a socket problem must not undo a
        recorded observation."""
        self._emitter(monkeypatch, fail_on=1)
        svc = CircuitSensingService()
        svc._emit([{"id": 1}])            # must not raise
        assert svc.status({})["ws_dropped"] == 1


class TestR1AmbientCountIsNeverOrderDependent:
    """F17 R1-15. `_ambient_fired_count` returned the FIRST armed layer's
    count. Measured with two monitored layers, the identical state produced 3
    or 9 purely by reordering `_armed_layers`.

    The field is documented as the count across the ENTIRE SAE and 'never
    estimated', and it shares a column name with Feature 11 rows. A circuit
    spans layers, so 'the entire SAE' has no single answer — picking one
    layer's number is a fabricated value on a comparison column.

    The old fixture could not see this: it assigned the SAME probe object to
    every layer, so the answer agreed with itself no matter which layer won."""

    class _P:
        def __init__(self, n, enabled=True):
            self.is_monitoring_enabled = enabled
            self._monitored_features = None
            self._n = n

        def get_feature_activations_for_item(self, _i):
            row = torch.zeros(10)
            row[: self._n] = 1.0
            return row.unsqueeze(0)

    def _svc(self, probes):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc._armed_saes = probes
        return svc

    def test_two_answering_layers_produce_the_SAME_result_either_order(self):
        svc = self._svc({10: self._P(3), 13: self._P(9)})
        svc._armed_layers = [10, 13]
        first = svc._ambient_fired_count()
        svc._armed_layers = [13, 10]
        second = svc._ambient_fired_count()
        assert first == second, (
            f"{first} vs {second} — the answer changed with layer order alone"
        )

    def test_an_ambiguous_count_is_declined_not_guessed(self):
        """None means 'not knowable', which is the contract. Reporting one
        layer's number would be an estimate on a column that promises never to
        estimate."""
        svc = self._svc({10: self._P(3), 13: self._P(9)})
        assert svc._ambient_fired_count() is None

    def test_exactly_one_answering_layer_still_reports(self):
        """The fix must not silence the case that genuinely has one answer."""
        svc = self._svc({10: self._P(3), 13: self._P(9, enabled=False)})
        assert svc._ambient_fired_count() == 3


class TestR1RecordIsActuallyExercised:
    """F17 R1-16. `CircuitSensingService.record()` had NO test caller at all.
    Untested: the row shape, `edge_rung_language` carried verbatim, the
    `truncated` flag reaching every row, `ambient_fired_count` placement, the
    persist-failure path, and — most importantly — the WebSocket privacy
    guarantee, whose only coverage was an `inspect.getsource` string grep for
    `"include_context=False"`. That grep passes if the substring appears in a
    comment and fails if someone writes `include_context = False`."""

    def _edge(self, **over):
        base = dict(
            edge_key="1@10->2@13", up_layer=10, up_feature_idx=1, up_pos=2,
            up_act=1.5, down_layer=13, down_feature_idx=2, down_pos=4,
            down_act=2.5, token_lag=2, phase="prefill", rung=0,
            rung_language="associated", edge_type="computed",
        )
        base.update(over)
        return SimpleNamespace(**base)

    def _armed_svc(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("req-1", saes)
        return svc

    def _run(self, monkeypatch, svc, edges, truncated=False, fail=False):
        """Drive record() against a stubbed session/repo, capturing the rows."""
        import contextlib

        captured = {"rows": None, "pruned": None, "broadcast_kwargs": []}

        class _Saved:
            def __init__(self, row):
                self._row = row

            def to_dict(self, **kwargs):
                captured["broadcast_kwargs"].append(kwargs)
                out = dict(self._row)
                if kwargs.get("include_context") is False:
                    out.pop("context_text", None)
                    out.pop("context_token_ids", None)
                    out.pop("context_parts", None)
                return out

        class _Repo:
            def __init__(self, _session):
                pass

            async def create_many(self, rows):
                if fail:
                    raise RuntimeError("db down")
                captured["rows"] = rows
                return [_Saved(r) for r in rows]

            async def prune(self, circuit_id, **kwargs):
                captured["pruned"] = (circuit_id, kwargs)

        @contextlib.asynccontextmanager
        async def _factory():
            class _S:
                async def commit(self):
                    pass

            yield _S()

        import millm.db.base as db_base
        import millm.db.repositories.circuit_edge_sensing_repository as repo_mod

        monkeypatch.setattr(db_base, "async_session_factory", _factory, raising=False)
        monkeypatch.setattr(
            repo_mod, "CircuitEdgeSensingRepository", _Repo, raising=False
        )
        import millm.sockets.progress as sp
        monkeypatch.setattr(
            sp.progress_emitter, "emit_circuit_sensing_event",
            lambda p: None, raising=False,
        )

        import asyncio
        payloads = asyncio.new_event_loop().run_until_complete(
            svc.record("req-1", edges, truncated, None, None)
        )
        return payloads, captured

    def test_the_broadcast_carries_NO_prompt_text(self, monkeypatch):
        """The privacy guarantee, asserted on the actual call rather than by
        grepping the source for a substring."""
        svc = self._armed_svc()
        _, cap = self._run(monkeypatch, svc, [self._edge()])
        assert cap["broadcast_kwargs"], "to_dict was never called"
        assert all(
            k.get("include_context") is False for k in cap["broadcast_kwargs"]
        ), f"broadcast built with {cap['broadcast_kwargs']} — prompt text leaks"

    def test_the_rung_language_is_carried_VERBATIM_onto_the_row(self, monkeypatch):
        """An observation must never restate the evidence claim. Re-deriving
        the phrase here would let a row drift from the rung it was observed
        under."""
        svc = self._armed_svc()
        _, cap = self._run(
            monkeypatch, svc,
            [self._edge(rung=0, rung_language="associated")],
        )
        row = cap["rows"][0]
        assert row["edge_rung"] == 0
        assert row["edge_rung_language"] == "associated"
        assert "causal" not in row["edge_rung_language"].lower()

    def test_truncated_reaches_every_row(self, monkeypatch):
        """The honesty flag is per-row; a row that lost its flag reads as a
        complete observation."""
        svc = self._armed_svc()
        _, cap = self._run(
            monkeypatch, svc, [self._edge(), self._edge(down_pos=9)],
            truncated=True,
        )
        assert [r["truncated"] for r in cap["rows"]] == [True, True]

    def test_rows_are_attributed_to_the_BOUNDARY_circuit(self, monkeypatch):
        """R2's defect: reading `self._circuit_id` at drain time attributed
        observations to whatever is armed NOW, not what was armed when they
        were observed."""
        svc = self._armed_svc()
        svc._circuit_id = "circ_SOMETHING_ELSE"
        _, cap = self._run(monkeypatch, svc, [self._edge()])
        assert cap["rows"][0]["circuit_id"] == "circ_1"

    def test_a_persist_failure_records_nothing_and_never_raises(self, monkeypatch):
        """An observation path must not break generation, and must not report
        events it failed to store."""
        svc = self._armed_svc()
        payloads, _ = self._run(monkeypatch, svc, [self._edge()], fail=True)
        assert payloads == []
        assert svc.status({})["events_recorded"] == 0

    def test_retention_pruning_runs_with_the_configured_bounds(self, monkeypatch):
        svc = self._armed_svc()
        _, cap = self._run(monkeypatch, svc, [self._edge()])
        assert cap["pruned"] is not None, "retention pruning never ran"
        pruned_circuit, kwargs = cap["pruned"]
        assert pruned_circuit == "circ_1"
        assert "cap" in kwargs and "max_age_days" in kwargs


class TestR2TheConcurrencyGuardCannotDEADLOCKSensing:
    """F17 R2-01. R1-03's guard refused a second boundary while one was open —
    and left the stale one open. Verified: after ONE request that never closed,
    every subsequent begin was refused FOREVER, and disarm + re-arm did not
    clear it either. The fix for a race became a permanent outage, which is
    strictly worse than the race.

    Reclaiming is safe because generation is serialised on the request queue:
    reaching `begin_request` at all means the previous request is no longer
    running, so its boundary is stale rather than concurrent."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_sensing_recovers_after_a_request_that_never_closes(self):
        svc, saes = self._armed()
        svc.begin_request("A", saes)          # never closed
        assert svc.begin_request("B", saes) is False, "B must still be refused"
        assert svc.begin_request("C", saes) is True, (
            "sensing never recovered — one hung request disabled it forever"
        )

    def test_the_recovered_boundary_attributes_correctly(self):
        """Self-healing must not resurrect the stale request's identity."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.begin_request("B", saes)          # refused, reclaims A
        svc.begin_request("C", saes)
        request_id, _, _ = svc.collect_edges(saes)
        assert request_id == "C"

    def test_the_refused_request_is_still_refused(self):
        """Reclaiming must not silently ADMIT the request that found the stale
        boundary — its observations would span two requests."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        assert svc.begin_request("B", saes) is False

    def test_disarm_releases_an_open_boundary(self):
        """The operator escape hatch: disarm ends observation entirely, so any
        boundary it owned is over by definition. This did NOT clear it."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.disarm(saes)
        svc.arm_for_circuit(circuit(), definition(), saes)
        assert svc.begin_request("E", saes) is True
        assert svc._ctx is not None and not svc._ctx.is_closed

    def test_the_stale_context_is_closed_not_merely_dropped(self):
        """A dropped-but-open context keeps its rings alive; closing is what
        makes a late write from the hung thread refuse (CTX-L2)."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        stale = svc._ctx
        svc.begin_request("B", saes)
        assert stale.is_closed is True
        assert all(s._edge_ctx is None for s in saes.values()) or svc._ctx is None


class TestR2APartiallyDarkRequestKeepsItsReason:
    """F17 R2-02. R1-06 added `note_paused(None)` on the success path to clear
    a stale reason; R1-02 made `begin_request` set `layer_unavailable` when
    some layers are dark. `begin_request` returns True when SOME layers began,
    so a partially dark circuit reached the clear and had its reason ERASED —
    one round-1 fix deleting another's signal.

    Verified before the fix: reason went to None while layer 13 was dark."""

    def _wire(self, monkeypatch, layers_for_begin):
        import millm.api.dependencies as deps
        from millm.services.inference_service import InferenceService

        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        monkeypatch.setattr(deps, "_circuit_sensing_service", svc, raising=False)
        chosen = {k: saes[k] for k in layers_for_begin}

        class Stub(InferenceService):
            _tokenizer = None

            def __init__(self):
                self._speculative_model_id = None

            def is_model_loaded(self):
                return False

            def _circuit_sensing_layer_saes(self):
                return chosen

        return svc, saes, Stub()

    def test_a_dark_layer_reason_survives_the_stale_clear(self, monkeypatch):
        svc, saes, stub = self._wire(monkeypatch, [10])      # layer 13 dark
        assert stub._circuit_sensing_begin("r1") is not None, "partial success"
        assert svc.status(saes)["paused_reason"] == "layer_unavailable", (
            "the reason for THIS request was wiped by the stale-clear"
        )

    def test_a_healthy_request_still_clears_a_stale_reason(self, monkeypatch):
        """The clear must keep working — otherwise the operator sees why
        sensing was paused long after it resumed."""
        svc, saes, stub = self._wire(monkeypatch, [10, 13])
        svc.note_paused("speculative_decoding")             # stale, prior request
        assert stub._circuit_sensing_begin("r1") is not None
        assert svc.status(saes)["paused_reason"] is None

    def test_a_reason_does_not_outlive_its_own_request(self, monkeypatch):
        """`layer_unavailable` from request 1 must not still be showing during
        a healthy request 2 — that would be the stale-reason bug in reverse."""
        import millm.api.dependencies as deps
        from millm.services.inference_service import InferenceService

        svc, saes, stub_partial = self._wire(monkeypatch, [10])
        stub_partial._circuit_sensing_begin("r1")
        assert svc.status(saes)["paused_reason"] == "layer_unavailable"
        svc.close_request()

        # Request 2 on the SAME service, this time with every layer present.
        class HealthyStub(InferenceService):
            _tokenizer = None

            def __init__(self):
                self._speculative_model_id = None

            def is_model_loaded(self):
                return False

            def _circuit_sensing_layer_saes(self):
                return saes

        assert HealthyStub()._circuit_sensing_begin("r2") is not None
        assert svc.status(saes)["paused_reason"] is None, (
            "request 1's reason is still showing during a healthy request 2"
        )


class TestR2AnAllDarkBeginLeavesNothingBehind:
    """F17 R2-04/R2-05. `self._ctx = ctx` was assigned BEFORE the dark-layer
    loop, so an all-dark begin returned False with the context already open.
    Nothing closes it — the caller returns None on False and
    `_notify_circuit_sensing` early-returns — so R1-03's concurrency guard then
    refused the NEXT request. One healthy request lost per orphan, flapping
    forever under a persistently dark condition.

    R1-02 (name the dark layers) and R1-03 (refuse a second boundary) were each
    correct alone; the collision is the finding."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_an_all_dark_begin_orphans_no_context(self):
        svc, saes = self._armed()
        assert svc.begin_request("r1", {}) is False
        assert svc._ctx is None, "the failed begin left a boundary open"

    def test_the_next_request_is_not_punished_for_it(self):
        svc, saes = self._armed()
        svc.begin_request("r1", {})              # all dark, returns False
        assert svc.begin_request("r2", saes) is True, (
            "a healthy request was refused because a dark one orphaned its "
            "context"
        )

    def test_requests_sensed_excludes_a_begin_that_observed_nothing(self):
        """`requests_sensed` promises 'ZERO while armed means no request
        reached sensing at all'. Counting an all-dark begin reports activity on
        exactly the wiring-failure path the field exists to expose."""
        svc, saes = self._armed()
        svc.begin_request("r1", {})
        assert svc.status(saes)["requests_sensed"] == 0

    def test_a_partially_dark_begin_IS_counted(self):
        """It genuinely observed something — the count must not swing the other
        way and hide real activity."""
        svc, saes = self._armed()
        assert svc.begin_request("r1", {10: saes[10]}) is True
        assert svc.status(saes)["requests_sensed"] == 1

    def test_the_dark_reason_survives_the_failed_begin(self):
        """Closing the context must not also wipe the explanation."""
        svc, saes = self._armed()
        svc.begin_request("r1", {})
        assert svc.status(saes)["paused_reason"] == "layer_unavailable"


class TestR2TheCircuitBudgetIsDerivedHonestly:
    """F17 R2-06/08/09. Three ways the budget and the truncation report could
    mislead an operator, all found by attacking round 1's fixes."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_a_recovered_layer_is_not_still_accused(self):
        """R2-06: `truncated_layers` was rebuilt only at drain time, so for the
        whole span between begin and drain the status named LAST request's dark
        layers. Accusing a healthy layer is R1-04's dishonesty inverted."""
        svc, saes = self._armed()
        svc.begin_request("r1", {10: saes[10]})      # layer 13 dark
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [13]
        svc.close_request()

        svc.begin_request("r2", saes)               # everything healthy
        assert svc.status(saes)["truncated_layers"] == [], (
            "a recovered layer is still reported as untrustworthy"
        )

    def test_the_budget_takes_the_MOST_RESTRICTIVE_layer(self):
        """R2-08: it used `_armed_layers[0]` — whichever layer sorts first.
        Verified with divergent configs: {10:5, 20:500} gave 5, reordered gave
        500. The same order-dependence R1-15 fixed for the ambient count, one
        function away, and F19's per-circuit configs make it live."""
        svc, saes = self._armed()
        svc._configs[10].max_events_per_request = 5
        svc._configs[13].max_events_per_request = 500
        svc.begin_request("r", saes)
        assert svc._ctx.budget.cap == 5

    def test_the_budget_is_order_independent(self):
        svc, saes = self._armed()
        svc._configs[10].max_events_per_request = 500
        svc._configs[13].max_events_per_request = 5
        svc.begin_request("r", saes)
        assert svc._ctx.budget.cap == 5, "the answer changed with layer order"

    def test_a_zero_cap_is_clamped_rather_than_arming_a_dead_circuit(self):
        """R2-09: `cap=0` meant armed, latched, and reporting truncation on
        every layer — sensing that looks ON and observes nothing. A
        misconfigured zero should mean OFF, and there is no lower-bound
        validation on the setting."""
        svc, saes = self._armed()
        for c in svc._configs.values():
            c.max_events_per_request = 0
        svc.begin_request("r", saes)
        assert svc._ctx.budget.cap >= 1


class TestR2ThrottlingIsVisibleWithoutBeingCalledLoss:
    """F17 R2-11. R1-14 correctly stopped counting cap-declined events as
    DROPPED — they are persisted and readable through the events API, so
    nothing is lost. But it left them invisible: with a 5-per-flush cap and a
    20-event request, 75% of a busy request's events never reach the live panel
    and no field said so. An operator comparing the panel against the events
    API had no way to explain the gap."""

    def _emitter(self, monkeypatch, fail_on=None):
        import millm.sockets.progress as sp

        calls = {"n": 0}

        def emit(payload):
            calls["n"] += 1
            if fail_on is not None and calls["n"] == fail_on:
                raise RuntimeError("socket died")

        monkeypatch.setattr(
            sp.progress_emitter, "emit_circuit_sensing_event", emit, raising=False
        )

    def test_throttling_is_counted_separately_from_loss(self, monkeypatch):
        self._emitter(monkeypatch)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(20)])
        st = svc.status({})
        assert st["ws_throttled"] == 15
        assert st["ws_dropped"] == 0, "throttling must not read as delivery loss"

    def test_a_healthy_small_flush_throttles_nothing(self, monkeypatch):
        self._emitter(monkeypatch)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(3)])
        assert svc.status({})["ws_throttled"] == 0

    def test_a_real_failure_still_counts_as_dropped_not_throttled(self, monkeypatch):
        """The two must not blur: one is by design, the other is a fault."""
        self._emitter(monkeypatch, fail_on=1)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(5)])
        st = svc.status({})
        assert st["ws_dropped"] == 5
        assert st["ws_throttled"] == 0

    def test_ws_throttled_reaches_the_STATUS_PAYLOAD(self, monkeypatch):
        from millm.api.schemas.circuit_sensing import CircuitSensingStatusResponse

        self._emitter(monkeypatch)
        svc = CircuitSensingService()
        svc._emit([{"id": i} for i in range(20)])
        model = CircuitSensingStatusResponse(**svc.status({}))
        assert model.model_dump()["ws_throttled"] == 15


class TestR2ARareTruncationCannotBeRacedAway:
    """F17 R2-13. R2-06 clears `truncated_layers` when a new boundary opens,
    which is correct — it describes the LAST DRAINED request, and leaving it
    would accuse a recovered layer. But it means a fast-arriving next request
    supersedes the report before an operator polls, so a rare truncation can
    never be seen at all.

    Both behaviours are right for their own field; the answer is a cumulative
    counter that no later request can erase."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_a_superseded_truncation_still_leaves_a_trace(self):
        svc, saes = self._armed()
        svc.begin_request("r1", saes)
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        svc.close_request()
        svc.begin_request("r2", saes)               # supersedes the report
        st = svc.status(saes)
        assert st["truncated_layers"] == []         # correct: r2 is clean
        assert st["requests_truncated"] == 1, (
            "r1's truncation vanished entirely — nothing records that the "
            "circuit has ever lost data"
        )

    def test_a_clean_circuit_reports_zero(self):
        svc, saes = self._armed()
        svc.begin_request("r1", saes)
        svc.collect_edges(saes)
        assert svc.status(saes)["requests_truncated"] == 0

    def test_it_counts_requests_not_layers(self):
        """Two layers truncating in ONE request is one truncated request."""
        svc, saes = self._armed()
        svc.begin_request("r1", saes)
        for s in saes.values():
            s._edge_truncated = True
        svc.collect_edges(saes)
        assert svc.status(saes)["requests_truncated"] == 1

    def test_it_resets_on_disarm(self):
        """The count belongs to the CURRENT arming, like requests_sensed."""
        svc, saes = self._armed()
        svc.begin_request("r1", saes)
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        svc.close_request()
        svc.disarm(saes)
        assert svc.status(saes)["requests_truncated"] == 0

    def test_it_reaches_the_STATUS_PAYLOAD(self):
        from millm.api.schemas.circuit_sensing import CircuitSensingStatusResponse

        svc, saes = self._armed()
        svc.begin_request("r1", saes)
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        model = CircuitSensingStatusResponse(**svc.status(saes))
        assert model.model_dump()["requests_truncated"] == 1


class TestR2APauseReasonCannotLagARequestBehind:
    """F17 R2-14, found by attacking R2-02's own fix. `_pause_is_current` was
    set by `reason is not None` — so ANY reason counted as current, including
    ones recorded OUTSIDE a request. The skip reasons (`speculative_decoding`,
    `no_attached_saes`) are set precisely when no boundary opens, so they
    survived the next request's clear and showed one request late."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_a_reason_set_outside_a_request_clears_on_the_first_try(self):
        svc, saes = self._armed()
        svc.note_paused("speculative_decoding")     # no boundary open
        svc.clear_stale_pause()
        assert svc._paused_reason is None, (
            "a skip reason survived its clear and will show one request late"
        )

    def test_a_reason_set_DURING_a_request_still_survives(self):
        """R2-02's guarantee must not be lost while fixing R2-14."""
        svc, saes = self._armed()
        svc.begin_request("r", saes)
        svc.note_paused("layer_unavailable")        # boundary IS open
        svc.clear_stale_pause()
        assert svc._paused_reason == "layer_unavailable"

    def test_an_in_request_reason_clears_on_the_NEXT_request(self):
        """It is current for its own request and stale afterwards."""
        svc, saes = self._armed()
        svc.begin_request("r1", saes)
        svc.note_paused("layer_unavailable")
        svc.collect_edges(saes)
        svc.close_request()
        svc.begin_request("r2", saes)
        svc.clear_stale_pause()
        assert svc._paused_reason is None


class TestR2AReclaimedRequestsDataIsDiscardedLOUDLY:
    """F17 R2-15, from attacking R2-01's reclaim path.

    Verified first that the reclaim does NOT leak: a later request drains 0
    edges and none of the stale observations, because
    `begin_edge_sensing_request` resets each buffer. So attribution is safe —
    the guard's whole purpose holds.

    But the discard was SILENT. On an evidence surface, quietly dropping a
    request's observations leaves the operator reading a clean circuit that
    lost data."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_stale_observations_are_never_attributed_to_a_later_request(self):
        """The guarantee that must hold no matter what: no fabrication."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        saes[10]._sensed_edges.append("A-observation")
        svc.begin_request("B", saes)          # refused, reclaims
        svc.begin_request("C", saes)          # succeeds
        request_id, edges, _ = svc.collect_edges(saes)
        assert request_id == "C"
        assert "A-observation" not in edges, (
            "A's observation was drained under C's identity — fabricated "
            "attribution, the exact failure the guard exists to prevent"
        )

    def test_a_lossy_reclaim_is_counted(self):
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        saes[10]._sensed_edges.append("obs")
        svc.begin_request("B", saes)
        assert svc.status(saes)["requests_truncated"] == 1, (
            "a request's observations were discarded and nothing recorded it"
        )

    def test_a_clean_reclaim_is_NOT_counted(self):
        """A reclaim that lost nothing must not raise a data-loss signal."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)          # observed nothing
        svc.begin_request("B", saes)
        assert svc.status(saes)["requests_truncated"] == 0


class TestR2ContextCaptureCannotGoSilentlyDark:
    """F17 R2-16/17. `_request_context_tokens` used `_armed_layers[0]` — the
    same order-dependence R2-08 fixed for the budget cap, one expression away.
    Measured with configs {10: 0, 13: 32}: order [10,13] gave 0, [13,10] gave
    32.

    Worse than the cap's version: `ctx_tokens == 0` hits the `k == 0` early
    return in `_context`, so ALL context capture is disabled — every event row
    loses its decoded window — depending on which layer sorted first.

    And nothing detected it. Mutating the whole expression to 0 left the suite
    green, because no test drove `record()` with a real tokenizer. That is the
    silently-dark mode this feature exists to remove, on the field that carries
    the operator's only view of WHAT fired."""

    def _armed(self, ctx_tokens_by_layer):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        for layer, n in ctx_tokens_by_layer.items():
            svc._configs[layer].context_tokens = n
        return svc, saes

    def test_context_tokens_are_order_independent(self):
        svc, saes = self._armed({10: 0, 13: 32})
        svc._armed_layers = [10, 13]
        svc.begin_request("a", saes)
        first = svc._request_context_tokens
        svc.close_request()
        svc._armed_layers = [13, 10]
        svc.begin_request("b", saes)
        assert svc._request_context_tokens == first, (
            "the decoded-window size changed with layer order alone"
        )

    def test_one_layer_configured_to_zero_does_not_silence_the_circuit(self):
        """MAX, not min: the cap bounds a shared resource so the strictest wins,
        but context capture is per-row enrichment and a single zero must not
        disable it everywhere."""
        svc, saes = self._armed({10: 0, 13: 32})
        svc.begin_request("r", saes)
        assert svc._request_context_tokens == 32

    def test_a_real_context_window_reaches_the_row(self):
        """C3: the coverage gap that let the above go unnoticed. Drives the
        real `_context` with a real token list."""
        import torch

        svc, saes = self._armed({10: 8, 13: 8})
        svc.begin_request("r", saes)

        class _Tok:
            def decode(self, ids, **kw):
                return " ".join(f"t{int(i)}" for i in ids)

        edge = SimpleNamespace(
            up_pos=2, down_pos=4, up_layer=10, down_layer=13,
            up_feature_idx=1, down_feature_idx=2,
        )
        ids = torch.arange(0, 12).unsqueeze(0)
        text, window, parts = svc._context(ids, edge, 8, _Tok())
        assert text, "context capture produced nothing with a live tokenizer"
        assert window, "no token window was captured"
        assert parts is None or isinstance(parts, dict)

    def test_zero_context_tokens_captures_nothing_by_design(self):
        """The other side: 0 genuinely means 'no capture', so the fix must not
        make it impossible to turn context off."""
        import torch

        svc, saes = self._armed({10: 0, 13: 0})
        svc.begin_request("r", saes)
        assert svc._request_context_tokens == 0

        class _Tok:
            def decode(self, ids, **kw):
                return "should not be called"

        edge = SimpleNamespace(
            up_pos=2, down_pos=4, up_layer=10, down_layer=13,
            up_feature_idx=1, down_feature_idx=2,
        )
        text, window, parts = svc._context(
            torch.arange(0, 12).unsqueeze(0), edge, 0, _Tok()
        )
        assert text is None and window is None


class TestR2CountersAreConsistentAcrossAnArmCycle:
    """F17 R2-18. `requests_sensed`/`requests_truncated` reset on disarm while
    `events_recorded`/`ws_dropped`/`ws_throttled` persisted, so a freshly
    re-armed circuit read

        requests_sensed: 0, events_recorded: 99

    which the schema defines as the WIRING-FAILURE signature ("zero while armed
    means no request reached sensing at all"). A healthy re-arm looked broken —
    the counter added to expose a failure mode was manufacturing one."""

    def test_every_counter_resets_together_on_rearm(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        svc.collect_edges(saes)
        svc.note_events_recorded(99)
        svc.note_ws_dropped(7)
        svc._ws_throttled = 5
        svc.close_request()
        svc.disarm(saes)
        svc.arm_for_circuit(circuit(), definition(), saes)

        st = svc.status(saes)
        stale = {
            k: st[k]
            for k in (
                "requests_sensed", "requests_truncated", "events_recorded",
                "ws_dropped", "ws_throttled",
            )
            if st[k] != 0
        }
        assert not stale, f"counters survived a re-arm: {stale}"


class TestR2TheStaleContextIsCLEAREDNotJustClosed:
    """F17 R2-19. `stale, self._ctx = self._ctx, None` — the `None` half was
    load-bearing and untested. Normally inert, because `stale.close()` sets
    `is_closed` and the guard tolerates that. But when `close()` RAISES, the
    real code still recovers while a version that only closed would deadlock
    permanently — the exact permanent outage R2-01 exists to prevent, on the
    hung-thread path it was written for."""

    def test_recovery_survives_a_context_that_fails_to_close(self, monkeypatch):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("A", saes)

        def boom():
            raise RuntimeError("close failed")

        monkeypatch.setattr(svc._ctx, "close", boom, raising=False)

        assert svc.begin_request("B", saes) is False, "B must still be refused"
        assert svc.begin_request("C", saes) is True, (
            "a context that failed to close deadlocked sensing permanently"
        )


class TestR2TruncationAccountingCannotDoubleCountOrCarryOver:
    """F17 R2-20. Two independent sites now increment `_requests_truncated` —
    the lossy-reclaim path (R2-15) and the drain (R2-13) — and two independent
    places record per-layer truncation. Attacked both for double-counting and
    for carry-over; they hold, and these pin them so a later change cannot
    quietly break either.

    Both failure modes matter on an evidence surface: double-counting
    manufactures data loss that did not happen, carry-over attributes one
    request's loss to another."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_one_lossy_request_is_counted_ONCE(self):
        """A request that both truncated AND was reclaimed with undrained data
        is still one truncated request."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        saes[13]._edge_truncated = True
        saes[10]._sensed_edges.append("undrained")
        svc.begin_request("B", saes)              # refused, lossy reclaim
        svc.begin_request("C", saes)
        svc.collect_edges(saes)
        assert svc.status(saes)["requests_truncated"] == 1, (
            "one request's loss was counted twice — manufactured data loss"
        )

    def test_a_truncation_flag_does_not_follow_into_the_next_request(self):
        """`begin_edge_sensing_request` resets the per-SAE flag; if it stopped,
        request B would be reported as having lost data that A lost."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        saes[13]._edge_truncated = True
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [13]
        svc.close_request()

        svc.begin_request("B", saes)
        assert saes[13]._edge_truncated is False
        svc.collect_edges(saes)
        assert svc.last_request_truncated_layers == [], (
            "A's truncation was attributed to B"
        )

    def test_two_separate_truncated_requests_count_twice(self):
        """The counter must still move: a fix for double-counting that stopped
        counting would be worse."""
        svc, saes = self._armed()
        for rid in ("A", "B"):
            svc.begin_request(rid, saes)
            saes[13]._edge_truncated = True
            svc.collect_edges(saes)
            svc.close_request()
        assert svc.status(saes)["requests_truncated"] == 2


class TestR3ASwappedOutSAEIsStillReleased:
    """F17 R3-02. The reclaim path (R2-01) and `close_request` unbound only
    `_armed_saes`. An SAE swapped out of that map since begin — a re-arm, an
    eviction — kept its reference to the now-dead context, and on its next
    begin self-bound a PRIVATE solo context whose observations no sibling can
    read (the R2-07 fallback, doing its job on a layer that should never have
    reached it).

    `disarm` already unioned the armed set with the caller's map for exactly
    this reason; the request paths did not."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_the_reclaim_releases_an_SAE_swapped_out_since_begin(self):
        from tests.unit.services.test_circuit_sensing_service import make_sae

        svc, saes = self._armed()
        svc.begin_request("A", saes)
        stranded = saes[13]
        svc._armed_saes[13] = make_sae()        # swapped in the armed map only

        svc.begin_request("B", saes)            # refused -> reclaim
        assert stranded._edge_ctx is None, (
            "a swapped-out SAE kept the dead context and will sense into a "
            "private ring no sibling can read"
        )

    def test_close_request_releases_it_too(self):
        from tests.unit.services.test_circuit_sensing_service import make_sae

        svc, saes = self._armed()
        svc.begin_request("A", saes)
        stranded = saes[13]
        svc._armed_saes[13] = make_sae()
        svc.collect_edges(saes)
        svc.close_request()
        assert stranded._edge_ctx is None


class TestR3PauseReasonsTransitionCorrectlyAcrossEveryPath:
    """F17 R3-03. `_pause_is_current` (R2-14) means "set while a boundary is
    open", and the reclaim path sets a reason and then nulls `_ctx` in the same
    call. Attacked every transition between the four reasons; they hold, and
    these pin them because the state machine now has enough paths that a later
    change could easily strand one.

    A stranded reason is not cosmetic: `paused_reason` is what an operator
    reads to decide whether an empty result means quiet traffic or a fault."""

    def _armed(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        return svc, saes

    def test_a_refusal_reason_does_not_outlive_the_reclaim(self):
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.begin_request("B", saes)              # refused
        assert svc._paused_reason == "concurrent_request"
        svc.begin_request("C", saes)              # succeeds on the reclaimed slot
        svc.clear_stale_pause()
        assert svc._paused_reason is None

    def test_a_dark_request_overrides_a_stale_refusal(self):
        """The newer, more specific reason must win — an operator debugging a
        dark layer must not be shown a refusal from two requests ago."""
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.begin_request("B", saes)              # refusal
        svc.begin_request("C", {10: saes[10]})    # partially dark
        assert svc._paused_reason == "layer_unavailable"

    def test_a_healthy_request_clears_everything(self):
        svc, saes = self._armed()
        svc.begin_request("A", saes)
        svc.begin_request("B", saes)
        svc.begin_request("C", {10: saes[10]})
        svc.collect_edges(saes)
        svc.close_request()
        svc.begin_request("D", saes)
        svc.clear_stale_pause()
        assert svc._paused_reason is None


class TestR3TheCapOrderingKeepsTheCountHonest:
    """F17 R3-04. R2-03 checks the per-SAE cap FIRST, then the shared budget.
    Attacked for a wrong total, a wrong spend, and a dishonest truncation
    report under an exhausted circuit budget."""

    def _circuit(self, sae_cap, circuit_cap, layers=(10, 11, 12)):
        from tests.unit.ml.test_edge_sensing_characterization import (
            config, ctx_for, real_sae, spec,
        )

        cfgs = {}
        for L in layers:
            c = config(
                edges=[spec(up_layer=L, down_layer=L, key=f"1@{L}->2@{L}")],
                max_lag=8, cap=sae_cap, layer=L,
            )
            c.circuit_id = "c"
            cfgs[L] = c
        ctx = ctx_for(cfgs[layers[0]], "r")
        ctx.budget.cap = circuit_cap
        saes = {}
        for L, c in cfgs.items():
            s = real_sae()
            s.arm_edge_sensing(c)
            s.bind_context(ctx)
            s.begin_edge_sensing_request("r")
            saes[L] = s
        return ctx, saes

    def _fire(self, sae, pairs=6):
        from tests.unit.ml.test_edge_sensing_characterization import hidden

        rows = []
        for _ in range(pairs):
            rows += [{1: 2.0}, {2: 2.0}]
        sae._sense_edges(hidden(*rows))

    def test_the_circuit_total_respects_the_shared_budget(self):
        ctx, saes = self._circuit(sae_cap=10, circuit_cap=3)
        for s in saes.values():
            self._fire(s)
        total = sum(len(s._sensed_edges) for s in saes.values())
        assert total == 3, f"{total} events against a circuit budget of 3"

    def test_the_recorded_spend_matches_what_was_emitted(self):
        """A spend that drifts from the emission count makes every later
        budget decision wrong."""
        ctx, saes = self._circuit(sae_cap=10, circuit_cap=3)
        for s in saes.values():
            self._fire(s)
        total = sum(len(s._sensed_edges) for s in saes.values())
        assert ctx.budget.spent("c") == total

    def test_every_layer_that_lost_a_match_is_named(self):
        """Layers emitting 0 are still flagged: they had matches REFUSED, so
        their view genuinely is incomplete. Silence here would read as 'this
        layer saw nothing', which is a different and false claim."""
        ctx, saes = self._circuit(sae_cap=10, circuit_cap=3)
        for s in saes.values():
            self._fire(s)
        assert ctx.budget.truncated_layers("c") == [10, 11, 12]


class TestR3TheMergedDrainIsOrderedByPosition:
    """F17 R3-06, found by mutation: deleting `merged.sort(...)` in
    `collect_edges` left the whole suite green.

    The sort is load-bearing for two things. Events reach the operator in
    CAUSAL order, and `_emit` keeps the LAST `_WS_MAX_PER_FLUSH` — a deliberate
    R2 fix so the live panel shows a request's most RECENT edges rather than
    its earliest. Without the sort, "last 5" means "whichever layer happened to
    drain last", which silently undoes that fix: a circuit whose late-position
    layer drains first would show the panel its oldest events."""

    def _edge(self, down_pos, up_pos):
        return SimpleNamespace(
            down_pos=down_pos, up_pos=up_pos, edge_key="e", phase="prefill",
            up_layer=10, up_feature_idx=1, up_act=1.0,
            down_layer=13, down_feature_idx=2, down_act=1.0,
            token_lag=down_pos - up_pos, rung=2,
            rung_language="causally validated (edge)", edge_type="computed",
        )

    def test_the_merge_is_sorted_by_downstream_position(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        # `collect_edges` walks `_armed_layers` in SORTED order (10 then 13),
        # so the LOWER layer must hold the LATER positions or the drain order
        # already matches position order and the sort is untested. The first
        # version of this test put late positions on layer 13 and passed
        # against the mutation — a fixture agreeing with the code by
        # construction (R1-12's anti-pattern, committed again here).
        saes[10]._sensed_edges = [self._edge(90, 88), self._edge(95, 93)]
        saes[13]._sensed_edges = [self._edge(5, 3), self._edge(9, 7)]

        _, merged, _ = svc.collect_edges(saes)
        positions = [e.down_pos for e in merged]
        assert positions == sorted(positions), (
            f"drained out of causal order: {positions}"
        )

    def test_the_live_panel_gets_the_NEWEST_events(self):
        """The property the sort exists to protect, asserted end to end."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        # Late positions on the LOWER layer, so drain order fights position
        # order — see the note above.
        saes[10]._sensed_edges = [self._edge(90, 88), self._edge(95, 93)]
        saes[13]._sensed_edges = [self._edge(5, 3), self._edge(9, 7)]

        _, merged, _ = svc.collect_edges(saes)
        newest = [e.down_pos for e in merged[-2:]]
        assert 95 in newest and 90 in newest, (
            f"the panel would show {newest}, missing the request's most recent "
            "edges — the flush cap keeps the LAST entries, so order is what "
            "makes 'last' mean 'newest'"
        )
