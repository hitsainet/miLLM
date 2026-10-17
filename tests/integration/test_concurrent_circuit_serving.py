"""Feature 19 task 6.1/6.4 — two circuits serving at once, end to end.

These tests exist because the unit-level owner-map tests could not catch the
thing that actually matters: whether the SERVING PATH is wired to the owner
map at all. Two mutations proved that gap — reverting `deactivate` to a global
`clear_circuit_steering()`, and disabling the owner routing inside the apply —
both passed the entire suite while reintroducing the defect the feature exists
to prevent.

So these drive the real `CircuitService.activate` / `deactivate` against a real
`SAEService` and assert what an operator would see: the OTHER circuit is still
steering.
"""

from types import SimpleNamespace

import pytest

from millm.api.schemas.circuit import CircuitMember
from millm.services.sae_service import AttachedEntry, AttachedSAEState, SAEService

pytestmark = pytest.mark.asyncio


class FakeSAE:
    def __init__(self):
        self._values: dict[int, float] = {}
        self.is_steering_enabled = False
        self.is_monitoring_enabled = False
        self.d_sae = 8192

    def get_steering_values(self):
        return dict(self._values)

    def clear_steering(self):
        self._values = {}

    def set_steering_batch(self, values):
        self._values = dict(values)

    def enable_steering(self, on):
        self.is_steering_enabled = on


@pytest.fixture(autouse=True)
def clean_state():
    AttachedSAEState().reset_for_tests()
    yield
    AttachedSAEState().reset_for_tests()


def attach(layer, sae_id=None):
    state = AttachedSAEState()
    sae = FakeSAE()
    sid = sae_id or f"sae-{layer}"
    state._entries[(sid, layer)] = AttachedEntry(
        sae=sae, sae_id=sid, layer=layer, hook_handle=None
    )
    return sae


class TestTwoCircuitsServeSimultaneously:
    async def test_disjoint_circuits_both_steer_and_neither_clears_the_other(
        self,
    ):
        """US-19.1. The whole point of the feature."""
        s10 = attach(10)
        s13 = attach(13)
        svc = SAEService.for_registry()

        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=13, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )

        assert s10.get_steering_values() == {1: 40.0}
        assert s13.get_steering_values() == {2: 30.0}, (
            "activating B cleared A's layer — the second activation wiped the "
            "first"
        )
        assert s10.is_steering_enabled and s13.is_steering_enabled

    async def test_releasing_one_leaves_the_other_serving(self):
        """US-19.5 / EC-19.2. The highest-consequence defect in the feature."""
        s10 = attach(10)
        s13 = attach(13)
        svc = SAEService.for_registry()

        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=13, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )

        AttachedSAEState().release_owner("circuit:A")

        assert s10.get_steering_values() == {}
        assert s13.get_steering_values() == {2: 30.0}, (
            "releasing A stopped B — a circuit the operator never touched"
        )
        assert s13.is_steering_enabled is True

    async def test_two_circuits_COMPOSED_on_one_layer_both_contribute(self):
        """The override case: distinct features on a shared layer sum, and
        each owner's contribution is individually removable."""
        s10 = attach(10)
        svc = SAEService.for_registry()

        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=10, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )
        assert s10.get_steering_values() == {1: 40.0, 2: 30.0}

        AttachedSAEState().release_owner("circuit:B")
        assert s10.get_steering_values() == {1: 40.0}, (
            "releasing the composed co-tenant took the incumbent with it"
        )

    async def test_an_owner_re_serving_replaces_only_its_own_contribution(self):
        s10 = attach(10)
        svc = SAEService.for_registry()

        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=10, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )
        # A re-serves at double intensity.
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            2.0,
            owner_id="circuit:A",
        )
        assert s10.get_steering_values() == {1: 80.0, 2: 30.0}, (
            "re-serving A disturbed B's contribution"
        )


class TestTheOwnerRoutingIsACTUALLYWired:
    """These pin the WIRING, not the mechanism.

    Both mutations below passed the entire suite before these existed:
    reverting `deactivate` to a global clear, and disabling the owner routing
    inside the apply. A mechanism nothing calls is not a fix, and the only way
    to tell the difference is to assert through the production entry point.
    """

    async def test_set_circuit_steering_with_an_owner_registers_ownership(self):
        attach(10)
        svc = SAEService.for_registry()
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        assert AttachedSAEState().owner_keys("circuit:A") == {10: [1]}, (
            "the apply did not route through the owner map, so nothing scopes "
            "this circuit's release"
        )

    async def test_without_an_owner_the_legacy_path_still_applies(self):
        """The unmigrated callers must keep working verbatim."""
        s10 = attach(10)
        svc = SAEService.for_registry()
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)], 1.0
        )
        assert s10.get_steering_values() == {1: 40.0}
        assert AttachedSAEState().owner_keys("circuit:A") == {}

    async def test_an_OFF_circuit_releases_rather_than_pinning_zeros(self):
        """λ=0 must free the layer for its co-tenants rather than holding it at
        zero — a departed circuit pinning a layer down is the same class of
        defect as clearing one."""
        s10 = attach(10)
        svc = SAEService.for_registry()
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=10, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )
        # A dials to zero.
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            0.0,
            owner_id="circuit:A",
        )
        assert AttachedSAEState().owner_keys("circuit:A") == {}
        assert s10.get_steering_values() == {2: 30.0}, (
            "a circuit dialled to zero still holds the layer against its "
            "co-tenant"
        )


class TestRungSuppressionOnComposedLayers:
    """Feature 19 task 4.1/6.3 — `X-miLLM-Circuit-Rung` is OMITTED while any
    layer is composed.

    The rung describes ONE circuit's evidence. When two circuits sum on a
    layer, no single rung describes what the user actually received, and
    emitting either one would overclaim — the same rule that already omits the
    header for slice-fallback.

    This is the honesty half of the override: an operator may compose, but the
    response stops making a claim it can no longer support.
    """

    async def test_a_composed_layer_suppresses_the_rung(self, monkeypatch):
        from millm.services import inference_service as inf_mod
        from millm.services.inference_service import InferenceService

        circuit = SimpleNamespace(
            id="c", rung=2, name="n", layers=[10], serving_mode="full"
        )
        svc = InferenceService.__new__(InferenceService)

        async def _steering():
            return circuit

        svc._steering_circuit = _steering

        async def _composed():
            return True

        svc._any_layer_composed = _composed

        assert await svc.active_circuit_rung() is None, (
            "a composed layer still advertised one circuit's rung — the "
            "response claims evidence that describes only part of what "
            "produced it"
        )

    async def test_an_UNCOMPOSED_circuit_still_reports_its_rung(self, monkeypatch):
        """The other side: suppression must be specific, or the feature
        silently deletes the disclosure it was built to protect."""
        from millm.services.inference_service import InferenceService

        circuit = SimpleNamespace(
            id="c", rung=2, name="n", layers=[10], serving_mode="full"
        )
        svc = InferenceService.__new__(InferenceService)

        async def _steering():
            return circuit

        svc._steering_circuit = _steering

        async def _composed():
            return False

        svc._any_layer_composed = _composed

        assert await svc.active_circuit_rung() == (
            2,
            "causally validated (edge)",
        )

    async def test_an_unreadable_claims_table_does_not_delete_the_disclosure(
        self, monkeypatch
    ):
        """A deliberate trade-off, recorded rather than assumed.

        Composition needs an explicit override and is rare; an unreachable
        claims table is comparatively common. Suppressing on every DB error
        would silently delete the rung disclosure for every request during a
        blip — losing an honesty signal far more often than it prevents a wrong
        one, and losing it in the direction that tells the operator LESS.

        So an unreadable table reports not-composed and says so loudly.
        """
        from millm.services import inference_service as inf_mod
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)

        def boom():
            raise RuntimeError("claims table unreachable")

        monkeypatch.setattr(inf_mod, "async_session_factory", boom, raising=False)
        assert await svc._any_layer_composed() is False


class TestR1CircuitServingIsOBSERVABLE:
    """F19 R1-09/10/11 — the operator can see what is happening.

    Before these, composition — the state in which the runtime deliberately
    STOPS making evidence claims — was invisible to every dashboard and
    unalertable. The only "circuit" metric on the surface is the unrelated
    HuggingFace HTTP breaker, which is an active trap for anyone alerting on
    the word.
    """

    async def test_metrics_report_circuits_layers_and_COMPOSITION(self):
        from millm.api.routes.system.health import get_metrics

        s10 = attach(10)
        attach(13)
        svc = SAEService.for_registry()
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=13, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )

        metrics = await get_metrics()
        assert metrics.circuits_serving == 2
        assert metrics.circuit_layers_served == 2
        assert metrics.circuit_layers_composed == 0, (
            "disjoint circuits are not composed"
        )

        # Now compose B onto A's layer.
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=9, layer=10, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )
        metrics = await get_metrics()
        assert metrics.circuit_layers_composed == 1, (
            "a composed layer is invisible to metrics — the one state where "
            "the rung header is suppressed cannot be alerted on"
        )
        assert s10.get_steering_values() == {1: 40.0, 9: 30.0}

    async def test_the_prometheus_surface_carries_the_same_gauges(self):
        from millm.api.routes.system.health import get_prometheus_metrics

        attach(10)
        SAEService.for_registry().set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        # The route takes its loader via Depends; call it directly with a stub.
        loader = SimpleNamespace(is_loaded=False, model_name=None)
        response = await get_prometheus_metrics(model_loader=loader)
        body = response.body.decode() if hasattr(response, "body") else str(response)
        assert "millm_circuits_serving 1" in body
        assert "millm_circuit_layers_served 1" in body
        assert "millm_circuit_layers_composed 0" in body

    async def test_a_bypassed_claim_gate_is_LOUD(self, monkeypatch):
        """R1-09. A repository without a session disables contention AND
        collision checking entirely. It still degrades rather than refusing —
        a persistence detail must not stop the server serving — but it can no
        longer do so silently."""
        # F19 R3-14: this greped the source for the event name and
        # "logger.error". Moving either string, or logging at a different
        # level in a different branch, passed it.
        #
        # Driven instead — but by patching the MODULE's logger rather than
        # reconfiguring structlog globally. The first version called
        # `structlog.configure()`, and `reset_defaults()` does not restore the
        # app's own configuration, so it passed alone and failed in the full
        # suite depending on test order. A test that mutates global logging
        # config is a test defect, not a finding about the code.
        from millm.api.routes.system.health import metrics_counter
        from millm.services import circuit_service as cs_mod
        from millm.services.circuit_service import CircuitService

        calls: list[tuple[str, str]] = []

        class Spy:
            def __getattr__(self, level):
                def record(event, **_kw):
                    calls.append((level, event))

                return record

        monkeypatch.setattr(cs_mod, "logger", Spy())

        before = metrics_counter.circuit_claim_faults
        svc = CircuitService.__new__(CircuitService)
        svc.repository = SimpleNamespace()  # no `.session`
        composed = await svc._claim_layers(
            SimpleNamespace(id="c", name="n"), [10, 13], None, False
        )

        assert composed == []
        bypass = [lvl for lvl, event in calls if event == "circuit_claim_gate_BYPASSED"]
        assert bypass, "a serve-without-claiming path emitted nothing"
        assert bypass[0] == "error", (
            f"the bypass logged at {bypass[0]!r} — a warning is "
            "indistinguishable from routine noise, and this restores pre-F19 "
            "silent-clobber behaviour behind a healthy-looking response"
        )
        assert metrics_counter.circuit_claim_faults > before


class TestR2TheComposedMetricSeesSLICECoTenants:
    """F19 R2-11. `circuit_layers_composed` counted only `circuit:` owners, so
    it was blind to the case it most needed to catch.

    A layer whose co-tenant arrived through SLICE-FALLBACK materialises a
    CLUSTER PROFILE, not a circuit owner. That layer has ONE circuit owner, so
    the metric read 0 — the documented alertable condition never firing — while
    `GET /circuits/claims` badged the layer composed and the rung header WAS
    being suppressed.

    Two authorities disagreeing is worse than either being wrong: the metric
    said "nothing composed" while the response headers said otherwise.
    """

    async def test_a_non_circuit_co_tenant_counts_as_composed(self):
        from millm.api.routes.system.health import get_metrics

        attach(10)
        svc = SAEService.for_registry()
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        # A cluster profile — NOT a `circuit:` owner — takes the same layer.
        AttachedSAEState().apply_owner("cluster:prof_1", {("sae-10", 10): {9: 20.0}})

        metrics = await get_metrics()
        assert metrics.circuit_layers_composed == 1, (
            "a slice-fallback co-tenant on a circuit's layer is invisible to "
            "the metric, so the alertable condition never fires while the "
            "rung header is already suppressed"
        )

    async def test_a_co_tenant_on_an_UNRELATED_layer_does_not_count(self):
        """The metric is about CIRCUIT-held layers. A cluster steering a layer
        no circuit touches is not a composition of anything."""
        from millm.api.routes.system.health import get_metrics

        attach(10)
        attach(13)
        SAEService.for_registry().set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        AttachedSAEState().apply_owner("cluster:prof_1", {("sae-13", 13): {9: 20.0}})

        metrics = await get_metrics()
        assert metrics.circuit_layers_composed == 0
        assert metrics.circuits_serving == 1


class TestR2ClaimFaultsAreCOUNTED:
    """F19 R2-16. Every one of F19's failure handlers logged and continued, so
    a claim leak, a failed rollback, or a bypassed gate was discoverable ONLY
    by grepping structured logs for a specific event name. An operator watching
    a dashboard saw a fully green system while layers leaked.

    These are the states where the claims table and what is actually steering
    can disagree — exactly the divergence F19 exists to remove — so they need a
    number, not just a log line.
    """

    async def test_the_counter_is_EXPOSED_on_both_surfaces(self):
        from millm.api.routes.system.health import (
            get_metrics,
            get_prometheus_metrics,
        )

        metrics = await get_metrics()
        assert hasattr(metrics, "circuit_claim_faults")

        loader = SimpleNamespace(is_loaded=False, model_name=None)
        body = (await get_prometheus_metrics(model_loader=loader)).body.decode()
        assert "millm_circuit_claim_faults_total" in body

    async def test_a_real_failure_INCREMENTS_it(self):
        """Wired, not merely declared.

        Driven through the CLAIM GATE's bypass path, which is the reachable
        divergence: a repository with no session means the activation was NOT
        checked for contention or collisions and took no claims — pre-F19
        silent-clobber behaviour behind a healthy-looking response.

        (The owner-rollback handler needs a rebuild that SUCCEEDS and then
        fails on restore. A collision raises during the merge, before any write,
        so no layer ever enters the rollback's `done` list — verified by
        instrumenting the path rather than assuming. That handler is covered by
        its own rollback tests; this asserts the COUNTER is reached from a real
        failure.)
        """
        from millm.api.routes.system.health import metrics_counter
        from millm.services.circuit_service import CircuitService

        before = metrics_counter.circuit_claim_faults

        svc = CircuitService.__new__(CircuitService)
        svc.repository = SimpleNamespace()  # no `.session`
        circuit = SimpleNamespace(id="c", name="n")

        composed = await svc._claim_layers(circuit, [10, 13], None, False)
        assert composed == []

        assert metrics_counter.circuit_claim_faults > before, (
            "the claim gate was BYPASSED — no contention or collision check, "
            "no claims taken — and nothing counted it, so an operator watching "
            "the dashboard sees a green system"
        )

    async def test_counting_cannot_break_the_caller(self, monkeypatch):
        """A metrics helper must never fail the serving path it reports on.

        Breaks the COUNTER's method rather than replacing the module attribute:
        an earlier version swapped `metrics_counter` for a raising property,
        which leaked into the sibling test above and made it read 0. A test
        that corrupts shared module state is a defect in the test, not a
        finding about the code.
        """
        from millm.api.routes.system.health import metrics_counter
        from millm.services import circuit_service as cs_mod

        def boom() -> None:
            raise RuntimeError("counter exploded")

        monkeypatch.setattr(
            metrics_counter, "increment_circuit_claim_faults", boom
        )
        cs_mod._note_claim_fault()  # must not raise


class TestR2BothMetricSurfacesAGREE:
    """F19 R2-20. The JSON and Prometheus surfaces each re-implemented the
    owner scan, and already DIFFERED in how they derived `layers_served` — one
    from a set of layers, the other from the length of a holder map.

    Two surfaces computing the same three numbers independently is a place for
    them to drift, and an operator whose dashboard reads one while their alert
    reads the other has no way to tell which is right. They now share one
    definition.
    """

    async def test_the_two_surfaces_report_identical_numbers(self):
        from millm.api.routes.system.health import (
            get_metrics,
            get_prometheus_metrics,
        )

        attach(10)
        attach(13)
        svc = SAEService.for_registry()
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        svc.set_circuit_steering(
            [CircuitMember(feature_idx=2, layer=13, budget=30.0, sign=1)],
            1.0,
            owner_id="circuit:B",
        )
        # A composed layer, so all three numbers are non-trivial.
        AttachedSAEState().apply_owner("cluster:p", {("sae-10", 10): {9: 5.0}})

        json_metrics = await get_metrics()
        loader = SimpleNamespace(is_loaded=False, model_name=None)
        prom = (await get_prometheus_metrics(model_loader=loader)).body.decode()

        for field, gauge in (
            ("circuits_serving", "millm_circuits_serving"),
            ("circuit_layers_served", "millm_circuit_layers_served"),
            ("circuit_layers_composed", "millm_circuit_layers_composed"),
        ):
            value = getattr(json_metrics, field)
            assert f"{gauge} {value}" in prom, (
                f"{gauge} disagrees with JSON {field}={value} — a dashboard "
                "and an alert built on these would contradict each other"
            )

        assert json_metrics.circuit_layers_composed == 1


class TestR3TheDialRefusesToGuessWhichCircuit:
    """F19 R3-06. `_active_full_circuit` read `get_active()`, which returns the
    most recently updated row.

    So with TWO circuits serving, the dial, the intensity resolution and the
    rung header all described ONE of them while the response carried BOTH
    circuits' summed steering. An operator dialling "the active circuit"
    changed a different circuit than the header named.

    Same rule as composition, for the same reason: no single circuit's evidence
    describes the response, so return None rather than name one arbitrarily.
    """

    @pytest.mark.asyncio
    async def test_two_serving_circuits_suppress_the_dial_and_rung(
        self, monkeypatch
    ):
        from millm.services import inference_service as inf_mod
        from millm.services.inference_service import InferenceService

        rows = [
            SimpleNamespace(id="cA", serving_mode="full", rung=2, name="A"),
            SimpleNamespace(id="cB", serving_mode="full", rung=2, name="B"),
        ]

        class Repo:
            def __init__(self, _session):
                pass

            async def list_active(self):
                return rows

        monkeypatch.setattr(
            "millm.db.repositories.circuit_repository.CircuitRepository", Repo
        )

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(
            "millm.db.base.async_session_factory", lambda: Session()
        )

        svc = InferenceService.__new__(InferenceService)
        assert await svc._active_full_circuit() is None, (
            "the dial named ONE of two serving circuits — an operator dialling "
            "'the active circuit' changes a different one than the rung header "
            "describes"
        )

    @pytest.mark.asyncio
    async def test_ONE_serving_circuit_still_resolves(self, monkeypatch):
        """Specificity: refusing whenever anything is active would delete the
        dial entirely."""
        from millm.services.inference_service import InferenceService

        row = SimpleNamespace(id="cA", serving_mode="full", rung=2, name="A")

        class Repo:
            def __init__(self, _session):
                pass

            async def list_active(self):
                return [row]

        monkeypatch.setattr(
            "millm.db.repositories.circuit_repository.CircuitRepository", Repo
        )

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(
            "millm.db.base.async_session_factory", lambda: Session()
        )

        svc = InferenceService.__new__(InferenceService)
        assert (await svc._active_full_circuit()) is row

    @pytest.mark.asyncio
    async def test_a_slice_fallback_circuit_does_not_count(self, monkeypatch):
        """Only FULL serving reaches the multi-SAE dial; a slice serve is
        steered by a cluster profile, not by this path."""
        from millm.services.inference_service import InferenceService

        rows = [
            SimpleNamespace(id="cA", serving_mode="full", rung=2, name="A"),
            SimpleNamespace(id="cB", serving_mode="slice_fallback", rung=2, name="B"),
        ]

        class Repo:
            def __init__(self, _session):
                pass

            async def list_active(self):
                return rows

        monkeypatch.setattr(
            "millm.db.repositories.circuit_repository.CircuitRepository", Repo
        )

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(
            "millm.db.base.async_session_factory", lambda: Session()
        )

        svc = InferenceService.__new__(InferenceService)
        resolved = await svc._active_full_circuit()
        assert resolved is not None and resolved.id == "cA"


class TestR3TheWORSTOutcomeIsCountedToo:
    """F19 R3-11. Only the PARTIAL-loss branch counted a fault.

    The TOTAL loss — circuit active and steering with ZERO claims, so anyone
    can take its layers — was the uncounted one. R2-16's stated goal ("all four
    handlers") held only because the partial and total conditions happen to
    overlap today, not by construction: an operator alerting on the fault rate
    would see nothing while a circuit steers with no claim at all.

    The activation rollback — the path R1 named "most easily missed" — was
    likewise uncounted.
    """

    @pytest.mark.asyncio
    async def test_a_TOTAL_restore_loss_counts_a_fault(self, monkeypatch):
        """The worst outcome, driven rather than greped.

        A window-based source assertion could not catch this: moving the call
        a few lines still passes. Verified by a mutation that SURVIVED the
        grep version of this test.

        Here the gate's re-claim fails AND every restore fails, so the circuit
        ends active and steering with ZERO claims — anyone can take its layers.
        """
        from millm.api.routes.system.health import metrics_counter
        from millm.services import circuit_claim_registry as reg_mod
        from millm.services.circuit_service import CircuitService

        before = metrics_counter.circuit_claim_faults

        held = [
            SimpleNamespace(
                circuit_id="cA", layer=10, composed=False, steering_keys=(1,)
            )
        ]

        class Registry:
            def __init__(self, _session):
                pass

            async def live_claims(self):
                return held

            async def assess(self, *a, **k):
                return SimpleNamespace(
                    has_collision=False,
                    has_contention=False,
                    contended_layers=(),
                    incumbents={},
                    colliding_keys=(),
                )

            async def release(self, _cid):
                return [10]

            async def claim(self, *a, **k):
                raise RuntimeError("every claim fails")

        monkeypatch.setattr(reg_mod, "CircuitClaimRegistry", Registry)

        svc = CircuitService.__new__(CircuitService)
        svc.repository = SimpleNamespace(session=object())
        definition = SimpleNamespace(
            members=[], edges=[], budget=None,
            sae_for_layer=lambda layer: None, layers=lambda: [10],
        )

        with pytest.raises(RuntimeError, match="every claim fails"):
            await svc._claim_layers(
                SimpleNamespace(id="cA", name="cA"), [10], definition, False
            )

        assert metrics_counter.circuit_claim_faults > before, (
            "the circuit is active and steering with NO claims — the worst "
            "outcome available here — and the dashboard counted nothing"
        )

    def test_the_rollback_path_also_counts(self):
        """The activation rollback is the path R1 named 'most easily missed'.
        Asserted structurally because driving it needs a failed DB write mid
        activation; the behavioural half is covered above."""
        import inspect

        from millm.services.circuit_service import CircuitService

        src = inspect.getsource(CircuitService.activate)
        handler = src[src.index("circuit_activate_rollback_clear_failed") - 400 :]
        assert "_note_claim_fault()" in handler[:400]


class TestR3TheTwoCompositionAUTHORITIESAreBothReported:
    """F19 R3-17. There are THREE authorities on "is a layer composed", and
    they can disagree:

      1. the OWNER MAP — what is contributing to each layer right now;
      2. the CLAIMS TABLE — what the gate recorded, and what
         `_any_layer_composed` reads to SUPPRESS the rung header;
      3. the gate's verdict at activation time.

    R2-11 widened `circuit_layers_composed` to count non-circuit co-tenants so
    it would "agree with header suppression". It does not — suppression reads
    the CLAIMS TABLE — so a cluster co-tenant makes the metric report composed
    while the header is still emitted. That is the same two-authorities
    defect R2-11 set out to remove, inverted.

    Reporting both is the honest fix: equal is healthy, a gap is the signal.
    """

    async def test_a_runtime_composition_with_no_CLAIM_shows_a_gap(self):
        from millm.api.routes.system.health import get_metrics

        attach(10)
        SAEService.for_registry().set_circuit_steering(
            [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)],
            1.0,
            owner_id="circuit:A",
        )
        # A cluster co-tenant: RUNTIME composition, no claim row.
        AttachedSAEState().apply_owner("cluster:p", {("sae-10", 10): {9: 5.0}})

        metrics = await get_metrics()
        assert metrics.circuit_layers_composed == 1, "runtime composition missed"
        assert metrics.circuit_layers_composed_claimed == 0, (
            "the claims table records no composition here, and that is the "
            "point: the rung header is NOT suppressed for this layer, so a "
            "metric reporting only the runtime view contradicts the headers"
        )

    async def test_a_CLAIMED_composition_is_reported(self, monkeypatch):
        """The half the previous assertion could not see.

        `assert claimed == 0` passes whether the gauge is computed or
        hard-wired to zero — verified by a mutation that SURVIVED it. Only a
        NON-zero case proves the claims table is actually read.
        """
        from millm.api.routes.system import health as health_mod

        async def two_composed_layers():
            return [
                SimpleNamespace(layer=10, composed=True),
                SimpleNamespace(layer=13, composed=True),
                SimpleNamespace(layer=20, composed=False),
            ]

        class Registry:
            def __init__(self, _session):
                pass

            live_claims = staticmethod(two_composed_layers)

        monkeypatch.setattr(
            "millm.services.circuit_claim_registry.CircuitClaimRegistry", Registry
        )

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(
            "millm.db.base.async_session_factory", lambda: Session()
        )

        metrics = await health_mod.get_metrics()
        assert metrics.circuit_layers_composed_claimed == 2, (
            "the claims table is not being read, so a drift between the gate "
            "and the runtime is invisible — and the gauge that would show it "
            "reports a constant"
        )
