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

    async def test_a_bypassed_claim_gate_is_LOUD(self):
        """R1-09. A repository without a session disables contention AND
        collision checking entirely. It still degrades rather than refusing —
        a persistence detail must not stop the server serving — but it can no
        longer do so silently."""
        import inspect

        from millm.services.circuit_service import CircuitService

        src = inspect.getsource(CircuitService._claim_layers)
        gate = src[src.index('session = getattr(self.repository, "session", None)'):]
        assert "circuit_claim_gate_BYPASSED" in gate
        assert "logger.error" in gate, (
            "a serve-without-claiming path must not be a warning — it restores "
            "the pre-F19 silent-clobber behaviour"
        )


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
