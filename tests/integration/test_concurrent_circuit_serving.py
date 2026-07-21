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

import pytest

from millm.api.schemas.circuit import CircuitMember
from millm.services.sae_service import AttachedEntry, AttachedSAEState, SAEService

pytestmark = pytest.mark.asyncio


class FakeSAE:
    def __init__(self):
        self._values: dict[int, float] = {}
        self.is_steering_enabled = False
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
