"""Feature 19 task 3.1/3.8 — per-owner steering provenance.

The defect these tests exist to prevent is the highest-consequence one in
concurrent serving: releasing circuit A clears a `(layer, feature_idx)` key
belonging to circuit B. B then silently stops steering while its row still
reads active, and nothing anywhere reports it.

The old apply cleared each layer's whole steering dict before writing. That is
correct when one circuit can serve and catastrophic when two can, so the
registry recomputes each layer from the surviving owners instead.
"""

import pytest

from millm.services.sae_service import AttachedSAEState


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


@pytest.fixture
def state():
    st = AttachedSAEState()
    st.reset_for_tests()
    yield st
    st.reset_for_tests()


def _attach(state, layer, sae_id=None):
    from millm.services.sae_service import AttachedEntry

    sae = FakeSAE()
    sid = sae_id or f"s{layer}"
    state._entries[(sid, layer)] = AttachedEntry(
        sae=sae, sae_id=sid, layer=layer, hook_handle=None
    )
    return sae


class TestCoTenantsSurvive:
    def test_releasing_one_owner_leaves_the_others_steering(self, state):
        """THE defect. Two circuits on the SAME layer with different features
        — the composed case — and releasing one must not disturb the other."""
        sae = _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        state.apply_owner("circuit:B", {("s10", 10): {99: 30.0}})
        assert sae.get_steering_values() == {42: 40.0, 99: 30.0}

        state.release_owner("circuit:A")
        assert sae.get_steering_values() == {99: 30.0}, (
            "releasing A tore out B's steering — B now serves nothing while "
            "its row still reads active"
        )
        assert sae.is_steering_enabled is True

    def test_releasing_on_a_disjoint_layer_does_not_touch_the_other(self, state):
        s10 = _attach(state, 10)
        s13 = _attach(state, 13)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        state.apply_owner("circuit:B", {("s13", 13): {99: 30.0}})

        state.release_owner("circuit:A")
        assert s10.get_steering_values() == {}
        assert s10.is_steering_enabled is False
        assert s13.get_steering_values() == {99: 30.0}
        assert s13.is_steering_enabled is True

    def test_the_last_owner_leaving_disables_the_layer(self, state):
        sae = _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        state.release_owner("circuit:A")
        assert sae.get_steering_values() == {}
        assert sae.is_steering_enabled is False, (
            "an unowned layer must be disabled, not left armed at zero"
        )

    def test_re_applying_an_owner_REPLACES_its_contribution(self, state):
        """An owner re-applying at a new intensity must not accumulate its own
        previous values on top of the new ones."""
        sae = _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0, 43: 10.0}})
        state.apply_owner("circuit:A", {("s10", 10): {42: 80.0}})
        assert sae.get_steering_values() == {42: 80.0}, (
            "the owner's stale key 43 survived its own re-apply"
        )

    def test_re_applying_drops_LAYERS_the_owner_no_longer_steers(self, state):
        """The half a same-key re-apply cannot see. An owner whose new claim
        set covers FEWER layers must stop steering the dropped ones — a dict
        `.update()` would leave the old layer's contribution resident forever,
        and a mutation to that effect SURVIVED the same-key test above until
        this was added."""
        s10 = _attach(state, 10)
        s13 = _attach(state, 13)
        state.apply_owner(
            "circuit:A", {("s10", 10): {42: 40.0}, ("s13", 13): {99: 30.0}}
        )
        assert s13.get_steering_values() == {99: 30.0}

        # Re-apply covering ONLY layer 10 — the circuit's members changed.
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        assert s13.get_steering_values() == {}, (
            "layer 13 kept steering for an owner that no longer claims it"
        )
        assert s13.is_steering_enabled is False
        assert s10.get_steering_values() == {42: 40.0}

    def test_releasing_an_unknown_owner_is_a_no_op(self, state):
        sae = _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        assert state.release_owner("circuit:GHOST") == []
        assert sae.get_steering_values() == {42: 40.0}


class TestCollisionRaisesRatherThanPickingAWinner:
    def test_two_owners_on_one_key_RAISE(self, state):
        """No honest composition exists: one strength would silently win and
        the served value would belong to neither author. The claim registry
        refuses this up front, so reaching the apply means the gate was
        bypassed — fail loudly rather than choosing."""
        _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        with pytest.raises(ValueError, match="collision"):
            state.apply_owner("circuit:B", {("s10", 10): {42: 30.0}})

    def test_the_raise_names_both_owners_and_the_key(self, state):
        _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        with pytest.raises(ValueError) as exc:
            state.apply_owner("circuit:B", {("s10", 10): {42: 30.0}})
        message = str(exc.value)
        assert "circuit:A" in message and "circuit:B" in message
        assert "42" in message and "L10" in message


class TestOwnerKeys:
    def test_owner_keys_reports_what_that_owner_steers(self, state):
        _attach(state, 10)
        _attach(state, 13)
        state.apply_owner(
            "circuit:A", {("s10", 10): {42: 1.0, 7: 2.0}, ("s13", 13): {99: 3.0}}
        )
        assert state.owner_keys("circuit:A") == {10: [7, 42], 13: [99]}

    def test_owner_keys_is_empty_for_an_unknown_owner(self, state):
        assert state.owner_keys("circuit:GHOST") == {}


class TestDetachedLayers:
    def test_a_detached_layer_is_skipped_not_crashed(self, state):
        """Attachment can change under a live owner map; rebuilding a layer
        that is gone must degrade rather than raise on the serving path."""
        _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {42: 40.0}})
        state.reset_for_tests()
        state.release_owner("circuit:A")  # must not raise
