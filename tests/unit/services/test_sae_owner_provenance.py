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


class TestR1PartialRebuildRollsBack:
    """F19 R1-12. `_rebuild_layer` RAISES on a colliding owner map, and the
    raise can land partway through a multi-layer apply — leaving earlier
    layers written at the NEW values, later ones at the OLD, and this owner's
    entry already replaced.

    The exception then propagates out of `set_circuit_steering` into
    `activate`, which has no handler there, so the circuit is left a chimera:
    partial steering, a live owner entry, and a claim row it already took.
    That is exactly the class F18 R3-09/10 fixed for the legacy apply path,
    reintroduced on the owner path.
    """

    def test_a_collision_partway_through_restores_the_previous_state(self, state):
        s10 = _attach(state, 10)
        s11 = _attach(state, 11)
        _attach(state, 12)

        state.apply_owner("circuit:A", {("s10", 10): {1: 10.0}, ("s11", 11): {2: 20.0}})
        # A co-tenant owns feature 9 on L12, so A cannot also take it.
        state.apply_owner("circuit:B", {("s12", 12): {9: 90.0}})

        with pytest.raises(ValueError, match="collision"):
            state.apply_owner(
                "circuit:A",
                {
                    ("s10", 10): {1: 99.0},
                    ("s11", 11): {2: 99.0},
                    ("s12", 12): {9: 50.0},  # collides with B
                },
            )

        assert state.owner_keys("circuit:A") == {10: [1], 11: [2]}, (
            "the failed apply left the owner map at its NEW value, so the "
            "circuit reads as owning a layer it never took"
        )
        assert s10.get_steering_values() == {1: 10.0}, (
            "layer 10 kept the new value from a partial apply that failed"
        )
        assert s11.get_steering_values() == {2: 20.0}

    def test_a_FIRST_apply_that_fails_leaves_no_owner_behind(self, state):
        _attach(state, 10)
        state.apply_owner("circuit:B", {("s10", 10): {1: 10.0}})

        with pytest.raises(ValueError, match="collision"):
            state.apply_owner("circuit:A", {("s10", 10): {1: 99.0}})

        assert state.owner_keys("circuit:A") == {}, (
            "a first apply that failed registered the owner anyway"
        )


class TestR1DetachedLayersDropTheirContributions:
    """F19 R1-13. Skipping a detached layer left the owner map desynchronised.

    After a detach and re-attach (a reload) the new entry has empty steering
    while `_owners` still records the old contribution: the circuit reads as
    serving that layer while steering nothing — and a LATER rebuild triggered
    by a co-tenant would include the stale contribution and RESURRECT steering
    the operator believed had stopped.
    """

    def test_the_contribution_is_dropped_when_the_layer_goes(self, state):
        """Driven through a rebuild triggered by SOMEONE ELSE, which is how
        this is actually reached in production. `release_owner` pops the owner
        regardless, so testing through it proves nothing about the drop —
        verified by a mutation that survived that version of this test."""
        _attach(state, 10)
        _attach(state, 11)
        state.apply_owner("circuit:A", {("s10", 10): {1: 40.0}})
        state.apply_owner("circuit:B", {("s11", 11): {2: 20.0}})
        assert state.owner_keys("circuit:A") == {10: [1]}

        # L10 detaches. B re-applies, which rebuilds its own layer only — so
        # A's stale contribution is dropped when something next touches L10.
        del state._entries[("s10", 10)]
        state.apply_owner("circuit:B", {("s11", 11): {2: 25.0}, ("s10", 10): {}})

        assert state.owner_keys("circuit:A") == {}, (
            "the detached layer's contribution survived in the owner map, so "
            "the circuit reads as steering a layer that no longer exists"
        )

    def test_a_re_attach_does_not_resurrect_stale_steering(self, state):
        """The operator-visible consequence, asserted directly."""
        _attach(state, 10)
        state.apply_owner("circuit:A", {("s10", 10): {1: 40.0}})

        # Detach, then re-attach a fresh SAE at the same key (a reload).
        state._entries.clear()
        state.apply_owner("circuit:A", {})  # any rebuild touching the gone key
        fresh = _attach(state, 10)

        # A co-tenant arriving must not drag A's stale contribution back.
        state.apply_owner("circuit:B", {("s10", 10): {9: 30.0}})
        assert fresh.get_steering_values() == {9: 30.0}, (
            "a detached circuit's steering was resurrected by a co-tenant's "
            "rebuild — the operator stopped it and it came back"
        )
