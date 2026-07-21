"""Feature 19 task 6.5 — two concurrent activations for one layer.

`assess()` and the claim INSERT are a check-then-act pair: two activations can
both pass `assess` before either inserts. So the service-level pre-check cannot
be what decides a race, and a test that only exercised the pre-check would
report the race as handled while the real window stayed open.

These assert that the DATABASE INDEX decides, with the pre-check deliberately
bypassed — `registry.claim()` is called directly, exactly as the losing side of
a race reaches it.
"""

import pytest

from millm.core.errors import CircuitLayerContentionError
from millm.db.models.circuit import Circuit
from millm.services.circuit_claim_registry import CircuitClaimRegistry

pytestmark = pytest.mark.asyncio


async def _circuit(session, cid, layers=(10,)):
    session.add(
        Circuit(
            id=cid,
            name=cid,
            circuit_meta={"kind": "mistudio.circuit-definition", "schema_version": "1"},
            rung=2,
            edge_count=0,
            layers=list(layers),
            per_sae_warnings=[],
            serveable=True,
            is_active=True,
            provenance={},
        )
    )
    await session.flush()


class TestTheIndexDecidesTheRace:
    async def test_exactly_one_claim_survives_with_the_precheck_BYPASSED(
        self, test_session
    ):
        """EC-19.7. No `assess()` call — this is the losing side of a race
        arriving at the INSERT with a stale verdict."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        registry = CircuitClaimRegistry(test_session)

        await registry.claim("cA", {10})

        with pytest.raises(CircuitLayerContentionError):
            await registry.claim("cB", {10})

        live = await registry.live_claims()
        holders = {c.circuit_id for c in live if c.layer == 10}
        assert holders == {"cA"}, (
            f"two circuits hold layer 10 ({holders}) — the index did not "
            "decide the race, so nothing did"
        )

    async def test_the_loser_gets_an_ordinary_CONTENTION_refusal(
        self, test_session
    ):
        """A race loss must read identically to the sequential case. An
        operator should not have to distinguish 'refused' from 'refused
        because of a race' — the remedy is the same either way."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        registry = CircuitClaimRegistry(test_session)
        await registry.claim("cA", {10})

        with pytest.raises(CircuitLayerContentionError) as exc:
            await registry.claim("cB", {10})

        assert exc.value.code == "CIRCUIT_LAYER_CONTENTION"
        assert exc.value.status_code == 200
        assert 10 in exc.value.details["contended_layers"]

    async def test_a_lost_race_leaves_NO_partial_claim(self, test_session):
        """The loser must not leave rows behind on the layers it DID get.

        A multi-layer circuit claiming {10, 13} where only 13 is free would
        otherwise insert 13, fail on 10, and leave a live claim for a circuit
        that never activated — refusing every future activation on 13 for a
        circuit nobody can deactivate.
        """
        await _circuit(test_session, "cA", layers=(10,))
        await _circuit(test_session, "cB", layers=(10, 13))
        registry = CircuitClaimRegistry(test_session)
        await registry.claim("cA", {10})

        with pytest.raises(CircuitLayerContentionError):
            await registry.claim("cB", {10, 13})

        live = await registry.live_claims()
        assert {c.circuit_id for c in live} == {"cA"}, (
            "the losing activation left a partial claim behind — layer 13 is "
            "now held by a circuit that never activated"
        )

    async def test_a_composed_claim_does_not_lose_the_race(self, test_session):
        """Composition is the explicit override, so the index must let those
        rows coexist — otherwise the override would be unreachable by
        construction."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        registry = CircuitClaimRegistry(test_session)

        await registry.claim("cA", {10}, composed=True)
        await registry.claim("cB", {10}, composed=True)  # must not raise

        holders = {c.circuit_id for c in await registry.live_claims() if c.layer == 10}
        assert holders == {"cA", "cB"}


class TestTheRefusalNamesTheWinner:
    """F19 R1-14. The race handler raised with `incumbent=None` and the FULL
    requested layer set.

    So a race loss reported "Layers [10, 13, 14] are already served by another
    active circuit" when only L13 collided and the winner is a named,
    discoverable circuit. Worse, the contention dialog renders its
    "Deactivate '<incumbent>'" action ONLY when `incumbent.id` is present — so
    the operator got a refusal with no route out of it.
    """

    async def test_the_loser_learns_WHO_won(self, test_session):
        await _circuit(test_session, "cA", layers=(10,))
        await _circuit(test_session, "cB", layers=(10, 13))
        registry = CircuitClaimRegistry(test_session)
        await registry.claim("cA", {10})

        with pytest.raises(CircuitLayerContentionError) as exc:
            await registry.claim("cB", {10, 13})

        details = exc.value.details
        assert details["incumbent"]["id"] == "cA", (
            "the refusal named nobody, so the dialog cannot offer the one "
            "action that resolves it"
        )
        assert details["incumbent"]["name"] == "cA"

    async def test_only_the_ACTUALLY_contended_layers_are_reported(
        self, test_session
    ):
        """Reporting the full requested set tells the operator three layers
        are taken when one is — and sends them editing circuits that were
        never in conflict."""
        await _circuit(test_session, "cA", layers=(13,))
        await _circuit(test_session, "cB", layers=(10, 13, 14))
        registry = CircuitClaimRegistry(test_session)
        await registry.claim("cA", {13})

        with pytest.raises(CircuitLayerContentionError) as exc:
            await registry.claim("cB", {10, 13, 14})

        assert exc.value.details["contended_layers"] == [13], (
            "layers 10 and 14 were free — naming them as contended sends the "
            "operator to fix circuits that were never in conflict"
        )
