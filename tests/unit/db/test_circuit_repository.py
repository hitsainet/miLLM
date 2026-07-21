"""Tests for the circuits table, model and repository (Feature 13, task 1.0).

Pins the model defaults, the single-active partial unique index
(``uq_circuits_active``), the ``validated`` rung property, and repository CRUD
including the deactivate-then-activate ordering that keeps the index safe.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from millm.db.models.circuit import Circuit
from millm.db.repositories.circuit_repository import CircuitRepository


@pytest.fixture
def make_circuit():
    def _make(**overrides):
        defaults = dict(
            id="circ_0001",
            name="fear→threat circuit",
            circuit_meta={"kind": "mistudio.circuit-definition", "schema_version": "1"},
            layers=[10, 13],
        )
        defaults.update(overrides)
        return Circuit(**defaults)

    return _make


class TestCircuitModel:
    async def test_defaults(self, test_session, make_circuit):
        circuit = make_circuit()
        test_session.add(circuit)
        await test_session.commit()

        row = (await test_session.execute(select(Circuit))).scalar_one()
        assert row.rung == 0
        assert row.edge_count == 0
        assert row.serveable is False
        assert row.is_active is False
        assert row.serving_mode is None
        assert row.intensity == 1.0
        assert row.layers == [10, 13]

    async def test_meta_round_trips_verbatim(self, test_session, make_circuit):
        """circuit_meta stores the RAW document (lossless re-export)."""
        doc = {
            "kind": "mistudio.circuit-definition",
            "schema_version": "1",
            "name": "c",
            "saes": [{"layer": 10, "n_features": 8192}],
            "members": [{"feature_idx": 1, "layer": 10}],
            "edges": [{"up": {"layer": 10, "feature_idx": 1},
                       "down": {"layer": 13, "feature_idx": 2}, "rung": 2}],
            "some_future_field": {"unknown": True},  # must survive
        }
        test_session.add(make_circuit(circuit_meta=doc))
        await test_session.commit()
        row = (await test_session.execute(select(Circuit))).scalar_one()
        assert row.circuit_meta == doc
        assert row.circuit_meta["some_future_field"] == {"unknown": True}

    @pytest.mark.parametrize("rung,expected", [(0, False), (1, False), (2, True), (3, True)])
    async def test_validated_property(self, test_session, make_circuit, rung, expected):
        """`validated` is the rung>=2 gate — below it, activation needs an ack
        and the word 'causal' is forbidden."""
        circuit = make_circuit(rung=rung)
        assert circuit.validated is expected


class TestSingleActiveIndex:
    async def test_SEVERAL_circuits_may_now_be_active(
        self, test_session, make_circuit
    ):
        """Feature 19 SUPERSEDES the single-active index.

        This test previously asserted that a second active circuit raises
        `IntegrityError`. That guarantee is deliberately gone: the constraint
        moved from "one circuit" to "one circuit PER LAYER", enforced by
        `circuit_layer_claims.uq_circuit_layer_claim_live`, because the layer
        is the unit contention actually has — two circuits on the same layer
        sum into one steering dict and nothing bounds that sum.

        Kept rather than deleted, and INVERTED rather than weakened, so the
        supersession is recorded where the old rule lived. Two circuits on
        DISJOINT layers is the whole point of the feature; the per-layer rule
        is asserted in `test_circuit_claim_registry.py`.
        """
        test_session.add(
            make_circuit(id="c1", name="one", is_active=True, layers=[10])
        )
        test_session.add(
            make_circuit(id="c2", name="two", is_active=True, layers=[13])
        )
        await test_session.commit()

        actives = (
            await test_session.execute(
                select(Circuit).where(Circuit.is_active.is_(True))
            )
        ).scalars().all()
        assert len(actives) == 2

    async def test_many_inactive_allowed(self, test_session, make_circuit):
        test_session.add(make_circuit(id="c1", name="one"))
        test_session.add(make_circuit(id="c2", name="two"))
        test_session.add(make_circuit(id="c3", name="three"))
        await test_session.commit()
        rows = (await test_session.execute(select(Circuit))).scalars().all()
        assert len(rows) == 3


class TestCircuitRepository:
    async def test_create_get_by_id_and_name(self, test_session):
        repo = CircuitRepository(test_session)
        created = await repo.create(
            id="c1", name="alpha", circuit_meta={"k": 1}, layers=[3]
        )
        assert created.id == "c1"
        assert (await repo.get("c1")).name == "alpha"
        assert (await repo.get_by_name("alpha")).id == "c1"
        assert await repo.get("nope") is None

    async def test_set_active_deactivates_previous(self, test_session):
        repo = CircuitRepository(test_session)
        await repo.create(id="c1", name="one", circuit_meta={}, layers=[1])
        await repo.create(id="c2", name="two", circuit_meta={}, layers=[2])

        await repo.set_active("c1", serving_mode="full")
        assert (await repo.get_active()).id == "c1"

        # Activating the second must deactivate the first (index safety).
        await repo.set_active("c2", serving_mode="slice_fallback")
        active = await repo.get_active()
        assert active.id == "c2" and active.serving_mode == "slice_fallback"
        assert (await repo.get("c1")).is_active is False

    async def test_deactivate_clears_serving_mode(self, test_session):
        repo = CircuitRepository(test_session)
        await repo.create(id="c1", name="one", circuit_meta={}, layers=[1])
        await repo.set_active("c1", serving_mode="full")
        await repo.deactivate("c1")
        row = await repo.get("c1")
        assert row.is_active is False and row.serving_mode is None

    async def test_set_active_unknown_id_returns_none(self, test_session):
        repo = CircuitRepository(test_session)
        assert await repo.set_active("ghost") is None

    async def test_filters_min_rung_and_serveable(self, test_session):
        repo = CircuitRepository(test_session)
        await repo.create(id="c1", name="mined", circuit_meta={}, layers=[1], rung=0)
        await repo.create(
            id="c2", name="validated", circuit_meta={}, layers=[2], rung=2, serveable=True
        )
        assert {c.id for c in await repo.get_all(min_rung=2)} == {"c2"}
        assert {c.id for c in await repo.get_all(serveable=True)} == {"c2"}
        assert len(await repo.get_all()) == 2
        assert await repo.count(min_rung=2) == 1

    async def test_pagination(self, test_session):
        repo = CircuitRepository(test_session)
        for i in range(5):
            await repo.create(id=f"c{i}", name=f"n{i}", circuit_meta={}, layers=[i])
        page = await repo.get_all(limit=2, offset=0)
        assert len(page) == 2
        assert len(await repo.get_all(limit=2, offset=4)) == 1

    async def test_update_and_delete(self, test_session):
        repo = CircuitRepository(test_session)
        await repo.create(id="c1", name="one", circuit_meta={}, layers=[1])
        updated = await repo.update("c1", rung=3, serveable=True)
        assert updated.rung == 3 and updated.serveable is True
        assert await repo.update("ghost", rung=1) is None
        assert await repo.delete("c1") is True
        assert await repo.delete("c1") is False

    async def test_deactivate_all(self, test_session):
        repo = CircuitRepository(test_session)
        await repo.create(id="c1", name="one", circuit_meta={}, layers=[1])
        await repo.set_active("c1")
        assert await repo.deactivate_all() == 1
        assert await repo.get_active() is None
