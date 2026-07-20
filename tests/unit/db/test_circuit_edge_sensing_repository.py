"""Feature 15 Task 1.3: CircuitEdgeSensingRepository unit tests.

CRUD, per-edge filtering, retention (cap + age prune) and CASCADE with circuit
deletion. The retention ordering assertions matter more here than they look:
one flush inserts many rows sharing a `created_at`, so without the `id`
tiebreak the cap keeps an arbitrary subset.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from millm.db.models.circuit import Circuit
from millm.db.models.circuit_edge_sensing_event import CircuitEdgeSensingEvent
from millm.db.repositories.circuit_edge_sensing_repository import (
    CircuitEdgeSensingRepository,
)


@pytest.fixture
async def circuit(test_session):
    row = Circuit(
        id="circ_sense01",
        name="fear→threat",
        circuit_meta={"kind": "mistudio.circuit-definition", "schema_version": "1"},
        rung=2,
        edge_count=1,
        layers=[10, 13],
        serveable=True,
        sensing_enabled=True,
    )
    test_session.add(row)
    await test_session.flush()
    return row


@pytest.fixture
def repo(test_session):
    return CircuitEdgeSensingRepository(test_session)


EDGE_A = "1@10->2@13"
EDGE_B = "3@10->4@13"


def make_event(circuit_id="circ_sense01", edge_key=EDGE_A, **overrides):
    base = dict(
        circuit_id=circuit_id,
        request_id="req-1",
        phase="decode",
        edge_key=edge_key,
        up_layer=10,
        up_feature_idx=1,
        up_pos=5,
        up_act=1.5,
        down_layer=13,
        down_feature_idx=2,
        down_pos=7,
        down_act=0.9,
        token_lag=2,
        edge_rung=2,
        edge_rung_language="causally validated (edge)",
        edge_type="computed",
        summary="edge fired",
    )
    base.update(overrides)
    return base


class TestCrud:
    async def test_create_many_and_count(self, repo, circuit):
        await repo.create_many([make_event(), make_event(request_id="req-2")])
        assert await repo.count(circuit_id="circ_sense01") == 2

    async def test_list_orders_newest_first(self, repo, circuit):
        await repo.create_many([make_event(request_id=f"req-{i}") for i in range(3)])
        rows = await repo.list_events(circuit_id="circ_sense01")
        ids = [r.id for r in rows]
        assert ids == sorted(ids, reverse=True)

    async def test_get_returns_none_for_a_missing_id(self, repo, circuit):
        assert await repo.get(999_999) is None

    async def test_clear_scoped_to_one_circuit(self, test_session, repo, circuit):
        other = Circuit(
            id="circ_other",
            name="other",
            circuit_meta={},
            rung=0,
            edge_count=0,
            layers=[10],
            serveable=True,
        )
        test_session.add(other)
        await test_session.flush()
        await repo.create_many(
            [make_event(), make_event(circuit_id="circ_other")]
        )

        assert await repo.clear(circuit_id="circ_sense01") == 1
        assert await repo.count(circuit_id="circ_sense01") == 0
        assert await repo.count(circuit_id="circ_other") == 1


class TestPerEdgeFiltering:
    async def test_list_and_count_filter_by_edge_key(self, repo, circuit):
        await repo.create_many(
            [make_event(edge_key=EDGE_A), make_event(edge_key=EDGE_A),
             make_event(edge_key=EDGE_B)]
        )
        assert await repo.count(circuit_id="circ_sense01", edge_key=EDGE_A) == 2
        rows = await repo.list_events(circuit_id="circ_sense01", edge_key=EDGE_B)
        assert len(rows) == 1 and rows[0].edge_key == EDGE_B


class TestRetention:
    async def test_prune_keeps_only_the_newest_cap(self, repo, circuit):
        await repo.create_many([make_event(request_id=f"req-{i}") for i in range(10)])
        deleted = await repo.prune("circ_sense01", cap=4, max_age_days=7)
        assert deleted == 6
        assert await repo.count(circuit_id="circ_sense01") == 4

    async def test_prune_keeps_the_NEWEST_rows_not_an_arbitrary_four(
        self, repo, circuit
    ):
        """All ten share a created_at, so only the id tiebreak makes this
        deterministic. Without it the cap keeps an arbitrary subset."""
        rows = await repo.create_many(
            [make_event(request_id=f"req-{i}") for i in range(10)]
        )
        newest = sorted((r.id for r in rows), reverse=True)[:4]
        await repo.prune("circ_sense01", cap=4, max_age_days=7)
        remaining = await repo.list_events(circuit_id="circ_sense01", limit=100)
        assert sorted(r.id for r in remaining) == sorted(newest)

    async def test_prune_aged_drops_rows_past_the_window(
        self, test_session, repo, circuit
    ):
        await repo.create_many([make_event()])
        old = CircuitEdgeSensingEvent(**make_event(request_id="ancient"))
        old.created_at = datetime.now(timezone.utc) - timedelta(days=30)
        test_session.add(old)
        await test_session.flush()

        assert await repo.prune_aged(max_age_days=7) == 1
        assert await repo.count(circuit_id="circ_sense01") == 1

    async def test_prune_is_scoped_to_its_circuit(self, test_session, repo, circuit):
        other = Circuit(
            id="circ_other2", name="o", circuit_meta={}, rung=0,
            edge_count=0, layers=[10], serveable=True,
        )
        test_session.add(other)
        await test_session.flush()
        await repo.create_many(
            [make_event(request_id=f"r{i}") for i in range(5)]
            + [make_event(circuit_id="circ_other2")]
        )
        await repo.prune("circ_sense01", cap=1, max_age_days=7)
        assert await repo.count(circuit_id="circ_sense01") == 1
        assert await repo.count(circuit_id="circ_other2") == 1


class TestCascade:
    async def test_deleting_the_circuit_deletes_its_events(
        self, test_session, repo, circuit
    ):
        await repo.create_many([make_event(), make_event(request_id="req-2")])
        await test_session.flush()

        await test_session.delete(circuit)
        await test_session.flush()

        remaining = await test_session.execute(
            select(CircuitEdgeSensingEvent).where(
                CircuitEdgeSensingEvent.circuit_id == "circ_sense01"
            )
        )
        assert remaining.scalars().all() == []


class TestSerialisation:
    async def test_to_dict_omits_context_when_asked(self, repo, circuit):
        rows = await repo.create_many(
            [make_event(context_text="secret prompt",
                        context_parts={"before": "a", "span": "b", "after": "c"})]
        )
        full = rows[0].to_dict(include_context=True)
        assert full["context_text"] == "secret prompt"

        ws = rows[0].to_dict(include_context=False)
        assert "context_text" not in ws
        assert "context_parts" not in ws
        assert "context_token_ids" not in ws

    async def test_to_dict_nests_both_endpoints(self, repo, circuit):
        rows = await repo.create_many([make_event()])
        d = rows[0].to_dict()
        assert d["up"] == {"layer": 10, "feature_idx": 1, "pos": 5, "act": 1.5}
        assert d["down"] == {"layer": 13, "feature_idx": 2, "pos": 7, "act": 0.9}
        assert d["token_lag"] == 2
        assert d["edge_rung_language"] == "causally validated (edge)"
