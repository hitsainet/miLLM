"""A re-run batch row records its sensing and circuit-edge events once (026 F-1, tracker 4.14).

The runner re-runs a row whose chunk died after the row's events were written but before its
state was recorded (the exactly-once window). Probe events were deduplicated (T-70); sensing and
edge events were not, so such a line's events were recorded twice. The repositories now skip a
(owner, batch, line) that already has events; migration 023's partial unique indexes back it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from millm.db.models.circuit import Circuit
from millm.db.models.circuit_edge_sensing_event import CircuitEdgeSensingEvent
from millm.db.models.profile import Profile
from millm.db.models.sensing_event import SensingEvent
from millm.db.repositories.circuit_edge_sensing_repository import CircuitEdgeSensingRepository
from millm.db.repositories.sensing_repository import SensingRepository


def _sense(**kw):
    base = dict(profile_id="prof_s", request_id="r", phase="decode", pos_start=1, pos_end=2,
                fired_members=[[7, 1.0]], fired_count=1, score=1.0, summary="s")
    base.update(kw)
    return base


def _edge(**kw):
    base = dict(circuit_id="circ_d", request_id="r", phase="decode", edge_key="1@10->2@13",
                up_layer=10, up_feature_idx=1, up_pos=5, up_act=1.5, down_layer=13,
                down_feature_idx=2, down_pos=7, down_act=0.9, token_lag=2, edge_rung=2,
                edge_rung_language="observed co-firing", summary="s")
    base.update(kw)
    return base


def _batch(line, **kw):
    return dict(origin="batch", batch_id="batch_1", batch_line=line, **kw)


@pytest.fixture
async def parents(test_session):
    test_session.add(Profile(id="prof_s", name="s", steering={"7": 1.0}, source_kind="cluster"))
    test_session.add(Circuit(id="circ_d", name="c", circuit_meta={}, rung=2, edge_count=1,
                             layers=[10, 13], serveable=True))
    await test_session.flush()


async def _count(session, model) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


class TestSensingEvents:
    async def test_a_rerun_line_is_recorded_once(self, test_session, parents):
        repo = SensingRepository(test_session)
        first = await repo.create_many([_sense(**_batch(3)), _sense(pos_start=4, pos_end=5,
                                                                    **_batch(3))])
        rerun = await repo.create_many([_sense(**_batch(3)), _sense(pos_start=4, pos_end=5,
                                                                    **_batch(3))])
        assert (len(first), len(rerun)) == (2, 0)
        assert await _count(test_session, SensingEvent) == 2

    async def test_a_rerun_that_fires_differently_still_adds_nothing(self, test_session, parents):
        """Per LINE, not per row: a sampled re-run can fire at other positions."""
        repo = SensingRepository(test_session)
        await repo.create_many([_sense(**_batch(3))])
        assert await repo.create_many([_sense(pos_start=9, pos_end=9, **_batch(3))]) == []

    async def test_other_lines_and_live_events_are_untouched(self, test_session, parents):
        repo = SensingRepository(test_session)
        await repo.create_many([_sense(**_batch(3))])
        assert len(await repo.create_many([_sense(**_batch(4))])) == 1
        assert len(await repo.create_many([_sense(), _sense()])) == 2  # live: never filtered
        assert await _count(test_session, SensingEvent) == 4

    async def test_the_index_refuses_an_identical_batch_row(self, test_session, parents):
        test_session.add_all([SensingEvent(**_sense(**_batch(3))),
                              SensingEvent(**_sense(**_batch(3)))])
        with pytest.raises(IntegrityError):
            await test_session.flush()


class TestCircuitEdgeEvents:
    async def test_a_rerun_line_is_recorded_once(self, test_session, parents):
        repo = CircuitEdgeSensingRepository(test_session)
        assert len(await repo.create_many([_edge(**_batch(3))])) == 1
        assert await repo.create_many([_edge(up_pos=6, down_pos=8, **_batch(3))]) == []
        assert len(await repo.create_many([_edge(**_batch(4)), _edge()])) == 2
        assert await _count(test_session, CircuitEdgeSensingEvent) == 3

    async def test_the_index_refuses_an_identical_batch_row(self, test_session, parents):
        test_session.add_all([CircuitEdgeSensingEvent(**_edge(**_batch(3))),
                              CircuitEdgeSensingEvent(**_edge(**_batch(3)))])
        with pytest.raises(IntegrityError):
            await test_session.flush()
