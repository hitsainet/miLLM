"""ENH-006 G19: a stored verdict definition serves again after a restart — the startup hook rebuilds the
verdict registry from `behaviour_verdicts`, and a stored definition that no longer checks is left out.

Control: drop `behaviour_verdict_store.rebuild` from `check_rule_sets_on_startup` → red.
"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from miserver.db.base import Base
from miserver.db.models.rule_set import BehaviourVerdict
from miserver.services import behaviour_verdict as bv
from tests.support import verdict_factory as vf

pytestmark = pytest.mark.asyncio


async def test_startup_rebuilds_the_verdict_registry_from_the_rows():
    from miserver.main import check_rule_sets_on_startup

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    good = vf.definition("g19-a", ["a" * 64, "b" * 64], candidate="any_of")
    broken = copy.deepcopy(vf.definition("g19-b", ["c" * 64], candidate="any_of"))
    broken["combiner_vectors"][0]["expect"]["fires"] = not broken["combiner_vectors"][0]["expect"]["fires"]
    async with factory() as session:
        session.add_all([
            BehaviourVerdict(behaviour_key="g19-a", definition=good, definition_sha256=bv.definition_sha256(good)),
            BehaviourVerdict(behaviour_key="g19-b", definition=broken,
                             definition_sha256=bv.definition_sha256(broken)),
        ])
        await session.commit()
    bv.forget_for_tests()
    try:
        await check_rule_sets_on_startup(factory)
        assert set(bv.current()) == {"g19-a"}
        assert bv.current()["g19-a"].sha256 == bv.definition_sha256(good)
    finally:
        bv.forget_for_tests()
        await engine.dispose()
