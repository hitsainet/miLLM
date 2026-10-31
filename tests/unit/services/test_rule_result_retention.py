"""Rule-result retention (039 TD39-17; FTASKS 14.4).

Rule results follow probe events: the age rule (`PROBE_MAX_AGE_DAYS`) and a cap per
`(rule_set_id, signal)` and origin — `PROBE_MAX_EVENTS_PER_PROBE` for live rows,
`PROBE_MAX_BATCH_EVENTS_PER_PROBE` for batch rows, so a batch never evicts live history. The
background retention pass runs it (the wiring is asserted by running the pass, not by reading it).

Controls (039 controls record): one cap for both origins → `test_each_origin_has_its_own_cap` red;
the call removed from `run_retention_pass` → `test_the_background_pass_prunes_rule_results` red.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from millm.core.config import settings
from millm.db.base import Base
from millm.db.models.rule_set import RuleSignalResult
from millm.db.repositories.rule_set_repository import prune_rule_results

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db():
    import millm.db.models  # noqa: F401 - every table

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


_n = 0


def _row(signal="hit", origin="live", age_days=0.0, rule_set_id="rs_000000000001", line=None):
    global _n
    _n += 1
    created = datetime.now(timezone.utc) - timedelta(days=age_days, seconds=_n)
    return RuleSignalResult(
        result={}, result_id=f"00000000-0000-7000-8000-{_n:012d}", event_time=created,
        outcome="completed", rule_set_id=rule_set_id, rule_set_digest="a" * 64,
        signal=signal, form="pattern", window="prompt", fires=True, count=1, origin=origin,
        batch_id="b1" if origin == "batch" else None, batch_line=line, created_at=created)


async def _count(db, **where) -> int:
    stmt = select(func.count()).select_from(RuleSignalResult)
    for k, v in where.items():
        stmt = stmt.where(getattr(RuleSignalResult, k) == v)
    return int((await db.execute(stmt)).scalar_one())


async def test_each_origin_has_its_own_cap(db):
    db.add_all([_row() for _ in range(5)] + [_row(origin="batch", line=i + 1) for i in range(7)])
    await db.commit()
    await prune_rule_results(db, max_age_days=None, live_cap=3, batch_cap=6)
    await db.commit()
    assert await _count(db, origin="live") == 3
    assert await _count(db, origin="batch") == 6


async def test_the_cap_is_per_signal_and_keeps_the_newest(db):
    rows = [_row(signal="a") for _ in range(4)] + [_row(signal="b") for _ in range(2)]
    db.add_all(rows)
    await db.commit()
    newest_a = sorted((r for r in rows if r.signal == "a"), key=lambda r: r.created_at)[-2:]
    await prune_rule_results(db, max_age_days=None, live_cap=2, batch_cap=2)
    await db.commit()
    kept = (await db.execute(select(RuleSignalResult.result_id)
                             .where(RuleSignalResult.signal == "a"))).scalars().all()
    assert set(kept) == {r.result_id for r in newest_a}
    assert await _count(db, signal="b") == 2


async def test_the_age_rule_is_table_wide(db):
    db.add_all([_row(age_days=40), _row(age_days=40, signal="z"), _row()])
    await db.commit()
    removed = await prune_rule_results(db, max_age_days=30, live_cap=100, batch_cap=100)
    await db.commit()
    assert removed == 2 and await _count(db) == 1


async def test_the_background_pass_prunes_rule_results(db, monkeypatch):
    from millm.services import probe_sessions

    monkeypatch.setattr(settings, "PROBE_MAX_EVENTS_PER_PROBE", 2)
    db.add_all([_row() for _ in range(5)] + [_row(age_days=400)])
    await db.commit()
    removed = await probe_sessions.run_retention_pass(db)
    assert removed["rule_results"] == 4
    assert await _count(db) == 2
