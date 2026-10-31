"""The restart's `paused_reason` is the constant (039 FTASKS 1.5, TD39-12, FTID Discrepancy 9).

"Armed before restart" is derived from `INVOLUNTARY_REASONS`, so the startup reset must write
EXACTLY `RESTART_REASON` — an inline copy that drifted by one character would silently drop every
restarted behaviour from that list. And because the startup reset writes with a direct UPDATE
(bypassing `mark_armed_rows_disarmed`'s length guard), the constant's length is asserted here.

Control: re-inline a different literal in `main.disarm_probes_on_startup` → red.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from millm.db.models.probe import Probe
from millm.services.probe_arming import (
    HANG_REASON,
    INVOLUNTARY_REASONS,
    RESTART_REASON,
    UNLOAD_REASON,
    paused_reason_limit,
)
from tests.support import behaviour_factory as bf

pytestmark = pytest.mark.asyncio


def test_the_three_involuntary_reasons_fit_the_column():
    assert INVOLUNTARY_REASONS == {UNLOAD_REASON, HANG_REASON, RESTART_REASON}
    for reason in INVOLUNTARY_REASONS:
        assert len(reason) <= paused_reason_limit(), reason


async def test_the_startup_reset_writes_exactly_the_constant(test_session):
    from millm.main import disarm_probes_on_startup

    row = await bf.import_probe(test_session, bf.definition("restarted", key=bf.random_key()))
    row.armed = True
    await test_session.commit()

    class _Factory:
        def __call__(self):
            return self

        async def __aenter__(self):
            return test_session

        async def __aexit__(self, *exc):
            return False

    assert await disarm_probes_on_startup(_Factory()) == [row.id]
    written = (await test_session.execute(
        select(Probe.paused_reason).where(Probe.id == row.id))).scalar_one()
    assert written == RESTART_REASON
    assert written in INVOLUNTARY_REASONS
