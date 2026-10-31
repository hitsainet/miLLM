"""039 S3 wiring, by RUNNING it (FTASKS 17.x; the reachability rule): each registration line, deleted,
turns one of these red.

- `lifespan` drains the after-response code jobs once on shutdown, and runs the floor check at
  startup (through `check_rule_sets_on_startup`);
- every probe record hands the request's code job to `code_followup.spawn` — exactly once, with the
  request's pending rule results and `live=True` (payload and call count);
- `rules_finish` builds that job only when an attached set has a pending (code) expression;
- the revision's socket event is emitted under its contract name.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.batch_fixtures import batch_db  # noqa: F401

REPO = Path(__file__).resolve().parents[2]


async def test_lifespan_drains_code_jobs_and_checks_the_floor(monkeypatch, batch_db):
    import millm.db.base as db_base
    import millm.main as main
    from millm.rules import code_followup
    from millm.services.inference_service import InferenceService

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    monkeypatch.setattr(InferenceService, "drain_probe_records", AsyncMock(return_value=0))
    drain = AsyncMock(return_value=0)
    monkeypatch.setattr(code_followup, "drain", drain)
    floor = MagicMock(return_value=[])
    monkeypatch.setattr(main, "check_rule_code_floor_on_startup", floor)

    async with main.lifespan(MagicMock()):
        assert drain.await_count == 0
        assert floor.call_count == 1
    assert drain.await_count == 1
    assert drain.await_args.args == (main.PROBE_RECORD_DRAIN_S,)


@pytest.mark.asyncio
async def test_every_record_hands_its_code_job_to_spawn_once(monkeypatch):
    from millm.core.config import settings
    from millm.rules import code_followup
    from tests.support import behaviour_factory as bf
    from tests.support import capture_pre038_goldens as cap
    from tests.unit.api.test_rule_code_followup import (
        BODY,
        _arm,
        _attach,
        code_document,
        envelope,
        rule_set_with_code,
    )

    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    calls: list[tuple] = []
    monkeypatch.setattr(code_followup, "spawn",
                        lambda job, results, *, live: calls.append((job, list(results), live)))
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        h.post_chat(BODY, {})
        assert len(calls) == 1 and calls[0][0] is None, "no code set attached: no job"
        rule_set_id = _attach(h, key, envelope(rule_set_with_code(key, code), code))
        response = h.post_chat(BODY, {})
        assert len(calls) == 2
        job, results, live = calls[1]
        assert live is True
        assert [a.rule_set_id for _k, _w, a in job.sets] == [rule_set_id]
        # Only the windows the attached set reports are held (nothing more of the text).
        assert set(job.texts) <= {w for _k, windows, _a in job.sets for w in windows}
        assert job.texts
        pending = {r.result_id for r in results if getattr(r, "kind", "") == "rule"
                   and r.reason == "rule_worker_unavailable"}
        live_pending = {e["result_id"] for e in response.json()["millm_probe_verdicts"]
                        if e.get("kind") == "rule" and e["reason"] == "rule_worker_unavailable"}
        assert pending and pending == live_pending


def test_the_revision_event_is_emitted_under_its_name():
    tree = ast.parse((REPO / "millm" / "sockets" / "progress.py").read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "emit_probe_rule_results")
    names = {c.args[0].value for c in ast.walk(method)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "emit" and c.args and isinstance(c.args[0], ast.Constant)}
    assert names == {"probe:rule_results"}
    assert json.dumps(sorted(names))
