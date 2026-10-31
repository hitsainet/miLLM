"""Code rules are refused below the runner contract's floor (039 FTASKS 17.4; D47).

A host grants at least `limits_floor`; a deployment may raise a limit, never lower it. The floor is
READ from the vendored contract (provisional per D47 — a re-vendor may raise it, and then this
check refuses a deployment still below it). `runner_client.floor_violations` is the one check;
`main.check_rule_code_floor_on_startup` runs it in `lifespan`, and every after-response job runs it
again before any call, so a value lowered at runtime is refused too.
"""

from __future__ import annotations

import asyncio
import logging

from millm.core.config import settings
from millm.rules import code_followup
from millm.rules import runner_contract as rc
from millm.rules.runner_client import call_limits, floor_violations


def test_the_defaults_sit_at_or_above_the_floor():
    floor = rc.floor()
    limits = call_limits(settings)
    assert floor_violations(settings) == []
    assert limits["memory_bytes"] >= floor["memory_bytes"]
    assert limits["output_bytes"] >= floor["output_bytes"]


def test_memory_below_the_floor_is_named(monkeypatch):
    monkeypatch.setattr(settings, "RULE_CODE_CALL_MEMORY_MIB", 32)
    assert floor_violations(settings) == ["memory_bytes"]


def test_output_below_the_floor_is_named(monkeypatch):
    monkeypatch.setattr(settings, "RULE_CODE_CALL_OUTPUT_BYTES", 1024)
    assert floor_violations(settings) == ["output_bytes"]


def test_a_raised_floor_refuses_the_old_setting(monkeypatch):
    """D47: a re-vendored contract that RAISES a floor turns the deployment's value into a refusal."""
    raised = {**rc.floor(), "memory_bytes": 128 << 20}
    monkeypatch.setattr(rc, "floor", lambda: raised)
    assert floor_violations(settings) == ["memory_bytes"]


def test_the_startup_check_refuses_and_says_which(monkeypatch, caplog):
    from millm.main import check_rule_code_floor_on_startup

    monkeypatch.setattr(settings, "RULE_CODE_CALL_MEMORY_MIB", 16)
    assert check_rule_code_floor_on_startup() == ["memory_bytes"]
    monkeypatch.setattr(settings, "RULE_CODE_CALL_MEMORY_MIB", 64)
    assert check_rule_code_floor_on_startup() == []


def test_lifespan_runs_the_floor_check(monkeypatch):
    """Wiring: the startup rule check calls the floor check (delete the call → red)."""
    import millm.main as main

    seen = []
    monkeypatch.setattr(main, "check_rule_code_floor_on_startup", lambda: seen.append(1) or [])

    class _Report:
        pass

    async def _selfcheck(factory):
        return _Report()

    import millm.rules.selfcheck as selfcheck

    monkeypatch.setattr(selfcheck, "run", _selfcheck)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def _rebuild(session, reason=""):
        return None

    import millm.rules.registry as registry

    monkeypatch.setattr(registry, "rebuild", _rebuild)
    asyncio.run(main.check_rule_sets_on_startup(lambda: _Session()))
    assert seen == [1]


def test_a_job_below_the_floor_calls_nothing(monkeypatch, caplog):
    """Every job re-checks: below the floor, no runner call is made and nothing is revised."""
    from types import SimpleNamespace

    from tests.support.stub_runner import StubRunner

    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(settings, "RULE_CODE_CALL_MEMORY_MIB", 16)
    pending = SimpleNamespace(kind="rule", reason="rule_worker_unavailable",
                              status="unavailable", rule_set_id="rs_000000000001",
                              signal="operator", window="all", result_id="r1")
    with StubRunner() as stub:
        monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
        job = code_followup.CodeJob(sets=(), texts={"all": "x"})
        assert asyncio.run(code_followup.score(job, [pending], settings=settings)) == []
        assert stub.received == []
    assert "rule_code_below_floor" in caplog.text
