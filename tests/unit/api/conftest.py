"""Fixtures for the API tests.

`RULE_TIME_BUDGET_MS` defaults to 20 ms (`miserver/core/config.py`). These tests assert WHAT a served rule decides (a
match, a veto, a behaviour verdict), not how fast; on a loaded CI runner the 20 ms deadline was missed, the rule
reported `rule_timeout`, its veto read unknown, and the member fired through it
(`test_behaviour_verdict_serving.py::test_the_veto_suppresses_the_in_scope_member_and_spares_the_other`, hitsainet
mirror run 42869716951, 2026-11-10; reproduced locally with the budget at 0.001 ms). The budget path keeps its own
tests (`tests/unit/rules/test_evaluator_budget.py`), which set their own values; a test here that wants the real
budget sets it with its own `monkeypatch`, which runs after this fixture.
"""
from __future__ import annotations

import pytest

from miserver.core.config import settings

#: Generous enough that no runner misses it; a hung evaluation still ends (the await is bounded by budget + grace).
API_TEST_RULE_BUDGET_MS = 10_000.0


@pytest.fixture(autouse=True)
def _rule_budget_the_clock_cannot_miss(monkeypatch):
    monkeypatch.setattr(settings, "RULE_TIME_BUDGET_MS", API_TEST_RULE_BUDGET_MS)
