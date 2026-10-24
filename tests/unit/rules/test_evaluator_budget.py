"""The per-request rule budget (039 FR-39.24, TD39-20; FTASKS 13.5).

- An INJECTED clock: signals past the deadline report `rule_timeout`, earlier ones complete, and
  the evaluator never reads wall time of its own.
- A thread that outlives the await: the request waits at most budget + `GRACE_S`, every signal
  reports `rule_timeout`, and the late result is DISCARDED by the generation counter.

Controls (039 controls record): remove the deadline check in `serve` → the injected-clock test red;
skip the generation check in `rules_finish.run` → the late-result test red.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from millm.core.config import settings
from millm.rules import registry as rule_registry
from millm.rules import serving
from millm.rules.evaluator import CompiledRuleSet
from tests.support import rule_set_factory as rf

pytestmark = pytest.mark.asyncio
KEY = "b-0budget0"


def _install(doc):
    compiled = CompiledRuleSet(doc)
    attached = rule_registry.AttachedSet(
        rule_set_id="rs_000000000abc", digest="a" * 64, version=1, behaviour_key=KEY,
        compiled=compiled, vectors_failed=False, only=None,
        forms=rule_registry.document_forms(doc))
    rule_registry._current = rule_registry.Registry({KEY: (attached,)})


def _context(windows=("prompt",)):
    probe = SimpleNamespace(descriptor=SimpleNamespace(behaviour_key=KEY), windows=windows,
                            scope="all")
    return SimpleNamespace(probes=[probe], not_scored_reason=None)


@pytest.fixture(autouse=True)
def _clean():
    rule_registry.forget_for_tests()
    yield
    rule_registry.forget_for_tests()


async def test_signals_past_the_deadline_time_out_and_earlier_ones_complete(monkeypatch):
    _install(rf.document(KEY))
    monkeypatch.setattr(settings, "RULE_TIME_BUDGET_MS", 3.0)
    ticks = iter([0.0, 0.0, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])

    def clock():
        return next(ticks, 1.0)

    context = _context()
    await serving.rules_finish(context, messages=[{"role": "user", "content": "ZQ-1234"}],
                               reply_text="x", clock=clock)
    by = [(i.signal, i.outcome, i.reason) for i in context.rule_items]
    names = sorted(rf.document(KEY)["expressions"])
    assert [n for n, _o, _r in by] == names
    assert by[0][1] == "completed"
    assert by[-1][1:] == ("failed", "rule_timeout")
    assert {o for _n, o, _r in by} == {"completed", "failed"}


async def test_a_late_thread_result_is_discarded(monkeypatch):
    _install(rf.document(KEY))
    monkeypatch.setattr(settings, "RULE_TIME_BUDGET_MS", 5.0)
    real = serving.evaluate_work

    def slow(*args, **kwargs):
        time.sleep(1.0)
        # No deadline here: the late thread produces COMPLETED items, so only the generation
        # counter can keep them off the context (control S2-18 survived while the late items
        # were timeouts too and agreed with the right answer by construction).
        return real(*args, **{**kwargs, "deadline": None})

    monkeypatch.setattr(serving, "evaluate_work", slow)
    context = _context()
    started = time.monotonic()
    await serving.rules_finish(context, messages=[{"role": "user", "content": "ZQ-1234"}],
                               reply_text="x")
    waited = time.monotonic() - started
    # Relative, not a CI wall-clock threshold: the request returned long before the 1 s thread.
    assert waited < 0.5, f"the request waited {waited:.3f}s for a 5 ms budget"
    assert context.rule_items and all(
        (i.outcome, i.reason) == ("failed", "rule_timeout") for i in context.rule_items)
    await asyncio.sleep(1.2)  # the thread finishes now — and must change nothing
    assert all(i.reason == "rule_timeout" for i in context.rule_items)


async def test_nothing_attached_returns_before_any_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(asyncio, "to_thread", lambda *a, **k: calls.append(a))
    context = _context()
    await serving.rules_finish(context, messages=[], reply_text="x")
    assert context.rule_items == [] and calls == []
