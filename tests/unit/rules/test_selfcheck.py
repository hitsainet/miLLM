"""Every stored rule set re-proves itself at startup (039 FR-39.20, TD39-26; FTASKS 11.9).

RUNS the real `lifespan` against a database holding two stored sets attached to one behaviour: one
that still agrees with its own cases and one that does not (a case stored wrong, as an evaluator
change would leave it). After startup the first is `passed`, the second `vectors_failed` with a
report naming the case, and the registry the request path reads has both — the failed one yielding
`rule_set_vectors_failed` for every signal.

Controls (039 controls record): delete the `check_rule_sets_on_startup` call from `lifespan` → red;
mutate the evaluator (a proximity bound off by one) → the good set is marked failed → red.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from millm.db.models.rule_set import BehaviourRuleSignals, RuleSet
from millm.rules import registry as rule_registry
from millm.rules import serving
from millm.rules.document import identity_digest
from tests.support import rule_set_factory as rf
from tests.unit.batch_fixtures import batch_db  # noqa: F401

pytestmark = pytest.mark.asyncio
KEY = "b-5e1fc4ec"


async def _seed(factory):
    good, bad = rf.document(KEY), rf.with_wrong_case(KEY)
    bad["name"] = "generated-stale"
    async with factory() as session:
        for rid, doc in (("rs_00000000600d", good), ("rs_0000000000bd", bad)):
            session.add(RuleSet(id=rid, name=doc["name"], version=1, behaviour_key=KEY,
                                origin="authored", digest=identity_digest(doc), document=doc,
                                vector_status="passed"))
        session.add(BehaviourRuleSignals(behaviour_key=KEY,
                                         rule_set_ids=["rs_00000000600d", "rs_0000000000bd"]))
        await session.commit()


async def test_startup_marks_a_disagreeing_set_and_builds_the_registry(monkeypatch,
                                                                       batch_db):  # noqa: F811
    import millm.db.base as db_base
    import millm.main as main

    rule_registry.forget_for_tests()
    await _seed(batch_db)
    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    async with main.lifespan(MagicMock()):
        async with batch_db() as session:
            good = await session.get(RuleSet, "rs_00000000600d")
            bad = await session.get(RuleSet, "rs_0000000000bd")
        assert (good.vector_status, good.checked_at is not None) == ("passed", True)
        assert bad.vector_status == "vectors_failed"
        assert bad.vector_report["case_index"] == 1 and bad.vector_report["where"] == "lexicon"
        assert bad.vector_report["build"].startswith("millm ")
        attached = {a.rule_set_id: a for a in rule_registry.current().for_behaviour(KEY)}
        assert set(attached) == {"rs_00000000600d", "rs_0000000000bd"}
        assert attached["rs_0000000000bd"].vectors_failed
        assert not attached["rs_00000000600d"].vectors_failed
        probe = SimpleNamespace(descriptor=SimpleNamespace(behaviour_key=KEY),
                                windows=("prompt",), scope="all")
        context = SimpleNamespace(probes=[probe], not_scored_reason=None)
        await serving.rules_finish(context, messages=[{"role": "user", "content": "ZQ-1234"}],
                                   reply_text="r")
        by_set = {}
        for item in context.rule_items:
            by_set.setdefault(item.rule_set_id, set()).add((item.outcome, item.reason))
        assert by_set["rs_0000000000bd"] == {("failed", "rule_set_vectors_failed")}
        assert by_set["rs_00000000600d"] == {("completed", None)}
    await asyncio.sleep(0)
    rule_registry.forget_for_tests()
