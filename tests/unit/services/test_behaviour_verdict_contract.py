"""The behaviour verdict's contract in miLLM (ENH-006 G19): the vendored files, the combiner on every
shared case (AC 2), the rule side by the ROW semantics, the header bound, the telemetry record (AC 6)
and the in-place revision of a verdict that waited on operator code (AC 7).

Controls (design list): drop `AND NOT veto` → `veto_full_scope_suppresses_blend` red; let the veto beat a
match → `match_beats_veto` red; `>=` into `>` on the stack bar → `stack_exactly_on_bar` red; `false` for
a missing member → `missing_member_nothing_fires` red; the veto applied to every member when scoped →
`veto_spares_out_of_scope_member` red; the digest check skipped → the members-digest test red.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from miserver.rules.evaluator import CompiledRuleSet
from miserver.rules.registry import AttachedSet
from miserver.services import behaviour_verdict as bv
from tests.support import rule_set_factory as rf
from tests.support import verdict_factory as vf
from tests.support.sibling_repos import cross_repo_checks_required, workbench_repo

ROOT = Path(__file__).resolve().parents[3]
CASES = json.loads(bv.CASES_PATH.read_text(encoding="utf-8"))


def test_the_vendored_files_are_the_ones_the_combiner_reads():
    assert bv.CASES_PATH == ROOT / "docs" / "schemas" / "behaviour-verdict-cases.json"
    assert bv.SCHEMA_PATH == ROOT / "docs" / "schemas" / "behaviour-verdict-v1.json"
    assert CASES["format"] == bv.CASES_FORMAT and len(CASES["cases"]) >= 30
    assert CASES["tolerance"]["value_abs"] == bv.VALUE_TOLERANCE


@pytest.mark.parametrize("name", ["behaviour-verdict-v1.json", "behaviour-verdict-cases.json"])
def test_the_copies_are_byte_identical_to_mistudios(name):
    repo = workbench_repo()
    theirs = Path(repo) / "docs" / "schemas" / name
    if not theirs.is_file():
        if cross_repo_checks_required():
            pytest.fail(f"miStudio's {name} is not at {theirs}")
        pytest.skip(f"miStudio sibling absent at {repo}: {name} not compared")
    assert (ROOT / "docs" / "schemas" / name).read_bytes() == theirs.read_bytes()


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["name"])
def test_millms_combiner_reproduces_each_shared_case(case):
    got = bv.combine({"candidate": case["candidate"], "members": case["members"]},
                     case["inputs"]["members"], case["inputs"]["rules"])
    assert bv.agrees(got, case["expect"]), (got, case["expect"])


def _definition(**kw):
    return vf.definition("g19-key", ["a" * 64, "b" * 64], **kw)


def test_check_recomputes_every_vector_and_refuses_a_tampered_one():
    good = _definition(candidate="any_of")
    report = bv.check(good, key="g19-key")
    assert report["combiner_vectors"] == len(good["combiner_vectors"]) >= 5
    for i in range(len(good["combiner_vectors"])):
        bad = copy.deepcopy(good)
        bad["combiner_vectors"][i]["expect"]["decided_by"] = "vetoed"
        with pytest.raises(bv.DefinitionRefused) as refused:
            bv.check(bad, key="g19-key")
        assert refused.value.code == "verdict_parity_failed"


def test_check_refuses_a_members_digest_that_is_not_the_members():
    bad = _definition(candidate="any_of")
    bad["members_digest"] = bv.members_digest(["c" * 64])
    with pytest.raises(bv.DefinitionRefused) as refused:
        bv.check(bad, key="g19-key")
    assert refused.value.code == "behaviour_verdict_invalid" and "computed" in refused.value.details


def test_check_refuses_a_partial_scope_on_a_score_kind():
    params = {"mean": [0.0, 0.0], "std": [1.0, 1.0], "weights": [1.0, 1.0], "intercept": 0.0}
    rule_doc = {"match": {"any_of": ["token"]}}
    bad = _definition(candidate="rules_or_stack", params=params, threshold=1.0, rule_doc=rule_doc,
                      veto_scope=[0], digest="e" * 64)
    with pytest.raises(bv.DefinitionRefused) as refused:
        bv.check(bad, key="g19-key")
    assert refused.value.details.get("candidate") == "rules_or_stack"


# ── the rule side: ROW semantics, never `serve` ──────────────────────────────────


def test_row_verdict_applies_the_vetoes_and_reads_only_the_match_expressions():
    compiled = CompiledRuleSet(vf.masked_document("g19-key"))
    assert compiled.row_verdict("marker ZQ-1234 here", ["token"]) == (True, False)
    assert compiled.row_verdict(f"see {vf.MASKED} here", ["token"]) == (False, True)
    assert compiled.row_verdict("nothing", ["token"]) == (False, False)
    # `serve` (the kind: "rule" results) is unchanged: the masked token still reports, pre-veto
    (outcome,) = compiled.serve({"all": f"see {vf.MASKED} here"}, only=["token"])
    assert outcome.fires is True
    # an expression outside the match is not a positive (D53)
    assert compiled.row_verdict(f"{rf.LUHN_OK}", ["token"]) == (False, False)


def test_row_verdict_is_pending_on_operator_code_until_its_count_arrives():
    doc = rf.document("g19-key", code_rule=True)
    compiled = CompiledRuleSet(doc)
    assert compiled.row_verdict("anything", ["operator"]) == (None, False)
    assert compiled.row_verdict("anything", ["operator", "token"]) == (None, False)
    assert compiled.row_verdict("ZQ-1234", ["operator", "token"]) == (True, False)
    assert compiled.row_verdict("anything", ["operator"], code={"op": 1}) == (True, False)
    assert compiled.row_verdict("anything", ["operator"], code={"op": 0}) == (False, False)


# ── the header bound keeps every verdict before any member ───────────────────────


def test_the_bounded_header_keeps_each_behaviours_verdict_first():
    from miserver.api.structured_header import bounded_members_keeping_leaders

    leaders = [['"behaviour:a";verdict=?1'], ['"behaviour:b";verdict=?0']]
    groups = [["m" * 60, "n" * 60], ["o" * 60]]
    out = bounded_members_keeping_leaders(leaders, groups, 120, total=5)
    assert out.startswith('"behaviour:a";verdict=?1, "behaviour:b";verdict=?0')
    assert "m" * 60 not in out and out.endswith("total=5")
    full = bounded_members_keeping_leaders(leaders, groups, 10_000, total=5)
    assert full.split(", ")[:3] == ['"behaviour:a";verdict=?1', "m" * 60, "n" * 60]


# ── AC 6: the telemetry record carries no text, no span, no list ─────────────────


def _context(definition, rows, members):
    probes = [SimpleNamespace(probe_id=f"pr_{i:012d}", descriptor=SimpleNamespace(
        behaviour_key="g19-key", detector_digest=d, decision=SimpleNamespace(decision_digest=None)))
        for i, d in enumerate(members)]
    return SimpleNamespace(probes=probes, request_id="req-1", trace=None, session=None, interrupted=None,
                           execution=None, verdict_rules=rows, not_scored_reason=None, verdict_pending={})


def _verdict(probe_id, fires, window="all"):
    return SimpleNamespace(probe_id=probe_id, window=window, scored=True, fires=fires, score=1.0,
                           provisional=False)


def test_the_telemetry_record_carries_only_listed_attributes(monkeypatch):
    from miserver import telemetry
    from miserver.telemetry import allowlist

    definition = _definition(candidate="any_of")
    bv.install([bv.attached_from("g19-key", definition)])
    try:
        context = _context(definition, {}, ["a" * 64, "b" * 64])
        (result,) = bv.build_results([_verdict("pr_000000000000", True), _verdict("pr_000000000001", False)],
                                     context)
    finally:
        bv.forget_for_tests()
    sent = []

    class Pipe:
        signals = {"logs"}
        cfg = SimpleNamespace(result_log_origins={"live"})

        def enqueue_log(self, name, signal, attrs, trace, ns):
            kept, dropped = allowlist.filter(signal, attrs)
            sent.append((kept, dropped))
            return True

    monkeypatch.setattr(telemetry, "_PIPELINE", Pipe())
    assert telemetry.emit_results([result]) == 1
    ((kept, dropped),) = sent
    assert dropped == 0, kept
    assert kept["mi.probe.kind"] == "behaviour" and kept["mi.probe.fires"] is True
    assert kept["mi.probe.decided_by"] == "members" and "mi.probe.members_fired" not in kept
    text = json.dumps(kept)
    for form in ("ZQ-", base64.b64encode(b"ZQ-1234").decode()):
        assert form not in text


# ── AC 7: a verdict that waited on operator code is revised IN PLACE ─────────────


class FakeSession:
    def __init__(self, count):
        self.calls, self.count = 0, count

    async def call(self, sha, doc, function, params, text):
        self.calls += 1
        return SimpleNamespace(reason=None, answered=True, count=self.count)


@pytest.mark.asyncio
async def test_a_pending_verdict_is_revised_in_place_keeping_its_result_id(monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    import miserver.db.base as db_base
    from miserver.db.base import Base
    from miserver.db.models.rule_set import BehaviourVerdictResult
    from miserver.rules import code_followup
    from miserver.rules import runner_contract as rc
    from miserver.services import behaviour_verdict_store as store

    doc = rf.document("g19-key", code_rule=True)
    doc["signals"][-1]["runner_contract"] = rc.contract_format()
    doc["match"] = {"version": 1, "any_of": ["operator"]}
    compiled = CompiledRuleSet(doc)
    aset = AttachedSet(rule_set_id="rs_1", digest="e" * 64, version=1, behaviour_key="g19-key",
                       compiled=compiled, vectors_failed=False, only=None,
                       code_documents=(("0" * 64, "code"),))
    definition = _definition(candidate="rules_or_any_of", rule_doc=doc, digest="e" * 64)
    attached = bv.attached_from("g19-key", definition)
    bv.install([attached])
    try:
        rows = {("g19-key", "all"): bv.RuleRow(reason="rule_worker_unavailable", pending=True)}
        context = _context(definition, rows, ["a" * 64, "b" * 64])
        (result,) = bv.build_results([_verdict("pr_000000000000", False), _verdict("pr_000000000001", False)],
                                     context)
    finally:
        bv.forget_for_tests()
    assert result.status == "partial" and result.fires is False and result.rules.unknown is True
    assert result.rules.reason == "rule_worker_unavailable"

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db_base, "async_session_factory", factory)
    async with factory() as session:
        session.add_all(store.rows_for([result]))
        await session.commit()

    job = code_followup.verdict_job_for([attached], SimpleNamespace(
        for_behaviour=lambda key: (aset,), __bool__=lambda self: True), {"all": "operator text"})
    assert job is not None and [e[:2] for e in job.entries] == [("g19-key", "all")]
    fake = FakeSession(count=1)
    decided = await code_followup.decide_verdict_rows(fake, job, cap=8)
    assert fake.calls == 1 and decided[("g19-key", "all")].match is True
    rows_out = await code_followup.revise_verdicts(job, context.verdict_pending, decided)
    assert len(rows_out) == 1
    async with factory() as session:
        (row,) = (await session.execute(__import__("sqlalchemy").select(BehaviourVerdictResult))).scalars().all()
    assert row.result_id == result.result_id and row.result["result_id"] == result.result_id
    assert row.status == "final" and row.fires is True and row.decided_by == "rule_match"
    assert row.result["rules"]["match"] is True and row.result["rules"]["reason"] is None
    # a second revision of the same row changes nothing (exactly once)
    assert await code_followup.revise_verdicts(job, context.verdict_pending, decided) == []
    await engine.dispose()
