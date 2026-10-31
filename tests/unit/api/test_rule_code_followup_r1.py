"""039 S3 Review R1: the after-response job's edges that no S3 test held (D41).

- batch traffic never emits the live `probe:rule_results` message, live traffic does (A1, A2);
- `lifespan` drains the probe records BEFORE the code jobs those records start (A3);
- a reply that came too late never answers the NEXT call on that connection (A5);
- a code rule naming another runner contract stays pending, nothing is sent (A28);
- a revision lands exactly once: a row no longer pending is never rewritten (R1-M5);
- the number of jobs in flight is bounded; past it results stay pending (R1-M4);
- the client waits for the worker's slot wait as well as the call (R1-L1);
- an import body may be the rule set's limit PLUS the code documents' (A29);
- a code document whose `params_schema` is not a usable schema is refused, not a 500 (R1-L2);
- a `within_n`, proximity `anchor` or validator `span_of` reading a code rule is REFUSED at
  validate and import (D65, 2026-10-25; it was R1's warning `code_rule_positions_never_resolve`),
  and so is a code rule named in a veto's `when` or `removes` or read by a `field` output (D74,
  2026-10-28, ENH-006 Q9), in the producer's order with its codes and details.
"""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from millm.core.config import settings
from millm.rules import code_followup
from millm.rules import runner_client as client
from millm.rules import runner_contract as rc
from millm.rules.evaluator import CompiledRuleSet
from millm.rules.registry import AttachedSet, document_forms
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support.stub_runner import StubRunner
from tests.unit.api.test_rule_code_followup import code_document, rule_set_with_code
from tests.unit.batch_fixtures import batch_db  # noqa: F401

REPO = Path(__file__).resolve().parents[3]


def _attached(doc: dict, code: dict) -> AttachedSet:
    return AttachedSet("rs_000000000001", "a" * 64, 1, "b-r1", CompiledRuleSet(doc), False,
                       ("operator",), document_forms(doc),
                       ((rc.code_sha256(code), rc.canonical_json(code).decode()),))


def _pending(result_id: str = "r1") -> SimpleNamespace:
    return SimpleNamespace(kind="rule", reason="rule_worker_unavailable", status="unavailable",
                           rule_set_id="rs_000000000001", signal="operator", window="all",
                           result_id=result_id)


# ── live vs batch (A1, A2) ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("live,emits", [(True, 1), (False, 0)])
async def test_only_live_traffic_emits_the_revision(monkeypatch, live, emits):
    rows = [SimpleNamespace(request_id="q")]
    monkeypatch.setattr(code_followup, "score", AsyncMock(return_value=[("r1", object())]))
    monkeypatch.setattr(code_followup, "revise", AsyncMock(return_value=rows))
    emit = MagicMock()
    monkeypatch.setattr(code_followup, "_emit", emit)
    assert await code_followup.run(MagicMock(), [], live=live) == 1
    assert emit.call_count == emits
    if emits:
        assert emit.call_args.args == (rows,)


def test_the_record_says_live_only_for_a_request_that_is_not_a_batch_row():
    """By AST of `_probe_record`: the one `code_followup.spawn` call passes
    `live=BATCH_ROW.get() is None` (a batch row's revision is stored, never broadcast)."""
    tree = ast.parse((REPO / "millm/services/inference_service.py").read_text(encoding="utf-8"))
    record = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_probe_record")
    calls = [c for c in ast.walk(record) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Attribute) and c.func.attr == "spawn"
             and isinstance(c.func.value, ast.Name) and c.func.value.id == "code_followup"]
    assert len(calls) == 1
    live = next(k.value for k in calls[0].keywords if k.arg == "live")
    assert ast.unparse(live) == "BATCH_ROW.get() is None"


# ── shutdown order (A3) ────────────────────────────────────────────────────────────────────────


async def test_lifespan_drains_the_records_before_the_code_jobs_they_start(monkeypatch, batch_db):  # noqa: F811
    import millm.db.base as db_base
    import millm.main as main
    from millm.services.inference_service import InferenceService

    order: list[str] = []
    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)

    async def records(_t):
        order.append("records")
        return 0

    async def jobs(_t):
        order.append("code_jobs")
        return 0

    monkeypatch.setattr(InferenceService, "drain_probe_records", records)
    monkeypatch.setattr(code_followup, "drain", jobs)
    async with main.lifespan(MagicMock()):
        pass
    assert order == ["records", "code_jobs"]


# ── the connection after a timeout (A5) and the slot wait (R1-L1) ─────────────────────────────


def _limits(**over):
    floor = rc.floor()
    out = {name: floor.get(name, 2000) for name in rc.contract()["request_limits"]}
    out.update(over)
    return out


def test_a_late_reply_never_answers_the_next_call(monkeypatch):
    """After a timeout the connection is dropped: the late `{count}` of call 1 must not be read
    as call 2's answer (which would attribute a count to the wrong signal or window)."""
    monkeypatch.setattr(client, "REPLY_MARGIN_S", 0.2)
    code = code_document()
    sha, text = rc.code_sha256(code), rc.canonical_json(code).decode()
    answers = iter([{"count": 7}, {"count": 1}])
    with StubRunner(reply=lambda m: next(answers), hold=True) as stub:
        async def go():
            async with client.RunnerSession(str(stub.socket), _limits(wall_ms=50)) as s:
                first = await s.call(sha, text, "f", {}, "abc")
                stub.release.set()
                await asyncio.sleep(0.3)
                second = await s.call(sha, text, "f", {}, "abcd")
                return first, second
        first, second = asyncio.run(go())
    assert first == client.CodeResult(reason="rule_timeout")
    assert second == client.CodeResult(reason="rule_worker_unavailable")


def test_the_client_waits_for_the_slot_wait_and_the_call(monkeypatch):
    """The worker waits up to the wall time for a slot, then runs up to the wall time."""
    seen: list[float] = []

    async def fake_wait_for(awaitable, timeout):
        seen.append(timeout)
        return await awaitable

    monkeypatch.setattr(client.asyncio, "wait_for", fake_wait_for)
    with StubRunner() as stub:
        async def go():
            async with client.RunnerSession(str(stub.socket), _limits(wall_ms=8000)) as s:
                code = code_document()
                return await s.call(rc.code_sha256(code), rc.canonical_json(code).decode(), "f",
                                    {}, "abc")
        assert asyncio.run(go()) == client.CodeResult(count=2)
    # Connect and close wait the margin; each exchange (load, call) waits both wall times.
    # (Review R2 bounded the close too, so the LAST wait is no longer the reply's.)
    exchanges = seen[1:-1]
    assert seen[0] == seen[-1] == client.REPLY_MARGIN_S and len(exchanges) == 2, seen
    assert all(t >= 2 * 8.0 for t in exchanges), seen


# ── another runner contract (A28) ─────────────────────────────────────────────────────────────


def test_a_code_rule_for_another_runner_contract_stays_pending():
    code = code_document()
    doc = rule_set_with_code("b-r1", code)
    next(s for s in doc["signals"] if s["kind"] == "code_rule")["runner_contract"] = (
        "midataworks.runner/v2")
    job = code_followup.job_for([("b-r1", ("all",), _attached(doc, code))], {"all": "a b c"})
    assert job is not None
    with StubRunner() as stub:
        cfg = SimpleNamespace(**{k: getattr(settings, k) for k in (
            "RULE_CODE_CALL_MEMORY_MIB", "RULE_CODE_CALL_OUTPUT_BYTES", "RULE_CODE_CALL_WALL_MS",
            "RULE_CODE_CALLS_PER_REQUEST")}, RULE_RUNNER_SOCKET=str(stub.socket))
        revised = asyncio.run(code_followup.score(job, [_pending()], settings=cfg))
        received = list(stub.received)
    assert revised == []
    assert [r for r in received if r["op"] in ("load", "call")] == []


# ── exactly once (R1-M5) ───────────────────────────────────────────────────────────────────────


async def test_a_revision_never_rewrites_a_row_that_is_no_longer_pending(monkeypatch, batch_db):  # noqa: F811
    import millm.db.base as db_base
    from millm.db.models.rule_set import RuleSignalResult
    from millm.rules.evaluator import Outcome

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    base = {"result_id": "r1", "kind": "rule", "outcome": "skipped", "status": "unavailable",
            "reason": "rule_worker_unavailable", "fires": None, "count": 0}
    async with batch_db() as session:
        session.add(RuleSignalResult(
            result=dict(base), result_id="r1", event_time=datetime.now(timezone.utc),
            outcome="skipped", reason="rule_worker_unavailable", status="unavailable",
            request_id="q", rule_set_id="rs_000000000001", rule_set_digest="a" * 64,
            signal="operator", form="signal", window="all", fires=None, count=0))
        await session.commit()
    fired = Outcome("operator", "signal", "all", True, 3, "completed", None)
    failed = Outcome("operator", "signal", "all", None, 0, "failed", "rule_error")
    assert len(await code_followup.revise([("r1", fired)])) == 1
    assert await code_followup.revise([("r1", failed)]) == []
    async with batch_db() as session:
        row = (await session.execute(select(RuleSignalResult))).scalar_one()
    assert (row.status, row.outcome, row.fires, row.count) == ("final", "completed", True, 3)
    assert row.result["fires"] is True and row.result["status"] == "final"


# ── bounded jobs (R1-M4) ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jobs_in_flight_are_bounded_and_the_rest_stay_pending(monkeypatch, caplog):
    gate = asyncio.Event()

    async def held(job, results, *, live):
        await gate.wait()
        return 0

    monkeypatch.setattr(code_followup, "run", held)
    monkeypatch.setattr(code_followup, "MAX_JOBS_IN_FLIGHT", 3)
    job = code_followup.CodeJob(sets=(("k", ("all",), object()),), texts={"all": "x"})
    started = [code_followup.spawn(job, [], live=True) for _ in range(5)]
    try:
        assert sum(t is not None for t in started) == 3
        assert started[3] is None and started[4] is None
        assert "rule_code_jobs_saturated" in caplog.text
    finally:
        gate.set()
        await asyncio.gather(*(t for t in started if t is not None))
    assert code_followup.spawn(job, [], live=True) is not None
    gate.set()
    assert await code_followup.drain(5) == 0


# ── the import body (A29) and an unusable params_schema (R1-L2) ─────────────────────────────


def _code(**over) -> dict:
    code = dict(code_document())
    code.update(over)
    return code


def _envelope(doc: dict, code: dict) -> bytes:
    return json.dumps({"rule_set": doc, "code_modules": {
        rc.code_sha256(code): rc.canonical_json(code).decode("utf-8")}}).encode("utf-8")


@pytest.mark.asyncio
async def test_the_body_may_be_the_rule_sets_limit_plus_the_code_documents(monkeypatch):
    code = _code(source=code_document()["source"] + "#" * 3000 + "\n")
    async with cap.Harness() as h:
        doc = rule_set_with_code(bf.random_key(), code)
        raw = _envelope(doc, code)
        doc_bytes = len(json.dumps(doc, separators=(",", ":")).encode())
        monkeypatch.setattr(settings, "RULE_SET_MAX_BYTES", max(doc_bytes + 64, 1024))
        assert len(raw) > settings.RULE_SET_MAX_BYTES, "the fixture must exceed the set's limit"
        r = h.client.post("/api/rule-sets/import", content=raw,
                          headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [{"type": 5}, {"$ref": "https://example.invalid/s.json"}])
async def test_an_unusable_params_schema_is_refused_not_a_500(schema):
    code = _code(params_schema=schema)
    async with cap.Harness() as h:
        r = h.client.post("/api/rule-sets/import",
                          content=_envelope(rule_set_with_code(bf.random_key(), code), code),
                          headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    assert r.json()["error"]["details"]["rule"] == "code_params_invalid"


# ── what reads a code rule's positions is REFUSED (D65, 2026-10-25; was R1's warning) ───────────


def _proximity(sid: str, anchor: str) -> dict:
    return {"id": sid, "kind": "proximity", "anchor": anchor, "phrases": ["marker"],
            "word_boundary": True, "max_distance": 8, "unit": "code_points", "side": "before",
            "ignore_case": True, "fold_fullwidth_ascii": False}


def _within_n(a: str, b: str) -> dict:
    return {"within_n": {"a": {"signal": a}, "b": {"signal": b}, "n": 5, "unit": "code_points"}}


def _positional(key: str, code: dict, use: str) -> dict:
    """A schema-valid set whose one extra use reads the code rule `op`'s POSITIONS."""
    doc = rule_set_with_code(key, code)
    if use == "within_n":
        doc["expressions"]["op_near_token"] = _within_n("op", "tok")
    elif use == "nested_within_n":  # inside an `all`, the code rule as operand `b`
        doc["expressions"]["nested"] = {"all": [{"signal": "tok"}, _within_n("tok", "op")]}
    elif use == "anchor":
        doc["signals"].append(_proximity("near_op", "op"))
        doc["expressions"]["near_operator"] = {"signal": "near_op"}
    elif use == "span_of":
        doc["signals"].append({"id": "luhn_op", "kind": "validator",
                               "validator": {"id": "luhn", "version": 1}, "params": {},
                               "candidates": {"span_of": "op"}})
        doc["expressions"]["valid_operator"] = {"signal": "luhn_op"}
    elif use == "veto_when":  # D74: `when` names an expression over the code rule
        doc["vetoes"] = [{"id": "v_op", "when": "op_and_token", "removes": ["tok"]}]
    elif use == "veto_removes":  # D74: a veto removes the code rule's (non-existent) spans
        doc["vetoes"] = [{"id": "v_tok", "when": "token", "removes": ["cue", "op"]}]
    elif use == "field":  # D74: a `field` output reads the code rule's candidate outputs
        doc["outputs"].append({"column": "op_hits", "type": "int", "value": {"field": ["op", "hits"]}})
        for case in doc["cases"]:
            case["expect"]["outputs"]["op_hits"] = 0
    return doc


#: use → (details.rule, the rest of details): names only, the producer's codes and shape.
POSITIONAL = {
    "within_n": ("code_rule_in_within_n", {"signal": "op", "expression": "op_near_token"}),
    "nested_within_n": ("code_rule_in_within_n", {"signal": "op", "expression": "nested"}),
    "anchor": ("code_rule_positions_needed",
               {"code_rule": "op", "use": "anchor", "signal": "near_op"}),
    "span_of": ("code_rule_positions_needed",
                {"code_rule": "op", "use": "span_of", "signal": "luhn_op"}),
    # D74 (2026-10-28): miDataworks `_check_code_positions`'s veto and field branches.
    "veto_when": ("code_rule_positions_needed",
                  {"code_rule": "op", "use": "veto_when", "veto": "v_op"}),
    "veto_removes": ("code_rule_positions_needed",
                     {"code_rule": "op", "use": "veto_removes", "veto": "v_tok"}),
    "field": ("code_rule_positions_needed",
              {"code_rule": "op", "use": "field", "column": "op_hits"}),
}


def _named(details: dict) -> str:
    """What the message must name: the expression, reading signal, veto or output column."""
    for key in ("expression", "signal", "veto", "column"):
        if key in details:
            return details[key]
    raise AssertionError(details)


def _refusal(doc: dict) -> tuple[str, dict]:
    from millm.rules import RuleSetNotRunnable, prepare

    with pytest.raises(RuleSetNotRunnable) as caught:
        prepare(doc, run_own_cases=False)
    return caught.value.code, caught.value.details


class TestProducerOrder:
    """D74: the FIRST refusal is the producer's first (`_check_code_positions`): within_n over
    every expression, then each signal's anchor and span_of, then each veto's `when` before its
    `removes`, then the outputs. A consumer naming a different first use than the producer for
    the same document would send the operator to fix the wrong thing."""

    def _doc(self, *uses: str) -> dict:
        doc = _positional("b-r1", code_document(), uses[0])
        extra = _positional("b-r1", code_document(), uses[1]) if len(uses) > 1 else doc
        for key in ("signals", "outputs"):
            for item in extra[key]:
                if item not in doc[key]:
                    doc[key].append(item)
        doc["expressions"].update(extra["expressions"])
        doc["vetoes"] = (doc.get("vetoes") or []) + [
            v for v in extra.get("vetoes") or [] if v not in (doc.get("vetoes") or [])]
        for case, other in zip(doc["cases"], extra["cases"]):
            case["expect"]["outputs"].update(other["expect"]["outputs"])
        return doc

    @pytest.mark.parametrize("first, second", [
        ("within_n", "anchor"), ("anchor", "veto_when"), ("span_of", "veto_removes"),
        ("veto_when", "field"), ("veto_removes", "field"),
    ])
    def test_earlier_use_wins_whatever_the_document_order(self, first, second):
        assert _refusal(self._doc(second, first)) == POSITIONAL[first]
        assert _refusal(self._doc(first, second)) == POSITIONAL[first]

    def test_a_vetos_when_is_checked_before_its_removes(self):
        doc = _positional("b-r1", code_document(), "veto_when")
        doc["vetoes"] = [{"id": "v_both", "when": "operator", "removes": ["op"]}]
        assert _refusal(doc) == ("code_rule_positions_needed",
                                 {"code_rule": "op", "use": "veto_when", "veto": "v_both"})

    def test_vetoes_are_checked_in_document_order(self):
        doc = _positional("b-r1", code_document(), "veto_when")
        doc["vetoes"] = [{"id": "v_tok", "when": "token", "removes": ["op"]},
                         {"id": "v_op", "when": "operator", "removes": ["tok"]}]
        assert _refusal(doc)[1] == {"code_rule": "op", "use": "veto_removes", "veto": "v_tok"}

    def test_count_uses_of_a_code_rule_in_vetoes_and_outputs_are_accepted(self):
        """A veto naming only declarative signals, and `fired`/`count` outputs over the code
        rule, need no positions: accepted (pending until run)."""
        from millm.rules import prepare

        doc = rule_set_with_code("b-r1", code_document())
        doc["vetoes"] = [{"id": "v_tok", "when": "token", "removes": ["cue"]}]
        doc["outputs"].append({"column": "op_n", "type": "int", "value": {"count": "operator"}})
        for case in doc["cases"]:
            case["expect"]["outputs"]["op_n"] = 0
        prepare(doc, run_own_cases=False)


@pytest.mark.parametrize("use", sorted(POSITIONAL))
def test_every_way_of_reading_a_code_rules_positions_is_refused(use):
    from millm.rules import RuleSetNotRunnable, prepare

    with pytest.raises(RuleSetNotRunnable) as caught:
        prepare(_positional("b-r1", code_document(), use), run_own_cases=False)
    rule, details = POSITIONAL[use]
    assert (caught.value.code, caught.value.details) == (rule, details)


def test_a_reader_of_a_reader_is_refused_through_the_first_reader():
    from millm.rules import RuleSetNotRunnable, prepare

    doc = _positional("b-r1", code_document(), "anchor")
    doc["signals"].append(_proximity("near_near", "near_op"))
    doc["expressions"]["far"] = _within_n("near_near", "tok")
    with pytest.raises(RuleSetNotRunnable) as caught:
        prepare(doc, run_own_cases=False)
    assert caught.value.details == {"code_rule": "op", "use": "anchor", "signal": "near_op"}


def test_a_set_without_code_or_with_counts_only_is_accepted():
    from millm.rules import prepare
    from tests.support import rule_set_factory as rf

    prepare(rf.document("b-r1"))  # within_n/proximity over declarative signals
    # D41/D44: `{signal: op}` and `all` over it resolve from the code rule's count.
    prepared = prepare(rule_set_with_code("b-r1", code_document()))
    assert {"operator", "op_and_token"} <= prepared.compiled.pending_expressions


def test_a_stored_set_with_a_positional_read_fails_its_startup_self_check():
    """Stored before D65 (accepted with the R1 warning): the next startup marks it
    `vectors_failed`, naming the rule, so it is never served as pending forever."""
    from millm.rules.selfcheck import check_document

    status, report = check_document(_positional("b-r1", code_document(), "within_n"))
    assert status == "vectors_failed"
    assert (report["where"], report["signal"]) == ("code_rule_in_within_n", "op")


@pytest.mark.parametrize("use", ["veto_when", "veto_removes", "field"])
def test_a_stored_set_with_a_d74_read_fails_its_self_check_naming_the_code_rule(use):
    """D74 refusals carry `code_rule`, not a reading `signal`: the self-check report still names
    the rule (it read only `signal` and reported None)."""
    from millm.rules.selfcheck import check_document

    status, report = check_document(_positional("b-r1", code_document(), use))
    assert status == "vectors_failed"
    assert (report["where"], report["signal"]) == ("code_rule_positions_needed", "op")


@pytest.mark.asyncio
@pytest.mark.parametrize("use", sorted(POSITIONAL))
async def test_validate_and_import_refuse_it_and_store_nothing(use):
    code = code_document()
    rule, details = POSITIONAL[use]
    async with cap.Harness() as h:
        key = bf.random_key()
        raw = _envelope(_positional(key, code, use), code)
        for path in ("/api/rule-sets/validate", "/api/rule-sets/import"):
            r = h.client.post(path, content=raw, headers={"content-type": "application/json"})
            assert r.status_code == 422, (path, r.text)
            error = r.json()["error"]
            assert error["code"] == "RULE_SET_INVALID", path
            want = {"rule": rule, **details}
            assert {k: error["details"].get(k) for k in want} == want, path
            # The message names the expression (within_n), the reading signal, the veto or the
            # output column.
            assert repr(_named(details)) in error["message"], path
            assert code["source"] not in r.text
        listed = h.client.get("/api/rule-sets", params={"behaviour": key})
        assert listed.status_code == 200, listed.text
        assert listed.json()["data"] == []


@pytest.mark.asyncio
async def test_validate_and_import_accept_a_count_only_code_rule_set():
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        raw = _envelope(rule_set_with_code(key, code), code)
        for path in ("/api/rule-sets/validate", "/api/rule-sets/import"):
            r = h.client.post(path, content=raw, headers={"content-type": "application/json"})
            assert r.status_code == 200, (path, r.text)
            data = r.json()["data"]
            assert "warnings" not in data, path  # D65: the R1 warning is gone, not empty
            assert data["code_modules"] == [rc.code_sha256(code)], path
        listed = h.client.get("/api/rule-sets", params={"behaviour": key}).json()["data"]
        assert len(listed) == 1
