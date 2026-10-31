"""Operator code on the STATELESS routes, run synchronously (suite decision D57; 039 follow-up).

`POST /api/rule-sets/{id}/evaluate` and `POST /api/probes/score` have no live budget, so D41's
reason for scoring code after the response does not apply there: the request waits for the
runner and returns the decided results. Proven through the LIVE app:

1. the worker answers → the code rule's expression is `completed` with the runner's count, the
   `all` over it resolves (a `within_n` over it is refused at import, D65) and no span is claimed for a code-decided result; the worker received exactly one call per
   (code rule, window), carrying the stored document's hash;
2. pending (`rule_worker_unavailable`, never `fires: false`) ONLY when the runner is absent or
   never answers within the client's wait — a call the worker never answered is NOT
   `rule_timeout` here;
3. one request makes at most `RULE_CODE_CALLS_PER_REQUEST` calls, across all its inputs;
4. the LIVE chat route still never waits (D41): the response arrives while the worker holds its
   reply, and the synchronous path is never entered from a chat.

Wiring controls (039 controls record, Follow-ups): remove `code.resolve` from either route, or the
route's `code_documents=` → red here.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from millm.core.config import settings
from millm.rules import code_followup, runner_client
from millm.rules import runner_contract as rc
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_runner as rr
from tests.support.stub_runner import StubRunner
from tests.unit.api.test_rule_code_followup import (
    BODY,
    CODE_NAMES,
    TEXT,
    _arm,
    _attach,
    _code,
    _live_loop,
    _pending,
    code_document,
    envelope,
    rule_set_with_code,
)

pytestmark = pytest.mark.asyncio


def _import(h, key, code) -> str:
    raw = envelope(rule_set_with_code(key, code), code)
    r = h.client.post("/api/rule-sets/import", content=raw,
                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    return r.json()["data"]["id"]


def _evaluate(h, rule_set_id, **body) -> list[dict[str, Any]]:
    r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate", json={"text": TEXT, **body})
    assert r.status_code == 200, r.text
    return r.json()["data"]["results"]


def _by(results, signal) -> dict[str, dict[str, Any]]:
    """`{window: entry}` for one signal, on the windows the input has text for (a window with no
    text reports why, e.g. `no_scored_tokens`, and is never sent to the runner)."""
    return {e["window"]: e for e in results
            if e["signal"] == signal and e["reason"] != "no_scored_tokens"}


async def test_the_evaluate_route_waits_for_the_runner_and_returns_its_result(monkeypatch):
    code = code_document()
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with StubRunner() as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            results = _evaluate(h, rule_set_id, return_rule_positions=True)
        op, both = (_by(results, s) for s in ("operator", "op_and_token"))
        assert op, results
        for window, entry in op.items():
            assert (entry["outcome"], entry["fires"], entry["count"], entry["reason"],
                    entry["status"]) == ("completed", True, 2, None, "final")
            # A code rule reports a count, never positions: no span is claimed for it.
            assert "spans" not in entry
            token_fired = both[window]["count"] > 0
            assert both[window]["outcome"] == "completed" and both[window]["status"] == "final"
            assert both[window]["fires"] is token_fired
            if token_fired:
                assert both[window]["count"] == 1 + 2
        # Declarative entries keep their spans on request.
        token = [e for e in results if e["signal"] == "token" and e["outcome"] == "completed"]
        assert token and all("spans" in e for e in token)
        calls = stub.calls()
        # One call per (code rule, window): the `operator` and `op_and_token` share it.
        assert len(calls) == len(op)
        assert {c["code_sha256"] for c in calls} == {rc.code_sha256(code)}
        assert {c["function"] for c in calls} == {"f"}
        assert all(c["text_len"] == len(TEXT) for c in calls)


async def test_a_runner_failure_is_a_final_failed_result_not_pending(monkeypatch):
    """The worker answered with a reason: the request reports it (final), never pending."""
    code = code_document()
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with StubRunner(reply=lambda _m: {"reason": "rule_timeout"}) as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            results = _evaluate(h, rule_set_id)
        op = _by(results, "operator")
        assert op and all((e["outcome"], e["reason"], e["fires"], e["status"]) ==
                          ("failed", "rule_timeout", None, "final") for e in op.values())


async def test_no_runner_leaves_code_expressions_pending(monkeypatch):
    code = code_document()
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", "/nonexistent/millm-runner.sock")
        results = _evaluate(h, rule_set_id)
        code_entries = [e for e in results if e["signal"] in CODE_NAMES]
        assert code_entries and all(_pending(e) for e in code_entries)


async def test_a_runner_that_never_answers_leaves_them_pending_not_timed_out(monkeypatch):
    """D57: pending when the runner times out — the call may never have run, so it is not a
    `rule_timeout` result. The request is bounded by the client's wait (2 x wall + margin)."""
    code = code_document()
    monkeypatch.setattr(settings, "RULE_CODE_CALL_WALL_MS", 50)
    monkeypatch.setattr(runner_client, "REPLY_MARGIN_S", 0.3)
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with StubRunner(hold=True) as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            started = time.monotonic()
            results = _evaluate(h, rule_set_id)
            elapsed = time.monotonic() - started
            stub.release.set()
        code_entries = [e for e in results if e["signal"] in CODE_NAMES]
        assert code_entries and all(_pending(e) for e in code_entries), code_entries
        # One unanswered call breaks the connection: the rest are not tried, so the wait is one
        # client wait, not one per window.
        assert len(stub.calls()) == 1
        assert elapsed < 5.0


async def test_one_stateless_request_spends_at_most_its_code_budget(monkeypatch):
    """Review R2: a request's operator code is bounded IN ALL (`STATELESS_CODE_BUDGET_S` from the
    request's start), not only per call — at the defaults 16 slow calls took over 64 s, past
    miStudio's 60 s client timeout. Past the budget nothing more is sent and the reply in flight
    is not awaited; those expressions stay pending (never `rule_timeout`, never `fires: false`)."""
    code = code_document()
    monkeypatch.setattr(code_followup, "STATELESS_CODE_BUDGET_S", 0.6)

    def slow(_message):
        time.sleep(0.45)
        return {"count": 2}

    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with StubRunner(reply=slow) as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            started = time.monotonic()
            results = _evaluate(h, rule_set_id)
            elapsed = time.monotonic() - started
        op = _by(results, "operator")
        assert len(op) >= 2, "needs two windows to show the budget"
        done = [e for e in op.values() if e["outcome"] == "completed"]
        assert len(done) == 1 and done[0]["count"] == 2
        assert all(_pending(e) for e in op.values() if e["outcome"] != "completed"), op
        # The second call was sent before the budget ran out and abandoned at it: two calls.
        assert len(stub.calls()) == 2
        assert elapsed < 3.0


async def test_the_score_route_runs_code_with_one_call_cap_for_the_whole_request(monkeypatch):
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(2):
                await bf.import_probe(session, bf.definition(f"{key}-s{i}", key=key, layer=i,
                                                             scale=0.13 + 0.2 * i))
        rule_set_id = _import(h, key, code)
        r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                         json={"rule_set_ids": [rule_set_id]})
        assert r.status_code == 200, r.text
        inputs = [{"text": TEXT}, {"text": "plain w1"}]
        with StubRunner() as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            r = h.client.post("/api/probes/score", json={"behaviours": [key], "inputs": inputs,
                                                         "return_rule_positions": True})
            assert r.status_code == 200, r.text
            first, second = r.json()["data"]["results"]
            # A code-decided result claims no spans; a declarative one keeps them on request.
            assert all("spans" not in e for e in _by(first["rules"], "operator").values())
            assert any("spans" in e for e in _by(first["rules"], "token").values())
            per_input = [len(_by(x["rules"], "operator")) for x in (first, second)]
            for result in (first, second):
                op = _by(result["rules"], "operator")
                assert op and all(
                    (e["outcome"], e["fires"], e["count"], e["status"], e["origin"]) ==
                    ("completed", True, 2, "final", "stateless") for e in op.values())
            assert len(stub.calls()) == sum(per_input)
            assert sorted(c["text_len"] for c in stub.calls()) == sorted(
                [len(TEXT)] * per_input[0] + [len("plain w1")] * per_input[1])

            # The cap is per REQUEST: one call in all, the rest pending.
            monkeypatch.setattr(settings, "RULE_CODE_CALLS_PER_REQUEST", 1)
            before = len(stub.calls())
            r = h.client.post("/api/probes/score", json={"behaviours": [key], "inputs": inputs})
            assert len(stub.calls()) - before == 1
        ops = [e for x in r.json()["data"]["results"] for e in _by(x["rules"], "operator").values()]
        assert len(ops) == sum(per_input)
        assert sum(e["outcome"] == "completed" for e in ops) == 1
        assert all(_pending(e) for e in ops if e["outcome"] != "completed")


async def test_the_real_worker_answers_the_evaluate_route(monkeypatch):
    """The value from the published guest: `count_tokens` counts whitespace tokens."""
    bundle = rr.bundle_dir()
    code = code_document("count_tokens")
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with rr.RunnerProcess(bundle) as runner:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(runner.socket))
            results = _evaluate(h, rule_set_id)
        op = _by(results, "operator")
        assert op["prompt"]["outcome"] == "completed"
        assert op["prompt"]["count"] == len(TEXT.split())


async def test_the_chat_route_still_never_waits_for_operator_code(monkeypatch):
    """D41 stays on the live path: with the worker HOLDING its reply and a 3 s wall (a waited call
    would take 2 x 3 s + margin), the chat answers at once with code pending, and the synchronous
    path is never entered from a chat."""
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    monkeypatch.setattr(settings, "RULE_CODE_CALL_WALL_MS", 3000)
    code = code_document()
    entered: list[int] = []
    real_resolve = code_followup.StatelessCode.resolve

    async def counting(self, *args, **kwargs):
        entered.append(1)
        return await real_resolve(self, *args, **kwargs)

    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        _attach(h, key, envelope(rule_set_with_code(key, code), code))
        with StubRunner(hold=True) as stub, \
                patch.object(code_followup.StatelessCode, "resolve", counting), \
                _live_loop(h) as client:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            h._fresh_socket_throttle()
            started = time.monotonic()
            response = client.post("/v1/chat/completions", json=BODY)
            elapsed = time.monotonic() - started
            assert response.status_code == 200, response.text
            live = _code(response.json()["millm_probe_verdicts"])
            assert live and all(_pending(e) for e in live)
            # The after-response job reaches the worker while the reply is still held.
            assert stub.call_seen.wait(20)
            assert not stub.release.is_set()
            stub.release.set()
            client.portal.call(code_followup.drain, 20.0)
        assert entered == []
        assert elapsed < 6.0, f"the chat waited {elapsed:.1f} s"


async def test_below_the_floor_the_stateless_routes_call_nothing(monkeypatch):
    """FTASKS 17.4 holds here too: a host never lowers a floor, so no call is made and the code
    expressions stay pending."""
    code = code_document()
    monkeypatch.setattr(settings, "RULE_CODE_CALL_MEMORY_MIB", 16)
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with StubRunner() as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            results = _evaluate(h, rule_set_id)
        assert [r for r in stub.received if r["op"] in ("load", "call")] == []
        code_entries = [e for e in results if e["signal"] in CODE_NAMES]
        assert code_entries and all(_pending(e) for e in code_entries)


async def test_after_the_response_a_worker_that_never_answers_leaves_them_pending(monkeypatch):
    """D63 (operator, 2026-10-24): after the response too, a worker that never answers within the
    client's wait (the slot wait AND the call, R1-L1) decides nothing — the stored row stays
    pending (`rule_worker_unavailable`), never a final `rule_timeout`: the call may never have
    run. Control: `score` back to `unanswered_pending=False` → red."""
    from types import SimpleNamespace

    from tests.unit.api.test_rule_code_followup_r1 import _attached
    from tests.unit.api.test_rule_code_followup_r1 import _pending as pending_result

    monkeypatch.setattr(runner_client, "REPLY_MARGIN_S", 0.2)
    code = code_document()
    doc = rule_set_with_code("b-r1", code)
    job = code_followup.job_for([("b-r1", ("all",), _attached(doc, code))], {"all": "a b c"})
    with StubRunner(hold=True) as stub:
        cfg = SimpleNamespace(**{k: getattr(settings, k) for k in (
            "RULE_CODE_CALL_MEMORY_MIB", "RULE_CODE_CALL_OUTPUT_BYTES",
            "RULE_CODE_CALLS_PER_REQUEST")}, RULE_CODE_CALL_WALL_MS=50,
            RULE_RUNNER_SOCKET=str(stub.socket))
        revised = await code_followup.score(job, [pending_result()], settings=cfg)
        stub.release.set()
    assert revised == []


async def test_after_the_response_the_runners_own_rule_timeout_is_final(monkeypatch):
    """D63's other half: when the RUNNER reports the guest overran its wall time, that is a final
    `rule_timeout` after the response (it answered; the call ran)."""
    from types import SimpleNamespace

    from tests.unit.api.test_rule_code_followup_r1 import _attached
    from tests.unit.api.test_rule_code_followup_r1 import _pending as pending_result

    code = code_document()
    doc = rule_set_with_code("b-r1", code)
    job = code_followup.job_for([("b-r1", ("all",), _attached(doc, code))], {"all": "a b c"})
    with StubRunner(reply=lambda _m: {"reason": "rule_timeout"}) as stub:
        cfg = SimpleNamespace(**{k: getattr(settings, k) for k in (
            "RULE_CODE_CALL_MEMORY_MIB", "RULE_CODE_CALL_OUTPUT_BYTES",
            "RULE_CODE_CALLS_PER_REQUEST", "RULE_CODE_CALL_WALL_MS")},
            RULE_RUNNER_SOCKET=str(stub.socket))
        revised = await code_followup.score(job, [pending_result()], settings=cfg)
    assert [(rid, o.outcome, o.reason) for rid, o in revised] == [("r1", "failed", "rule_timeout")]


# ── D56: the runner's own reasons, published ────────────────────────────────────────────────────


@pytest.mark.parametrize("reason,reported", [
    ("rule_resource_limit", "rule_resource_limit"),
    ("rule_output_invalid", "rule_output_invalid"),
    ("rule_load_refused", "rule_load_refused"),
    ("rule_input_too_large", "rule_input_too_large"),
    ("rule_timeout", "rule_timeout"),
    ("rule_error", "rule_error"),
    # A contract reason the published list does not carry is still `rule_error`.
    ("rule_not_approved", "rule_error"),
])
async def test_a_failed_code_rule_reports_the_runners_own_reason(monkeypatch, reason, reported):
    """D56: each result is a strict `mi.probe.result/v1` object carrying the runner's reason."""
    import json as _json

    import jsonschema

    from tests.unit.api.test_probe_result_contract import STRICT

    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            await bf.import_probe(session, bf.definition(f"{key}-s0", key=key, layer=0,
                                                         scale=0.13))
        rule_set_id = _import(h, key, code)
        assert h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                            json={"rule_set_ids": [rule_set_id]}).status_code == 200
        with StubRunner(reply=lambda _m: {"reason": reason}) as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            r = h.client.post("/api/probes/score",
                              json={"behaviours": [key], "inputs": [{"text": TEXT}]})
        rules = r.json()["data"]["results"][0]["rules"]
        op = _by(rules, "operator")
        assert op and all((e["outcome"], e["reason"], e["fires"], e["status"]) ==
                          ("failed", reported, None, "final") for e in op.values())
        validator = jsonschema.Draft202012Validator(_json.loads(STRICT.read_text()))
        for entry in op.values():
            validator.validate(entry)


async def test_a_text_over_the_input_limit_reports_rule_input_too_large(monkeypatch):
    """The client never sends it (FTASKS 17.2) and the result now says why (D56)."""
    code = code_document()
    long_text = "w " * (rc.floor()["input_code_points"] // 2 + 1)
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        with StubRunner() as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate", json={"text": long_text})
        assert stub.calls() == []
        op = _by(r.json()["data"]["results"], "operator")
        assert op and all((e["outcome"], e["reason"]) == ("failed", "rule_input_too_large")
                          for e in op.values())


async def test_only_a_runner_reason_on_the_published_list_passes_through():
    """`published` needs BOTH: on `mi.probe.result/v1`'s list and a runner contract reason. A
    published code the runner does not own (`judge_traffic`) is never taken from a runner reply."""
    assert code_followup.published("rule_resource_limit") == "rule_resource_limit"
    assert code_followup.published("rule_not_approved") == "rule_error"
    assert code_followup.published("judge_traffic") == "rule_error"


async def test_the_stateless_run_logs_no_text(monkeypatch, caplog):
    """FR-39.28 on the synchronous path: the caller's text reaches the worker and the response,
    never a log line — including a failed call's line."""
    import logging
    import random
    import string

    sentinel = "".join(random.SystemRandom().choice(string.ascii_letters) for _ in range(14))
    caplog.set_level(logging.DEBUG)
    code = code_document()
    async with cap.Harness() as h:
        rule_set_id = _import(h, bf.random_key(), code)
        for reply in ({"count": 2}, {"reason": "rule_output_invalid"}):
            with StubRunner(reply=lambda _m, reply=reply: reply) as stub:
                monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
                r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate",
                                  json={"text": f"marker {sentinel} w1"})
            assert r.status_code == 200 and stub.calls()
    assert "rule_code_failed" in caplog.text
    assert not any(sentinel[i:i + 4] in caplog.text for i in range(len(sentinel) - 3))


async def test_the_score_route_closes_its_runner_session_once_and_logs_the_cap(monkeypatch, caplog):
    """Review R2 (mutations N2, N12 survived): the score route's `finally: code.aclose()` and the
    capped-calls warning. Asserted by count: one close per request, and the warning names how many
    calls the cap skipped."""
    import logging

    code = code_document()
    closes: list[int] = []
    real_aclose = code_followup.StatelessCode.aclose

    async def counting(self):
        closes.append(1)
        await real_aclose(self)

    monkeypatch.setattr(code_followup.StatelessCode, "aclose", counting)
    caplog.set_level(logging.WARNING, logger="millm.rules.code_followup")
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            await bf.import_probe(session, bf.definition(f"{key}-c0", key=key, layer=0,
                                                         scale=0.13))
        rule_set_id = _import(h, key, code)
        assert h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                            json={"rule_set_ids": [rule_set_id]}).status_code == 200
        monkeypatch.setattr(settings, "RULE_CODE_CALLS_PER_REQUEST", 1)
        with StubRunner() as stub:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            r = h.client.post("/api/probes/score",
                              json={"behaviours": [key], "inputs": [{"text": TEXT}]})
            assert r.status_code == 200, r.text
        ops = _by(r.json()["data"]["results"][0]["rules"], "operator")
    assert len(closes) == 1
    skipped = len(ops) - 1
    assert skipped >= 1
    assert f"rule_code_calls_capped skipped={skipped} cap=1" in caplog.text
