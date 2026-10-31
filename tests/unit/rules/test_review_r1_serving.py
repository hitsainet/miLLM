"""039 review R1 (S1–S2): what the serving path owes the shared vectors and D41.

Four gaps the builder's controls left open, each found by a mutation that survived the suite:

- R1-01: the case runner compared outputs with Python `==`, so a `fired` column emitted as `1`
  passed every case. The producer compares WITH JSON TYPES (`True` is not `1`); so does miLLM now.
- R1-02: every shared row case ran through `CompiledRuleSet.evaluate` (the row semantics) and none
  through `serve`, the path a live request takes. A `not` expression served as `bool(spans)`
  (never fires) stayed green. Now every text case runs through `serve` too, on every window.
- R1-03: an expression over a signal that READS operator code (a proximity anchored on a
  `code_rule`, a validator over its spans) must be pending as well (D41): otherwise it reports
  `completed, fires: false` — an honest-looking answer computed from nothing.
- R1-08: a window the request has no text for must report `skipped` with why, never vanish
  ("nothing dropped silently", FR-39.23).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from millm.rules import registry as rule_registry
from millm.rules.cases import case_input, run_cases, same
from millm.rules.evaluator import PENDING_REASON, WINDOW_ORDER, CompiledRuleSet
from millm.rules.serving import evaluate_work
from millm.rules.windows import window_texts
from tests.support import rule_set_factory as rf

VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "docs" / "schemas" / "rule-signal-cases-v1.json")
    .read_text(encoding="utf-8")
)


def _text_cases():
    for kind, cases in VECTORS["rule_kinds"].items():
        for case in cases:
            if not isinstance(case.get("document"), str):
                continue
            if "text" not in case["input"]:
                continue  # an element input has no serving equivalent (serving reads text)
            yield pytest.param(case, id=f"{kind}:{case['id']}")


TEXT_CASES = list(_text_cases())


def test_same_is_typed():
    assert same(True, True) and same(1, 1) and same([[0, 1]], [[0, 1]])
    assert not same(1, True) and not same(True, 1) and not same(0, False)
    assert not same(1, 1.0) and not same([[0, 1]], [[0, True]])
    assert not same({"a": 1}, {"a": 1, "b": None})


def test_a_case_expecting_true_fails_on_one():
    """The runner itself: a column whose expectation is `1` where the evaluator says `True`."""
    doc = rf.document("b-0review0")
    case = copy.deepcopy(doc["cases"][0])
    case["expect"]["outputs"]["token"] = 1
    report = run_cases(CompiledRuleSet(doc), [case])
    assert not report.ok and report.failures[0].where == "token"


@pytest.mark.parametrize("case", TEXT_CASES)
def test_serving_agrees_with_every_shared_row_case(case):
    """`serve` on every window gives each expression the truth and spans of the row semantics.
    Where the document declares vetoes, serving does not apply them (D9) — so the comparison is
    against the PRE-veto evaluation, and the case's stated spans are checked only without vetoes.
    """
    doc = VECTORS["rule_documents"][case["document"]]
    compiled = CompiledRuleSet(doc)
    text, _ = case_input(case)
    windows = dict.fromkeys(WINDOW_ORDER, text)
    served = compiled.serve(windows, return_positions=True)
    assert len(served) == len(compiled.expressions) * len(WINDOW_ORDER)
    if doc.get("vetoes"):
        unvetoed = copy.deepcopy(doc)
        unvetoed["vetoes"] = []
        row = CompiledRuleSet(unvetoed).evaluate(text)
    else:
        row = compiled.evaluate(text)
        for name, spans in (case["expect"].get("spans") or {}).items():
            got = [o for o in served if o.signal == name]
            assert all(same([list(s) for s in o.spans], [list(s) for s in spans]) for o in got), \
                (name, got)
    for o in served:
        want = row.expressions[o.signal]
        assert (o.outcome, o.reason) == ("completed", None)
        assert o.fires is want["fired"], (o.signal, o.window)
        assert [list(s) for s in o.spans] == want["spans"], (o.signal, o.window)
        assert o.count == len(want["spans"])


def test_the_serving_comparison_covers_every_composition():
    forms = {CompiledRuleSet(VECTORS["rule_documents"][c.values[0]["document"]]).form_of(n)
             for c in TEXT_CASES
             for n in VECTORS["rule_documents"][c.values[0]["document"]]["expressions"]}
    assert {"all", "any", "not", "within_n", "pattern", "proximity", "validator",
            "lexical_weights"} <= forms


def _with_readers_of_operator_code(key: str) -> dict:
    doc = rf.document(key, code_rule=True)
    doc["signals"] += [
        {"id": "near_op", "kind": "proximity", "anchor": "op", "phrases": ["marker"],
         "word_boundary": True, "max_distance": 8, "unit": "code_points", "side": "before",
         "ignore_case": True, "fold_fullwidth_ascii": False},
        {"id": "luhn_op", "kind": "validator", "validator": {"id": "luhn", "version": 1},
         "params": {}, "candidates": {"span_of": "op"}},
    ]
    doc["expressions"]["near_operator"] = {"signal": "near_op"}
    doc["expressions"]["valid_operator"] = {"signal": "luhn_op"}
    doc["expressions"]["token_or_near"] = {"any": [{"signal": "tok"}, {"signal": "near_op"}]}
    return doc


def test_a_signal_reading_operator_code_is_pending_too():
    """D41: whatever reads a `code_rule`'s spans — directly or through an anchor or `span_of` —
    is pending: skipped, unavailable, `rule_worker_unavailable`, never `completed`."""
    compiled = CompiledRuleSet(_with_readers_of_operator_code("b-0review0"))
    assert {"op", "near_op", "luhn_op"} <= compiled.pending_signals
    served = compiled.serve({"prompt": "marker ZQ-1234"})
    pending = {"operator", "near_operator", "valid_operator", "token_or_near"}
    for o in served:
        if o.signal in pending:
            assert (o.outcome, o.status, o.reason, o.fires) == (
                "skipped", "unavailable", PENDING_REASON, None), o
        else:
            assert o.outcome == "completed", o
    assert {o.signal for o in served} >= pending


def test_a_window_without_text_reports_why():
    """A text completion has no user turn: the behaviour's `last_user` window is reported
    `skipped` with the reason the reduction gives, beside the completed `prompt` window."""
    doc = rf.document("b-0review0")
    attached = rule_registry.AttachedSet(
        rule_set_id="rs_000000000abc", digest="a" * 64, version=1, behaviour_key="b-0review0",
        compiled=CompiledRuleSet(doc), vectors_failed=False, only=None,
        forms=rule_registry.document_forms(doc))
    texts = window_texts(prompt="marker ZQ-1234")
    assert "last_user" in texts.missing
    items = evaluate_work([("b-0review0", ("prompt", "last_user"), attached)], texts,
                          clock=lambda: 0.0, deadline=None)
    by_window: dict[str, list] = {}
    for item in items:
        by_window.setdefault(item.window, []).append(item)
    assert len(by_window["prompt"]) == len(by_window["last_user"]) == len(doc["expressions"])
    for item in by_window["last_user"]:
        assert (item.outcome, item.reason) == ("skipped", texts.missing["last_user"])
        assert item.fires is None and item.count == 0
    assert all(i.outcome == "completed" for i in by_window["prompt"])


@pytest.mark.asyncio
async def test_the_evaluate_route_runs_the_evaluator_off_the_event_loop(monkeypatch):
    """`POST /api/rule-sets/{id}/evaluate` has no budget (it is a tool, not serving), so it must
    not run on the event loop that streams every other request. Control R1-17: call
    `evaluate_work` directly in the route → red."""
    import asyncio

    from millm.rules import serving
    from tests.support import behaviour_factory as bf
    from tests.support import capture_pre038_goldens as cap

    on_loop: list[bool] = []
    real = serving.evaluate_work

    def spy(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)  # called from a coroutine: the loop is blocked meanwhile
        except RuntimeError:
            on_loop.append(False)  # a worker thread has no running loop
        return real(*args, **kwargs)

    monkeypatch.setattr(serving, "evaluate_work", spy)
    async with cap.Harness() as h:
        r = h.client.post("/api/rule-sets/import", content=rf.raw(rf.document(bf.random_key())),
                          headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        rule_set_id = r.json()["data"]["id"]
        r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate", json={"text": "ZQ-1234"})
        assert r.status_code == 200, r.text
        assert any(x["fires"] for x in r.json()["data"]["results"])
    assert on_loop == [False]
