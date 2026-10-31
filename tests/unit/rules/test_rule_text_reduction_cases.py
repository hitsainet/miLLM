"""The text each window gives a rule, as published (039 TD39-21 – TD39-23; FTASKS 11.8).

`docs/schemas/rule-text-reduction-cases.json` is OWNED by miLLM (miStudio 040's `reduce.py` vendors
it byte-identically). Every case is recomputed through THE builder, `millm.rules.windows.
window_texts`, and a few are pinned again here by hand so the file cannot drift together with the
code it was generated from.

The serving side: `_rules_finish` must read the same VISIBLE reply each site already records
(reasoning excluded, first choice only). `test_each_site_passes_the_reply_it_records` compares,
by AST, the `reply_text=` of every `_rules_finish` call with the `reply_text=` of the record call in
the same method. Control: pass the raw completion text at one site → red.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import millm.services.inference_service as inference_module
from millm.rules.windows import window_texts

CASES = json.loads(
    (Path(__file__).resolve().parents[3] / "docs" / "schemas" / "rule-text-reduction-cases.json")
    .read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_case_is_recomputed(case):
    data = case["input"]
    got = window_texts(messages=data.get("messages"), prompt=data.get("prompt"),
                       reply_text=data.get("reply_text"))
    assert dict(got.texts) == case["expect"]["texts"]
    assert dict(got.missing) == case["expect"]["missing"]


def test_hand_pinned_cases():
    by_id = {c["id"]: c["expect"] for c in CASES}
    assert by_id["chat.multi_turn_last_user"]["texts"] == {
        "prompt": "first\nans\nsecond", "last_user": "second", "response": "r",
        "all": "first\nans\nsecond\nr"}
    assert by_id["chat.reasoning_not_read"]["texts"]["prompt"] == "prev\ngo"
    assert by_id["chat.no_user_turn"]["missing"] == {"last_user": "no_user_turn"}
    assert by_id["completion.prompt_string"]["missing"] == {
        "last_user": "text_completion_has_no_user_turn"}
    assert by_id["chat.text_as_given"]["texts"]["last_user"] == "  ＡＢＣ  Café\tTAB "
    assert len(CASES) >= 12


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == name]


def _reply(call: ast.Call) -> str:
    kw = next(k for k in call.keywords if k.arg == "reply_text")
    return ast.unparse(kw.value)


def test_each_site_passes_the_reply_it_records():
    tree = ast.parse(Path(inference_module.__file__).read_text())
    checked = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        rules = _calls(fn, "_rules_finish")
        if not rules or fn.name == "_rules_finish":
            continue
        records = _calls(fn, "_record_probe_shielded")
        assert records, f"{fn.name}: runs rules but records nothing"
        recorded = {_reply(r) for r in records if any(k.arg == "reply_text" for k in r.keywords)}
        for call in rules:
            assert _reply(call) in recorded, (fn.name, _reply(call), recorded)
            checked += 1
    assert checked == 10, f"expected the ten `_probe_finish` sites, found {checked}"
