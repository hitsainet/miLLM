"""Rule-set documents built per run (039 S2; TD39-36: no behaviour key typed in a test).

`document(key)` is a valid `midataworks.rule-set/v1` document with ONE SIGNAL PER DECLARATIVE FORM
(pattern, proximity, validator, lexical_weights) and a composition over them, keyed to `key`, with
its own cases. The values are constructed: a token shape no identifier uses (`ZQ-` + digits) and
a Luhn-valid digit run of the published test-card kind is not used — the validator here is the
generic `luhn@1` over a constructed `79927398713` (the textbook Luhn example, not an issued value).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

#: The textbook Luhn example (ISO/IEC 7812-1 Annex B's worked number), not an issued value.
LUHN_OK = "79927398713"
SENTINEL_PREFIX = "ZQ-"


def document(key: str, *, name: str = "generated", extra_cases: Optional[list] = None,
             code_rule: bool = False) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "format": "midataworks.rule-set/v1",
        "name": name,
        "behaviour_key": key,
        "description": "A generated rule set: one signal per declarative form.",
        "signals": [
            {"id": "tok", "kind": "pattern", "pattern": r"\bZQ-[0-9]{4}\b", "ignore_case": False,
             "fold_fullwidth_ascii": False, "reads": "text"},
            {"id": "cue", "kind": "proximity", "anchor": "tok", "phrases": ["marker"],
             "word_boundary": True, "max_distance": 8, "unit": "code_points", "side": "before",
             "ignore_case": True, "fold_fullwidth_ascii": False},
            {"id": "num", "kind": "validator", "validator": {"id": "luhn", "version": 1},
             "params": {}, "candidates": {"scan": True}},
            {"id": "lex", "kind": "lexical_weights",
             "terms": [{"term": "zorbel", "weight": 1.5}, {"term": "quint", "weight": 1.0}],
             "cut": 2.5, "counting": "presence", "ignore_case": False,
             "fold_fullwidth_ascii": False, "word_boundary": True},
        ],
        "expressions": {
            "token": {"signal": "tok"},
            "cued": {"signal": "cue"},
            "number": {"signal": "num"},
            "lexicon": {"signal": "lex"},
            "token_and_number": {"all": [{"signal": "tok"}, {"signal": "num"}]},
        },
        "outputs": [
            {"column": "token", "type": "bool", "value": {"fired": "token"}},
            {"column": "tokens", "type": "int", "value": {"count": "token"}},
            {"column": "cued", "type": "bool", "value": {"fired": "cued"}},
            {"column": "number", "type": "bool", "value": {"fired": "number"}},
            {"column": "lexicon", "type": "bool", "value": {"fired": "lexicon"}},
            {"column": "both", "type": "bool", "value": {"fired": "token_and_number"}},
        ],
        "cases": [
            _case("all-fire", f"marker ZQ-1234 and {LUHN_OK}, zorbel quint",
                  token=True, tokens=1, cued=True, number=True, lexicon=True, both=True),
            _case("none-fire", "nothing to see here",
                  token=False, tokens=0, cued=False, number=False, lexicon=False, both=False),
            # "1234" fails Luhn and passes a Luhn with the doubling step removed: this case is
            # what makes a broken check digit visible (control S2-16; a fixture whose values
            # agree under the mutant proves nothing).
            _case("uncued", "ZQ-1234 alone, 79927398710",
                  token=True, tokens=1, cued=False, number=False, lexicon=False, both=False),
        ] + list(extra_cases or []),
    }
    if code_rule:
        doc["signals"].append({
            "id": "op", "kind": "code_rule", "code_sha256": "0" * 64,
            "runner_contract": "midataworks.code-rule-runner/v1", "function": "detect",
            "params": {}, "outputs": [{"name": "op_hit", "type": "bool"}]})
        doc["expressions"]["operator"] = {"signal": "op"}
        doc["outputs"].append({"column": "operator", "type": "bool",
                               "value": {"fired": "operator"}})
        for case in doc["cases"]:
            case["expect"]["outputs"]["operator"] = False
    return doc


#: D28: the one case whose generic Luhn signal fires states why its value cannot be issued.
_UNISSUABLE = {"kind": "unissuable", "citation": (
    "79927398713 is ISO/IEC 7812-1 Annex B's worked Luhn example: 11 digits, a length no card "
    "(12-19), routing (9) or account format issues")}


def _case(case_id: str, text: str, **outputs: Any) -> dict[str, Any]:
    source = _UNISSUABLE if outputs.get("number") else {"kind": "constructed"}
    return {"id": case_id, "input": {"text": text}, "note": "constructed",
            "source": source, "expect": {"outputs": outputs, "vetoed": False}}


def raw(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def with_wrong_case(key: str) -> dict[str, Any]:
    doc = document(key)
    doc = copy.deepcopy(doc)
    doc["cases"][1]["expect"]["outputs"]["lexicon"] = True
    return doc
