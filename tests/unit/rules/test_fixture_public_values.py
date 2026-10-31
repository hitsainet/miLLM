"""Every valid value in a test rule set is public or provably unissuable (039 D28, D30; FTASKS 8.2).

miLLM's `tests/` tree reaches the public mirror. A case whose text makes a VALUE-BEARING validator
fire on a valid candidate (a well-formed card, account or routing number) must carry a `source` of
kind `published_test_value`, `published_reference_value` or `unissuable` with a citation. Masks
(`masked@1`, `value_bearing=False`) and constructed shapes that no validator accepts need nothing.

Which cases are valid is COMPUTED by the evaluator, never read from the case's own label: a case
mislabelled `constructed` around a real-looking valid number is exactly what this catches.

Negative control (in this file): a copy of a pilot set with one valid case relabelled
`constructed` must be reported.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from millm.rules.cases import case_input
from millm.rules.evaluator import CompiledRuleSet
from tests.support import rule_set_factory as rf

ROOT = Path(__file__).resolve().parents[3]
SETS_DIR = ROOT / "tests" / "fixtures" / "behaviour_packs" / "rule_sets"
PUBLIC = {"published_test_value", "published_reference_value", "unissuable"}


def fixture_documents() -> dict[str, dict]:
    docs = {p.name: json.loads(p.read_text()) for p in sorted(SETS_DIR.glob("*.json"))}
    docs["factory:generated"] = rf.document("b-00000000")
    docs["factory:code_rule"] = rf.document("b-00000000", code_rule=True)
    return docs


def violations(doc: dict) -> list[str]:
    compiled = CompiledRuleSet(doc)
    bearing = {s["id"] for s in doc["signals"] if s["kind"] == "validator"
               and compiled._validators[s["id"]].decl.value_bearing}
    out = []
    for case in doc["cases"]:
        text, element = case_input(case)
        results = compiled.signal_results(text, element)
        if any(results[sid].spans for sid in bearing):
            source = case.get("source") or {}
            if source.get("kind") not in PUBLIC or not source.get("citation"):
                out.append(case["id"])
    return out


def test_the_fixtures_are_found():
    docs = fixture_documents()
    assert {"card-pan.json", "iban.json", "aba-routing.json", "dlp-ssn-us.json"} <= set(docs)


def test_every_valid_case_is_public_or_unissuable():
    bad = {name: v for name, doc in fixture_documents().items() if (v := violations(doc))}
    assert not bad, bad


def test_a_valid_case_relabelled_constructed_is_reported():
    doc = copy.deepcopy(json.loads((SETS_DIR / "card-pan.json").read_text()))
    case = next(c for c in doc["cases"] if c["id"] == "valid-named")
    case["source"] = {"kind": "constructed"}
    assert violations(doc) == ["valid-named"]


def test_every_fixture_set_proves_itself():
    """The pilot sets and the producer's published set pass their own cases (FTDD §16: a range
    bound mutated → the set's own cases fail at import)."""
    from millm.rules import prepare

    for name, doc in fixture_documents().items():
        assert prepare(doc).cases.passed >= 3, name
