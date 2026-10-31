"""Every vendored shared vector passes against miLLM's evaluator (039 FR-39.18 – FR-39.20; D11).

`docs/schemas/rule-signal-cases-v1.json` is miDataworks' (vendored byte-identically, pinned by
`test_vendored_rule_contracts.py`). "The vector is the truth": a case red here is a miLLM defect,
never a vector to edit. Sections: rule documents and their row cases, refusals, renderings, the
validator sections (value checks and scans) and the rule-set identity (RFC 8785).

One deliberate difference, recorded as discrepancy X-15: under D41 (2026-10-24) miLLM ACCEPTS a
rule set naming operator code (its expressions report `rule_worker_unavailable` and nothing runs
inline), where the vector `refuse.document.code_rule` expects the producer's refusal. That case is
a strict xfail, so the day the vector or D41 changes, this file goes red. Under D65 (2026-10-25)
the positional case `refuse.document.code_rule_in_within_n` is refused like the producer and is no
longer part of that difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from millm.rules import RuleSetInvalid, RuleSetNotRunnable, dialect, prepare, renderings
from millm.rules.cases import run_cases
from millm.rules.document import identity_digest
from millm.rules.evaluator import CompiledRuleSet
from millm.rules.validators import VALIDATORS, resolve

VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "docs" / "schemas" / "rule-signal-cases-v1.json")
    .read_text(encoding="utf-8")
)
DOCUMENTS = VECTORS["rule_documents"]
#: D41: miLLM accepts operator code; the producer refuses it (discrepancy X-15).
#: D65 (2026-10-25): `refuse.document.code_rule_in_within_n` is NOT here — miLLM refuses a
#: positional read of a code rule exactly as the producer does.
D41_ACCEPTED = {"refuse.document.code_rule"}


def _row_cases():
    for kind, cases in VECTORS["rule_kinds"].items():
        for case in cases:
            if "document" in case and isinstance(case["document"], str):
                yield pytest.param(case, id=f"{kind}:{case['id']}")


def _refusals():
    for case in VECTORS["rule_kinds"]["refusals"]:
        marks = ()
        if case["id"] in D41_ACCEPTED:
            marks = (pytest.mark.xfail(strict=True, reason="D41: operator code is accepted and "
                                       "deferred, never refused (X-15)"),)
        yield pytest.param(case, id=case["id"], marks=marks)


def _validator_cases():
    for ref, section in VECTORS["validators"].items():
        for case in section["cases"]:
            yield pytest.param(ref, case, id=case["id"])


@pytest.mark.parametrize("case", list(_row_cases()))
def test_row_case(case):
    compiled = CompiledRuleSet(DOCUMENTS[case["document"]])
    report = run_cases(compiled, [case])
    assert report.ok, [f.report() for f in report.failures]
    assert not report.unverified


def test_every_row_case_document_is_published():
    named = {c["document"] for cases in VECTORS["rule_kinds"].values() for c in cases
             if isinstance(c.get("document"), str)}
    assert named <= set(DOCUMENTS)


@pytest.mark.parametrize("case", list(_refusals()))
def test_refusal(case):
    refuse = case["refuse"]
    if "pattern" in refuse:
        refused = dialect.scan(refuse["pattern"])
        assert refused is not None, "the dialect admitted a refused pattern"
        assert refused.rule == case["rule"]
        assert refused.position == case["at"]
        return
    with pytest.raises((RuleSetInvalid, RuleSetNotRunnable)) as caught:
        prepare(refuse["document"], run_own_cases=False)
    assert caught.value.code == case["rule"]


@pytest.mark.parametrize("case", VECTORS["rule_kinds"]["renderings"], ids=lambda c: c["id"])
def test_rendering(case):
    rendered = renderings.render(case["rendering"], case["input"]["value"])
    assert rendered == case["expect"]["rendered"]
    assert renderings.normalise(case["rendering"], rendered) == case["expect"]["normalised"]


@pytest.mark.parametrize("ref,case", list(_validator_cases()))
def test_validator_case(ref, case):
    vid, version = ref.split("@")
    validator = resolve(vid, int(version))
    params = case.get("params", {})
    expect = case["expect"]
    data = case["input"]
    if "candidates" in expect:
        got = [{"start": c.start, "end": c.end, "valid": c.verdict.valid,
                "reason": c.verdict.reason, "tier": c.verdict.tier}
               for c in validator.scan(data["text"], params)]
        want = [{k: c.get(k, g.get(k)) for k in ("start", "end", "valid", "reason", "tier")}
                for c, g in zip(expect["candidates"], got)]
        assert len(got) == len(expect["candidates"])
        assert got == want
        return
    verdict = validator.check(data["value"], data.get("before", ""), data.get("after", ""), params)
    assert (verdict.valid, verdict.reason) == (expect["valid"], expect["reason"])
    for name, value in (expect.get("outputs") or {}).items():
        assert getattr(verdict, name) == value, name


@pytest.mark.parametrize("case", VECTORS["rule_set_identity"], ids=lambda c: c["id"])
def test_identity(case):
    from millm.rules.document import canonical_bytes

    assert canonical_bytes(case["document"]).decode("utf-8") == case["canonical_utf8"]
    assert identity_digest(case["document"]) == case["sha256"]


def test_the_registry_and_the_vector_sections_are_the_same_set():
    """Coverage, both ways (FR-39.19): a registered validator with no section, or a section no
    validator answers, is red. The route-level form of this test reads the LIVE registry through
    `GET /api/rule-sets/validators` (test_rule_set_routes.py)."""
    registered = {f"{i}@{v}" for i, v in VALIDATORS}
    assert registered == set(VECTORS["validators"])
    for ref, section in VECTORS["validators"].items():
        assert section["cases"], f"{ref} has no case"
