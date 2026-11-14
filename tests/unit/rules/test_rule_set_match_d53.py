"""A rule set's explicit positive (miDataworks D53, ENH-006 G19-4): miLLM accepts, checks and reports `match`.

`midataworks.rule-set/v1` gained an optional `match: {version: 1, any_of: [expression names]}` (miDataworks
`5a75301`, re-vendored here). Serving does not combine expressions (D9): every expression is reported on its own,
so nothing at serving decides a positive yet; G19's verdict will. What is pinned now:

* a document carrying `match` imports (before the re-vendor it was refused: the schema's top level is closed), its
  stored document keeps the field, and its identity differs from the same document without it;
* `match_of` reads the positive: `explicit` (the stated names, exactly) or `inferred` (every `fired` output, the
  producer's pre-D53 rule), and the rule-set detail and the evaluate answer report it;
* the producer's document rules: absent or an object, never null; each name an expression, once; version 1; no
  other key.
"""

from __future__ import annotations

import copy

import pytest

from miserver.rules import document as rule_document
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf

INFERRED = ["token", "cued", "number", "lexicon", "token_and_number"]  # every `fired` output, output order


def _with_match(names, key=None):
    doc = rf.document(key or bf.random_key())
    doc["match"] = {"version": 1, "any_of": list(names)}
    return doc


# ── reading the positive ─────────────────────────────────────────────────────────────────────────


def test_without_match_the_positive_is_inferred_and_says_so():
    doc = rf.document(bf.random_key())
    assert "match" not in doc
    assert rule_document.match_of(doc) == {"source": "inferred", "version": None, "any_of": INFERRED}


def test_an_explicit_match_is_read_as_stated():
    """`cued` alone: not widened to the `fired` outputs, and a name no output reads is kept."""
    doc = _with_match(["cued"])
    assert rule_document.match_of(doc) == {"source": "explicit", "version": 1, "any_of": ["cued"]}
    doc["outputs"] = [o for o in doc["outputs"] if o["column"] != "cued"]
    for case in doc["cases"]:
        case["expect"]["outputs"].pop("cued")
    rule_document.check(doc)
    assert rule_document.match_of(doc)["any_of"] == ["cued"]


def test_a_document_with_match_passes_the_check():
    doc = _with_match(["cued", "token_and_number"])
    assert rule_document.check(copy.deepcopy(doc)) == doc


# ── the producer's document rules ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["null", "unknown", "duplicate", "version_2", "extra_key", "empty"])
def test_a_malformed_match_is_refused(bad):
    doc = _with_match(["cued"])
    if bad == "null":
        doc["match"] = None
    elif bad == "unknown":
        doc["match"]["any_of"] = ["cued", "zz_not_an_expression"]
    elif bad == "duplicate":
        doc["match"]["any_of"] = ["cued", "cued"]
    elif bad == "version_2":
        doc["match"]["version"] = 2
    elif bad == "extra_key":
        doc["match"]["all_of"] = ["cued"]
    elif bad == "empty":
        doc["match"]["any_of"] = []
    with pytest.raises(rule_document.RuleSetInvalid) as refused:
        rule_document.check(doc)
    assert refused.value.code == "rule_set_invalid"
    expected = {
        "null": {"pointer": "/match"},
        "unknown": {"pointer": "/match/any_of", "expression": "zz_not_an_expression"},
        "duplicate": {"what": "match expression", "name": "cued"},
    }.get(bad)
    if expected is not None:
        assert refused.value.details == expected


# ── the routes report it ─────────────────────────────────────────────────────────────────────────


def _post(h, path, doc, **params):
    return h.client.post(path, content=rf.raw(doc), params=params,
                         headers={"content-type": "application/json"})


@pytest.mark.asyncio
@pytest.mark.parametrize("match, expected", [
    (None, {"source": "inferred", "version": None, "any_of": INFERRED}),
    (["cued"], {"source": "explicit", "version": 1, "any_of": ["cued"]}),
])
async def test_detail_and_evaluate_report_the_positive(match, expected):
    async with cap.Harness() as h:
        doc = rf.document(bf.random_key()) if match is None else _with_match(match)
        imported = _post(h, "/api/rule-sets/import", doc)
        assert imported.status_code == 200, imported.text
        rule_set_id = imported.json()["data"]["id"]
        detail = h.client.get(f"/api/rule-sets/{rule_set_id}").json()["data"]
        assert detail["match"] == expected
        assert ("match" in detail["document"]) is (match is not None)  # the stored bytes keep it
        assert detail["digest"] == rule_document.identity_digest(doc)
        answer = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate",
                               json={"text": "marker ZQ-1234 alone"})
        assert answer.status_code == 200, answer.text
        data = answer.json()["data"]
        assert data["match"] == expected
        # serving still reports EVERY expression on its own (D9): `match` narrows nothing here
        assert {r["signal"] for r in data["results"]} == set(doc["expressions"])


def test_the_identity_moves_with_match():
    doc = rf.document("k-a")
    assert rule_document.identity_digest(_with_match(["cued"], key="k-a")) != rule_document.identity_digest(doc)
