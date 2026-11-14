"""D53 review (2026-11-08), miLLM: the inferred positive reads only `fired` outputs, once each, in output order
(controls L2, L10), and every top-level field of the vendored schema is classified for serving, `match` among the
fields serving does not use (control L1)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from miserver.rules import document as rule_document
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf

ROOT = Path(__file__).resolve().parents[3]


def test_inferred_reads_only_fired_outputs_once_each_in_output_order():
    """The producer's `inferred_match`: every expression an output reads with `fired`, de-duplicated, in output
    order. The base fixture's only non-`fired` output reads an expression a `fired` output also reads, and no two
    `fired` outputs share an expression, so it cannot tell either rule apart."""
    doc = copy.deepcopy(rf.document(bf.random_key()))
    doc["outputs"] = [
        {"column": "lex_count", "type": "int", "value": {"count": "lexicon"}},   # count only: never the positive
        {"column": "number", "type": "bool", "value": {"fired": "number"}},
        {"column": "token", "type": "bool", "value": {"fired": "token"}},
        {"column": "number_again", "type": "bool", "value": {"fired": "number"}},  # named twice: once in any_of
    ]
    for case in doc["cases"]:
        old = case["expect"]["outputs"]
        case["expect"]["outputs"] = {"lex_count": 0, "number": old.get("number", False),
                                     "token": old.get("token", False), "number_again": old.get("number", False)}
    rule_document.check(copy.deepcopy(doc))
    assert rule_document.match_of(doc) == {"source": "inferred", "version": None, "any_of": ["number", "token"]}


def test_every_top_level_field_is_classified_for_serving():
    """Each top-level property of the vendored schema is either read at serving or listed as not used there, so a
    re-vendor adding a field (as D53 added `match`) fails here until someone classifies it."""
    used_at_serving = {"format", "name", "behaviour_key", "description", "signals", "expressions", "cases",
                       "provenance"}
    props = set(json.loads((ROOT / "docs" / "schemas" / "midataworks-rule-set-v1.json").read_text())["properties"])
    assert props - used_at_serving == set(rule_document.NOT_USED_AT_SERVING)


@pytest.mark.asyncio
async def test_detail_lists_match_as_not_used_at_serving():
    async with cap.Harness() as h:
        doc = rf.document(bf.random_key())
        doc["match"] = {"version": 1, "any_of": ["cued"]}
        r = h.client.post("/api/rule-sets/import", content=rf.raw(doc), headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        detail = h.client.get(f"/api/rule-sets/{r.json()['data']['id']}").json()["data"]
        assert detail["not_used_at_serving"]["match"] == {"version": 1, "any_of": ["cued"]}
