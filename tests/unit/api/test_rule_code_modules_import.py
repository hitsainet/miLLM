"""Operator-code documents travel beside their rule set (039 FTASKS 17.3; OQ-39.7 (a); TD39-34).

`POST /api/rule-sets/import` (and `/validate`) accept the S3 envelope
`{"rule_set": <document>, "code_modules": {"<code_sha256>": "<canonical UTF-8>"}}`. Each document is
verified by DATA checks only — hash, canonical bytes, runner contract, the function the signal calls,
the signal's parameters against `params_schema` — and stored in `rule_sets.code_modules` (migration
036). Nothing in it runs in the API process. A code rule whose document did not come is still
accepted (D44) and listed under `code_modules_missing`. Refusals name the hash and the signal, never
a byte of the source.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import func, select

from millm.db.models.rule_set import RuleSet
from millm.rules import runner_contract as rc
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.unit.api.test_rule_code_followup import code_document, rule_set_with_code

pytestmark = pytest.mark.asyncio

SECRET_SOURCE = "def f(text, params):\n    marker_qzvwxy = 1\n    return 0\n"


def _code(source: str = "def f(text, params):\n    return 1\n", **over) -> dict:
    doc = {"derived_from": {"sources": [], "statement": "constructed"},
           "functions": [{"name": "f", "output": {"type": "count"}}],
           "params_schema": {"type": "object"}, "runner_contract": "midataworks.runner/v1",
           "source": source}
    doc.update(over)
    return doc


def _post(h, path, rule_set, modules, **params):
    body = {"rule_set": rule_set, "code_modules": modules} if modules is not None else rule_set
    return h.client.post(path, content=json.dumps(body).encode(), params=params,
                         headers={"content-type": "application/json"})


async def _rows(h):
    async with h.factory() as session:
        return list((await session.execute(select(RuleSet))).scalars().all())


def _modules(code: dict) -> dict:
    return {rc.code_sha256(code): rc.canonical_json(code).decode("utf-8")}


async def test_an_envelope_stores_each_verified_document_beside_its_set():
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        doc = rule_set_with_code(key, code)
        r = _post(h, "/api/rule-sets/validate", doc, _modules(code))
        assert r.status_code == 200, r.text
        assert r.json()["data"]["code_modules"] == [rc.code_sha256(code)]
        assert await _rows(h) == [], "validate writes nothing"
        r = _post(h, "/api/rule-sets/import", doc, _modules(code))
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["code_modules"] == [rc.code_sha256(code)]
        assert data["code_modules_missing"] == []
        (row,) = await _rows(h)
        assert row.code_modules == _modules(code)
        detail = h.client.get(f"/api/rule-sets/{row.id}").json()["data"]
        assert detail["code_modules"] == [rc.code_sha256(code)]
        # Hashes only: the detail never carries the code's source.
        assert "code_modules_text" not in detail and "def f(" not in json.dumps(
            {k: v for k, v in detail.items() if k != "document"})


async def test_a_bare_document_naming_code_is_accepted_and_lists_what_is_missing():
    code = code_document()
    async with cap.Harness() as h:
        doc = rule_set_with_code(bf.random_key(), code)
        r = _post(h, "/api/rule-sets/import", doc, None)
        assert r.status_code == 200, r.text
        assert r.json()["data"]["code_modules_missing"] == [rc.code_sha256(code)]
        # The same set again WITH its document: stored beside it, the same id.
        again = _post(h, "/api/rule-sets/import", doc, _modules(code))
        assert again.status_code == 200 and again.json()["data"]["id"] == r.json()["data"]["id"]
        assert again.json()["data"]["created"] is False
        (row,) = await _rows(h)
        assert row.code_modules == _modules(code)


@pytest.mark.parametrize("break_it,rule", [
    ("hash", "code_hash_mismatch"),
    ("not_canonical", "code_document_invalid"),
    ("unused", "code_module_unused"),
    ("function", "code_function_not_declared"),
    ("params", "code_params_invalid"),
    ("contract", "runner_contract_unknown"),
])
async def test_a_document_that_is_not_what_it_claims_is_refused(break_it, rule):
    code = _code(SECRET_SOURCE)
    async with cap.Harness() as h:
        doc = rule_set_with_code(bf.random_key(), code)
        modules = _modules(code)
        sha = rc.code_sha256(code)
        if break_it == "hash":
            modules = {sha: rc.canonical_json(_code(SECRET_SOURCE + "#")).decode()}
        elif break_it == "not_canonical":
            # Bytes that DO hash to the name they are filed under, but are not the canonical
            # form: another host would hash the document differently.
            indented = json.dumps(code, indent=1)
            fake = rc.bytes_sha256(indented.encode("utf-8"))
            doc["signals"][-1]["code_sha256"] = fake
            modules = {fake: indented}
        elif break_it == "unused":
            other = _code("def f(text, params):\n    return 2\n")
            modules = {**modules, **_modules(other)}
        elif break_it == "function":
            doc["signals"][-1]["function"] = "g"
        elif break_it == "params":
            strict = _code(SECRET_SOURCE, params_schema={
                "type": "object", "required": ["min"], "properties": {"min": {"type": "integer"}}})
            doc = rule_set_with_code(bf.random_key(), strict)
            modules = _modules(strict)
        elif break_it == "contract":
            doc["signals"][-1]["runner_contract"] = "midataworks.runner/v2"
        r = _post(h, "/api/rule-sets/import", doc, modules)
        assert r.status_code == 422, r.text
        error = r.json()["error"]
        assert error["details"]["rule"] == rule, error
        assert "marker_qzvwxy" not in r.text, "a refusal must never echo the source"
        async with h.factory() as session:
            assert (await session.execute(
                select(func.count()).select_from(RuleSet))).scalar_one() == 0


async def test_the_code_documents_have_their_own_size_limit(monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "RULE_CODE_MODULES_MAX_BYTES", 1024)
    code = _code("def f(text, params):\n    return 1\n" + "#" * 2000 + "\n")
    async with cap.Harness() as h:
        r = _post(h, "/api/rule-sets/import", rule_set_with_code(bf.random_key(), code),
                  _modules(code))
        assert r.status_code == 422
        assert r.json()["error"]["details"]["rule"] in ("code_modules_too_large",
                                                        "rule_set_too_large")
