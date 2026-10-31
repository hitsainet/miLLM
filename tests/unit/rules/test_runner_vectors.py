"""Every vector of the vendored `runner-cases-v1.json` through miLLM's host (039 FTASKS 18.3).

miLLM's host is the API-side client (`runner_client.RunnerSession`) talking to the REAL worker
(`python -m millm.rules.runner_worker`, a subprocess with no database and no secret) that runs the
PUBLISHED guest snapshot. Conformance, every hostile vector the file HAS, the hash vectors (re-hashed
here by miLLM's own hash rule), the import vectors and the limit vectors all run.

⚠ PENDING, NOT COVERED (ENH-006 D48). The escape-form hostile vectors — guest code reaching files,
the environment, sockets or the clock past the prelude (classes 1, 2, 3, 4, 9) and the guest forms
of 17, 18, 19 — are deliberately NOT in the cases file: the operator writes them. They are named in
:data:`PENDING_HOSTILE_CLASSES`, which mirrors the producer's `coverage.PENDING_HOSTILE_CLASSES`
(pinned by `test_the_pending_set_is_the_producers`), so nothing here reports them as covered. The
host defences they will exercise are pinned at host level by `test_runner_wasi_table.py`.

Skips loudly without the snapshot or `wasmtime`; fails instead under `MILLM_REQUIRE_RUNNER=1`.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
from typing import Any

import pytest

from millm.rules import runner_contract as rc
from tests.support import rule_runner as rr
from tests.support.sibling_repos import sibling_repo

CASES: dict[str, Any] = json.loads(rc.CASES_PATH.read_text(encoding="utf-8"))
DOCUMENTS = CASES["documents"]
#: Mirrors miDataworks' `runner_contract.coverage.PENDING_HOSTILE_CLASSES` (D48).
PENDING_HOSTILE_CLASSES = frozenset({1, 2, 3, 4, 9, 17, 18, 19})
#: Reasons a guest-source vector cannot produce (the producer's `REASONS_WITHOUT_VECTOR`).
REASONS_WITHOUT_VECTOR = frozenset({"rule_worker_unavailable", "rule_not_approved"})


@pytest.fixture(scope="module")
def runner():
    bundle = rr.bundle_dir()
    with rr.RunnerProcess(bundle) as process:
        yield process


def _call(runner, doc_id: str, function: str, params: dict, text: str,
          limits: dict[str, int] | None = None) -> Any:
    from millm.rules.runner_client import RunnerSession

    document = DOCUMENTS[doc_id]

    async def go():
        async with RunnerSession(str(runner.socket), limits or rr.floor_limits()) as session:
            return await session.call(rc.code_sha256(document),
                                      rc.canonical_json(document).decode("utf-8"),
                                      function, params, text)

    return asyncio.run(go())


def _match(expect: dict, got: Any) -> bool:
    if "count" in expect:
        return got.count == expect["count"] and got.reason is None
    return got.reason == expect["reason"] and got.count is None


SINGLE = [c for c in CASES["conformance"] if "calls" not in c]


@pytest.mark.parametrize("case", SINGLE, ids=[c["id"] for c in SINGLE])
def test_conformance(runner, case):
    got = _call(runner, case["doc"], case["function"], case["params"], case["text"])
    assert _match(case["expect"], got), (case["id"], got)


def test_each_call_is_a_fresh_instance(runner):
    case = next(c for c in CASES["conformance"] if c["id"] == "conf.fresh_instance")
    counts = [_call(runner, case["doc"], case["function"], case["params"], case["text"]).count
              for _ in range(case["calls"])]
    assert counts == case["expect"]["counts"]


@pytest.mark.parametrize("case", CASES["hostile"], ids=[c["id"] for c in CASES["hostile"]])
def test_hostile_vectors_the_file_has(runner, case):
    got = _call(runner, case["doc"], case["function"], case["params"], case["text"])
    assert _match(case["expect"], got), (case["id"], got)


@pytest.mark.parametrize("case", CASES["imports"], ids=[c["id"] for c in CASES["imports"]])
def test_imports(runner, case):
    assert _match(case["expect"], _call(runner, case["doc"], "f", {}, "a a a"))


@pytest.mark.parametrize("case", CASES["limits"], ids=[c["id"] for c in CASES["limits"]])
def test_limits(runner, case):
    limits, text = rr.floor_limits(), case["text"]
    if case["id"] == "limit.wall_ms.timeout":
        limits = rr.floor_limits(wall_ms=50)
    if case["id"] == "limit.input_code_points.exceed":
        text = "x" * (rc.floor()["input_code_points"] + 1)
    got = _call(runner, case["doc"], case["function"], case["params"], text, limits)
    assert _match(case["expect"], got), (case["id"], got)


def test_the_worker_itself_refuses_an_input_over_the_limit(runner):
    """The client never sends such a text; the worker refuses it anyway (two walls)."""
    import socket

    from millm.rules import runner_protocol as proto

    document = DOCUMENTS["count_ascii_digits"]
    sha = rc.code_sha256(document)
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(runner.socket))
        proto.write_frame(s, {"op": "load", "code_sha256": sha,
                              "canonical_utf8": rc.canonical_json(document).decode()})
        assert proto.read_frame(s) == {"ok": True}
        proto.write_frame(s, {"op": "call", "code_sha256": sha, "function": "f", "params": {},
                              "text": "1" * (rc.floor()["input_code_points"] + 1),
                              "limits": rr.floor_limits()})
        assert proto.read_frame(s) == {"reason": "rule_input_too_large"}


@pytest.mark.parametrize("case", CASES["hash"], ids=[c["id"] for c in CASES["hash"]])
def test_hash(case):
    document = DOCUMENTS[case["document"]]
    assert rc.canonical_json(document).decode("utf-8") == case["canonical_utf8"]
    assert rc.code_sha256(document) == case["sha256"]
    parsed = rc.parse_code_document(case["sha256"], case["canonical_utf8"])
    assert parsed.code_sha256 == case["sha256"]


# ── coverage: derived from the CONTRACT, the pending set named ──────────────────────────────


def _hostile_present() -> set[int]:
    return {c["number"] for c in CASES["hostile"]}


def test_every_hostile_class_is_covered_or_named_pending():
    classes = {h["number"] for h in rc.contract()["hostile_classes"]}
    assert classes == set(range(1, 20))
    assert _hostile_present() == classes - PENDING_HOSTILE_CLASSES, (
        "a class moved: the operator's vectors arrived (empty the pending set for them) or one "
        "vanished")


def test_no_pending_class_is_reported_as_covered():
    """⚠ D48: nothing here may claim the escape classes. If the file gains one, this goes red
    until the pending set loses it — the producer's coverage rule, mirrored."""
    assert not (_hostile_present() & PENDING_HOSTILE_CLASSES)


def test_every_reason_module_and_limit_has_a_vector():
    contract = rc.contract()
    reasons = set()
    for section in ("conformance", "hostile", "limits"):
        reasons |= {c["expect"]["reason"] for c in CASES[section] if "reason" in c["expect"]}
    assert set(contract["reasons"]) - REASONS_WITHOUT_VECTOR <= reasons
    assert set(contract["stdlib_allowed"]) == {c["module"] for c in CASES["imports"]}
    assert set(contract["limits_floor"]) <= {c["limit"] for c in CASES["limits"]}


def _producer_pending() -> frozenset[int] | None:
    path = (sibling_repo("miDataworks", "MIDATAWORKS_REPO") / "backend" / "src"
            / "runner_contract" / "coverage.py")
    if not path.exists():
        return None
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == \
                "PENDING_HOSTILE_CLASSES":
            call = node.value
            assert isinstance(call, ast.Call)
            return frozenset(ast.literal_eval(call.args[0]))
    raise AssertionError("PENDING_HOSTILE_CLASSES not found in the producer's coverage.py")


def test_the_pending_set_is_the_producers():
    pending = _producer_pending()
    if pending is None:
        message = "miDataworks is not checked out — the pending-set mirror did NOT run"
        if os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(message + " and MILLM_REQUIRE_CROSS_REPO_CHECKS=1")
        pytest.skip(message)
    assert pending == PENDING_HOSTILE_CLASSES


@pytest.mark.parametrize("name", ["call_fuel", "memory_bytes", "output_bytes", "input_code_points"])
def test_the_worker_refuses_a_limit_below_the_floor(runner, name):
    """A host may raise a floor, never lower it: the worker itself refuses (D47; FTASKS 17.4)."""
    import socket

    from millm.rules import runner_protocol as proto

    document = DOCUMENTS["return_zero"]
    sha = rc.code_sha256(document)
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(runner.socket))
        proto.write_frame(s, {"op": "load", "code_sha256": sha,
                              "canonical_utf8": rc.canonical_json(document).decode()})
        assert proto.read_frame(s) == {"ok": True}
        limits = rr.floor_limits(**{name: rc.floor()[name] - 1})
        proto.write_frame(s, {"op": "call", "code_sha256": sha, "function": "f", "params": {},
                              "text": "x", "limits": limits})
        assert proto.read_frame(s) == {"error": "limit_below_floor"}
        proto.write_frame(s, {"op": "call", "code_sha256": sha, "function": "f", "params": {},
                              "text": "x", "limits": rr.floor_limits()})
        assert proto.read_frame(s) == {"count": 0}
