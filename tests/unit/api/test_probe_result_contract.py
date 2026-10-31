"""`mi.probe.result/v1`: generated, pinned, closed to free text (FR-38.2, D38.22).

Both schemas are GENERATED from `millm/api/schemas/probe_result.py`. The published, open one
(`docs/schemas/probe-result-v1.json`) is byte-pinned by miStudio; the strict one
(`tests/fixtures/contracts/probe-result-v1.strict.json`) is what every object miLLM's tests emit is
validated against. Regenerate both with `MILLM_WRITE_CONTRACTS=1 pytest <this file>`.

Control M2: add a field to the model without regenerating — `test_generated_equals_committed` red.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterator

import jsonschema
import pytest

from millm.api.schemas.probe_result import open_schema, strict_schema
from millm.api.schemas.weights_contract import render

ROOT = Path(__file__).resolve().parents[3]
OPEN = ROOT / "docs" / "schemas" / "probe-result-v1.json"
STRICT = ROOT / "tests" / "fixtures" / "contracts" / "probe-result-v1.strict.json"

#: String properties allowed without an enum, pattern or format — each with its reason. Empty:
#: every identifier the contract carries has a pattern.
FREE_TEXT_ALLOWLIST: dict[str, str] = {}


@pytest.mark.parametrize("path,build", [(OPEN, open_schema), (STRICT, strict_schema)],
                         ids=["open", "strict"])
def test_generated_equals_committed(path, build):
    generated = render(build())
    if os.environ.get("MILLM_WRITE_CONTRACTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated, encoding="utf-8")
    assert path.is_file(), f"{path} is missing; run with MILLM_WRITE_CONTRACTS=1"
    assert path.read_text(encoding="utf-8") == generated, (
        f"{path.name} no longer matches the model. Regenerate (MILLM_WRITE_CONTRACTS=1); the open "
        f"schema is byte-pinned by miStudio, so its copy must move in the same release."
    )


def _string_properties(node: Any, where: str = "") -> Iterator[tuple[str, dict]]:
    if isinstance(node, dict):
        if node.get("type") == "string":
            yield where, node
        for key, value in node.items():
            if key in ("if", "then", "allOf"):
                continue  # the conditional rules restate types; their values are checked above
            yield from _string_properties(value, f"{where}.{key}" if where else key)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _string_properties(item, f"{where}[{i}]")


def test_no_free_text_in_the_strict_schema():
    """⚠ No property can carry a prompt, a reply, an exception message or a credential."""
    bare = [
        where for where, prop in _string_properties(strict_schema())
        if not ({"enum", "pattern", "format", "const"} & set(prop))
        and where not in FREE_TEXT_ALLOWLIST
    ]
    assert not bare, f"free-text string properties in the strict schema: {bare}"


def test_the_walk_sees_strings_at_all():
    """A walk that finds nothing would pass the rule above by checking nothing."""
    assert len(list(_string_properties(strict_schema()))) >= 30


def _probe_example() -> dict:
    from types import SimpleNamespace

    from millm.api.schemas.probe_result import wire
    from millm.services.probe_result import build_probe_results
    from millm.services.probe_runtime import Verdict

    verdict = Verdict(probe_id="pr_0123456789ab", name="probe one", rung=2,
                      rung_language="detects on unseen tasks", scored=True, score=2.31,
                      threshold=1.07, fires=True, n_scored_tokens=12, served_model="tiny")
    context = SimpleNamespace(request_id="chatcmpl-abc", probes=(), session=None, trace=None)
    return wire(build_probe_results([verdict], context)[0])


def _rule_example() -> dict:
    probe = _probe_example()
    keep = ("schema", "result_id", "event_time", "event_time_source", "origin", "producer",
            "migrated", "request_id", "trace_id", "span_id", "batch_id", "batch_line",
            "session_id", "turn_index", "outcome", "reason", "window", "status")
    rule = {k: probe[k] for k in keep}
    rule.update({
        "kind": "rule", "behaviour": {"key": "b-placeholder", "version": 1},
        "set": {"armed_members": 2, "total_members": 3, "complete": False,
                "missing_probe_ids": ["pr_0123456789ac"], "members_digest": "a" * 64,
                "attached_rule_sets": [{"rule_set_id": "rs_1", "rule_set_digest": "b" * 64}]},
        "rule_set_id": "rs_1", "rule_set_digest": "b" * 64, "rule_set_version": 1,
        "signal": "sig_1", "form": "count", "fires": True, "count": 2,
    })
    return rule


@pytest.mark.parametrize("schema", [open_schema, strict_schema], ids=["open", "strict"])
def test_both_kinds_validate(schema):
    validator = jsonschema.Draft202012Validator(schema())
    validator.validate(_probe_example())
    validator.validate(_rule_example())


def test_a_value_added_later_in_v1_is_open_and_strict_refuses_it():
    example = {**_probe_example(), "engine": "vllm", "extra_field": 1}
    jsonschema.Draft202012Validator(open_schema()).validate(example)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(strict_schema()).validate(example)


#: One violation per conditional rule (FTDD §5.1), each applied to a valid completed result.
VIOLATIONS = {
    "completed_with_a_reason": {"reason": "no_scored_tokens"},
    "skipped_with_a_score": {"outcome": "skipped", "scored": False, "reason": "judge_traffic",
                             "verdict": None, "fires": None},
    "interrupted_with_a_verdict": {"outcome": "interrupted", "status": "partial",
                                   "scored": False, "reason": "client_disconnected"},
    "partial_without_a_score": {"outcome": "interrupted", "status": "partial", "scored": False,
                                "reason": "cancelled", "score": None, "verdict": None,
                                "fires": None},
    "unavailable_with_a_score": {"outcome": "interrupted", "status": "unavailable",
                                 "scored": False, "reason": "cancelled", "verdict": None,
                                 "fires": None},
    "no_threshold_but_a_verdict": {"threshold": None},
    "an_unlisted_reason": {"outcome": "skipped", "scored": False, "score": None,
                           "verdict": None, "fires": None, "reason": "made_up_code"},
}


@pytest.mark.parametrize("name", sorted(VIOLATIONS))
def test_each_conditional_rule_is_refused_once(name):
    validator = jsonschema.Draft202012Validator(strict_schema())
    valid = _probe_example()
    validator.validate(valid)
    broken = copy.deepcopy(valid)
    broken.update(VIOLATIONS[name])
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(broken)


def test_the_finalized_object_has_no_recorded_only_fields():
    example = _probe_example()
    assert "persistence" not in example and "source_event_id" not in example
