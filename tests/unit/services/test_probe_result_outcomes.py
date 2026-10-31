"""A skip is never a zero, a missing threshold is never a "no", reasons are codes (FR-38.5).

* every reason a writer can emit is in the published list (`probe_reasons.REASONS`) — collected by
  AST over `millm/`, with a floor so a parser that finds nothing cannot pass;
* the builder's outcome mapping (AT03's fixtures, non-finite scores, every code);
* the free-text reasons are gone: the detail goes to the log, the reason is the code.

Controls M5: a skip serialized as score 0 or verdict false; a reason outside the list; the
finiteness check dropped — each turns a test here red.
"""

from __future__ import annotations

import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import torch

from millm.api.schemas.probe_result import strict_schema, wire
from millm.services.probe_reasons import OUTCOME_OF, REASON_TABLE, REASONS
from millm.services.probe_result import build_probe_results
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext, Verdict

MILLM = Path(__file__).resolve().parents[3] / "millm"
STRICT = jsonschema.Draft202012Validator(strict_schema())


# ── the code list, by AST ─────────────────────────────────────────────────────────


def _module_constants(tree: ast.Module) -> dict[str, str]:
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _reason_positions(tree: ast.Module):
    """Every expression in a reason position (FTID §7.1)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "mark_not_scored" and node.args:
                yield node.args[0]
            elif name == "_probe_begin_detached" and len(node.args) >= 2:
                yield node.args[1]
            elif name == "set_last_user_span" and len(node.args) >= 2:
                yield node.args[1]
            elif name == "Verdict":
                for kw in node.keywords:
                    if kw.arg == "not_scored_reason":
                        yield kw.value
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == "last_user_token_span":
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple) \
                        and len(ret.value.elts) == 2:
                    yield ret.value.elts[1]


def collect_reasons() -> tuple[set[str], list[str]]:
    literals: set[str] = set()
    fstrings: list[str] = []
    for path in sorted(MILLM.rglob("*.py")):
        tree = ast.parse(path.read_text())
        constants = _module_constants(tree)
        for expr in _reason_positions(tree):
            if any(isinstance(n, ast.JoinedStr) for n in ast.walk(expr)):
                fstrings.append(f"{path.relative_to(MILLM)}:{expr.lineno}")
                continue
            for n in ast.walk(expr):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    literals.add(n.value)
                elif isinstance(n, ast.Name) and n.id in constants:
                    literals.add(constants[n.id])
    return literals, fstrings


def test_every_emitted_reason_is_a_published_code():
    literals, fstrings = collect_reasons()
    assert not fstrings, f"a reason built from text (FR-38.5): {fstrings}"
    assert len(literals) >= 15, f"found only {len(literals)} reason literals — parser broken?"
    unlisted = sorted(literals - REASONS)
    assert not unlisted, f"reasons written but not in probe_reasons.REASONS: {unlisted}"


def test_every_code_has_an_outcome_and_a_meaning():
    for code, (outcome, meaning) in REASON_TABLE.items():
        assert outcome in ("skipped", "failed", "interrupted"), code
        assert meaning.strip(), code


# ── the builder's mapping (AT03) ──────────────────────────────────────────────────


def _context():
    return SimpleNamespace(request_id="chatcmpl-x", probes=(), session=None, trace=None)


def _verdict(**over) -> Verdict:
    base = dict(probe_id="pr_0123456789ab", name="probe", rung=2,
                rung_language="detects on unseen tasks", scored=True, score=2.31,
                threshold=1.07, fires=True, n_scored_tokens=5)
    base.update(over)
    return Verdict(**base)


def _one(**over) -> dict:
    data = wire(build_probe_results([_verdict(**over)], _context())[0])
    STRICT.validate(data)
    return data


@pytest.mark.parametrize("score,threshold,fires,expected", [
    (2.31, 1.07, True, True),        # above the bar
    (1.07, 1.07, True, True),        # EXACTLY on the bar fires (>=, P-03)
    (0.5, 1.07, False, False),       # below
    (2.31, None, None, None),        # no threshold: ranks, does not decide
])
def test_at03_verdicts(score, threshold, fires, expected):
    data = _one(score=score, threshold=threshold, fires=fires)
    assert data["outcome"] == "completed" and data["scored"] is True
    assert data["verdict"] is expected and data["fires"] is expected
    assert data["score"] == score and data["reason"] is None


def test_a_skip_has_no_score_and_no_verdict():
    data = _one(scored=False, score=None, threshold=1.07, fires=None,
                not_scored_reason="judge_traffic")
    assert data["outcome"] == "skipped" and data["reason"] == "judge_traffic"
    assert data["score"] is None, "a skip reported as a score is a measurement never taken"
    assert data["verdict"] is None, "a skip reported as `false` is a decision never made"
    assert data["scored"] is False


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_score_is_a_failure_with_no_score(value):
    data = _one(score=value, fires=False)
    assert data["outcome"] == "failed" and data["reason"] == "non_finite_score"
    assert data["score"] is None and data["verdict"] is None and data["scored"] is False


@pytest.mark.parametrize("code", sorted(c for c, o in OUTCOME_OF.items() if o != "interrupted"))
def test_every_code_maps_to_its_outcome(code):
    data = _one(scored=False, score=None, fires=None, not_scored_reason=code)
    assert data["outcome"] == OUTCOME_OF[code] and data["reason"] == code


def test_an_unlisted_reason_is_a_failure_never_a_skip():
    """A defect path: kept (it matches the pattern) but reported as `failed`."""
    data = wire(build_probe_results(
        [_verdict(scored=False, score=None, fires=None, not_scored_reason="not_a_code")],
        _context())[0])
    assert data["outcome"] == "failed" and data["reason"] == "not_a_code"
    with pytest.raises(jsonschema.ValidationError):
        STRICT.validate(data)


# ── the free-text reasons are gone (2.2) ──────────────────────────────────────────


def _probe(**over) -> ArmedProbe:
    from millm.ml.probe_head import ProbeHead

    base = dict(probe_id="pr_0123456789ab", name="probe", head=ProbeHead(weight=torch.ones(8)),
                rule="mean", scope="all", layer=0, rung=2,
                rung_language="detects on unseen tasks", threshold=0.0)
    base.update(over)
    return ArmedProbe(**base)


def test_a_long_exception_is_a_code_and_its_text_stays_in_the_log(caplog):
    text = "boom " * 40  # 200 characters
    assert len(text) == 200

    def broken(_x):
        raise RuntimeError(text)

    context = ProbeRequestContext("req", [_probe(encoder=broken)])
    with caplog.at_level(logging.ERROR):
        context.observe(0, torch.randn(1, 4, 8))
    verdict = context.finish()[0]
    assert verdict.not_scored_reason == "encoder_failed"
    assert any(text.strip() in r.getMessage() for r in caplog.records), "the detail must be logged"
    results = build_probe_results([verdict], context)
    data = wire(results[0])
    assert data["reason"] == "encoder_failed" and text.strip() not in str(data)

    from millm.api.routes.openai.chat import build_probe_verdicts_header

    header = build_probe_verdicts_header(results)
    assert 'reason="encoder_failed"' in header and "boom" not in header


@pytest.mark.parametrize("calls,code", [((-5,), "prompt_boundary_invalid"),
                                         ((6, 9), "prompt_boundary_conflict")])
def test_a_prompt_boundary_problem_is_a_code(calls, code, caplog):
    context = ProbeRequestContext("req", [_probe()])
    with caplog.at_level(logging.WARNING):
        for n in calls:
            context.set_prompt_length(n)
    assert context.not_scored_reason == code
    assert any(code in r.getMessage() for r in caplog.records)
