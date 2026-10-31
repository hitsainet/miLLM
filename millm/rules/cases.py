"""Each rule set proves itself: its own cases, run at import and at every startup (039 FR-39.20).

A case states every output column, whether a veto fired and (optionally) an expression's spans; the
row semantics are the producer's (`evaluator.CompiledRuleSet.evaluate`). The first mismatch is
reported by case INDEX and id and by the column or expression that disagreed — never by the text.

Operator code (D41): a column that depends on a pending (`code_rule`) signal — directly, through an
expression, or through a veto whose condition is pending — cannot be checked here and is listed as
`unverified`, never compared and never counted as a pass of that column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from millm.rules.document import output_value_key, signals_named
from millm.rules.evaluator import CompiledRuleSet, RuleSignalError


@dataclass(frozen=True)
class CaseFailure:
    index: int
    case_id: str
    #: The output column, `vetoed`, or `spans:<expression>` that disagreed; or `error`.
    where: str
    #: The signal (expression or output) the failure is attributed to, for `vector_report`.
    signal: Optional[str]

    def report(self) -> dict[str, Any]:
        return {"case_index": self.index, "case_id": self.case_id, "where": self.where,
                "signal": self.signal}


@dataclass
class CaseReport:
    passed: int = 0
    failures: list[CaseFailure] = field(default_factory=list)
    #: `{case_id: [column, ...]}` not checked because they depend on operator code (D41).
    unverified: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def same(a: Any, b: Any) -> bool:
    """JSON equality WITH TYPES, as the producer compares (miDataworks `rules/cases.py::same`):
    `True` is not `1`, `1` is not `1.0`. Python's `==` would pass a `fired` column emitted as
    `1`, or a `count` emitted as `True` (039 review R1, control R1-01)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return bool(a == b)


def case_input(case: dict[str, Any]) -> tuple[str, Optional[str]]:
    """`(text, element)`: a value input is `before + value + after` with the value as element."""
    data = case["input"]
    if "text" in data:
        return data["text"], None
    value = data["value"]
    return data.get("before", "") + value + data.get("after", ""), value


def _case_pending(compiled: CompiledRuleSet) -> tuple[frozenset[str], frozenset[str], bool]:
    """(signals, expressions, vetoed-flag pending) for case checking."""
    pending = set(compiled.pending_signals)
    vetoes = compiled.document.get("vetoes") or []
    veto_pending = False
    for veto in vetoes:
        if veto["when"] in compiled.pending_expressions:
            veto_pending = True
            pending.update(veto["removes"])
    expressions = frozenset(name for name, expr in compiled.expressions.items()
                            if any(s in pending for s in signals_named(expr)))
    return frozenset(pending), expressions, veto_pending


def _column_signal(out: dict[str, Any]) -> str:
    value = out["value"]
    key = output_value_key(value)
    return value["field"][0] if key == "field" else value[key]


def run_cases(compiled: CompiledRuleSet, cases: Optional[list[dict[str, Any]]] = None,
              *, stop_at_first: bool = True) -> CaseReport:
    """Run `cases` (default: the document's own). Never raises for a disagreement."""
    cases = compiled.document["cases"] if cases is None else cases
    pending_signals, pending_exprs, veto_pending = _case_pending(compiled)
    outputs = compiled.document["outputs"]
    skip_columns = {
        out["column"] for out in outputs
        if (output_value_key(out["value"]) == "field" and out["value"]["field"][0]
            in pending_signals)
        or (output_value_key(out["value"]) != "field" and _column_signal(out) in pending_exprs)
    }
    report = CaseReport()
    for index, case in enumerate(cases):
        text, element = case_input(case)
        expect = case["expect"]
        try:
            row = compiled.evaluate(text, element)
        except RuleSignalError as exc:
            report.failures.append(CaseFailure(index, case["id"], "error", exc.signal_id))
            if stop_at_first:
                return report
            continue
        failure: Optional[CaseFailure] = None
        for out in outputs:
            column = out["column"]
            if column in skip_columns:
                continue
            if not same(row.outputs.get(column), expect["outputs"].get(column)):
                failure = CaseFailure(index, case["id"], column, _column_signal(out))
                break
        if failure is None and not veto_pending and not same(row.vetoed, expect["vetoed"]):
            failure = CaseFailure(index, case["id"], "vetoed", None)
        if failure is None:
            for name, spans in (expect.get("spans") or {}).items():
                if name in pending_exprs:
                    continue
                if not same(row.expressions[name]["spans"], [list(s) for s in spans]):
                    failure = CaseFailure(index, case["id"], f"spans:{name}", name)
                    break
        skipped = sorted(skip_columns) + (["vetoed"] if veto_pending else [])
        if skipped:
            report.unverified[case["id"]] = skipped
        if failure is not None:
            report.failures.append(failure)
            if stop_at_first:
                return report
        else:
            report.passed += 1
    return report
