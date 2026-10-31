"""Rule sets as data and the one evaluator (039 slice S2; FTDD §6.6).

`prepare` is the ONE path from bytes (or a parsed document) to a compiled, self-proven rule set:
size → strict JSON → forms → schema and references → compile → own cases. Import, validate, the
startup self-check and the tests all call it, so they cannot disagree about what is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from millm.rules.cases import CaseReport, run_cases
from millm.rules.document import (
    RuleSetInvalid,
    check,
    identity_digest,
    origin_of,
    parse_bytes,
)
from millm.rules.evaluator import CompiledRuleSet, RuleSetNotRunnable
from millm.rules.forms import check_forms


@dataclass(frozen=True)
class Prepared:
    document: dict[str, Any]
    digest: str
    origin: str
    compiled: CompiledRuleSet
    cases: CaseReport
    #: 039 S3: the verified code-rule documents the import carried (`code_modules.verify`).
    code_modules: dict[str, str] = field(default_factory=dict)


class RuleVectorsFailed(ValueError):
    """The rule set disagrees with one of its own cases (`RULE_VECTORS_FAILED`)."""

    def __init__(self, report: CaseReport) -> None:
        first = report.failures[0]
        super().__init__(f"case {first.index} ({first.case_id}) disagrees at {first.where}")
        self.report = report
        self.details = first.report()


def prepare(source: Any, *, max_bytes: Optional[int] = None,
            declared_origin: Optional[str] = None, run_own_cases: bool = True) -> Prepared:
    """Check, compile and prove one rule set. Raises `RuleSetInvalid`, `RuleSetNotRunnable`,
    `dialect.PatternRefused` (wrapped as `RuleSetNotRunnable`) or `RuleVectorsFailed`."""
    if isinstance(source, (bytes, bytearray)):
        if max_bytes is None:
            from millm.core.config import settings

            max_bytes = int(settings.RULE_SET_MAX_BYTES)
        document = parse_bytes(bytes(source), max_bytes)
    else:
        document = source
    if isinstance(document, dict):
        check_forms(document)
    check(document)
    origin = origin_of(document)
    if declared_origin is not None and declared_origin != origin:
        raise RuleSetInvalid(
            "rule_set_invalid",
            "origin 'derived' needs provenance" if declared_origin == "derived"
            else "a document with provenance is derived",
            {"declared_origin": declared_origin, "origin": origin, "pointer": "/provenance"},
        )
    # D65: a code rule's positions are never reported, so what reads them is refused here, before
    # compiling — import, validate and the startup self-check alike.
    from millm.rules.code_modules import refuse_positional_reads

    refuse_positional_reads(document)
    digest = identity_digest(document)
    compiled = CompiledRuleSet(document)
    report = run_cases(compiled) if run_own_cases else CaseReport()
    if not report.ok:
        raise RuleVectorsFailed(report)
    return Prepared(document, digest, origin, compiled, report)


__all__ = ["Prepared", "RuleSetInvalid", "RuleSetNotRunnable", "RuleVectorsFailed", "prepare"]
