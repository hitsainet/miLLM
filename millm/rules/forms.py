"""The forms a rule set may use, derived from the vendored schema (039 FR-39.18; FTID §3.5).

The signal kinds are READ from `docs/schemas/midataworks-rule-set-v1.json` (`document.signal_kinds`),
never restated: a kind the producer adds and this evaluator lacks is refused at import
`RULE_FORM_UNAVAILABLE`, never ignored, and `tests/unit/rules/test_forms.py` fails until it is
implemented or deferred here.

- `IMPLEMENTED`: run inline by `evaluator.py`.
- `DEFERRED`: accepted and scored off the live path. `code_rule` is here per D41 (2026-10-24):
  operator code is never evaluated on miLLM's live path; its expressions report
  `rule_worker_unavailable` until the runner (slice S3) scores them after the response.
- Composition (`all`, `any`, `not`, `within_n`) is over expressions and signals of the SAME rule
  set: the schema has no way to name another set's signal, and a nested `rule_set` reference is
  refused by the schema (vector `refuse.document.nested_rule_set`).
"""

from __future__ import annotations

from typing import Any

from millm.rules.document import COMPOSITION_FORMS, RuleSetInvalid, signal_kinds

IMPLEMENTED: frozenset[str] = frozenset({"pattern", "proximity", "validator", "lexical_weights"})
DEFERRED: frozenset[str] = frozenset({"code_rule"})


def schema_forms() -> tuple[str, ...]:
    """Every form the vendored schema declares: its signal kinds and the compositions."""
    return tuple(sorted(set(signal_kinds()) | set(COMPOSITION_FORMS)))


def known(kind: str) -> bool:
    return kind in IMPLEMENTED or kind in DEFERRED


def check_forms(document: dict[str, Any]) -> None:
    """Refuse a signal kind this evaluator neither implements nor defers."""
    for index, signal in enumerate(document.get("signals") or []):
        kind = signal.get("kind")
        if not known(kind):
            raise RuleSetInvalid("form_unavailable", f"signal {signal.get('id')!r}: the form "
                                 f"{kind!r} is not available in this miLLM",
                                 {"signal": signal.get("id"), "form": kind, "index": index})


def listing() -> list[dict[str, Any]]:
    """The forms, as `GET /api/rule-sets/validators` reports them beside the validators."""
    return [
        {"form": f, "evaluated": "inline" if f in IMPLEMENTED or f in COMPOSITION_FORMS
         else ("after_response" if f in DEFERRED else "unavailable")}
        for f in schema_forms()
    ]
