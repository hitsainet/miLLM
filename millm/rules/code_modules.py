"""Operator-code documents arriving beside a rule set (039 S3; FTDD §4.4; TD39-34; OQ-39.7 (a)).

A rule set's `code_rule` signal names its code by content hash only (`code_sha256`, the runner
contract's hash rule). The import request may carry the documents themselves:

    {"rule_set": <the rule-set document>, "code_modules": {"<code_sha256>": "<canonical UTF-8>"}}

Each document is checked HERE, in the API process, by data checks only — its bytes hash to the key,
they are the canonical bytes, it names this miLLM's runner contract, it declares the function the
signal calls, and the signal's parameters satisfy its `params_schema`. Nothing in it is run, compiled
or imported here (FR-39.26: the API process never evaluates operator code). A document no signal
names is refused (nothing is stored that nothing uses).

A code rule whose document did not arrive is still ACCEPTED (D44): it stays pending
(`rule_worker_unavailable`) and the rule set's detail lists it under `code_modules_missing`.
"""

from __future__ import annotations

from typing import Any, Optional

from jsonschema import Draft202012Validator

from millm.rules import runner_contract as rc
from millm.rules.document import RuleSetInvalid

ENVELOPE_KEYS = frozenset({"rule_set", "code_modules"})


def split_envelope(parsed: Any) -> tuple[Any, Optional[dict[str, Any]]]:
    """`(document, modules)` for an envelope, `(parsed, None)` for a bare rule-set document."""
    if isinstance(parsed, dict) and set(parsed) == ENVELOPE_KEYS:
        modules = parsed["code_modules"]
        if not isinstance(modules, dict):
            raise RuleSetInvalid("code_module_invalid", "code_modules must be an object",
                                 {"pointer": "/code_modules"})
        return parsed["rule_set"], modules
    return parsed, None


def code_signals(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in document.get("signals") or [] if s.get("kind") == "code_rule"]


def verify(document: dict[str, Any], modules: Optional[dict[str, Any]], *,
           max_bytes: int) -> dict[str, str]:
    """The verified `{code_sha256: canonical text}` to store. Raises `RuleSetInvalid` naming the
    hash and the signal — never a byte of the source."""
    if not modules:
        return {}
    total = sum(len(v.encode("utf-8", errors="replace")) for v in modules.values()
                if isinstance(v, str))
    if total > max_bytes:
        raise RuleSetInvalid("code_modules_too_large",
                             f"the code documents are {total} bytes; the limit is {max_bytes}",
                             {"bytes": total, "limit": max_bytes,
                              "setting": "RULE_CODE_MODULES_MAX_BYTES"})
    signals = code_signals(document)
    named = {s["code_sha256"] for s in signals}
    out: dict[str, str] = {}
    parsed: dict[str, rc.CodeDocument] = {}
    for key, text in sorted(modules.items()):
        if key not in named:
            raise RuleSetInvalid("code_module_unused",
                                 "a code document is not named by any code_rule signal",
                                 {"code_sha256": key if isinstance(key, str) else None})
        try:
            parsed[key] = rc.parse_code_document(key, text)
        except rc.CodeDocumentRefused as exc:
            raise RuleSetInvalid(exc.code, f"code document {key}: {exc}",
                                 {"code_sha256": key}) from None
        out[key] = text
    for signal in signals:
        doc = parsed.get(signal["code_sha256"])
        if doc is None:
            continue
        where = {"signal": signal["id"], "code_sha256": doc.code_sha256}
        if signal.get("runner_contract") != rc.contract_format():
            raise RuleSetInvalid("runner_contract_unknown",
                                 f"signal {signal['id']!r} names another runner contract than "
                                 + rc.contract_format(), where)
        if signal["function"] not in doc.functions:
            raise RuleSetInvalid("code_function_not_declared",
                                 f"signal {signal['id']!r}: the code document does not declare "
                                 "the function it calls", where)
        try:
            # Review R1 (L2): an operator's schema that is not a usable JSON Schema (a bad
            # `type`, an unresolvable `$ref` — never fetched) made the import a 500.
            Draft202012Validator.check_schema(doc.params_schema)
            errors = list(Draft202012Validator(doc.params_schema).iter_errors(signal["params"]))
        except Exception:  # noqa: BLE001 - any failure of the schema itself is a refusal
            errors = [None]
        if errors:
            raise RuleSetInvalid("code_params_invalid",
                                 f"signal {signal['id']!r}: parameters refused by the code "
                                 "document's params_schema", where)
    return out


def _within_n_nodes(expr: Any) -> list[dict[str, Any]]:
    """Every `within_n` node inside one expression, nested ones included."""
    from millm.rules.document import expression_form

    if not isinstance(expr, dict) or not expr:
        return []
    form = expression_form(expr)
    if form == "within_n":
        # Its operands' signals are its own (`signals_named`): an inner within_n adds nothing.
        return [expr]
    if form in ("all", "any"):
        return [n for e in expr[form] or [] for n in _within_n_nodes(e)]
    if form == "not":
        return _within_n_nodes(expr["not"])
    return []


def refuse_positional_reads(document: dict[str, Any]) -> None:
    """REFUSE every use of a code rule that needs its positions (suite decision D65, 2026-10-25;
    supersedes 039 S3 R1's import-time warning `code_rule_positions_never_resolve`; widened by
    D74, 2026-10-28, ENH-006 Q9).

    A code rule reports a count and never a span (runner contract; producer FR-018.38), so such a
    use could only stay pending (`rule_worker_unavailable`) for good — "a signal that silently
    never fires". As the producer's compiler does (miDataworks `rules/compose.py`
    `_check_code_positions`), in the same order and with the same codes and details:

    1. a `within_n` (nested anywhere in an expression) with a code-rule operand →
       `code_rule_in_within_n`, details `{signal, expression}` (the shared vector
       `refuse.document.code_rule_in_within_n`);
    2. per signal, in document order: a proximity `anchor`, then a validator `candidates.span_of`,
       naming a code rule → `code_rule_positions_needed`, details `{code_rule, use, signal}`;
    3. per veto, in document order (D74): a code rule named in the expression its `when` names →
       `use: "veto_when"`, then one in its `removes` → `use: "veto_removes"`; details
       `{code_rule, use, veto}` — a veto removes a signal's SPANS, and a code rule has none;
    4. per output, in document order (D74): a `field` output reading a code rule →
       `use: "field"`, details `{code_rule, use, column}`.

    A signal reading such a reader is refused through the reader itself. Count-based uses
    (`{signal: op}`, `all`/`any`/`not` over it, `fired`/`count` outputs) stay ACCEPTED and pending
    until run (D41/D44). Names only: signal ids, expression, veto and column names, never source or
    parameters.
    """
    from millm.rules.document import output_value_key, signals_named
    from millm.rules.evaluator import RuleSetNotRunnable

    signals = [s for s in document.get("signals") or [] if isinstance(s, dict)]
    code = {s.get("id") for s in signals if s.get("kind") == "code_rule"}
    if not code:
        return
    expressions = document.get("expressions") or {}
    for name, expr in expressions.items():
        for node in _within_n_nodes(expr):
            found = [s for s in signals_named(node) if s in code]
            if found:
                raise RuleSetNotRunnable(
                    "code_rule_in_within_n",
                    f"expression {name!r}: within_n needs positions, and code rule "
                    f"{found[0]!r} reports a count, never a span",
                    {"signal": found[0], "expression": name})

    def needs(where: str, code_rule: str, use: str, **details: Any) -> RuleSetNotRunnable:
        return RuleSetNotRunnable(
            "code_rule_positions_needed",
            f"{where}: {use} of code rule {code_rule!r} needs its positions; a code rule "
            "reports a count, never a span",
            {"code_rule": code_rule, "use": use, **details})

    for s in signals:
        candidates = s.get("candidates") if isinstance(s.get("candidates"), dict) else {}
        for use, target in (("anchor", s.get("anchor")), ("span_of", candidates.get("span_of"))):
            if isinstance(target, str) and target in code:
                raise needs(f"signal {s.get('id')!r}", target, use, signal=s.get("id"))
    for veto in document.get("vetoes") or []:
        if not isinstance(veto, dict):
            continue
        when = expressions.get(veto.get("when"))
        for named in signals_named(when) if isinstance(when, dict) else []:
            if named in code:
                raise needs(f"veto {veto.get('id')!r}", named, "veto_when", veto=veto.get("id"))
        for named in veto.get("removes") or []:
            if named in code:
                raise needs(f"veto {veto.get('id')!r}", named, "veto_removes", veto=veto.get("id"))
    for out in document.get("outputs") or []:
        value = out.get("value") if isinstance(out, dict) else None
        if not isinstance(value, dict) or not value or output_value_key(value) != "field":
            continue
        field = value.get("field")
        if isinstance(field, (list, tuple)) and field and field[0] in code:
            raise needs(f"output {out.get('column')!r}", field[0], "field",
                        column=out.get("column"))


def missing(document: dict[str, Any], stored: Optional[dict[str, str]]) -> list[str]:
    """Hashes a code_rule signal names whose document miLLM does not hold (sorted)."""
    held = set(stored or {})
    return sorted({s["code_sha256"] for s in code_signals(document)} - held)


def names_operator_code(body: Any) -> bool:
    """True when an import body names operator code anywhere (ENH-008: the approval gate on
    `POST /api/rule-sets/import`). Port of miStudio's `names_operator_code`. FAILS CLOSED: any
    `code_modules` key, or any object anywhere whose `kind` is `code_rule` — not only the top-level
    `signals` the importer reads — so a shape this check did not foresee is gated, never run
    unapproved. A declarative rule set never asks."""
    if isinstance(body, dict) and "code_modules" in body:
        return True
    stack: list[Any] = [body]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if value.get("kind") == "code_rule":
                return True
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return False
