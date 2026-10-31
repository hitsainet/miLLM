"""The one evaluator of the declarative forms (039 FR-39.18, FR-39.24; TD39-20; FTID §3.5).

A port of miDataworks' `backend/src/rules/compose.py` (origin/main 2082e01) over the parsed
document (dicts, TD39-25), so the vendored vectors pin both apps alike. Two entry points:

- :meth:`CompiledRuleSet.evaluate` — the producer's ROW semantics (signals, vetoes, expressions,
  output columns). Only the rule set's own cases and the shared vectors use it (FR-39.20).
- :meth:`CompiledRuleSet.serve` — SERVING: every named expression on every window, in a fixed
  order, as `{signal, form, window, fires, count, outcome, reason}`. Vetoes are NEVER applied at
  serving (D9, OQ-39.2 (a)): a declared veto is dataset-building data. Output columns are not
  read either. `count` is the number of the expression's spans — never a value.

⚠ OPERATOR CODE IS NEVER RUN INLINE (D41, 2026-10-24). A `code_rule` signal is ACCEPTED: it is
"pending" — scored after the response by the runner (slice S3), never on the live path. Every
signal that reads it (`anchor`, `span_of`) and every expression that names one is pending too, and
reports `outcome: "skipped"`, `status: "unavailable"`, `reason: "rule_worker_unavailable"` (the
FTDD's code; discrepancy X-15 in the controls record). Nothing about it runs here.

The clock is injected (`clock`, monotonic seconds) and read between units of work; at the
deadline every unfinished expression reports `outcome: "failed"`, `reason: "rule_timeout"`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from jsonschema import Draft202012Validator

from millm.rules import dialect
from millm.rules.document import (
    expression_form,
    output_value_key,
    signal_reference,
    signals_named,
)
from millm.rules.validators import resolve

TEXT = "text"
ELEMENT = "element"
#: Serving windows in their fixed order (TD39-21).
WINDOW_ORDER = ("prompt", "last_user", "response", "all")
#: The reason a pending (operator-code) expression reports (D41; FTDD §5.5).
PENDING_REASON = "rule_worker_unavailable"
TIMEOUT_REASON = "rule_timeout"
ERROR_REASON = "rule_error"
VECTORS_FAILED_REASON = "rule_set_vectors_failed"


class RuleSetNotRunnable(ValueError):
    """The document cannot be evaluated here: `validator_unknown`, `validator_params_invalid`,
    `pattern_refused` or `field_unknown` (the producer's codes)."""

    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class RuleSignalError(RuntimeError):
    """A validator raised. Carries the signal and validator, NEVER the text."""

    def __init__(self, signal_id: str, validator: str, error: BaseException) -> None:
        super().__init__(f"signal {signal_id!r} ({validator}) raised {type(error).__name__}")
        self.signal_id = signal_id
        self.validator = validator
        self.error_type = type(error).__name__


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    source: str = TEXT
    outputs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.source, self.start, self.end)


@dataclass
class SignalResult:
    spans: list[Span]
    candidates: list[Span]


@dataclass
class RowResult:
    vetoed: bool
    outputs: dict[str, Any]
    expressions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Outcome:
    """One expression on one window at serving. `spans` only when positions were asked for."""

    signal: str
    form: str
    window: str
    fires: Optional[bool]
    count: int
    outcome: str
    reason: Optional[str] = None
    status: str = "final"
    spans: Optional[tuple[tuple[int, int], ...]] = None


def overlaps(a: Span, b: Span) -> bool:
    return a.source == b.source and a.start < b.end and b.start < a.end


def _sorted_unique(spans: list[Span]) -> list[Span]:
    seen: dict[tuple[str, int, int], Span] = {}
    for s in spans:
        seen.setdefault(s.key, s)
    return sorted(seen.values(), key=lambda s: (s.start, s.end, s.source))


def _units(text: str, lo: int, hi: int, unit: str) -> int:
    if unit == "utf8_bytes":
        return len(text[lo:hi].encode("utf-8"))
    return hi - lo


def _gap(a: Span, b: Span, unit: str, strings: Mapping[str, str]) -> Optional[int]:
    """Edge-to-edge gap of two spans of one string; overlap is 0; None across strings."""
    if a.source != b.source:
        return None
    if a.start < b.end and b.start < a.end:
        return 0
    lo, hi = (a.end, b.start) if a.end <= b.start else (b.end, a.start)
    return _units(strings[a.source], lo, hi, unit)


def _apply_vetoes(results: dict[str, SignalResult],
                  vetoes: list[tuple[str, list[Span], list[str]]]) -> dict[str, int]:
    removed: dict[str, int] = {}
    for veto_id, when_spans, removes in vetoes:
        count = 0
        for signal_id in removes:
            result = results[signal_id]
            keep = [s for s in result.spans if not any(overlaps(s, w) for w in when_spans)]
            keep_all = [s for s in result.candidates if not any(overlaps(s, w) for w in when_spans)]
            count += len(result.spans) - len(keep)
            result.spans, result.candidates = keep, keep_all
        removed[veto_id] = count
    return removed


def _spans_json(spans: list[Span]) -> str:
    import json

    return json.dumps([[s.start, s.end] for s in spans], separators=(",", ":"))


class CompiledRuleSet:
    """A checked document compiled once: patterns, validators and their params."""

    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        self.signals: list[dict[str, Any]] = list(document["signals"])
        self.expressions: dict[str, dict[str, Any]] = dict(document["expressions"])
        self._by_id = {s["id"]: s for s in self.signals}
        self._compiled: dict[str, Any] = {}
        self._phrases: dict[str, list[Any]] = {}
        self._lexical: dict[str, tuple[Any, list[tuple[str, float]]]] = {}
        self._validators: dict[str, Any] = {}
        for signal in self.signals:
            self._compile_signal(signal)
        self.pending_signals = self._pending_signals()
        self.pending_expressions = frozenset(
            name for name, expr in self.expressions.items()
            if any(s in self.pending_signals for s in signals_named(expr))
        )
        self._check_fields()

    # ── compile ──────────────────────────────────────────────────────────────────────────

    def _pattern(self, signal_id: str, pattern: str, ignore_case: bool) -> Any:
        try:
            return dialect.compile_pattern(pattern, ignore_case)
        except dialect.PatternRefused as exc:
            raise RuleSetNotRunnable(
                "pattern_refused", f"signal {signal_id!r}: {exc.rule}",
                {"signal": signal_id, "rule": exc.rule, "position": exc.position},
            ) from None

    @staticmethod
    def _literal(text: str, word_boundary: bool) -> str:
        body = dialect.escape_literal(text)
        return rf"\b{body}\b" if word_boundary else body

    def _compile_signal(self, signal: dict[str, Any]) -> None:
        kind = signal["kind"]
        sid = signal["id"]
        if kind == "pattern":
            self._compiled[sid] = self._pattern(sid, signal["pattern"], signal["ignore_case"])
        elif kind == "proximity":
            self._phrases[sid] = [
                self._pattern(sid, self._literal(p, signal["word_boundary"]),
                              signal["ignore_case"])
                for p in signal["phrases"]
            ]
        elif kind == "lexical_weights":
            ordered = sorted(enumerate(signal["terms"]), key=lambda it: (-len(it[1]["term"]), it[0]))
            alternation = "|".join(f"({dialect.escape_literal(t['term'])})" for _, t in ordered)
            body = f"(?:{alternation})"
            pattern = rf"\b{body}\b" if signal["word_boundary"] else body
            self._lexical[sid] = (
                self._pattern(sid, pattern, signal["ignore_case"]),
                [(t["term"], float(t["weight"])) for _, t in ordered],
            )
        elif kind == "validator":
            ref = signal["validator"]
            try:
                validator = resolve(ref["id"], ref["version"])
            except LookupError:
                raise RuleSetNotRunnable(
                    "validator_unknown",
                    f"signal {sid!r}: validator {ref['id']}@{ref['version']} is not registered",
                    {"signal": sid, "validator": ref["id"], "version": ref["version"]},
                ) from None
            errors = sorted(
                Draft202012Validator(validator.decl.params_schema).iter_errors(signal["params"]),
                key=str,
            )
            if errors:
                raise RuleSetNotRunnable(
                    "validator_params_invalid", f"signal {sid!r}: parameters refused",
                    {"signal": sid, "validator": validator.decl.ref},
                )
            self._validators[sid] = validator
        elif kind == "code_rule":
            # D41: accepted and never run here; see `_pending_signals`.
            return
        else:  # a kind the vendored schema has and this evaluator lacks (forms.py covers it)
            raise RuleSetNotRunnable("form_unavailable", f"signal {sid!r}: form {kind!r}",
                                     {"signal": sid, "form": kind})

    def _pending_signals(self) -> frozenset[str]:
        pending: set[str] = set()
        for signal in self.signals:  # in order: a reference is always to an earlier signal
            if signal["kind"] == "code_rule" or signal_reference(signal) in pending:
                pending.add(signal["id"])
        return frozenset(pending)

    def _field_names(self, signal_id: str) -> set[str]:
        signal = self._by_id[signal_id]
        kind = signal["kind"]
        if kind == "validator":
            return {name for name, _ in self._validators[signal_id].decl.outputs}
        if kind == "proximity":
            return {"distance", "phrase", "side"}
        if kind == "lexical_weights":
            return {"term", "weight"}
        if kind == "code_rule":
            return {o["name"] for o in signal["outputs"]}
        return {"text"}

    def _check_fields(self) -> None:
        for out in self.document["outputs"]:
            if output_value_key(out["value"]) == "field":
                sig, name = out["value"]["field"]
                if name not in self._field_names(sig):
                    raise RuleSetNotRunnable(
                        "field_unknown", f"output {out['column']!r}: no output {name!r}",
                        {"column": out["column"], "signal": sig, "field": name},
                    )

    # ── step 1: signals ──────────────────────────────────────────────────────────────────

    def _signal_spans(self, text: str, element: Optional[str]) -> dict[str, SignalResult]:
        strings = {TEXT: text, ELEMENT: element or ""}
        folded: dict[tuple[str, bool], str] = {}

        def view(source: str, fold: bool) -> str:
            key = (source, fold)
            if key not in folded:
                raw = strings[source]
                folded[key] = dialect.fold_fullwidth_ascii(raw) if fold else raw
            return folded[key]

        results: dict[str, SignalResult] = {}
        for signal in self.signals:
            sid, kind = signal["id"], signal["kind"]
            if sid in self.pending_signals:
                results[sid] = SignalResult([], [])  # never run inline (D41)
                continue
            if kind == "pattern":
                source = ELEMENT if signal["reads"] == "element" else TEXT
                if source == ELEMENT and element is None:
                    spans: list[Span] = []
                else:
                    raw = strings[source]
                    spans = [
                        Span(m.start(), m.end(), source, {"text": raw[m.start(): m.end()]})
                        for m in dialect.finditer(self._compiled[sid],
                                                  view(source, signal["fold_fullwidth_ascii"]))
                    ]
                results[sid] = SignalResult(spans, spans)
            elif kind == "proximity":
                spans = self._proximity(signal, results[signal["anchor"]].spans, text, view)
                results[sid] = SignalResult(spans, spans)
            elif kind == "validator":
                try:
                    candidates = self._validate(signal, results, text, element)
                except Exception as exc:  # noqa: BLE001 - re-raised named, without the text
                    ref = signal["validator"]
                    raise RuleSignalError(sid, f"{ref['id']}@{ref['version']}", exc) from None
                results[sid] = SignalResult(
                    [c for c in candidates if c.outputs.get("valid") is True], candidates)
            elif kind == "lexical_weights":
                spans = self._lexical_spans(signal, view)
                results[sid] = SignalResult(spans, spans)
        return results

    def _proximity(self, signal: dict[str, Any], anchors: list[Span], text: str,
                   view: Callable[[str, bool], str]) -> list[Span]:
        haystack = view(TEXT, signal["fold_fullwidth_ascii"])
        occurrences: list[tuple[int, int, int]] = []
        for order, compiled in enumerate(self._phrases[signal["id"]]):
            occurrences.extend((m.start(), m.end(), order)
                               for m in dialect.finditer(compiled, haystack))
        side_rule, unit = signal["side"], signal["unit"]
        out: list[Span] = []
        for anchor in anchors:
            if anchor.source != TEXT:
                continue
            best: Optional[tuple[int, int, int, int, int]] = None
            for s, e, order in occurrences:
                if e <= anchor.start and side_rule in ("before", "either"):
                    gap, side = _units(text, e, anchor.start, unit), 0
                elif s >= anchor.end and side_rule in ("after", "either"):
                    gap, side = _units(text, anchor.end, s, unit), 1
                else:
                    continue
                candidate = (gap, order, side, s, e)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            if best is not None and best[0] <= signal["max_distance"]:
                gap, _o, side, s, e = best
                out.append(Span(anchor.start, anchor.end, TEXT,
                                {"distance": gap, "phrase": text[s:e],
                                 "side": "before" if side == 0 else "after"}))
        return out

    def _validate(self, signal: dict[str, Any], results: dict[str, SignalResult], text: str,
                  element: Optional[str]) -> list[Span]:
        validator = self._validators[signal["id"]]
        names = [name for name, _ in validator.decl.outputs]
        params = signal["params"]

        def span(start: int, end: int, source: str, verdict: Any) -> Span:
            return Span(start, end, source, {n: getattr(verdict, n) for n in names})

        candidates = signal["candidates"]
        if "span_of" in candidates:
            out = []
            for s in results[candidates["span_of"]].spans:
                string = text if s.source == TEXT else (element or "")
                out.append(span(s.start, s.end, s.source, validator.check(
                    string[s.start: s.end], string[: s.start], string[s.end:], params)))
            return out
        if "element" in candidates:
            if element is None:
                return []
            return [span(0, len(element), ELEMENT, validator.check(element, "", "", params))]
        return [span(c.start, c.end, TEXT, c.verdict) for c in validator.scan(text, params)]

    def _lexical_spans(self, signal: dict[str, Any],
                       view: Callable[[str, bool], str]) -> list[Span]:
        compiled, terms = self._lexical[signal["id"]]
        score = 0.0
        first: dict[int, Span] = {}
        for m in dialect.finditer(compiled, view(TEXT, signal["fold_fullwidth_ascii"])):
            index = next(i for i, g in enumerate(m.groups()) if g is not None)
            term, weight = terms[index]
            if signal["counting"] == "occurrences" or index not in first:
                score += weight
            if index not in first:
                first[index] = Span(m.start(), m.end(), TEXT, {"term": term, "weight": weight})
        if score >= signal["cut"]:
            return sorted(first.values(), key=lambda s: (s.start, s.end))
        return []

    def signal_results(self, text: str, element: Optional[str] = None) -> dict[str, SignalResult]:
        """Step 1 alone: every signal's spans and candidates before any veto (the producer's
        `signal_results`; the public-values test reads which value-bearing validators fired)."""
        return self._signal_spans(text, element)

    # ── step 3: expressions ──────────────────────────────────────────────────────────────

    def _expr(self, expr: dict[str, Any], results: dict[str, SignalResult],
              strings: Mapping[str, str]) -> tuple[bool, list[Span]]:
        form = expression_form(expr)
        if form == "signal":
            spans = results[expr["signal"]].spans
            return bool(spans), list(spans)
        if form == "all":
            members = [self._expr(e, results, strings) for e in expr["all"]]
            if not all(t for t, _ in members):
                return False, []
            return True, _sorted_unique([s for _, ss in members for s in ss])
        if form == "any":
            members = [self._expr(e, results, strings) for e in expr["any"]]
            if not any(t for t, _ in members):
                return False, []
            return True, _sorted_unique([s for t, ss in members if t for s in ss])
        if form == "not":
            truth, _ = self._expr(expr["not"], results, strings)
            return not truth, []
        body = expr["within_n"]
        _, a_spans = self._expr(body["a"], results, strings)
        _, b_spans = self._expr(body["b"], results, strings)
        hits: list[Span] = []
        for a in a_spans:
            for b in b_spans:
                gap = _gap(a, b, body["unit"], strings)
                if gap is not None and gap <= body["n"]:
                    hits.extend((a, b))
        return bool(hits), _sorted_unique(hits)

    def form_of(self, name: str) -> str:
        """The expression's form: its signal's kind for `{signal}`, else the composition."""
        expr = self.expressions[name]
        form = expression_form(expr)
        if form == "signal":
            return str(self._by_id[expr["signal"]]["kind"])
        return form

    # ── the row (cases only) ─────────────────────────────────────────────────────────────

    def evaluate(self, text: str, element: Optional[str] = None) -> RowResult:
        strings = {TEXT: text, ELEMENT: element or ""}
        results = self._signal_spans(text, element)
        before_veto = {k: SignalResult(list(v.spans), list(v.candidates))
                       for k, v in results.items()}
        vetoes = [
            (v["id"], self._expr(self.expressions[v["when"]], before_veto, strings)[1],
             v["removes"])
            for v in self.document.get("vetoes") or []
        ]
        removed = _apply_vetoes(results, vetoes)
        evaluated = {name: self._expr(expr, results, strings)
                     for name, expr in self.expressions.items()}
        outputs: dict[str, Any] = {}
        for out in self.document["outputs"]:
            value = out["value"]
            key = output_value_key(value)
            if key == "field":
                sig, name = value["field"]
                remaining = sorted(results[sig].candidates, key=lambda s: (s.start, s.end))
                outputs[out["column"]] = remaining[0].outputs.get(name) if remaining else None
                continue
            truth, spans = evaluated[value[key]]
            if key == "fired":
                outputs[out["column"]] = truth
            elif key == "count":
                outputs[out["column"]] = len(spans)
            elif key == "first":
                outputs[out["column"]] = (
                    strings[spans[0].source][spans[0].start: spans[0].end] if spans else None)
            else:
                outputs[out["column"]] = _spans_json(spans)
        return RowResult(
            vetoed=any(removed.values()),
            outputs=outputs,
            expressions={name: {"fired": t, "spans": [[s.start, s.end] for s in ss]}
                         for name, (t, ss) in evaluated.items()},
        )

    # ── serving ──────────────────────────────────────────────────────────────────────────

    def serve(
        self,
        windows: Mapping[str, str],
        *,
        only: Optional[Iterable[str]] = None,
        clock: Callable[[], float] = time.monotonic,
        deadline: Optional[float] = None,
        return_positions: bool = False,
    ) -> list[Outcome]:
        """Every named expression (or `only` those) on every window given, fixed order.

        Order: expression name, then window in `WINDOW_ORDER`. The clock is read before each
        window's signal pass and before each expression; past `deadline`, the rest report
        `rule_timeout`. Vetoes are not applied (D9). Never raises: a validator that raises yields
        `rule_error` for that window's expressions.
        """
        names = sorted(self.expressions if only is None
                       else (n for n in self.expressions if n in set(only)))
        ordered_windows = [w for w in WINDOW_ORDER if w in windows] + sorted(
            w for w in windows if w not in WINDOW_ORDER)
        passes: dict[str, Optional[dict[str, SignalResult]]] = {}
        out: list[Outcome] = []
        for name in names:
            form = self.form_of(name)
            for window in ordered_windows:
                if name in self.pending_expressions:
                    out.append(Outcome(name, form, window, None, 0, "skipped", PENDING_REASON,
                                       "unavailable"))
                    continue
                if deadline is not None and clock() >= deadline:
                    out.append(Outcome(name, form, window, None, 0, "failed", TIMEOUT_REASON))
                    continue
                if window not in passes:
                    try:
                        passes[window] = self._signal_spans(windows[window], None)
                    except RuleSignalError:
                        passes[window] = None
                    if deadline is not None and clock() >= deadline:
                        out.append(Outcome(name, form, window, None, 0, "failed",
                                           TIMEOUT_REASON))
                        continue
                results = passes[window]
                if results is None:
                    out.append(Outcome(name, form, window, None, 0, "failed", ERROR_REASON))
                    continue
                truth, spans = self._expr(self.expressions[name], results,
                                          {TEXT: windows[window], ELEMENT: ""})
                out.append(Outcome(
                    name, form, window, truth, len(spans), "completed", None,
                    spans=tuple((s.start, s.end) for s in spans) if return_positions else None,
                ))
        return out


    # ── operator code, after the response (S3; D41) ──────────────────────────────────────

    def code_signals(self) -> tuple[dict[str, Any], ...]:
        """The `code_rule` signals, in document order."""
        return tuple(s for s in self.signals if s["kind"] == "code_rule")

    def _resolvable(self, expr: dict[str, Any]) -> bool:
        """Can a pending expression be computed from the code rules' COUNTS alone?

        A code rule reports a count and never a position (the runner contract; miDataworks 018
        FR-018.38, `spans: null`). So `{signal: <code rule>}` and `all` / `any` / `not` over such
        terms resolve; a signal that READS a code rule's positions (a proximity `anchor`, a
        validator `span_of`) and a `within_n` over one never can, and stay pending.
        """
        form = expression_form(expr)
        if form == "signal":
            sid = expr["signal"]
            return sid not in self.pending_signals or self._by_id[sid]["kind"] == "code_rule"
        if form in ("all", "any"):
            return all(self._resolvable(e) for e in expr[form])
        if form == "not":
            return self._resolvable(expr["not"])
        return not any(s in self.pending_signals for s in signals_named(expr))

    def resolvable_code_signals(self, names: Iterable[str]) -> dict[str, tuple[str, ...]]:
        """`{pending expression: the code-rule signal ids it needs}` for every resolvable one."""
        out: dict[str, tuple[str, ...]] = {}
        for name in sorted(set(names)):
            if name not in self.pending_expressions:
                continue
            expr = self.expressions[name]
            if self._resolvable(expr):
                out[name] = tuple(dict.fromkeys(
                    s for s in signals_named(expr) if self._by_id[s]["kind"] == "code_rule"))
        return out

    def _mixed(self, expr: dict[str, Any], results: dict[str, SignalResult],
               strings: Mapping[str, str], code: Mapping[str, int]
               ) -> tuple[bool, list[Span], int]:
        """`(truth, positions, code count)`: `_expr` with code-rule terms counted, not placed.
        With no code term inside it equals `_expr` exactly (code count 0)."""
        form = expression_form(expr)
        if form == "signal":
            sid = expr["signal"]
            if sid in code:
                return code[sid] >= 1, [], code[sid]
            spans = results[sid].spans
            return bool(spans), list(spans), 0
        if form in ("all", "any"):
            members = [self._mixed(e, results, strings, code) for e in expr[form]]
            truth = all(t for t, _s, _c in members) if form == "all" else any(
                t for t, _s, _c in members)
            if not truth:
                return False, [], 0
            kept = members if form == "all" else [m for m in members if m[0]]
            return True, _sorted_unique([s for _t, ss, _c in kept for s in ss]), sum(
                c for _t, _s, c in kept)
        if form == "not":
            truth, _s, _c = self._mixed(expr["not"], results, strings, code)
            return not truth, [], 0
        truth, spans = self._expr(expr, results, strings)
        return truth, spans, 0

    def resolve_pending(
        self,
        windows: Mapping[str, str],
        code_results: Mapping[tuple[str, str], tuple[Optional[int], Optional[str]]],
        *,
        only: Optional[Iterable[str]] = None,
    ) -> list[Outcome]:
        """The pending expressions that the runner's results now decide (D41).

        `code_results` is `{(code signal id, window): (count, reason)}` — exactly one of the two.
        An expression is returned only when it is resolvable and every code rule it names has a
        result on that window; anything else stays pending (never `fires: false`). A failed code
        rule fails the expression with its reason. Fixed order: expression name, then window.
        """
        names = self.pending_expressions if only is None else (
            self.pending_expressions & set(only))
        needs = self.resolvable_code_signals(names)
        ordered_windows = [w for w in WINDOW_ORDER if w in windows] + sorted(
            w for w in windows if w not in WINDOW_ORDER)
        passes: dict[str, Optional[dict[str, SignalResult]]] = {}
        out: list[Outcome] = []
        for name in sorted(needs):
            form = self.form_of(name)
            for window in ordered_windows:
                got = [code_results.get((sid, window)) for sid in needs[name]]
                if any(g is None for g in got):
                    continue
                failed = next((reason for count, reason in got if reason is not None), None)
                if failed is not None:
                    out.append(Outcome(name, form, window, None, 0, "failed", failed))
                    continue
                if window not in passes:
                    try:
                        passes[window] = self._signal_spans(windows[window], None)
                    except RuleSignalError:
                        passes[window] = None
                results = passes[window]
                if results is None:
                    out.append(Outcome(name, form, window, None, 0, "failed", ERROR_REASON))
                    continue
                counts = {sid: int(count or 0) for sid, (count, _r) in zip(needs[name], got)}
                truth, spans, code_count = self._mixed(
                    self.expressions[name], results, {TEXT: windows[window], ELEMENT: ""},
                    counts)
                out.append(Outcome(name, form, window, truth, len(spans) + code_count,
                                   "completed", None))
        return out


def compile_rule_set(document: dict[str, Any]) -> CompiledRuleSet:
    return CompiledRuleSet(document)
