"""Parse and evaluate the job-level repository gates of the workflows (040 FR-040.14, TPD-8; TID
§3.5.6). A small evaluator for the forms the workflows use:

    atom := github.repository == '<s>'  |  '<s>' == github.repository  |  any other condition
    expr := expr || expr  |  expr && expr  |  ( expr )  |  ! expr (never around a repository atom)

String comparison is case-insensitive, as GitHub's is. Any other construct next to
`github.repository` (`!=`, `contains(`, `startsWith(`, a negation) RAISES `UnsupportedGate` naming
the workflow and job: a gate this cannot read must not be read as passing. Conditions that do not
mention the repository (`needs.x.outputs.y != ''`) evaluate to True: they are not this test's subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REPO_ATOM = "github.repository"
_TOKEN = re.compile(r"\s*(\|\||&&|==|!=|!|\(|\)|'[^']*'|[A-Za-z_][\w.\-\[\]*]*\(?|[^\s()|&!=']+)")


class UnsupportedGate(ValueError):
    pass


@dataclass(frozen=True)
class Gate:
    workflow: str
    job: str
    expr: str


def gates(workflows_dir: Path = WORKFLOWS) -> list[Gate]:
    out = []
    for path in sorted(workflows_dir.glob("*.yml")):
        jobs = (yaml.safe_load(path.read_text()) or {}).get("jobs") or {}
        for name, job in jobs.items():
            cond = str((job or {}).get("if", ""))
            if REPO_ATOM in cond:
                out.append(Gate(path.name, name, cond.strip().removeprefix("${{").removesuffix("}}")))
    return out


def _tokens(expr: str) -> list[str]:
    tokens, pos = [], 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip():
                raise UnsupportedGate(f"cannot read {expr[pos:]!r}")
            break
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str], repository: str, where: str) -> None:
        self.t, self.i, self.repo, self.where = tokens, 0, repository.lower(), where

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        self.i += 1
        return self.t[self.i - 1]

    def or_expr(self) -> bool:
        value = self.and_expr()
        while self.peek() == "||":
            self.take()
            value = self.and_expr() or value
        return value

    def and_expr(self) -> bool:
        value = self.unary()
        while self.peek() == "&&":
            self.take()
            value = self.unary() and value
        return value

    def unary(self) -> bool:
        if self.peek() == "!":
            self.take()
            start = self.i
            value = self.unary()
            if REPO_ATOM in self.t[start:self.i]:
                raise UnsupportedGate(f"{self.where}: negation next to {REPO_ATOM}")
            return not value
        if self.peek() == "(":
            self.take()
            value = self.or_expr()
            if self.take() != ")":
                raise UnsupportedGate(f"{self.where}: unbalanced parentheses")
            return value
        return self.atom()

    def atom(self) -> bool:
        parts: list[str] = []
        depth = 0
        while self.peek() is not None and not (depth == 0 and self.peek() in ("||", "&&", ")")):
            token = self.take()
            depth += token.endswith("(") - (token == ")")
            parts.append(token)
        if REPO_ATOM not in parts:
            return True  # not a repository condition; recorded by the caller, not tested here
        if len(parts) == 3 and parts[1] == "==" and REPO_ATOM in (parts[0], parts[2]):
            literal = parts[2] if parts[0] == REPO_ATOM else parts[0]
            if literal.startswith("'") and literal.endswith("'"):
                return literal[1:-1].lower() == self.repo
        raise UnsupportedGate(f"{self.where}: unsupported repository condition {' '.join(parts)!r}")


def evaluate(expr: str, repository: str, where: str = "<gate>") -> bool:
    parser = _Parser(_tokens(expr), repository, where)
    value = parser.or_expr()
    if parser.peek() is not None:
        raise UnsupportedGate(f"{where}: trailing {parser.t[parser.i:]!r}")
    return value


def violations(gate_list: list[Gate], renames: list[tuple[str, str]], new_only: bool) -> list[str]:
    """Each gate must run under the new name and, before P1 closes, the old one too; from P1 it must
    skip under the old name (CF-17). `renames` are the export's `repo` (old, new) pairs."""
    problems = []
    for g in gate_list:
        where = f"{g.workflow}:{g.job}"
        for old, new in renames:
            if old.split("/")[0].lower() not in g.expr.lower():
                continue  # a public gate is not judged by the private pair, and the reverse
            if old.split("/")[1].lower() not in ("millm",):
                continue  # this repo's own rename only
            if not evaluate(g.expr, new, where):
                problems.append(f"{where} skips under the new name {new}")
            runs_old = evaluate(g.expr, old, where)
            if new_only and runs_old:
                problems.append(f"{where} still runs under the old name {old} after P1 closed")
            if not new_only and not runs_old:
                problems.append(f"{where} skips under the old name {old} before P1 closed")
    return problems
