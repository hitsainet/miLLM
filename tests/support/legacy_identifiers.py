"""Parser of the legacy-identifier export (040 FR-040.1, FR-040.2, FR-040.15; FTDD §4.2; TID §3.5.4).

`docs/legacy-identifiers.md` (full, private repo) and `docs/legacy-identifiers.public.md` (the only
variant the mirror keeps, CF-2) are written by the suite's `census.py --export` (S001 FTDD §4.3, CF-1)
and never edited here. This module is the only reader. Every rule REFUSES loudly, naming the file, the
line and the value: a parser that skipped an unknown record would let the frozen test and the brand
guard pass over a rule they never saw.

Nothing in `miserver/` reads the export (the server keeps its own hand-written alias tables, CF-3).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "docs" / "legacy-identifiers.md"
PUBLIC = ROOT / "docs" / "legacy-identifiers.public.md"

#: The allowed values, one constant per field, so a CF-1 correction is a one-line change.
FORMAT = 1
SECTIONS = frozenset({"entry", "env", "host", "history_tool", "repo", "guard"})
STATUSES = frozenset({"frozen", "scheduled", "retired"})
TRANSITION = frozenset({"transition", "retired"})
RETIRE_PHASES = frozenset({"P8", None})
FLAGS = frozenset({"history_only", "text_only", "public_withheld", "pending"})
EFFECTS = frozenset({"deny", "allow"})
PHASES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
JS_FLAGS = frozenset({"", "i"})

HEADER_FIELDS = frozenset({"format", "repo", "variant", "register_hash", "phase", "closed", "withheld"})
#: Optional header keys and the only values each may take. `forbidden_list: "unread"` is written by
#: `--public-unchecked` while S001 has not read this repo's forbidden-content list (FTASKS 0.53); a
#: `--public` export after that carries no such key.
OPTIONAL_HEADER: dict[str, frozenset[str]] = {"forbidden_list": frozenset({"unread"})}
FIELDS: dict[str, frozenset[str]] = {
    "entry": frozenset({"section", "id", "kind", "group", "status", "retire_phase", "pattern",
                        "pattern_js", "js_flags", "literals", "exercise", "flags"}),
    "env": frozenset({"section", "old", "new", "status"}),
    "host": frozenset({"section", "old", "new", "status", "observed"}),
    "history_tool": frozenset({"section", "old", "new"}),
    "repo": frozenset({"section", "old", "new", "status"}),
    "guard": frozenset({"section", "id", "pattern", "pattern_js", "js_flags", "effect", "from_phase",
                        "until_phase"}),
}

#: CF-25 m4: an inline flag group, not a lookaround or `(?:…)`.
INLINE_FLAG = re.compile(r"\(\?[a-zA-Z]+[:)]")
#: CF-25 m5: constructs with no identical JavaScript meaning.
FORBIDDEN = ("\\A", "\\Z", "(?P", "(?#")
EXERCISE = re.compile(
    r"^(?:pytest:(?P<py>tests/[\w./-]+\.py)::(?P<pytest>[\w\[\]./:-]+)"
    r"|vitest:(?P<vt>admin-ui/[\w./-]+\.test\.tsx?)::(?P<vitest>.+))$")


class LegacyFileError(ValueError):
    """The export breaks a rule of FTDD §4.2 / S001 FTDD §4.3."""


class LegacyFileMissing(LegacyFileError):
    """Neither variant exists: the frozen test and the brand guard must not pass on nothing."""


@dataclass(frozen=True)
class Export:
    path: Path
    header: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    lines: tuple[int, ...]

    # ── views ────────────────────────────────────────────────────────────────────
    def section(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["section"] == name]

    def entries(self, status: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.section("entry") if status is None or r["status"] == status]

    def env(self) -> list[dict[str, Any]]:
        return self.section("env")

    def hosts(self) -> list[dict[str, Any]]:
        return self.section("host")

    def history_tools(self) -> list[dict[str, Any]]:
        return self.section("history_tool")

    def repos(self) -> list[dict[str, Any]]:
        return self.section("repo")

    def guards(self) -> list[dict[str, Any]]:
        return self.section("guard")

    def phase(self) -> str:
        return self.header["phase"]

    def closed(self) -> str | None:
        return self.header["closed"]

    def withheld(self) -> list[str]:
        return list(self.header["withheld"])

    def is_withheld(self, section: str, old: str) -> bool:
        """An `env`/`host`/`history_tool`/`repo` record the public variant leaves out, named either
        `<section>:<old>` or by its stand-in `<section>:sha256:<16 hex>` (CF-2)."""
        held = set(self.header["withheld"])
        return f"{section}:{old}" in held or stand_in(section, old) in held

    def closed_at_least(self, phase: str) -> bool:
        """The header's `closed` is `phase` or later (CF-21: behaviour keyed to closure, never to
        `phase`). `None` (nothing closed yet) is before P0."""
        closed = self.header["closed"]
        return closed is not None and PHASES.index(closed) >= PHASES.index(phase)

    def active_guards(self) -> list[dict[str, Any]]:
        """`guard` records in force at the header's `phase`: at or after `from_phase`, before
        `until_phase` (null: unbounded)."""
        now = PHASES.index(self.header["phase"])
        out = []
        for g in self.guards():
            start = -1 if g["from_phase"] is None else PHASES.index(g["from_phase"])
            end = len(PHASES) if g["until_phase"] is None else PHASES.index(g["until_phase"])
            if start <= now < end:
                out.append(g)
        return out

    def allowed(self, text: str, start: int, end: int) -> bool:
        """FTDD §6.7 allowance: the span `text[start:end]` lies inside a match of an active `allow`
        guard, or of a `frozen`/`scheduled` entry's pattern, or is the `old` name of a `transition`
        `repo`, `env` or `host` record, or of a `history_tool` record. `retired` allows nothing."""
        for g in self.active_guards():
            if g["effect"] == "allow" and _covers(g["pattern"], text, start, end):
                return True
        for r in self.entries():
            if r["status"] in ("frozen", "scheduled") and _covers(r["pattern"], text, start, end):
                return True
        for r in [*self.repos(), *self.env(), *self.hosts()]:
            if r["status"] == "transition" and _covers(re.escape(r["old"]), text, start, end, True):
                return True
        for r in self.history_tools():
            if _covers(re.escape(r["old"]), text, start, end):
                return True
        return False


def stand_in(section: str, old: str) -> str:
    """The name a public variant gives a withheld record whose old name it must not print:
    `<section>:sha256:` + the first 16 hex characters of the SHA-256 of the old name."""
    return f"{section}:sha256:" + hashlib.sha256(old.encode("utf-8")).hexdigest()[:16]


def _covers(pattern: str, text: str, start: int, end: int, ignore_case: bool = False) -> bool:
    flags = re.IGNORECASE if ignore_case else 0
    return any(m.start() <= start and end <= m.end() for m in re.finditer(pattern, text, flags))


def skip_if_withheld(export: Export, record_id: str) -> None:
    """The ONLY skip in this module (CF-2): a record the public variant leaves out is skipped by
    name on the mirror, never silently."""
    if record_id in export.header["withheld"]:
        pytest.skip(f"withheld from the public variant: {record_id}")


def default_path() -> Path:
    if FULL.exists():
        return FULL
    if PUBLIC.exists():
        return PUBLIC
    raise LegacyFileMissing(f"neither {FULL} nor {PUBLIC} exists: the export is required")


def load(path: Path | None = None) -> Export:
    path = default_path() if path is None else Path(path)
    if not path.exists():
        raise LegacyFileMissing(f"{path} does not exist")
    return parse(path.read_text(encoding="utf-8"), path)


def _fail(path: Path, line: int, why: str, value: Any) -> LegacyFileError:
    return LegacyFileError(f"{path}:{line}: {why}: {value!r}")


def _block(text: str, path: Path) -> Iterator[tuple[int, str]]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "```jsonl"]
    if len(starts) != 1:
        raise _fail(path, 0, "expected exactly one ```jsonl block", len(starts))
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "```"), None)
    if end is None:
        raise _fail(path, start + 1, "the ```jsonl block is not closed", "```")
    for i in range(start + 1, end):
        if lines[i].strip():
            yield i + 1, lines[i]


def _check_pattern(path: Path, line: int, rec: dict[str, Any]) -> None:
    pattern, pattern_js, js_flags = rec["pattern"], rec["pattern_js"], rec["js_flags"]
    if not isinstance(pattern, str) or not pattern:
        raise _fail(path, line, "pattern must be a non-empty string", pattern)
    try:
        re.compile(pattern)
    except re.error as exc:
        raise _fail(path, line, f"pattern does not compile ({exc})", pattern) from exc
    if not isinstance(pattern_js, str) or not pattern_js:
        raise _fail(path, line, "pattern_js must be a non-empty string", pattern_js)
    if INLINE_FLAG.search(pattern_js):
        raise _fail(path, line, "pattern_js holds an inline flag group (CF-16, CF-25 m4)", pattern_js)
    for bad in FORBIDDEN:
        if bad in pattern or bad in pattern_js:
            raise _fail(path, line, f"pattern uses {bad} (no identical JavaScript meaning, CF-25 m5)",
                        pattern if bad in pattern else pattern_js)
    if js_flags not in JS_FLAGS:
        raise _fail(path, line, "js_flags must be '' or 'i'", js_flags)


def parse(text: str, path: Path) -> Export:
    items = list(_block(text, path))
    if not items:
        raise _fail(path, 0, "the jsonl block is empty", "")
    parsed: list[tuple[int, dict[str, Any]]] = []
    for line, raw in items:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _fail(path, line, f"not JSON ({exc.msg})", raw[:80]) from exc
        if not isinstance(value, dict):
            raise _fail(path, line, "a record must be an object", value)
        parsed.append((line, value))

    # ── header ───────────────────────────────────────────────────────────────────
    hline, header = parsed[0]
    if "section" in header or not HEADER_FIELDS <= set(header) \
            or not set(header) - HEADER_FIELDS <= set(OPTIONAL_HEADER):
        raise _fail(path, hline, f"the first line must be the header with exactly {sorted(HEADER_FIELDS)} "
                    f"(optional: {sorted(OPTIONAL_HEADER)})", sorted(header))
    for key, allowed in OPTIONAL_HEADER.items():
        if key in header and header[key] not in allowed:
            raise _fail(path, hline, f"{key} must be one of {sorted(allowed)}", header[key])
    if header["format"] != FORMAT:
        raise _fail(path, hline, f"format must be {FORMAT}", header["format"])
    expected_variant = "public" if path.name.endswith(".public.md") else "full"
    if header["variant"] != expected_variant:
        raise _fail(path, hline, f"variant must be {expected_variant!r} for this file name",
                    header["variant"])
    for key in ("repo", "register_hash", "phase"):
        if not isinstance(header[key], str) or not header[key]:
            raise _fail(path, hline, f"{key} must be a non-empty string", header[key])
    if header["phase"] not in PHASES:
        raise _fail(path, hline, "unknown phase", header["phase"])
    closed = header["closed"]
    if closed is not None and closed not in PHASES:
        raise _fail(path, hline, "closed must be a phase name or null", closed)
    after = PHASES.index(closed) + 1 if closed is not None else 0
    if PHASES.index(header["phase"]) not in (after, after - 1) or (closed is None and header["phase"] != "P0"):
        raise _fail(path, hline, "phase must equal closed or the phase after it (CF-21)",
                    (header["phase"], closed))
    withheld = header["withheld"]
    if not isinstance(withheld, list) or not all(isinstance(w, str) for w in withheld):
        raise _fail(path, hline, "withheld must be a list of strings", withheld)
    if expected_variant == "full" and withheld:
        raise _fail(path, hline, "the full variant withholds nothing", withheld)
    if expected_variant == "full" and set(header) - HEADER_FIELDS:
        raise _fail(path, hline, "optional public-only header keys in the full variant",
                    sorted(set(header) - HEADER_FIELDS))

    # ── records ──────────────────────────────────────────────────────────────────
    records: list[dict[str, Any]] = []
    lines: list[int] = []
    seen_ids: set[str] = set()
    seen_old: set[tuple[str, str]] = set()
    for line, rec in parsed[1:]:
        section = rec.get("section")
        if section not in SECTIONS:
            raise _fail(path, line, "unknown section", section)
        if set(rec) != FIELDS[section]:
            extra = sorted(set(rec) - FIELDS[section])
            missing = sorted(FIELDS[section] - set(rec))
            raise _fail(path, line, f"{section} fields wrong (extra {extra}, missing {missing})",
                        sorted(rec))
        if section in ("entry", "guard"):
            if rec["id"] in seen_ids:
                raise _fail(path, line, "duplicate id", rec["id"])
            seen_ids.add(rec["id"])
            _check_pattern(path, line, rec)
        else:
            key = (section, rec["old"])
            if key in seen_old:
                raise _fail(path, line, "duplicate (section, old)", key)
            seen_old.add(key)
            for f in ("old", "new"):
                if not isinstance(rec[f], str) or not rec[f]:
                    raise _fail(path, line, f"{f} must be a non-empty string", rec[f])
        if section == "entry":
            _check_entry(path, line, rec)
        elif section in ("env", "host", "repo"):
            if rec["status"] not in TRANSITION:
                raise _fail(path, line, f"{section} status must be transition or retired", rec["status"])
            if section == "host" and not isinstance(rec["observed"], bool):
                raise _fail(path, line, "host.observed must be a boolean (CF-24)", rec["observed"])
        elif section == "guard":
            if rec["effect"] not in EFFECTS:
                raise _fail(path, line, "guard effect must be deny or allow", rec["effect"])
            for f in ("from_phase", "until_phase"):
                if rec[f] is not None and rec[f] not in PHASES:
                    raise _fail(path, line, f"guard {f} must be a phase or null", rec[f])
        records.append(rec)
        lines.append(line)
    if not any(r["section"] == "entry" for r in records):
        raise _fail(path, hline, "zero entry records", 0)
    return Export(path, header, tuple(records), tuple(lines))


def _check_entry(path: Path, line: int, rec: dict[str, Any]) -> None:
    if rec["status"] not in STATUSES:
        raise _fail(path, line, "entry status must be frozen, scheduled or retired", rec["status"])
    if rec["retire_phase"] not in RETIRE_PHASES:
        raise _fail(path, line, "retire_phase must be 'P8' or null", rec["retire_phase"])
    flags = rec["flags"]
    if not isinstance(flags, list) or not set(flags) <= FLAGS:
        raise _fail(path, line, f"flags must be a subset of {sorted(FLAGS)}", flags)
    if "public_withheld" in flags and path.name.endswith(".public.md"):
        raise _fail(path, line, "a public_withheld record is in the public variant", rec["id"])
    literals = rec["literals"]
    if not isinstance(literals, list) or not all(isinstance(x, str) and x for x in literals):
        raise _fail(path, line, "literals must be a list of non-empty strings", literals)
    for literal in literals:
        if not re.search(rec["pattern"], literal):
            raise _fail(path, line, "a literal its own pattern misses (CF-25 m5)", literal)
    exercise = rec["exercise"]
    if exercise is None:
        if not set(flags) & {"history_only", "text_only", "pending"}:
            raise _fail(path, line, "exercise null needs history_only, text_only or pending", rec["id"])
    elif not isinstance(exercise, str) or not EXERCISE.match(exercise):
        raise _fail(path, line, "exercise must be pytest:<tests/...py>::<test> or "
                                "vitest:<admin-ui/...test.ts(x)>::<name>", exercise)
    if "pending" in flags and exercise is not None:
        raise _fail(path, line, "a pending record has an exercise", rec["id"])
