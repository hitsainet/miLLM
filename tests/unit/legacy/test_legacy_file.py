"""T01: the legacy-identifier export parser refuses every malformed input, naming it
(040 FR-040.1; FTDD §4.2; TID §3.5.4). Each refusal runs on a temp copy of the real export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import legacy_identifiers as li


def _real() -> li.Export:
    return li.load()


def _lines() -> list[str]:
    return li.default_path().read_text(encoding="utf-8").splitlines()


def _write(tmp_path: Path, lines: list[str], name: str | None = None) -> Path:
    name = name or li.default_path().name
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _edit(tmp_path: Path, pick, change, name: str | None = None) -> Path:
    """Copy the real export, apply `change(record)` to the first record `pick(record)` selects."""
    lines = _lines()
    for i, raw in enumerate(lines):
        if raw.startswith("{"):
            rec = json.loads(raw)
            if pick(rec):
                change(rec)
                lines[i] = json.dumps(rec)
                return _write(tmp_path, lines, name)
    raise AssertionError("no record picked")


def _refused(path: Path, words: str) -> None:
    with pytest.raises(li.LegacyFileError) as exc:
        li.load(path)
    assert words in str(exc.value), str(exc.value)
    assert str(path) in str(exc.value), "the refusal must name the file"


def _header(rec):
    return "format" in rec


def _entry(rec):
    return rec.get("section") == "entry"


def _guard(rec):
    return rec.get("section") == "guard"


def test_real_export_parses():
    export = _real()
    assert export.header["format"] == 1 and export.entries()
    assert {r["section"] for r in export.records} <= li.SECTIONS
    held_hosts = [w for w in export.withheld() if w.startswith("host:")]
    assert len(export.hosts()) + len(held_hosts) >= 7 and len(export.env()) >= 1


def test_refuses_format_2(tmp_path):
    _refused(_edit(tmp_path, _header, lambda r: r.update(format=2)), "format must be 1")


def test_refuses_unknown_section_hosts(tmp_path):
    _refused(_edit(tmp_path, lambda r: r.get("section") == "host",
                   lambda r: r.update(section="hosts")), "unknown section")


def test_refuses_locations_field(tmp_path):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(locations=["x.py:1"])), "extra ['locations']")


def test_refuses_regex_flag_in_flags(tmp_path):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(flags=["i"])), "flags must be a subset")


def test_refuses_transition_status_on_entry(tmp_path):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(status="transition")), "entry status")


def test_refuses_backend_exercise(tmp_path):
    _refused(_edit(tmp_path, lambda r: _entry(r) and r["exercise"],
                   lambda r: r.update(exercise="backend:millm.x::y")), "exercise must be")


def test_refuses_full_variant_in_public_file(tmp_path):
    _refused(_edit(tmp_path, _header, lambda r: r.update(variant="full", withheld=[]),
                   "legacy-identifiers.public.md"), "variant must be 'public'")


def test_refuses_missing_header(tmp_path):
    lines = [line for line in _lines() if not (line.startswith("{") and '"format"' in line)]
    _refused(_write(tmp_path, lines), "the first line must be the header")


def test_refuses_guard_effect_warn(tmp_path):
    _refused(_edit(tmp_path, _guard, lambda r: r.update(effect="warn")), "guard effect")


def test_refuses_status_on_guard(tmp_path):
    _refused(_edit(tmp_path, _guard, lambda r: r.update(status="frozen")), "extra ['status']")


def test_refuses_inline_flag_group_in_pattern_js(tmp_path):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(pattern_js="(?i:x)" + r["pattern_js"])),
             "inline flag group")


def test_accepts_lookahead_and_noncapturing_group(tmp_path):
    def change(r):
        r.update(pattern="(?:mi)(?=llm)" + r["pattern"], pattern_js="(?:mi)(?=llm)(?!x)(?<=y)?" + r["pattern_js"],
                 literals=[])
    li.load(_edit(tmp_path, _entry, change))


def test_refuses_js_flags_g(tmp_path):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(js_flags="g")), "js_flags")


def test_refuses_frozen_repo_record(tmp_path):
    _refused(_edit(tmp_path, lambda r: r.get("section") == "repo", lambda r: r.update(status="frozen")),
             "repo status must be transition or retired")


def test_refuses_header_without_closed(tmp_path):
    _refused(_edit(tmp_path, _header, lambda r: r.pop("closed")), "the first line must be the header")


def test_refuses_phase_not_equal_or_next_to_closed(tmp_path):
    _refused(_edit(tmp_path, _header, lambda r: r.update(phase="P3", closed="P1")), "CF-21")
    _refused(_edit(tmp_path, _header, lambda r: r.update(phase="P1", closed=None)), "CF-21")
    for phase, closed in (("P1", "P1"), ("P2", "P1"), ("P0", None)):
        li.load(_edit(tmp_path, _header, lambda r, p=phase, c=closed: r.update(phase=p, closed=c)))


def test_refuses_host_without_boolean_observed(tmp_path):
    _refused(_edit(tmp_path, lambda r: r.get("section") == "host", lambda r: r.update(observed="yes")),
             "observed must be a boolean")
    _refused(_edit(tmp_path, lambda r: r.get("section") == "host", lambda r: r.pop("observed")),
             "missing ['observed']")


@pytest.mark.parametrize("construct", ["\\A", "\\Z", "(?P<x>a)", "(?#c)"])
def test_refuses_anchor_and_named_group_constructs(tmp_path, construct):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(pattern=construct + r["pattern"], literals=[])),
             "no identical JavaScript meaning")


def test_refuses_literal_its_pattern_misses(tmp_path):
    _refused(_edit(tmp_path, _entry, lambda r: r.update(literals=["zz-not-matched-zz"])),
             "a literal its own pattern misses")


def test_refuses_duplicate_ids(tmp_path):
    lines = _lines()
    first = next(line for line in lines if line.startswith('{"section":"entry"'))
    end = lines.index("```", lines.index("```jsonl"))
    _refused(_write(tmp_path, lines[:end] + [first] + lines[end:]), "duplicate id")


def test_refuses_zero_entries(tmp_path):
    lines = [line for line in _lines() if not line.startswith('{"section":"entry"')]
    _refused(_write(tmp_path, lines), "zero entry records")


def test_refuses_null_exercise_without_a_flag(tmp_path):
    _refused(_edit(tmp_path, lambda r: _entry(r) and r["exercise"], lambda r: r.update(exercise=None)),
             "exercise null needs")


def test_a_missing_file_is_refused_not_skipped(tmp_path):
    with pytest.raises(li.LegacyFileMissing):
        li.load(tmp_path / "legacy-identifiers.md")


def test_phase_helpers():
    export = _real()
    assert li.PHASES.index(export.phase()) >= 0
    assert export.closed_at_least("P0") == (export.closed() is not None)


def _public_header_lines(tmp_path: Path, **extra) -> Path:
    """A public variant made from the loaded export: records flagged `public_withheld` dropped and
    named in `withheld`, as census does."""
    lines = _lines()
    i = next(n for n, line in enumerate(lines) if line.startswith('{"format"'))
    header = json.loads(lines[i])
    kept, held = [], list(header["withheld"])
    for line in lines:
        if line.startswith('{"section":"entry"') and "public_withheld" in json.loads(line)["flags"]:
            held.append(json.loads(line)["id"])
            continue
        kept.append(line)
    header.pop("forbidden_list", None)
    header.update(variant="public", withheld=held, **extra)
    kept[i] = json.dumps(header)
    return _write(tmp_path, kept, "legacy-identifiers.public.md")


def test_accepts_forbidden_list_unread_in_the_public_variant(tmp_path):
    """`--public-unchecked` (S001 FTASKS 0.53 not done) adds `forbidden_list: "unread"`; a later
    `--public` export drops the key. Both load."""
    assert li.load(_public_header_lines(tmp_path, forbidden_list="unread")).header["forbidden_list"] == "unread"
    assert "forbidden_list" not in li.load(_public_header_lines(tmp_path)).header


def test_refuses_another_forbidden_list_value_or_key(tmp_path):
    _refused(_public_header_lines(tmp_path, forbidden_list="read"), "forbidden_list must be one of")
    _refused(_public_header_lines(tmp_path, other_key=1), "the first line must be the header")
    _refused(_full_variant(tmp_path, header={"forbidden_list": "unread"}),
             "optional public-only header keys in the full variant")


def test_stand_in_names_a_withheld_record():
    """CF-2: `<section>:sha256:` + 16 hex of the SHA-256 of the old name."""
    assert li.stand_in("host", "k8s-millm.hitsai.local") == "host:sha256:565e698c25fc2516"
    export = li.load()
    for record in export.hosts():
        assert export.is_withheld("host", record["old"]) is False, "a present record is not withheld"


def test_public_withheld_flag_is_accepted_in_the_full_variant_only(tmp_path):
    li.load(_full_variant(tmp_path, flag_first_entry=True))
    _refused(_public_header_lines_with_flag(tmp_path), "a public_withheld record is in the public variant")


def _public_header_lines_with_flag(tmp_path: Path) -> Path:
    path = _public_header_lines(tmp_path)
    lines = path.read_text().splitlines()
    for i, raw in enumerate(lines):
        if raw.startswith('{"section":"entry"'):
            rec = json.loads(raw)
            rec["flags"] = ["public_withheld"]
            lines[i] = json.dumps(rec)
            break
    path.write_text("\n".join(lines) + "\n")
    return path


def _full_variant(tmp_path: Path, header: dict | None = None, flag_first_entry: bool = False) -> Path:
    """A full variant made from whichever export is loaded (the mirror carries only the public one):
    `variant: full`, nothing withheld, no public-only header key."""
    lines, flagged = _lines(), False
    for i, raw in enumerate(lines):
        if raw.startswith('{"format"'):
            rec = json.loads(raw)
            rec.pop("forbidden_list", None)
            rec.update(variant="full", withheld=[], **(header or {}))
            lines[i] = json.dumps(rec)
        elif flag_first_entry and not flagged and raw.startswith('{"section":"entry"'):
            rec = json.loads(raw)
            rec["flags"] = ["public_withheld"]
            lines[i] = json.dumps(rec)
            flagged = True
    sub = tmp_path / "full"
    sub.mkdir(exist_ok=True)
    return _write(sub, lines, "legacy-identifiers.md")
