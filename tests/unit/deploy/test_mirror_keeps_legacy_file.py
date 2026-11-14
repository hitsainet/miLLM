"""T02: the mirror keeps the public legacy-identifier export and nothing else of `docs/` but the
schemas (040 FR-040.1, CF-2). The strip step is read from `sync-to-clean.yml` and run for real."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support import legacy_identifiers as li
from tests.support.mirror_strip import ROOT, strip_like_the_mirror


def _scratch(tmp_path: Path) -> Path:
    tree = tmp_path / "mirror"
    shutil.copytree(ROOT / "docs", tree / "docs")
    for d in ("0xcc", "scripts", ".claude", "backups", "nested/docs"):
        (tree / d).mkdir(parents=True, exist_ok=True)
        (tree / d / "x.md").write_text("x\n")
    (tree / "CLAUDE.md").write_text("x\n")
    (tree / "nested" / "docs" / "legacy-identifiers.md").write_text("x\n")
    return tree


def test_strip_keeps_public_variant_and_schemas_only(tmp_path):
    if not (ROOT / "docs" / "legacy-identifiers.md").is_file() and not (ROOT / "0xcc").exists():
        pytest.skip("this is the stripped mirror: the full export is (rightly) absent; the strip "
                    "rule is checked in the private repo")
    assert (ROOT / "docs" / "legacy-identifiers.md").is_file(), "the full export must exist here"
    assert (ROOT / "docs" / "legacy-identifiers.public.md").is_file(), "the public export must exist"
    tree = _scratch(tmp_path)
    strip_like_the_mirror(tree)
    kept = sorted(p.name for p in (tree / "docs").iterdir())
    assert kept == ["legacy-identifiers.public.md", "schemas"], kept
    assert not (tree / "0xcc").exists() and not (tree / "CLAUDE.md").exists()
    assert li.load(tree / "docs" / "legacy-identifiers.public.md").header["variant"] == "public"


def test_frozen_test_skips_withheld_ids_by_name_on_the_stripped_tree(tmp_path):
    tree = _scratch(tmp_path)
    strip_like_the_mirror(tree)
    public = tree / "docs" / "legacy-identifiers.public.md"
    lines = public.read_text().splitlines()
    header_at = next(i for i, line in enumerate(lines) if line.startswith('{"format"'))
    header = json.loads(lines[header_at])
    victim = next(json.loads(line) for line in lines if line.startswith('{"section":"entry"'))
    header["withheld"] = [victim["id"]]
    lines[header_at] = json.dumps(header)
    lines = [line for line in lines if not (line.startswith("{") and json.loads(line).get("id") == victim["id"])]
    public.write_text("\n".join(lines) + "\n")
    export = li.load(public)
    assert victim["id"] not in [r["id"] for r in export.entries()]
    with pytest.raises(pytest.skip.Exception, match=f"withheld from the public variant: {victim['id']}"):
        li.skip_if_withheld(export, victim["id"])
    li.skip_if_withheld(export, "DR-99")  # not withheld: no skip
