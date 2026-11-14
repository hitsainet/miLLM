"""T31: repository names in origin and reference text follow the renames (040 FR-040.44, parent
BR-075).

The tracked tree is read whole (comments and tests included), except the generated export
(`docs/legacy-identifiers*.md`, which names every old repository by design) and `0xcc/` (records).

* The old WORKBENCH repository names (either owner, then the workbench's old name; any case)
  appear nowhere, except inside a match of a frozen or scheduled `entry` record: the vendored
  schemas' `$id` URLs (DR-04) are identifiers, not links, and stay.
* The old names of THIS repository (its old name under either owner) appear only where a record of
  the export allows them. From P1 its `repo` records were `transition` and allowed them; P7
  (FR-040.40) retired them, so since P7 the names appear nowhere (tests that need one assemble it,
  like this file). A name the public variant withholds cannot be judged on the mirror and is
  skipped by name there.

Control NC-90 (0xcc/reviews/040_P4_controls_2026-11-11.md): put the workbench's old GitHub URL
(private owner) back into README.md ->
`test_no_old_workbench_repo_name_outside_records` red.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from tests.support import legacy_identifiers as li

ROOT = Path(__file__).resolve().parents[3]
OWNERS = "onegaishimas|hitsainet"
WORKBENCH = re.compile(rf"(?i)(?<![\w.-])({OWNERS})/mistudio(?![\w-])")
SERVER = re.compile(rf"(?i)(?<![\w.-])({OWNERS})/millm(?![\w-])")
EXCLUDED = (":(exclude,glob)**/docs/legacy-identifiers*.md", ":(exclude)0xcc")


def tracked_lines(root: Path = ROOT):
    """`(path, line number, line)` of every tracked text file outside the excluded paths."""
    listed = subprocess.run(["git", "ls-files", "-z", "--", ".", *EXCLUDED], cwd=root,
                            capture_output=True, text=True, check=True).stdout.split("\0")
    files = [rel for rel in listed if rel and (root / rel).is_file()]
    assert len(files) > 500 or root != ROOT, f"only {len(files)} files listed: the walk is failing open"
    for rel in files:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # an image or other binary file
        for number, line in enumerate(text.splitlines(), 1):
            yield rel, number, line


def _inside_an_entry(export: li.Export, line: str, start: int, end: int) -> bool:
    return any(m.start() <= start and end <= m.end()
               for record in export.entries() if record["status"] in ("frozen", "scheduled")
               for m in re.finditer(record["pattern"], line))


def workbench_findings(export: li.Export, root: Path = ROOT) -> list[str]:
    return [f"{rel}:{number}: {line.strip()[:160]}"
            for rel, number, line in tracked_lines(root)
            for m in WORKBENCH.finditer(line)
            if not _inside_an_entry(export, line, m.start(), m.end())]


def server_findings(export: li.Export, root: Path = ROOT) -> list[str]:
    out = []
    for rel, number, line in tracked_lines(root):
        for m in SERVER.finditer(line):
            owner = {"onegaishimas": "Onegaishimas", "hitsainet": "hitsainet"}[m.group(1).lower()]
            if export.is_withheld("repo", f"{owner}/miLLM"):
                continue  # the public variant cannot name it: not judged on the mirror
            if not export.allowed(line, m.start(), m.end()):
                out.append(f"{rel}:{number}: {line.strip()[:160]}")
    return out


def test_no_old_workbench_repo_name_outside_records():
    found = workbench_findings(li.load())
    assert not found, ("an old miWorkbench repository name outside the export's rows (040 FR-040.44):\n  "
                       + "\n  ".join(found))


def test_old_server_repo_names_only_through_transition_records():
    found = server_findings(li.load())
    assert not found, ("an old miServer repository name no record allows (040 FR-040.44):\n  "
                       + "\n  ".join(found))


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    return root


def test_the_scan_finds_plants_and_reads_the_records(tmp_path):
    """Self-test on a planted tree: what is reported, what a record allows, what is excluded."""
    export = li.load()
    # The old names are assembled here so that this file never holds one (it is tracked, and read).
    old, private, public = "mi" + "Studio", "Onegaishimas/", "hitsainet/"
    server = "mi" + "LLM"
    readme = f"See https://github.com/{private}{old} and {public.upper()}{old.upper()}."
    root = _tree(tmp_path, {
        "README.md": readme + "\n",
        "docs/schemas/x.json": json.dumps({
            "$id": f"https://raw.githubusercontent.com/{public}{old}/main/docs/schemas/x.json"}),
        "docs/legacy-identifiers.md": f"{private}{old}\n",
        "0xcc/reviews/r.md": f"{public}{old}\n",
        "CONTRIBUTING.md": f"git clone https://github.com/{public}{server}.git\n",
        "notes/other.md": f"{public}{old}-fork and x{public}{old} are other names\n",
    })
    assert workbench_findings(export, root) == [f"README.md:1: {readme}"] * 2, (
        "two names on the README line; the $id (DR-04), the export and 0xcc/ are not findings")
    clone = f"CONTRIBUTING.md:1: git clone https://github.com/{public}{server}.git"
    assert server_findings(export, root) == [clone], "since P7 the `repo` records are retired"

    transition = tmp_path / li.default_path().name
    transition.write_text("\n".join(
        json.dumps({**json.loads(line), "status": "transition"})
        if line.startswith("{") and json.loads(line).get("section") == "repo" else line
        for line in li.default_path().read_text().splitlines()) + "\n")
    assert server_findings(li.load(transition), root) == [], (
        "the record decides: in `transition`, the same line is allowed")
