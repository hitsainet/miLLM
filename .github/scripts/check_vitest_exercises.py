"""Every `vitest:` exercise of the legacy-identifier export PASSED in this run (040 FR-040.2; rev2
review of 040 P0). T03 (backend) can only read an exercise's source; this reads Vitest's own JSON
report in Frontend CI, so a skipped, todo, failed or missing exercise fails the job.

usage: check_vitest_exercises.py <vitest JSON report> <docs dir>
Standard library only. The export is `<docs>/legacy-identifiers.md`, else `.public.md` (the mirror).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def exercises(docs: Path) -> list[tuple[str, str, str]]:
    """`(entry id, file under admin-ui/, test name)` for every vitest exercise in the export."""
    path = docs / "legacy-identifiers.md"
    if not path.is_file():
        path = docs / "legacy-identifiers.public.md"
    text = path.read_text(encoding="utf-8")
    block = text.split("```jsonl\n", 1)[1].split("\n```", 1)[0]
    out = []
    for line in block.splitlines()[1:]:
        if not line.strip():
            continue
        rec = json.loads(line)
        ex = rec.get("exercise") if rec.get("section") == "entry" else None
        if isinstance(ex, str) and ex.startswith("vitest:"):
            file, _, name = ex[len("vitest:"):].partition("::")
            out.append((rec["id"], file, name))
    return out


def problems(report: dict, wanted: list[tuple[str, str, str]]) -> list[str]:
    out = []
    for entry, file, name in wanted:
        suffix = "/" + file  # report names are absolute paths ending in admin-ui/…
        statuses = [a.get("status") for r in report.get("testResults", [])
                    if str(r.get("name", "")).endswith(suffix)
                    for a in r.get("assertionResults", []) if a.get("title") == name]
        if statuses != ["passed"]:
            out.append(f"{entry}: {file} :: {name!r} reported {statuses or 'nothing'} (must be one 'passed')")
    return out


def main(argv: list[str]) -> int:
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    wanted = exercises(Path(argv[2]))
    found = problems(report, wanted)
    for line in found:
        print(f"::error::{line}")
    print(f"vitest exercises checked: {len(wanted)}, problems: {len(found)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
