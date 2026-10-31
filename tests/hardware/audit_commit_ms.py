#!/usr/bin/env python3
"""commit_ms percentiles from miLLM's own audit lines (031 9.4: "from both the harness and
miLLM's audit lines"). Reads a captured server log (JSON lines) on stdin or as an argument and
summarises every accepted `weights_commit_end` audit line, optionally for one attachment_ref.

  python tests/hardware/audit_commit_ms.py watchdog_server.log [--ref 0775a4db]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_common import summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--ref")
    a = ap.parse_args(argv)
    fh = open(a.log) if a.log else sys.stdin
    values, refs = [], set()
    for line in fh:
        if '"weights_commit_end"' not in line or '"weights_audit"' not in line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("outcome") != "accepted" or (a.ref and rec.get("attachment_ref") != a.ref):
            continue
        refs.add(rec.get("attachment_ref"))
        values.append(float(rec["commit_ms"]))
    print(json.dumps({"source": "miLLM weights_audit weights_commit_end lines",
                      "attachment_refs": sorted(r for r in refs if r), "commit_ms": summary(values)},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
