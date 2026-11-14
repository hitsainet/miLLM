"""Apply the mirror's strip step exactly as `sync-to-clean.yml` runs it (040 T02, TID §8).

The step's script is read from the parsed workflow, never retyped, so the mirror check before a push
strips what the mirror strips."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "sync-to-clean.yml"
STEP = "Remove excluded files and folders"


def strip_script() -> str:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    steps = [s for s in jobs["sync"]["steps"] if s.get("name") == STEP]
    assert len(steps) == 1, f"{WORKFLOW}: expected one step named {STEP!r}, found {len(steps)}"
    return steps[0]["run"]


def strip_like_the_mirror(tree: Path) -> None:
    """Run the strip step in `tree` (a scratch copy; it deletes files)."""
    subprocess.run(["bash", "-euo", "pipefail", "-c", strip_script()], cwd=tree, check=True,
                   capture_output=True, text=True)
