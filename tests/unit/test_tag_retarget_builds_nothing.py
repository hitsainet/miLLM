"""A retargeted mirror tag must not rebuild the images (2026-10-23).

sync-to-clean force-moves every existing mirror tag onto each new snapshot. That fires
`docker-images.yml` a second time with a tag ref. When `detect` treated any tag push as a
release, it rebuilt both images with new digests and Image Updater rolled the backend, even
for a docs-only push whose `main` run had correctly built nothing (2026-10-22, 2026-10-23).

These tests RUN the `Detect changed images` step's own script, taken from the parsed
workflow, with the tag-push environment. The tag branch exits before any registry or git
call, so no network or history is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "docker-images.yml"


def _detect_step() -> dict:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    return next(s for s in jobs["detect"]["steps"] if s.get("name") == "Detect changed images")


def _run_detect(tmp_path: Path, ref: str, created: str) -> tuple[list[str], str]:
    out = tmp_path / "github_output"
    out.write_text("")
    env = {**os.environ, "REF": ref, "CREATED": created, "GITHUB_OUTPUT": str(out)}
    proc = subprocess.run(
        ["bash", "-c", _detect_step()["run"]],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in out.read_text().splitlines() if l.startswith("images="))
    return json.loads(line.removeprefix("images=")), proc.stdout


def test_the_step_receives_whether_the_tag_was_created():
    assert _detect_step()["env"]["CREATED"] == "${{ github.event.created }}"


def test_a_retargeted_tag_builds_nothing(tmp_path):
    images, log = _run_detect(tmp_path, "refs/tags/v0.5.0", "false")
    assert images == [], log
    assert "retargeted" in log


def test_an_unset_created_flag_builds_nothing(tmp_path):
    # A tag event that does not report `created` is not evidence of a release.
    images, _ = _run_detect(tmp_path, "refs/tags/v0.5.0", "")
    assert images == []


def test_a_new_tag_still_builds_every_image(tmp_path):
    images, _ = _run_detect(tmp_path, "refs/tags/v0.6.0", "true")
    assert images == ["backend", "admin-ui"]
