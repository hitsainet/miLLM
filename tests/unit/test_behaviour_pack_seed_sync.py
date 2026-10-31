"""The guard's seeded pack equals miStudio's seeded behaviour catalogue (FR-38.1, FTDD 038 §10.4).

`tests/fixtures/behaviour_packs/seeded.json` lists the behaviours miStudio ships seeded: the
`behaviour` key and `behaviour_label` of every judge-prompt seed file in miStudio's
`backend/src/data/judge_prompt_seeds/*.json`. The generality guard (`test_generality_guard.py`)
reads that pack, so a seed miStudio adds without the pack changing is a name the guard does not
look for. This test fails on that drift.

Skipped LOUDLY when miStudio is not checked out (the public mirror; a worktree outside the suite
folder — set `MISTUDIO_REPO`). Required under `MILLM_REQUIRE_CROSS_REPO_CHECKS=1` (the
`cross-repo-contract` CI job) or `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`.

Imports nothing from `millm` and needs pytest alone, so the cross-repo job runs it without the ML
stack (`test_cross_repo_sync_files_need_only_pytest.py` holds that for the job's files).
"""

from __future__ import annotations

import json
import os

import pytest

from tests.support.sibling_repos import REPO_ROOT, mistudio_repo

PACK = REPO_ROOT / "tests" / "fixtures" / "behaviour_packs" / "seeded.json"
SEEDS = mistudio_repo() / "backend" / "src" / "data" / "judge_prompt_seeds"
REQUIRED = "1" in (
    os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS"),
    os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"),
)


def _sibling_catalogue() -> set[tuple[str, str]]:
    files = sorted(SEEDS.glob("*.json")) if SEEDS.is_dir() else []
    if not files:
        message = f"miStudio's seed files are not at {SEEDS}; set MISTUDIO_REPO"
        if REQUIRED:
            pytest.fail(message + " (cross-repo checks are required here)")
        pytest.skip(message)
    out = set()
    for path in files:
        doc = json.loads(path.read_text())
        out.add((doc["behaviour"], doc.get("behaviour_label") or doc["behaviour"]))
    return out


def test_the_seeded_pack_equals_mistudios_seed_files():
    pack = {(b["key"], b["name"]) for b in json.loads(PACK.read_text())["behaviours"]}
    sibling = _sibling_catalogue()
    assert pack == sibling, (
        f"tests/fixtures/behaviour_packs/seeded.json has drifted from miStudio's seed files: "
        f"only in the pack {sorted(pack - sibling)}, only in miStudio {sorted(sibling - pack)}. "
        f"Update the pack so the generality guard looks for every seeded name."
    )


def test_the_pack_is_not_empty():
    assert json.loads(PACK.read_text())["behaviours"], "an empty pack guards nothing"
