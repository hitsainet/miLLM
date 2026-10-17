"""The vendored checkpoint manifest schema must stay byte-identical to miForge's (FR-32.3.9).

miForge owns `miforge.checkpoint-manifest` v1 and writes it beside every checkpoint it saves; miLLM
vendors it to validate the manifests of checkpoints it loads and, in the attached mode, to write
them. If this fails, someone edited the copy instead of the source, or miForge republished v1
without a version bump. Re-vendor with `cp`; never edit the copy.

The ownership split (`millm/services/checkpoint_manifest.py`) is derived from this file's
descriptions, so a drifted copy would also silently move fields between the trainer and the
writer; its partition is tested in `tests/unit/services/test_checkpoint_manifest.py`.

Skipped LOUDLY when miForge is not checked out; REQUIRED under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1`. ⚠ miForge is private: CI checks it out with the read-only
deploy key `MIFORGE_DEPLOY_KEY` in the `adapter-manifest-sync` job, and until that secret exists
the cross-repo switch stays OFF there (addendum §5). Imports nothing from `millm`, so that job
runs it with `--noconftest` and only pytest installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import sibling_repo

VENDORED = (
    Path(__file__).resolve().parents[3] / "docs" / "schemas" / "checkpoint-manifest-v1.json"
)
MIFORGE = sibling_repo("miForge", "MIFORGE_REPO") / "docs/schemas/checkpoint-manifest-v1.json"
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"


def test_it_is_byte_identical():
    assert VENDORED.exists(), f"vendored checkpoint manifest schema missing at {VENDORED}"
    if not MIFORGE.exists():
        if REQUIRED:
            pytest.fail(
                f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but miForge's schema is not at {MIFORGE}"
            )
        pytest.skip(
            f"miForge not checked out at {MIFORGE}: the checkpoint manifest byte-identity check "
            "did NOT run (set MIFORGE_REPO, or MILLM_REQUIRE_CROSS_REPO_CHECKS=1 to fail)"
        )
    assert VENDORED.read_bytes() == MIFORGE.read_bytes(), (
        "the vendored checkpoint manifest schema has drifted from miForge's. Re-vendor with cp; "
        "never edit the copy — the two repos would then disagree about what a checkpoint is."
    )


def test_the_vendored_file_is_the_v1_contract():
    schema = json.loads(VENDORED.read_text())
    assert schema["$ref"] == "#/$defs/CheckpointManifestV1"
    props = schema["$defs"]["CheckpointManifestV1"]["properties"]
    assert props["kind"]["const"] == "miforge.checkpoint-manifest"
    assert props["schema_version"]["const"] == "1"
    # D-30: a miForge manifest may omit both constants, so neither may become required.
    required = schema["$defs"]["CheckpointManifestV1"]["required"]
    assert "kind" not in required and "schema_version" not in required
