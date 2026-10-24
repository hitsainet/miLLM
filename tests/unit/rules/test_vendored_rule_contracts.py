"""The vendored rule contracts are miDataworks' bytes (039 FR-39.17, FR-39.20; FTASKS 11.1).

`docs/schemas/midataworks-rule-set-v1.json` (the rule-set document) and
`docs/schemas/rule-signal-cases-v1.json` (the shared conformance vectors) are owned by miDataworks
016 and vendored byte-identically. Re-vendor with `cp`; never edit the copy.

The byte check runs against `$MIDATAWORKS_REPO` (default: the sibling checkout), is REQUIRED under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1`, and skips LOUDLY otherwise. ⚠ No CI job checks miDataworks
out yet (no deploy key for it in this repo): until one does, this pin runs on developer machines
only — tracked in the 039 controls record.

⚠ IMPORTS NOTHING FROM `millm` (a pytest-only runner can run it). Control: flip one byte of either
copy → red.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import sibling_repo

ROOT = Path(__file__).resolve().parents[3]
FILES = ("midataworks-rule-set-v1.json", "rule-signal-cases-v1.json")
#: The shas this build vendored (miDataworks origin/main 2082e01, 2026-10-24).
PINNED = {
    "midataworks-rule-set-v1.json":
        "24925f35a7e731453e9693319be79ab25a69aef29f16c4d6f74c94e304b5464b",
    "rule-signal-cases-v1.json":
        "1d50c5a0e37964959253de61402d9391b5632614469d26dea39fefccbe1a6a9a",
}
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"


@pytest.mark.parametrize("name", FILES)
def test_the_vendored_copy_is_the_pinned_bytes(name):
    data = (ROOT / "docs" / "schemas" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name], (
        f"{name} is not the bytes this build vendored; re-vendor from miDataworks and update the "
        "pin in the same commit")


@pytest.mark.parametrize("name", FILES)
def test_the_vendored_copy_is_byte_identical_to_midataworks(name):
    source = sibling_repo("miDataworks", "MIDATAWORKS_REPO") / "docs" / "schemas" / name
    if not source.exists():
        message = f"{source} is not checked out — the miDataworks byte pin did NOT run"
        if REQUIRED:
            pytest.fail(message + " and MILLM_REQUIRE_CROSS_REPO_CHECKS=1")
        pytest.skip(message)
    assert (ROOT / "docs" / "schemas" / name).read_bytes() == source.read_bytes(), (
        f"{name} differs from miDataworks'; re-copy it, never edit the copy")
