"""The vendored runner contract and vectors are miDataworks' bytes (039 FTASKS 18.1; FR-39.26).

`docs/schemas/midataworks-runner-v1.json` (the isolated runner contract) and
`docs/schemas/runner-cases-v1.json` (the shared runner vectors) are owned by miDataworks 018 and
vendored byte-identically. Re-vendor with `cp`; never edit the copy. The contract's floors are
PROVISIONAL (`limits_floor_status`, suite decision D47): a re-vendor may RAISE one, never lower it,
and `test_runner_settings_floor.py` then refuses a deployment below it.

The byte check runs against `$MIDATAWORKS_REPO` (default: the sibling checkout), is REQUIRED under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1`, and skips LOUDLY otherwise.

⚠ IMPORTS NOTHING FROM `millm` (a pytest-only runner can run it). Control: flip one byte of either
copy → red.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import sibling_repo

ROOT = Path(__file__).resolve().parents[3]
FILES = ("midataworks-runner-v1.json", "runner-cases-v1.json")
#: The shas this build vendored (miDataworks main 6c71e7e, 018 S1 live, 2026-10-24).
PINNED = {
    "midataworks-runner-v1.json":
        "7a9319b63bf3862c0b313035debda3a6f75e57d088a58217fde5aaf5088ea0bf",
    "runner-cases-v1.json":
        "e6e0b4b80be0cc28b157f5b28150c0f7156730a5fb2e0fa702bbbf175a036052",
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


def test_the_floor_is_marked_provisional_and_may_only_rise():
    """D47: the file itself says its floors are provisional. miLLM reads them, never restates
    them; this pins that the status block is there to read."""
    contract = json.loads((ROOT / "docs" / "schemas" / FILES[0]).read_text(encoding="utf-8"))
    status = contract["limits_floor_status"]
    assert status["provisional"] is True
    assert "never lowered" in status["change_rule"]
