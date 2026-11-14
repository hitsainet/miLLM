"""Rev2 review of 040 P0 (2026-11-08): the two cross-repo guards converted after the G38/D53 merge
(`test_vendored_rule_contracts.py`, `test_template_prefix_skip.py`) must FAIL, not skip, when their
sibling is absent and the switch is on - under the old name (what CI's jobs set today) and the new.

Controls C5 (`REQUIRED = False`) and C6 (`if False:` before `pytest.fail`) turned each guard into a
silent skip with every test green: nothing ran them with the sibling missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import env_flags

ROOT = Path(__file__).resolve().parents[2]
NODES = (
    "tests/unit/rules/test_vendored_rule_contracts.py::test_the_vendored_copy_is_byte_identical_to_midataworks",
    "tests/unit/services/test_template_prefix_skip.py::test_the_case_file_is_identical_in_mistudio",
)


@pytest.mark.parametrize("switch", [env_flags.new_name("REQUIRE_CROSS_REPO_CHECKS"),
                                    "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"], ids=["new", "suite"])
def test_absent_sibling_with_the_switch_on_fails(tmp_path, switch):
    """Each switch spelling that is still read (040 P7 retired the old `MILLM_` one)."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_REQUIRE_CROSS_REPO_CHECKS")}
    env.update({"MIWORKBENCH_REPO": str(tmp_path / "no-wb"), "MIDATAWORKS_REPO": str(tmp_path / "no-dw"),
                switch: "1"})
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfs", "-o", "addopts=",
                          "-p", "no:cacheprovider", *NODES], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    out = run.stdout + run.stderr
    assert run.returncode == 1, out[-2000:]
    for node in NODES:
        assert f"FAILED {node}" in out, (node, out[-2000:])
