"""The exercise for DR-20 (the package `millm`, 040 FR-040.7; CF-7), which feature 041 retires at F1.

041's rename commit (R) renamed the package. DR-20 stays "in progress" until F1 (its remaining hits are not
package uses: stored approval keys, a body field, the probe-result contract's path citation, the siblings'
either-name lookups), so its exercise pins the RENAMED behaviour: the package read from `pyproject.toml`
names the console script and the image's `CMD`, and both resolve. The exercises of the two other entries
this file held were deleted with their records' retirement (041 ID-3); 041's permanent tests (P6, P7, P9,
P11) cover the same ground.
"""

from __future__ import annotations

import importlib
import json
import re
import tomllib
from pathlib import Path

from tests.support.package_root import package_name

REPO = Path(__file__).resolve().parents[4]


def _resolve(target: str):
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def test_package_entry_points_resolve():
    pkg = package_name()
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {pkg: f"{pkg}.main:run"}
    assert callable(_resolve(scripts[pkg]))
    cmds = re.findall(r'^CMD \[(.*)\]$', (REPO / "Dockerfile").read_text(), flags=re.M)
    targets = {arg for cmd in cmds for arg in json.loads(f"[{cmd}]") if ":" in arg and "." in arg}
    assert targets == {f"{pkg}.main:app"}, targets
    for target in targets:
        assert _resolve(target) is not None
