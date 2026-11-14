"""Locate this repository and its sibling suite repos for cross-repo tests.

The suite repos (miStudio, miLLM, miDataworks, miForge) are checked out side by
side, so a sibling is found from THIS repository's own root — never from an
absolute path. They moved once already (``/home/x-sean/app/<repo>`` to
``/home/x-sean/app/miSuite/<repo>`` on 2026-10-16) with no symlinks left
behind; a hard-coded default would have turned every cross-repo check into a
quiet skip.

An environment variable (``MIWORKBENCH_REPO``, ``MIFORGE_REPO``, ...) still
overrides the sibling default for a checkout somewhere else.

040 FR-040.13: miStudio became miWorkbench. ``workbench_repo()`` reads ``$MIWORKBENCH_REPO``, else
``../miWorkbench``, and the "a missing sibling fails" switch is one function,
``cross_repo_checks_required()``, true under ``MISERVER_REQUIRE_CROSS_REPO_CHECKS=1`` or the
suite-wide ``MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`` (the suite's own name, which stays, Q11/R17).
P7 (FR-040.40) retired the old spellings: ``$MISTUDIO_REPO``, the ``../miStudio`` folder (since P5
the suite's own folder), the ``MILLM_`` switch and the ``mistudio_repo`` name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.env_flags import flag

#: The miServer repository root (this file is ``<root>/tests/support/sibling_repos.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def sibling_repo(name: str, env_var: str | None = None) -> Path:
    """Return the path of sibling repo ``name``, or ``$env_var`` when it is set.

    The default is ``REPO_ROOT.parent / name``. The path is not checked for
    existence: each caller decides whether a missing sibling skips or fails
    (``cross_repo_checks_required()``).
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return Path(override)
    return REPO_ROOT.parent / name


def workbench_repo() -> Path:
    """The miWorkbench checkout (040 FR-040.13, TPD-18): ``$MIWORKBENCH_REPO``, else
    ``../miWorkbench``. Not checked for existence: callers decide."""
    return sibling_repo("miWorkbench", "MIWORKBENCH_REPO")


def cross_repo_checks_required() -> bool:
    """True when this environment declared every cross-repo guard mandatory (FR-040.13)."""
    return (flag("REQUIRE_CROSS_REPO_CHECKS") == "1"
            or os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1")


def require_or_skip(reason: str) -> None:
    """Skip loudly, or FAIL when ``cross_repo_checks_required()`` (the one copy of this body)."""
    if cross_repo_checks_required():
        pytest.fail(f"cross-repo checks are required (MISERVER_ or MISTUDIO_"
                    f"REQUIRE_CROSS_REPO_CHECKS=1) but this guard cannot run: {reason}")
    pytest.skip(reason)
