"""Locate this repository and its sibling suite repos for cross-repo tests.

The suite repos (miStudio, miLLM, miDataworks, miForge) are checked out side by
side, so a sibling is found from THIS repository's own root — never from an
absolute path. They moved once already (``/home/x-sean/app/<repo>`` to
``/home/x-sean/app/miSuite/<repo>`` on 2026-10-16) with no symlinks left
behind; a hard-coded default would have turned every cross-repo check into a
quiet skip.

An environment variable (``MISTUDIO_REPO``, ``MIFORGE_REPO``, ...) still
overrides the sibling default for a checkout somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The miLLM repository root (this file is ``<root>/tests/support/sibling_repos.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def sibling_repo(name: str, env_var: str | None = None) -> Path:
    """Return the path of sibling repo ``name``, or ``$env_var`` when it is set.

    The default is ``REPO_ROOT.parent / name``. The path is not checked for
    existence: each caller decides whether a missing sibling skips or fails
    (``MILLM_REQUIRE_CROSS_REPO_CHECKS=1``).
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return Path(override)
    return REPO_ROOT.parent / name


def mistudio_repo() -> Path:
    """The miStudio checkout: ``$MISTUDIO_REPO``, else the sibling ``../miStudio``."""
    return sibling_repo("miStudio", "MISTUDIO_REPO")
