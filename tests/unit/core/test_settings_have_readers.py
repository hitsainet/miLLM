"""ENH-003 decision D7: every `Settings` field has a reader in `millm/` (or an exemption here).

Eight fields were declared and never read — `REDIS_URL`, `MAX_DOWNLOAD_WORKERS`,
`MAX_LOAD_WORKERS`, `DOWNLOAD_TIMEOUT` and four `CIRCUIT_*` limits — so a deployment could set them
and nothing would change. They were deleted; this keeps a new one from appearing.

A READER is an attribute access named for the field (`settings.X`, `_settings.X`, `self.X` in the
config's own properties) or a string constant equal to it (`getattr(settings, self.TTL_SETTING)`
names its field as a string), anywhere in `millm/` outside the `Settings` class body. The AST is
read, not the text, so a comment mentioning a field is not a reader.
"""

from __future__ import annotations

import ast
from pathlib import Path

from millm.core.config import RETIRED_SETTINGS, Settings

REPO = Path(__file__).resolve().parents[3]

#: Fields read only outside `millm/`, each with where and why. Empty today.
EXEMPT: dict[str, str] = {}


def _settings_class_span(tree: ast.Module) -> tuple[int, int]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return node.lineno, node.end_lineno or node.lineno
    raise AssertionError("no Settings class in millm/core/config.py")


def names_read(source: str, skip: tuple[int, int] | None = None) -> set[str]:
    """Attribute names and string constants in `source`, outside the line span `skip`, except
    the field declarations themselves (annotated assignments in the class body)."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        in_class = skip is not None and line is not None and skip[0] <= line <= skip[1]
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and not in_class:
            found.add(node.value)
    return found


def _readers() -> set[str]:
    found: set[str] = set()
    for path in sorted((REPO / "millm").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if "/migrations/" in rel:
            continue
        source = path.read_text()
        skip = _settings_class_span(ast.parse(source)) if rel == "millm/core/config.py" else None
        found |= names_read(source, skip)
    return found


def test_the_scan_reads_real_code():
    """Guard the guard: known per-call readers are found, and a comment is not a reader."""
    readers = _readers()
    assert {"MAX_PENDING_REQUESTS", "GRACEFUL_UNLOAD_TIMEOUT", "CLUSTER_HUB_TAG"} <= readers
    assert "UNREAD_FIELD" not in names_read("x = 1  # settings.UNREAD_FIELD\n")
    assert "UNREAD_FIELD" in names_read("y = settings.UNREAD_FIELD\n")


def test_every_field_has_a_reader():
    readers = _readers()
    unread = sorted(set(Settings.model_fields) - readers - set(EXEMPT))
    assert not unread, (
        f"declared but never read in millm/: {unread}. Wire each, or delete it (and add its name "
        "to RETIRED_SETTINGS so an old .env still starts), or exempt it here with the reason."
    )


def test_the_exemptions_are_still_needed():
    stale = sorted(k for k in EXEMPT if k in _readers() or k not in Settings.model_fields)
    assert not stale, f"exemptions no longer needed: {stale}"


def test_retired_names_are_not_fields():
    assert not (RETIRED_SETTINGS & set(Settings.model_fields))
