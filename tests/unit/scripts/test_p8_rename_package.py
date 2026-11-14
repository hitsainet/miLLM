"""P12 (041 FR-041.2, FR-041.3, FR-041.17; TID §3.8): the rename tool on a fixture tree.

`scripts/p8_rename_package.py` rewrites every reference to the moved package that resolves, and nothing
else. The fixture (`tests/fixtures/p8_rename/`, a miniature repository right after `git mv oldpkg
newpkg`) holds one line per rule and one per form the tool must leave alone; it uses a synthetic name so
it adds no hit to any census record of the real package. Each test runs the real script in a subprocess
on a fresh git repository copied from the fixture.

The public mirror strips `scripts/` (and `0xcc/`): this file skips there, and only there (041 ID-2).
With `0xcc/` present and the tool missing it FAILS, so the private repository's CI always runs it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.package_root import REPO_ROOT

TOOL = REPO_ROOT / "scripts" / "p8_rename_package.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "p8_rename"
OLD, NEW = "oldpkg", "newpkg"
MIRROR_REASON = "the mirror strips scripts/ (041 ID-2): the rename tool is not there"


def mirror_skip_reason(root: Path) -> str | None:
    """Skip only on the public mirror, recognised by the absence of `0xcc/`, which it strips with
    `scripts/`. A private tree without the tool is a failure, not a skip."""
    return MIRROR_REASON if not (root / "0xcc").is_dir() else None


if mirror_skip_reason(REPO_ROOT):
    pytest.skip(mirror_skip_reason(REPO_ROOT), allow_module_level=True)


# ── harness ─────────────────────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                          check=True, capture_output=True, text=True).stdout


def make_tree(dest: Path, extra: dict[str, str] | None = None) -> Path:
    assert TOOL.is_file(), f"{TOOL} is missing from a private tree (0xcc/ present): it must run here"
    assert FIXTURE.is_dir(), FIXTURE
    root = dest / "repo"
    shutil.copytree(FIXTURE, root)
    for path in list(root.rglob("*.py_")):
        path.rename(path.with_suffix(".py"))
    for rel, text in (extra or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def run_tool(root: Path, *flags: str) -> tuple[int, dict, str]:
    report = root.parent / f"report{len(list(root.parent.glob('report*.json')))}.json"
    proc = subprocess.run([sys.executable, str(TOOL), "--old", OLD, "--new", NEW, "--report", str(report),
                           "--root", str(root), *flags], cwd=root, capture_output=True, text=True, timeout=300)
    body = json.loads(report.read_text()) if report.is_file() else {}
    return proc.returncode, body, proc.stdout + proc.stderr


def digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and ".git" not in p.relative_to(root).parts}


def line(root: Path, rel: str, tag: str) -> str:
    hits = [ln for ln in (root / rel).read_text(encoding="utf-8").splitlines() if ln.endswith(tag)]
    assert len(hits) == 1, (tag, hits)
    return hits[0]


@pytest.fixture(scope="module")
def renamed(tmp_path_factory):
    root = make_tree(tmp_path_factory.mktemp("p8"))
    code, report, out = run_tool(root)
    assert code == 0, out + json.dumps(report, indent=1)
    return root, report


# ── each rule rewrites ──────────────────────────────────────────────────────────────

SAMPLE = "tests/sample.py"
REWRITTEN = {
    "import_top": "import newpkg  # case: import_top",
    "import_dotted": "import newpkg.core.config  # case: import_dotted",
    "import_from": "from newpkg.services import svc  # case: import_from",
    "import_from_pkg": "from newpkg import __version__  # case: import_from_pkg",
    "import_lazy": "    from newpkg.core.config import settings  # case: import_lazy",
    "module_name": "PATHS = newpkg.__path__  # case: module_name",
    "package_prefix": 'MODS = pkgutil.walk_packages(newpkg.__path__, "newpkg.")  # case: package_prefix',
    "module_string": 'TARGET = "newpkg.services.svc.helper"  # case: module_string',
    "module_attr": 'APP = "newpkg.main:app"  # case: module_attr',
    "dash_m": 'CMD = "run python -m newpkg.rules.runner_worker fetch <dir>"  # case: dash_m',
    "path_string": 'REL = "newpkg/core/config.py"  # case: path_string',
    "fstring": 'NAME = f"newpkg.services.{svc.__name__}"  # case: fstring',
    "logger": 'LOG = logging.getLogger("newpkg.rule_runner")  # case: logger',
    "comment_path": "X = 1  # see newpkg/services/svc.py and newpkg.core.config (case: comment_path)",
    "semicolon_import": 'ONE_LINE = "import json; from newpkg.core.config import settings"  # case: semicolon_import',
    "fstring_dash_m": 'FCMD = f"python -m newpkg.rules.runner_worker fetch {svc}"  # case: fstring_dash_m',
    "app_path": 'COPIES = ("newpkg/", "/app/newpkg/")  # case: app_path',
    "tool_arg": 'LINT = "python -m ruff check newpkg --select E9"  # case: tool_arg',
    "script_import": "import sys, os, newpkg  # case: script_import",
    "script_from": "from newpkg.core import config  # case: script_from",
    "script_quoted": 'loaded = [m for m in sys.modules if m in ("newpkg.rules.runner_worker",)]  # case: script_quoted',
}
#: Rules on the other files (FTDD §6.8): (file, a unique fragment of the line, the line after).
REWRITTEN_ELSEWHERE = {
    "docstring": (SAMPLE, '"""Cites', '"""Cites newpkg/core/config.py and newpkg.services.svc in prose."""'),
    "package_logger": ("newpkg/rules/runner_worker.py", "getLogger", 'logger = logging.getLogger("newpkg.rule_runner")'),
    "package_run_target": ("newpkg/main.py", "uvicorn.run", '    uvicorn.run("newpkg.main:app")'),
    "toml_name": ("pyproject.toml", "name =", 'name = "newpkg"'),
    "toml_include": ("pyproject.toml", "include", 'include = ["newpkg*"]'),
    "toml_script": ("pyproject.toml", ".main:run", 'newpkg = "newpkg.main:run"'),
    "toml_source": ("pyproject.toml", "source", 'source = ["newpkg"]'),
    "toml_omit": ("pyproject.toml", "core/*", '    "newpkg/core/*",'),
    "toml_isort": ("pyproject.toml", "known-first-party", 'known-first-party = ["newpkg"]'),
    "alembic": ("alembic.ini", "script_location", "script_location = newpkg/core"),
    "dockerfile_copy": ("Dockerfile", "COPY oldpkg/ /app", "COPY newpkg/ /app/newpkg/"),
    "dockerfile_run": ("Dockerfile", "RUN python", "RUN python -m newpkg.rules.runner_worker fetch /app/bundle"),
    "dockerfile_cmd": ("Dockerfile", "CMD", 'CMD ["python", "-m", "uvicorn", "newpkg.main:app"]'),
    "compose_mount": ("docker-compose.yml", ":ro", "      - ./newpkg:/app/newpkg:ro"),
    "compose_entrypoint": ("docker-compose.yml", "entrypoint", '    entrypoint: ["python", "-m", "newpkg.rules.runner_worker"]'),
    "readme": ("README.md", "uvicorn", "Run `uvicorn newpkg.main:app`; the code is in newpkg/services/svc.py. "
                                      "Coverage: `pytest --cov=newpkg`."),
    "docs_page": ("docs/guide.md", "See", "See newpkg/core/config.py and `newpkg.core.config`."),
    "manual_page": ("manual/docs/dev.md", "See", "See newpkg/core/config.py and `newpkg.core.config`."),
}


@pytest.mark.parametrize("cls", sorted(REWRITTEN) + sorted(REWRITTEN_ELSEWHERE))
def test_rewrites_each_class(renamed, cls):
    root, _report = renamed
    if cls in REWRITTEN:
        assert line(root, SAMPLE, f"case: {cls})" if cls == "comment_path" else f"case: {cls}") == REWRITTEN[cls]
    else:
        rel, fragment, expected = REWRITTEN_ELSEWHERE[cls]
        hits = [ln for ln in (root / rel).read_text(encoding="utf-8").splitlines()
                if fragment.replace(OLD, NEW) in ln or fragment in ln]
        assert expected in hits, (cls, hits)


# ── each never-rewritten form stays ─────────────────────────────────────────────────

KEPT = {
    "bare": 'BARE = "oldpkg"  # case: bare',
    "keyword": "KW = dict(oldpkg=1)  # case: keyword",
    "attribute": "ATTR = LOG.oldpkg  # case: attribute",
    "span": 'SPAN = "oldpkg.queue.wait"  # case: span',
    "kind": 'KIND = "oldpkg.steering-set/v1"  # case: kind',
    "field": "    oldpkg: dict  # case: field",
    "local_bind": '    oldpkg = {"a": 1}  # case: local_bind',
    "local": '    return oldpkg["a"]  # case: local',
    "pragma": 'KEPT = "oldpkg.services.svc.helper"  # p8: keep (case: pragma)',
    "comment_bare_dir": "Y = 2  # prose about the bare directory oldpkg/ stays (case: comment_bare_dir)",
    "prose_string": 'MSG = "the oldpkg package stays a word here"  # case: prose_string',
}
KEPT_ELSEWHERE = {
    "dockerfile_user": ("Dockerfile", "RUN groupadd --gid 1000 oldpkg && useradd --uid 1000 --gid 1000 oldpkg"),
    "dockerfile_user_directive": ("Dockerfile", "USER oldpkg"),
}


@pytest.mark.parametrize("form", sorted(KEPT) + sorted(KEPT_ELSEWHERE))
def test_leaves_each_never_rewritten_form(renamed, form):
    root, report = renamed
    if form in KEPT:
        tag = f"case: {form})" if form in ("pragma", "comment_bare_dir") else f"case: {form}"
        assert line(root, SAMPLE, tag) == KEPT[form]
        if form == "pragma":
            assert any("case: pragma" in k for k in report["kept_by_pragma"]), report["kept_by_pragma"]
    else:
        rel, expected = KEPT_ELSEWHERE[form]
        assert expected in (root / rel).read_text(encoding="utf-8").splitlines(), (form, rel)


def test_refuses_a_deny_listed_edit(renamed):
    """`"oldpkg.core.frozen_name"` resolves (`core` is a module), so the module-path rule would rewrite it;
    the fixture's frozen record DR-99 matches it, so the deny list refuses the edit and reports it."""
    root, report = renamed
    assert line(root, SAMPLE, "case: plant") == 'PLANT = "oldpkg.core.frozen_name"  # case: plant'
    assert [(r["file"], r["record"]) for r in report["refused"]] == [(SAMPLE, "DR-99")], report["refused"]


def test_invariant_failure_exits_2(tmp_path):
    """An edit the deny list does not see (it does not overlap the record's match) but that changes a
    frozen record's count: the tool restores every file and exits 2."""
    export = (FIXTURE / "docs" / "legacy-identifiers.md").read_text(encoding="utf-8")
    record = json.dumps({"section": "entry", "id": "DR-97", "status": "frozen",
                         "pattern": r"(?<=oldpkg\.)services\.svc\.helper\b"})
    export = export.replace("\n```\n", "\n" + record + "\n```\n", 1)
    assert "DR-97" in export
    root = make_tree(tmp_path, {"docs/legacy-identifiers.md": export})
    before = digest(root)
    code, report, out = run_tool(root)
    assert code == 2, out
    assert "DR-97" in report["error"], report
    assert digest(root) == before, "a failed run must leave every file as it was"


def test_bare_path_segment_exits_3(tmp_path):
    root = make_tree(tmp_path, {"tests/walk.py": 'from pathlib import Path\nROOT = Path(".")\nPKG = ROOT / "oldpkg"\n'})
    before = digest(root)
    code, report, out = run_tool(root)
    assert code == 3, out
    assert "tests/walk.py:3" in report["error"], report
    assert digest(root) == before


def test_second_run_changes_nothing(renamed, tmp_path):
    root, _ = renamed
    before = digest(root)
    code, report, out = run_tool(root, "--check")
    assert code == 0, out + json.dumps(report.get("files_changed"))
    code, report, out = run_tool(root)
    assert code == 0 and report["files_changed"] == [], out
    assert digest(root) == before
    # and --check does see work when there is some
    fresh = make_tree(tmp_path)
    code, report, _ = run_tool(fresh, "--check")
    assert code == 1 and SAMPLE in report["files_changed"]
    assert digest(fresh)[SAMPLE] == hashlib.sha256((FIXTURE / "tests" / "sample.py_").read_bytes()).hexdigest()


def test_export_file_is_never_touched(renamed):
    """The export is census output and the deny list itself (CF-20): the root-level and nested
    `docs/legacy-identifiers*.md` stay byte-identical, while pages beside them with the same text are
    rewritten (so the exclusion, not a missed rule, is what keeps them)."""
    root, _ = renamed
    for rel in ("docs/legacy-identifiers.md", "docs/legacy-identifiers.public.md",
                "manual/docs/legacy-identifiers.md"):
        assert (root / rel).read_bytes() == (FIXTURE / rel).read_bytes(), rel
    assert "newpkg/core/config.py" in (root / "docs" / "guide.md").read_text()
    assert "newpkg/core/config.py" in (root / "manual" / "docs" / "dev.md").read_text()


def test_skips_only_on_the_mirror(tmp_path):
    mirror, private, built = tmp_path / "mirror", tmp_path / "private", tmp_path / "built"
    for root in (mirror, private, built):
        root.mkdir()
    (private / "0xcc").mkdir()
    (built / "0xcc").mkdir()
    (built / "scripts").mkdir()
    (built / "scripts" / "p8_rename_package.py").write_text("")
    assert mirror_skip_reason(mirror) == MIRROR_REASON
    assert mirror_skip_reason(private) is None, "a private tree without the tool must fail, not skip"
    assert mirror_skip_reason(built) is None
    assert mirror_skip_reason(REPO_ROOT) is None and TOOL.is_file()
