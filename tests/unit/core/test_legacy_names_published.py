"""T11 (040 FR-040.8, FR-040.11; CF-3, CF-14, CF-24; the DR-29 exercise): the server's hand-kept alias
tables equal the census export's `env` and `host` records.

Two independent sources on purpose: `census.py` writes the export, `miserver/core/legacy_names.py` is
typed by hand, and this compares them, so a row added to one side only is red. A table generated
from the export would agree with any census mistake by construction.

Since P7 (FR-040.40) the export's `env` and `host` records are `retired`: `HOST_ALIASES` is empty,
every old env name is in `RETIRED_ENV`, and no deployable file names an old host.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from miserver.core import legacy_names
from tests.support import legacy_identifiers as li
from tests.support.package_root import package_path, package_rel

ROOT = Path(__file__).resolve().parents[3]
MODULE = package_path("core", "legacy_names.py")
#: Old hosts as they appear in deployable files (manifests, workflows, code, integrations).
OLD_HOST = re.compile(r"(?<![\w.-])millm-backend\.millm(?:\.svc(?:\.cluster\.local)?)?\b"
                      r"|(?<![\w.-])(?:k8s-)?millm(?:-otel)?\.hitsai\.(?:local|net)\b")


def _export():
    return li.load()


def test_host_aliases_equal_the_export():
    export = _export()
    mine = {(a.old, a.new, a.observed) for a in legacy_names.HOST_ALIASES}
    theirs = {(r["old"], r["new"], r["observed"]) for r in export.hosts() if r["status"] == "transition"}
    retired = {r["old"] for r in export.hosts() if r["status"] == "retired"}
    assert theirs or retired, "the export carries no host record: the comparison would be vacuous"
    assert not {m[0] for m in mine} & retired, "a host the export retired is still in HOST_ALIASES"
    # The public variant (the mirror) withholds the `.local` hosts by stand-in (CF-2): compared by
    # name, never silently. In the full variant nothing is withheld.
    held = {m for m in mine if export.is_withheld("host", m[0])}
    mine -= held
    assert mine == theirs, {"only in legacy_names.py": mine - theirs, "only in the export": theirs - mine}


def test_env_aliases_equal_the_export():
    export = _export()
    mine = {(a.old, a.new) for a in legacy_names.ENV_ALIASES if a.old not in legacy_names.RETIRED_ENV
            and not export.is_withheld("env", a.old)}
    theirs = {(r["old"], r["new"]) for r in export.env() if r["status"] == "transition"}
    assert export.env(), "the export carries no env record: the comparison would be vacuous"
    assert mine == theirs, {"only in legacy_names.py": mine - theirs, "only in the export": theirs - mine}


def test_retired_env_names_are_retired_in_the_export():
    export = _export()
    retired = {r["old"] for r in export.env() if r["status"] == "retired"}
    mine = {old for old in legacy_names.RETIRED_ENV if not export.is_withheld("env", old)}
    assert mine == retired
    assert {a.old for a in legacy_names.ENV_ALIASES} >= set(legacy_names.RETIRED_ENV), (
        "a retired name keeps its pair in ENV_ALIASES, so `retired_but_set` can report it")


def test_the_tables_are_hand_kept():
    """Standard library only, no file opened or read, and both tables are literal tuples of
    `EnvAlias(...)` / `HostAlias(...)` calls with constant arguments (CF-3, CF-25 m13); `RETIRED_ENV`
    a literal tuple of strings. `HOST_ALIASES` may be empty once P7 retires every host."""
    tree = ast.parse(MODULE.read_text())
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= stdlib, ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in stdlib, node.module
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in {"open", "read_text", "read_bytes", "load", "loads", "import_module",
                                "__import__", "exec", "eval"}, f"legacy_names.py calls {name}()"
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tables[node.target.id] = node.value
    for name, ctor in (("ENV_ALIASES", "EnvAlias"), ("HOST_ALIASES", "HostAlias")):
        value = tables.get(name)
        assert isinstance(value, ast.Tuple), f"{name} must be a literal tuple"
        assert value.elts or name == "HOST_ALIASES", f"{name} must not be empty"
        for elt in value.elts:
            assert isinstance(elt, ast.Call) and getattr(elt.func, "id", None) == ctor, ast.dump(elt)
            args = [*elt.args, *(k.value for k in elt.keywords)]
            assert all(isinstance(a, ast.Constant) for a in args), ast.dump(elt)
    retired = tables.get("RETIRED_ENV")
    assert isinstance(retired, ast.Tuple), "RETIRED_ENV must be a literal tuple"
    assert all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in retired.elts)


def test_no_deployable_file_names_an_old_host():
    """P7 (FR-040.40): with `HOST_ALIASES` empty, no deployable file (manifests, workflows, code,
    integrations, compose) may name an old host: each one found would be an address that no longer
    answers, or one the server no longer watches. Before P7 every one had to be an alias (T11, M8)."""
    files = subprocess.run(
        ["git", "ls-files", "--", "k8s", "integrations", package_rel(), "admin-ui/src", "admin-ui/vite.config.ts",
         "nginx", ".github", "docker-compose*.yml", ":(exclude)" + package_rel("core", "legacy_names.py"),
         ":(exclude,glob)admin-ui/src/**/__tests__/**"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert len(files) > 100, f"only {len(files)} files listed: the scan is failing open"
    aliases = {a.old for a in legacy_names.HOST_ALIASES}
    seen: dict[str, str] = {}
    for rel in files:
        path = ROOT / rel
        if path.is_file():
            for host in OLD_HOST.findall(path.read_text(errors="replace")):
                seen.setdefault(host.lower(), rel)
    found = {h: f for h, f in seen.items() if h not in aliases}
    assert not found, f"old hosts in deployable files with no alias: {found}"


def test_the_old_host_scan_finds_each_form(tmp_path):
    """The scan above must not pass by matching nothing: every form it is meant to catch is found."""
    forms = ["k8s-millm.hitsai.local", "k8s-millm.hitsai.net", "millm.hitsai.local", "millm.hitsai.net",
             "k8s-millm-otel.hitsai.local", "millm-backend.millm", "millm-backend.millm.svc",
             "millm-backend.millm.svc.cluster.local"]
    text = " ".join(f"http://{f}:8000/x" for f in forms)
    assert OLD_HOST.findall(text) == forms
