"""P6 (041 FR-041.3, TID §3.6): every command that starts the server or a worker names a module that exists.

A `python -m <module>` or `uvicorn <module>:<attr>` naming a module that is not there fails only when the
container starts; nothing at build time notices. P8 renames the package, so each target is collected
from where it is written - the Dockerfile (`RUN` and `CMD`), the compose file, every Kubernetes
container's `command`/`args`, the workflows' `run:` blocks, the hardware pod specs, the test rule
runner's argv and `run()` in `<package>/main.py` - and resolved here.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import re
import shlex
from dataclasses import dataclass

import yaml

from tests.support.package_root import REPO_ROOT, package_name, package_path

#: `python -m` modules that are tools, not the package (they start the package; they are not it).
TOOLS = frozenset({"pytest", "ruff", "mypy", "alembic", "uvicorn", "pip", "build", "coverage"})
FLOOR = 6


@dataclass(frozen=True)
class Target:
    where: str
    module: str
    attr: str | None = None

    def __str__(self) -> str:
        return f"{self.where}: {self.module}{':' + self.attr if self.attr else ''}"


def _from_argv(where: str, argv: list[str]) -> list[Target]:
    out = []
    for i, word in enumerate(argv):
        runs_python = i == 0 or argv[i - 1].rsplit("/", 1)[-1].startswith("python")
        if word == "-m" and runs_python and i + 1 < len(argv) and argv[i + 1] not in TOOLS:
            out.append(Target(where, argv[i + 1]))
        elif word == "uvicorn" and i + 1 < len(argv) and ":" in argv[i + 1]:
            module, attr = argv[i + 1].split(":", 1)
            out.append(Target(where, module, attr))
    return out


def _from_shell(where: str, text: str) -> list[Target]:
    out = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line.startswith("#") or "python" not in line:
            continue
        for part in re.split(r"&&|\|\||;", line):
            try:
                words = shlex.split(part)
            except ValueError:
                words = part.split()
            out += _from_argv(where, words)
    return out


def _dockerfile() -> list[Target]:
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    out = []
    for n, line in enumerate(text.replace("\\\n", " ").splitlines(), 1):
        if line.startswith("CMD ") or line.startswith("ENTRYPOINT "):
            rest = line.split(" ", 1)[1].strip()
            argv = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
            out += _from_argv(f"Dockerfile {line.split()[0]}", argv)
        elif line.startswith("RUN "):
            out += _from_shell("Dockerfile RUN", line[4:])
    return out


def _containers(doc: object):
    if isinstance(doc, dict):
        if "image" in doc and ("command" in doc or "args" in doc):
            yield doc
        for value in doc.values():
            yield from _containers(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from _containers(value)


def _argv_of(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return shlex.split(value)
    return []


def _manifests() -> list[Target]:
    out = []
    files = [REPO_ROOT / "docker-compose.yml", *sorted((REPO_ROOT / "k8s").rglob("*.yaml")),
             *sorted((REPO_ROOT / "tests" / "hardware").glob("*.yaml"))]
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc is None:
                continue
            if path.name.startswith("docker-compose"):
                for name, svc in doc.get("services", {}).items():
                    for key in ("entrypoint", "command"):
                        if key in svc:
                            out += _from_argv(f"{rel} {name}.{key}", _argv_of(svc[key]))
                continue
            for container in _containers(doc):
                where = f"{rel} {container.get('name')}"
                argv = [*_argv_of(container.get("command")), *_argv_of(container.get("args"))]
                if "-c" in argv:  # `sh -c <script>`: the script is shell
                    out += _from_shell(where, "\n".join(argv[argv.index("-c") + 1:]))
                else:
                    out += _from_argv(where, argv)
    return out


def _workflows() -> list[Target]:
    out = []
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        jobs = yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs", {})
        for job in jobs.values():
            for step in job.get("steps", []):
                if isinstance(step.get("run"), str):
                    out += _from_shell(f".github/workflows/{path.name}", step["run"])
    return out


def _rule_runner() -> list[Target]:
    tree = ast.parse((REPO_ROOT / "tests" / "support" / "rule_runner.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            words = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            out += _from_argv("tests/support/rule_runner.py", words)
    return out


def _run_target() -> Target:
    tree = ast.parse(package_path("main.py").read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    call = next(n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "run" and isinstance(n.func.value, ast.Name) and n.func.value.id == "uvicorn")
    first = call.args[0]
    assert isinstance(first, ast.Constant) and isinstance(first.value, str), ast.dump(first)
    module, attr = first.value.split(":", 1)
    return Target(f"{package_name()}/main.py run()", module, attr)


def targets() -> list[Target]:
    return [*_dockerfile(), *_manifests(), *_workflows(), *_rule_runner(), _run_target()]


def _resolve(target: Target) -> str | None:
    """None when the target resolves, else why not."""
    try:
        spec = importlib.util.find_spec(target.module)
    except ModuleNotFoundError:  # a parent package that does not exist
        spec = None
    if spec is None:
        return "no such module"
    if target.attr is not None:
        value = getattr(importlib.import_module(target.module), target.attr, None)
        if not callable(value):
            return f"{target.attr} is missing or not callable"
    return None


def test_every_command_target_resolves():
    found = targets()
    assert len(found) >= FLOOR, f"only {len(found)} command targets found (floor {FLOOR}): {found}"
    sources = {t.where.split(" ")[0] for t in found}
    for expected in ("Dockerfile", "docker-compose.yml", "k8s/base/backend.yaml",
                     ".github/workflows/backend-tests.yml", "tests/support/rule_runner.py",
                     "tests/hardware/millm_hwtest_pod.yaml"):
        assert expected in sources, f"no command target collected from {expected}: {sorted(sources)}"
    broken = [f"{t} ({why})" for t in found if (why := _resolve(t))]
    assert not broken, "these commands name nothing that exists:\n" + "\n".join(broken)


def test_run_target_resolves():
    target = _run_target()
    assert _resolve(target) is None, target


def test_targets_name_the_package():
    wrong = [str(t) for t in targets() if t.module.split(".")[0] != package_name()]
    assert not wrong, f"these command targets do not name the package {package_name()!r}:\n" + "\n".join(wrong)


def test_the_collector_reads_each_form():
    """The collector is a scan; prove each form it claims to read is read (and `pytest -m "not slow"`
    is not a module)."""
    assert _from_argv("x", ["python", "-m", "pkg.mod", "fetch"]) == [Target("x", "pkg.mod")]
    assert _from_argv("x", ["-m", "pkg.mod"]) == [Target("x", "pkg.mod")]  # [sys.executable, "-m", ...]
    assert _from_argv("x", ["python", "-m", "uvicorn", "pkg.main:app", "--port", "1"]) == \
        [Target("x", "pkg.main", "app")]
    assert _from_argv("x", ["python", "-m", "pytest", "-m", "not slow"]) == []
    assert _from_shell("x", "cd /a && exec python -m uvicorn pkg.main:app --host 0.0.0.0") == \
        [Target("x", "pkg.main", "app")]
