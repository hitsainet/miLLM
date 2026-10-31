"""Every Python dependency is pinned exactly, and the image and CI install the same set (2026-10-21).

Before this, pyproject.toml carried open ranges (`fastapi>=0.115.0`, `sqlalchemy>=2.0.36`,
`redis>=5.0.0` ...) and nothing pinned the transitive set, so each image build installed whatever
was newest that day: the pod ran fastapi 0.158.0 while CI had tested whatever its runner resolved
on its own day. Now:

* `pyproject.toml` pins every direct dependency, every extra and the build backend with `==`;
* `constraints.txt` (THE LOCK) pins the complete install set, transitive packages included, and
  agrees with pyproject on every name both carry;
* the Dockerfile and every CI job install with `-c constraints.txt`, and run
  `docker/check_installed_against_constraints.py`, which fails on anything installed that the lock
  does not pin or pins at another version;
* this process's own environment matches the lock (so a dev venv or the image's test run that
  drifted says so).

Cross-repo halves (skip loudly without the sibling, fail under MILLM_REQUIRE_CROSS_REPO_CHECKS=1):
the lock follows the suite baseline except for the recorded exceptions, and the dev `mcp` pin
equals miStudio's, because miLLM installs mcp ONLY to load miStudio's tool registry.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tomllib
from pathlib import Path

import pytest

from tests.support.sibling_repos import REPO_ROOT, sibling_repo

LOCK = REPO_ROOT / "constraints.txt"
CHECKER = REPO_ROOT / "docker" / "check_installed_against_constraints.py"
DOCKERFILE = REPO_ROOT / "Dockerfile"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: Where the lock departs from the suite baseline (miSuite/0xcc/dependency_baseline_2026-10-21.md),
#: and why. Recorded in 0xcc/reviews/deps_sync_2026-10-21.md. Anything not listed must equal it.
BASELINE_EXCEPTIONS = {
    # sympy 1.14.0 (the latest release; torch's dependency) requires mpmath<1.4, so the latest
    # mpmath (1.4.1, the baseline's) cannot be installed beside it.
    "mpmath": "1.3.0",
    # Not held: ahead. alembic 1.20.1 was released 2026-10-22 03:24 UTC, after the baseline was
    # written (it says 1.20.0); the latest-release rule takes it. Remove once the baseline rises.
    "alembic": "1.20.1",
}


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_installed_against_constraints", CHECKER)
    assert spec and spec.loader, CHECKER
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _name(requirement: str) -> str:
    return checker.normalise(re.split(r"[\[<>=!~;\s]", requirement.strip(), maxsplit=1)[0])


def _pyproject_requirements() -> list[tuple[str, str]]:
    """`(where, requirement)` for every requirement pyproject.toml declares."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    found = [("dependencies", d) for d in data["project"]["dependencies"]]
    for extra, deps in data["project"].get("optional-dependencies", {}).items():
        found.extend((f"extra {extra}", d) for d in deps)
    found.extend(("build-system", d) for d in data["build-system"]["requires"])
    return found


EXACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,_-]+\])?==[^\s,;<>=!~*]+$")


# ------------------------------------------------------------------------------------------------
# The declarations
# ------------------------------------------------------------------------------------------------


def test_every_pyproject_requirement_is_one_exact_pin():
    loose = [
        f"{where}: {req}"
        for where, req in _pyproject_requirements()
        if not EXACT.match(req.replace(" ", ""))
    ]
    assert not loose, (
        "every dependency must be pinned `==` (a range re-resolves on the next image build): "
        + "; ".join(loose)
    )


def test_the_lock_is_exact_pins_only_and_names_each_package_once():
    pins = checker.read_constraints(LOCK)  # raises on a range, a bare name or a duplicate
    assert len(pins) > 100, f"the lock names only {len(pins)} packages; is it the full set?"


def test_pyproject_and_the_lock_agree_on_every_name_both_carry():
    pins = checker.read_constraints(LOCK)
    missing, differ = [], []
    for where, req in _pyproject_requirements():
        name, version = _name(req), req.replace(" ", "").split("==", 1)[1]
        if name not in pins:
            missing.append(f"{name} ({where})")
        elif pins[name] != version:
            differ.append(f"{name}: pyproject {version}, constraints.txt {pins[name]}")
    assert not missing, f"pinned in pyproject.toml but absent from constraints.txt: {missing}"
    assert not differ, f"pyproject.toml and constraints.txt disagree: {differ}"


# ------------------------------------------------------------------------------------------------
# The installers: the image and CI must both install from the lock
# ------------------------------------------------------------------------------------------------


def _run_blocks(text: str) -> list[str]:
    """Each Dockerfile RUN instruction, continuation lines joined."""
    joined = re.sub(r"\\\n", " ", text)
    return [line for line in joined.splitlines() if line.lstrip().startswith("RUN ")]


def _pip_installs(command: str) -> list[str]:
    """Each `pip install ...` segment of a shell command, up to the next `&&`, `||` or `;`."""
    return [m.group(0) for m in re.finditer(r"pip install[^&|;\n]*", command)]


def _quoted_specs(install: str) -> list[str]:
    return re.findall(r'"([A-Za-z0-9][A-Za-z0-9._\[\]-]*[<>=!~][^"]*)"', install)


def test_every_dockerfile_pip_install_uses_the_lock_and_names_only_its_pins():
    pins = checker.read_constraints(LOCK)
    installs = [i for block in _run_blocks(DOCKERFILE.read_text()) for i in _pip_installs(block)]
    assert len(installs) >= 6, installs  # wheel, ., -e . fallback, cuda extra, llama-cpp, dev tools
    for install in installs:
        assert "-c constraints.txt" in install, f"Dockerfile installs without the lock: {install}"
        for spec in _quoted_specs(install):
            assert EXACT.match(spec), f"Dockerfile installs a range: {spec}"
            name, version = _name(spec), spec.split("==", 1)[1]
            assert pins.get(name) == version, (
                f"Dockerfile installs {spec}; the lock says {pins.get(name)}"
            )


def test_the_dockerfile_copies_the_lock_before_installing_and_checks_after():
    # Comments mention `pip install` too; only instructions count.
    text = "\n".join(
        line for line in DOCKERFILE.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    runtime = text.split("AS development", 1)[0]
    first_install = runtime.index("pip install")
    assert re.search(r"^COPY [^\n]*constraints\.txt", runtime[:first_install], re.M), (
        "constraints.txt must be copied in before the first pip install"
    )
    assert re.search(
        r"^COPY docker/check_installed_against_constraints\.py ", runtime[:first_install], re.M
    )
    check = "RUN python docker/check_installed_against_constraints.py constraints.txt"
    assert check in runtime, "the image build no longer runs the lock check"
    assert runtime.index(check) > runtime.rindex("pip install"), (
        "the lock check must run AFTER the runtime stage's last pip install"
    )
    development = text.split("AS development", 1)[1]
    assert development.rindex(
        "python docker/check_installed_against_constraints.py"
    ) > development.rindex("pip install"), (
        "the development stage installs tools after its lock check"
    )


def _workflow_texts() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(WORKFLOWS.glob("*.yml"))}


def test_every_ci_pip_install_is_locked():
    """`-c` with the lock (or the cross-repo job's lock minus mcp), or only exact pins the lock agrees
    with (the CPU torch pre-install, the mirror gate's pyyaml)."""
    pins = checker.read_constraints(LOCK)
    seen = 0
    for name, text in _workflow_texts().items():
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for install in _pip_installs(line):
                seen += 1
                if re.search(
                    r"-c (miLLM/)?constraints\.txt\b|-c \"\$RUNNER_TEMP/constraints-no-mcp\.txt\"",
                    install,
                ):
                    continue
                specs = re.findall(r"\"([^\"]+)\"|\b([A-Za-z0-9_.-]+==[^\s\"]+)", install)
                specs = [a or b for a, b in specs]
                assert specs, f"{name}: unlocked install: {install}"
                for spec in specs:
                    assert EXACT.match(spec), f"{name}: unlocked install of {spec!r}: {install}"
                    assert pins.get(_name(spec)) == spec.split("==", 1)[1], (
                        f"{name}: installs {spec}; the lock says {pins.get(_name(spec))}"
                    )
    assert seen >= 7, f"found only {seen} pip installs in the workflows; is the parser blind?"


def test_the_cross_repo_job_strips_only_mcp_from_the_lock():
    text = (WORKFLOWS / "backend-tests.yml").read_text()
    assert (
        "grep -vE '^mcp(-types)?==' miLLM/constraints.txt > \"$RUNNER_TEMP/constraints-no-mcp.txt\""
        in text
    )


def test_both_full_install_jobs_check_the_environment_after_installing():
    text = (WORKFLOWS / "backend-tests.yml").read_text()
    pair = 'pip install -c constraints.txt -e ".[dev]"\n          python docker/check_installed_against_constraints.py constraints.txt'
    # 039 S3: the rule-runner job installs the same package the unit job does.
    assert text.count(pair) == 3, (
        "the unit, schema and rule-runner jobs must each install from the lock and check it"
    )
    assert (
        text.count(
            "cache-dependency-path: |\n            pyproject.toml\n            constraints.txt"
        )
        == 3
    )


def test_the_image_rebuilds_when_the_lock_changes():
    text = (WORKFLOWS / "docker-images.yml").read_text()
    backend = re.search(r"backend\)\s+echo '([^']+)'", text)
    assert backend, "docker-images.yml no longer lists the backend image's paths"
    pattern = re.compile(backend.group(1))
    for path in (
        "constraints.txt",
        "docker/check_installed_against_constraints.py",
        "pyproject.toml",
    ):
        assert pattern.search(path), f"a change to {path} would not rebuild the backend image"


# ------------------------------------------------------------------------------------------------
# The environment this test runs in (CI's venv, the image, the dev venv)
# ------------------------------------------------------------------------------------------------


def test_this_environment_matches_the_lock():
    found = checker.problems(checker.read_constraints(LOCK), checker.installed())
    assert not found, "this environment drifted from constraints.txt:\n" + "\n".join(found)


def test_the_checker_flags_unpinned_and_mismatched_and_ignores_local_labels():
    pins = {"torch": "2.14.1", "fastapi": "0.143.3"}
    assert (
        checker.problems(pins, {"torch": "2.14.1+cpu", "fastapi": "0.143.3", "pip": "26.0"}) == []
    )
    assert checker.problems(pins, {"torch": "2.14.1+cu130", "fastapi": "0.158.0"}) == [
        "fastapi is installed at 0.158.0; constraints.txt pins 0.143.3"
    ]
    assert checker.problems(
        pins, {"torch": "2.14.1", "fastapi": "0.143.3", "starlette": "1.9.0"}
    ) == ["starlette==1.9.0 is installed but not pinned in constraints.txt"]
    assert checker.normalise("Typing_Extensions") == "typing-extensions"


@pytest.mark.parametrize(
    "line", ["fastapi>=0.115.0", "fastapi", "fastapi==0.143.3,<1", "fastapi ~= 0.143"]
)
def test_the_checker_refuses_a_lock_line_that_is_not_an_exact_pin(tmp_path, line):
    lock = tmp_path / "constraints.txt"
    lock.write_text(f"# header\n{line}\n")
    with pytest.raises(ValueError, match="not an exact"):
        checker.read_constraints(lock)


def test_the_checker_refuses_a_duplicate(tmp_path):
    lock = tmp_path / "constraints.txt"
    lock.write_text("Jinja2==3.1.6\njinja2==3.1.5\n")
    with pytest.raises(ValueError, match="pinned twice"):
        checker.read_constraints(lock)


# ------------------------------------------------------------------------------------------------
# Cross-repo
# ------------------------------------------------------------------------------------------------


def _sibling_file(path: Path, what: str) -> Path:
    if not path.is_file():
        message = f"{path} not found: {what} NOT compared"
        if os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return path


def test_the_lock_follows_the_suite_baseline():
    suite = Path(os.environ.get("MISUITE_REPO") or REPO_ROOT.parent)
    baseline = _sibling_file(
        suite / "0xcc" / "dependency_baseline_2026-10-21.md", "the suite baseline"
    )
    section = baseline.read_text().split("## Python", 1)[1].split("## JavaScript", 1)[0]
    wanted = dict(re.findall(r"^- ([A-Za-z0-9._-]+)==(\S+)$", section, re.M))
    assert len(wanted) > 100, "the baseline's Python list did not parse"
    pins = checker.read_constraints(LOCK)
    differ = []
    for name, version in wanted.items():
        name = checker.normalise(name)
        expected = BASELINE_EXCEPTIONS.get(name, version)
        if name in pins and pins[name] != expected:
            differ.append(f"{name}: lock {pins[name]}, baseline {version}")
    assert not differ, (
        f"the lock departs from the suite baseline without a recorded reason: {differ}"
    )
    stale = [n for n, v in BASELINE_EXCEPTIONS.items() if pins.get(n) != v]
    assert not stale, f"BASELINE_EXCEPTIONS no longer describe the lock: {stale}"


def test_the_dev_mcp_pin_is_mistudios():
    requirements = _sibling_file(
        sibling_repo("miStudio", "MISTUDIO_REPO") / "backend" / "requirements.txt",
        "miStudio's mcp pin",
    )
    theirs = re.search(r"^mcp==(\S+)", requirements.read_text(), re.M)
    assert theirs, "miStudio's requirements.txt no longer pins mcp exactly"
    assert checker.read_constraints(LOCK)["mcp"] == theirs.group(1), (
        f"miStudio pins mcp {theirs.group(1)}; miLLM's dev pin must follow (it exists only to load "
        "miStudio's registry) — move pyproject.toml, constraints.txt and BASELINE_EXCEPTIONS together"
    )


# ------------------------------------------------------------------------------------------------
# OpenTelemetry moves in lockstep (038 FR-38.24, FTDD §3)
# ------------------------------------------------------------------------------------------------

#: Every package the telemetry export needs (FTDD §3). Each must be in the lock; the direct ones
#: also in pyproject (the agreement test above holds the versions equal).
OTEL_SET = (
    "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-exporter-otlp-proto-common", "opentelemetry-proto",
    "opentelemetry-semantic-conventions", "opentelemetry-exporter-http-transport",
    "opentelemetry-exporter-otlp-common", "protobuf", "googleapis-common-protos",
    "requests", "urllib3", "charset-normalizer",
)


def test_otel_lockstep():
    """Every `opentelemetry-*` pin versioned 1.x equals the API's, and every 0.xbN pin equals
    semantic-conventions'. The SDK requires `opentelemetry-api==` its own version exactly, so a
    pair out of step is a resolver failure in the image build — this turns it into a named red."""
    pins = checker.read_constraints(LOCK)
    missing = [n for n in OTEL_SET if checker.normalise(n) not in pins]
    assert not missing, f"the telemetry set is not fully locked: {missing}"
    api = pins["opentelemetry-api"]
    semconv = pins["opentelemetry-semantic-conventions"]
    assert api.startswith("1.") and re.match(r"^0\.\d+b\d+$", semconv), (api, semconv)
    otel = {n: v for n, v in pins.items() if n.startswith("opentelemetry-")}
    assert len(otel) >= 8, otel
    off = {n: v for n, v in otel.items()
           if v != (api if v.startswith("1.") else semconv)}
    assert not off, (
        f"opentelemetry packages out of lockstep (1.x must equal api {api}, 0.xb must equal "
        f"semantic-conventions {semconv}): {off}"
    )
    direct = {_name(r) for _, r in _pyproject_requirements()}
    assert {"opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http"} <= direct


# ------------------------------------------------------------------------------------------------
# JavaScript: the admin UI and the manual
# ------------------------------------------------------------------------------------------------

JS_PROJECTS = ("admin-ui", "manual")


@pytest.mark.parametrize("project", JS_PROJECTS)
def test_every_js_dependency_is_one_exact_version_and_the_lockfile_agrees(project):
    """No `^`/`~`: the version in package.json is the one installed, and the lockfile's root entry
    says so too (a hand-edited package.json without `npm install` leaves `npm ci` refusing in CI)."""
    import json

    manifest = json.loads((REPO_ROOT / project / "package.json").read_text())
    lock = json.loads((REPO_ROOT / project / "package-lock.json").read_text())["packages"]
    loose, unlocked = [], []
    for section in ("dependencies", "devDependencies"):
        for name, spec in manifest.get(section, {}).items():
            if not re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?", spec):
                loose.append(f"{name}@{spec}")
            elif lock.get(f"node_modules/{name}", {}).get("version") != spec:
                unlocked.append(
                    f"{name}: package.json {spec}, lock {lock.get(f'node_modules/{name}', {}).get('version')}"
                )
            if lock[""].get(section, {}).get(name) != spec:
                unlocked.append(f"{name}: lock root says {lock[''].get(section, {}).get(name)}")
    assert not loose, f"{project}: not exact versions: {loose}"
    assert not unlocked, f"{project}: package.json and package-lock.json disagree: {unlocked}"


def test_every_js_install_is_npm_ci():
    """CI and the admin UI image install from the lockfile alone (`npm ci`), never `npm install`,
    which may rewrite it - so the image and CI cannot install different trees."""
    sources = {"admin-ui/Dockerfile": (REPO_ROOT / "admin-ui" / "Dockerfile").read_text()}
    sources.update(_workflow_texts())
    installs = 0
    for name, text in sources.items():
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"\bnpm (install|i|add)\b", line):
                pytest.fail(f"{name}: `{line.strip()}` installs without the lockfile")
            installs += len(re.findall(r"\bnpm ci\b", line))
    assert installs >= 4, f"found only {installs} `npm ci`; is the scan blind?"
