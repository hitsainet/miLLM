"""No compose file carries a default password (operator decision R28, 2026-11-11).

`docker-compose.yml` used to start PostgreSQL with `${POSTGRES_PASSWORD:-<a shared default>}` and built
the API's `DATABASE_URL` from the same default: every developer who never wrote a `.env` ran the same
published password. Compose now REQUIRES the value (`${POSTGRES_PASSWORD:?<what to do>}`): missing or
empty, it stops with a plain message and starts nothing. `.env.example` carries the key with no value.

What is held here, over every `docker-compose*.yml` in the repository:
* no reference to a secret-named variable has a default (`:-x`, `-x`) or an alternative (`:+x`), and
  no secret-named key or URL carries a literal credential;
* the database password is referenced in its required form, with a message that names the variable
  and the file to set it in, everywhere it is used (the database and the API's connection string);
* `.env.example` names `POSTGRES_PASSWORD` and gives it no value, and no URL in it holds a password;
* when the `docker` CLI is here, compose itself refuses the file with that message while the value is
  missing or empty, and accepts it once set (read-only: `config`, nothing is started).

Control NC-96 (0xcc/reviews/040_P4_controls_2026-11-11.md).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
#: A variable or key whose value is a credential. HF_TOKEN and SETTINGS_ENCRYPTION_KEY are optional and
#: written `${NAME:-}` (an EMPTY default: absent means absent), which is not a default value.
SECRET_NAME = re.compile(r"PASSWORD|PASSWD|SECRET|TOKEN|CREDENTIAL|(?<![A-Z])KEY(?![A-Z])")
#: `${NAME}`, `${NAME:-default}`, `${NAME-default}`, `${NAME:?message}`, `${NAME:+alt}` ...
REFERENCE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<op>:?[-?+])?(?P<arg>[^}]*)\}")
#: `scheme://user:password@host` with a literal (not interpolated) password.
URL_PASSWORD = re.compile(r"[a-z][a-z0-9+.-]*://[^\s:/@$]+:(?P<password>[^\s@$<][^\s@]*)@")
REQUIRED = "POSTGRES_PASSWORD"


def compose_files(root: Path = REPO) -> list[Path]:
    return sorted(root.glob("docker-compose*.yml")) + sorted(root.glob("docker-compose*.yaml"))


def _scalars(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _scalars(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from _scalars(item, path)
    elif value is not None:
        yield path, str(value)


def password_default_problems(text: str) -> list[str]:
    """Every way a compose document can carry a secret's value, as readable findings."""
    problems = []
    for path, scalar in _scalars(yaml.safe_load(text)):
        for ref in REFERENCE.finditer(scalar):
            if not SECRET_NAME.search(ref["name"].upper()):
                continue
            if ref["op"] in (":-", "-", ":+", "+") and ref["arg"]:
                problems.append(f"{path}: ${{{ref['name']}}} has the default {ref['arg']!r}")
        key = path.rsplit(".", 1)[-1]
        name, _, assigned = scalar.partition("=") if "=" in scalar and path.endswith("environment") else (key, "", scalar)
        if SECRET_NAME.search(name.upper()) and assigned and "${" not in assigned:
            problems.append(f"{path}: {name} is set to a literal value")
        for url in URL_PASSWORD.finditer(scalar):
            problems.append(f"{path}: a URL carries the literal password {url['password']!r}")
    return problems


def test_no_compose_file_carries_a_password_default():
    files = compose_files()
    assert [f.name for f in files] == ["docker-compose.otel.yml", "docker-compose.yml"], files
    for path in files:
        assert password_default_problems(path.read_text()) == [], path.name


def test_the_check_finds_each_form_of_a_default():
    """The checker on planted documents: each form is found, and the forms in use are not."""
    def found(environment) -> list[str]:
        return password_default_problems(yaml.safe_dump({"services": {"db": {"environment": environment}}}))

    assert found({"POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:-devpass}"})
    assert found({"POSTGRES_PASSWORD": "${POSTGRES_PASSWORD-devpass}"})
    assert found({"POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:+devpass}"})
    assert found({"POSTGRES_PASSWORD": "devpass"})
    assert found(["POSTGRES_PASSWORD=devpass"])
    assert found(["DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER:-u}:${POSTGRES_PASSWORD:-devpass}@db:5432/x"])
    assert found(["DATABASE_URL=postgresql+asyncpg://user:devpass@db:5432/x"])
    assert found({"GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD:-admin}"})
    assert found(["API_TOKEN=${API_TOKEN:-abc}"])
    assert not found({"POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:?set it in .env}"})
    assert not found(["HF_TOKEN=${HF_TOKEN:-}", "SETTINGS_ENCRYPTION_KEY=${SETTINGS_ENCRYPTION_KEY:-}"])
    assert not found(["DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER:-u}:${POSTGRES_PASSWORD:?m}@db:5432/x"])
    assert not found({"POSTGRES_USER": "${POSTGRES_USER:-millm}", "REDIS_URL": "redis://redis:6379/0"})


def test_the_database_password_is_required_everywhere_it_is_used():
    text = (REPO / "docker-compose.yml").read_text()
    uses = [(path, ref) for path, scalar in _scalars(yaml.safe_load(text))
            for ref in REFERENCE.finditer(scalar) if ref["name"] == REQUIRED]
    assert sorted(path for path, _ in uses) == [".services.api.environment", ".services.db.environment.POSTGRES_PASSWORD"]
    for path, ref in uses:
        assert ref["op"] == ":?", f"{path}: {REQUIRED} is not required (`:?`), so an empty value would start the stack"
        message = ref["arg"]
        assert REQUIRED in message and ".env" in message and len(message.split()) >= 8, (path, message)
    assert len({ref["arg"] for _, ref in uses}) == 1, "the two uses give different messages"


def test_env_example_names_the_password_and_gives_no_value():
    lines = [ln.strip() for ln in (REPO / ".env.example").read_text().splitlines()]
    assigned = [ln for ln in lines if ln.startswith(f"{REQUIRED}=")]
    assert assigned == [f"{REQUIRED}="], assigned
    for line in lines:
        if line and not line.startswith("#"):
            assert not URL_PASSWORD.search(line), f".env.example carries a password: {line}"
            name, _, value = line.partition("=")
            assert not (SECRET_NAME.search(name.upper()) and value), f".env.example gives {name} a value"


def _config(env_text: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env_file = tmp_path / "compose.env"
    env_file.write_text(env_text)
    environ = {k: v for k, v in os.environ.items() if k != REQUIRED}
    return subprocess.run(["docker", "compose", "--env-file", str(env_file), "-f", str(REPO / "docker-compose.yml"),
                           "config", "--quiet"], capture_output=True, text=True, env=environ, cwd=tmp_path, timeout=120)


def test_compose_itself_refuses_to_start_without_the_password(tmp_path):
    """The real CLI, read-only (`config` renders the file; nothing is pulled, built or started)."""
    if shutil.which("docker") is None or subprocess.run(
            ["docker", "compose", "version"], capture_output=True).returncode != 0:
        pytest.skip("the docker compose CLI is not installed here (the checks above read the file itself)")
    for label, env_text in (("missing", ""), ("empty, as .env.example ships it", (REPO / ".env.example").read_text())):
        refused = _config(env_text, tmp_path)
        assert refused.returncode != 0, f"compose accepted the file with the password {label}"
        assert f"Set {REQUIRED} in your .env file" in refused.stderr, (label, refused.stderr[-300:])
    accepted = _config(f"{REQUIRED}=chosen-by-the-developer\n", tmp_path)
    assert accepted.returncode == 0, accepted.stderr[-300:]
