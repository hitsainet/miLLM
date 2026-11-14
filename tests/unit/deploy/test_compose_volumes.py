"""T32: new names in compose and CI, and the one thing that must not move (040 FR-040.37, PD-8).

* A compose named volume keeps its `name:`. A new name is a new, EMPTY volume: the database, the
  model cache and the checkpoints would silently start from nothing (DR-41).
* Containers and the network are named `miserver-*`; the otel file names the main file's network
  (its `default` network has the same `name:`), so the two must agree (FTDD C-6).
* Workflows set a test switch under its `MISERVER_*` name only (the old spelling was retired at
  P7, FR-040.40: `tests/unit/core/test_legacy_names_env.py`).

Control NC-89 (0xcc/reviews/040_P4_controls_2026-11-11.md): rename `millm-postgres-data`'s `name:`
in docker-compose.yml -> `test_compose_volume_names_unchanged` red.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests.support import env_flags

#: The old switch prefix (040 FR-040.37; retired at P7, FR-040.40).
OLD_PREFIX = "MILLM_"

REPO = Path(__file__).resolve().parents[3]
COMPOSE = ("docker-compose.yml", "docker-compose.otel.yml")

#: compose key -> the Docker volume name it has always had.
VOLUMES = {
    "docker-compose.yml": {
        "postgres_data": "millm-postgres-data",
        "redis_data": "millm-redis-data",
        "model_cache": "millm-model-cache",
        "sae_cache": "millm-sae-cache",
        "batch_files": "millm-batch-files",
        "checkpoints": "millm-checkpoints",
        "frontend_node_modules": "millm-frontend-node-modules",
        "nginx_logs": "millm-nginx-logs",
        "rule_runner_socket": "millm-rule-runner-socket",
    },
    "docker-compose.otel.yml": {"otel_lgtm_data": "millm-otel-lgtm-data"},
}


def _compose(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())


def test_compose_volume_names_unchanged():
    for rel, expected in VOLUMES.items():
        volumes = _compose(rel).get("volumes") or {}
        named = {key: (spec or {}).get("name") for key, spec in volumes.items()}
        assert named == expected, f"{rel}: a volume's name changed, was added or lost its `name:`"


def test_compose_containers_and_network_say_miserver():
    seen = 0
    for rel in COMPOSE:
        document = _compose(rel)
        for key, service in document["services"].items():
            name = service.get("container_name")
            assert name and name.startswith("miserver-"), (rel, key, name)
            seen += 1
        networks = {key: (spec or {}).get("name") for key, spec in document["networks"].items()}
        assert set(networks.values()) == {"miserver-network"}, (rel, networks)
    assert seen >= 8, f"only {seen} services read: the walk is failing open"


def _env_blocks(node, path=""):
    """Every `env:` mapping in a workflow (workflow, job and step level)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "env" and isinstance(value, dict):
                yield path, value
            else:
                yield from _env_blocks(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _env_blocks(item, f"{path}[{index}]")


def test_workflows_set_only_miserver_flags():
    old_set, new_set = [], set()
    assign = re.compile(rf"\b{OLD_PREFIX}[A-Z0-9_]+=")
    files = sorted((REPO / ".github" / "workflows").glob("*.yml"))
    assert len(files) >= 7, [f.name for f in files]
    for path in files:
        text = path.read_text()
        for where, env in _env_blocks(yaml.safe_load(text)):
            for key in env:
                if key.startswith(OLD_PREFIX):
                    old_set.append(f"{path.name}{where}: {key}")
                if key.startswith(env_flags.NEW):
                    new_set.add((path.name, key))
        # a `run:` script exporting the old spelling counts too (comments do not)
        for number, line in enumerate(text.splitlines(), 1):
            if not line.lstrip().startswith("#") and assign.search(line):
                old_set.append(f"{path.name}:{number}: {line.strip()}")
    assert not old_set, "a workflow sets an old-name switch (040 FR-040.37):\n  " + "\n  ".join(old_set)
    expected = {("backend-tests.yml", env_flags.new_name(name)) for name in (
        "TEST_PG_BASE", "REQUIRE_PG", "REQUIRE_RUNNER", "REQUIRE_CROSS_REPO_CHECKS")}
    assert expected <= new_set, f"the CI switches are gone: {sorted(expected - new_set)}"
