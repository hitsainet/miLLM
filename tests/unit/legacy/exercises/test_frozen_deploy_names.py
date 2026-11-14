"""Exercises for the frozen deployment names (040 FR-040.7): DR-19 (database and role `millm`, R6),
DR-40 (rule-runner seccomp profile and socket path), DR-41 (Compose named volumes), DR-42 (backup
dump prefix `millm-*.dump`) and DR-43 (dashboard uid `millm-probes`).

Each names something that already exists on a host: a database, a node-installed seccomp profile, a
Docker volume holding data, dump files the retention must keep pruning, a dashboard bookmarked by
uid. Manifests are read PARSED (YAML/JSON), and the backup retention runs for real in bash.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]

#: Every Compose named volume's `name:` (DR-41), pinned by value.
COMPOSE_VOLUMES = {
    "docker-compose.yml": {
        "millm-postgres-data", "millm-redis-data", "millm-model-cache", "millm-sae-cache",
        "millm-batch-files", "millm-checkpoints", "millm-frontend-node-modules", "millm-nginx-logs",
        "millm-rule-runner-socket",
    },
    "docker-compose.otel.yml": {"millm-otel-lgtm-data"},
}


def _docs(rel: str) -> list[dict]:
    return [d for d in yaml.safe_load_all((REPO / rel).read_text()) if d]


def _cronjob_container() -> dict:
    (job,) = [d for d in _docs("k8s/base/db-backup.yaml") if d.get("kind") == "CronJob"]
    (container,) = job["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"]
    return container


def test_database_and_role_stay_millm():
    env = {e["name"]: e.get("value") for e in _cronjob_container()["env"]}
    assert (env["PGUSER"], env["PGDATABASE"]) == ("millm", "millm")
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    db_env = compose["services"]["db"]["environment"]
    assert db_env["POSTGRES_USER"] == "${POSTGRES_USER:-millm}"
    assert db_env["POSTGRES_DB"] == "${POSTGRES_DB:-millm}"


def test_rule_runner_profile_and_socket_path():
    assert (REPO / "k8s" / "seccomp" / "millm-rule-runner.json").is_file()
    (deploy,) = [d for d in _docs("k8s/base/backend.yaml") if d.get("kind") == "Deployment"]
    containers = deploy["spec"]["template"]["spec"]["containers"]
    profiles = [c.get("securityContext", {}).get("seccompProfile", {}).get("localhostProfile")
                for c in containers]
    assert "profiles/millm-rule-runner.json" in profiles
    sockets = {e["value"] for c in containers for e in c.get("env", [])
               if e["name"] == "RULE_RUNNER_SOCKET"}
    assert sockets == {"/run/millm-rules/runner.sock"}
    # The code's own defaults (review M10): the API's setting and the sidecar's reader.
    from miserver.core.config import Settings
    from miserver.rules.runner_worker import WorkerConfig

    assert Settings.model_fields["RULE_RUNNER_SOCKET"].default == "/run/millm-rules/runner.sock"
    assert str(WorkerConfig.from_env({}).socket_path) == "/run/millm-rules/runner.sock"
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    for service in ("api", "rule-runner"):
        assert "RULE_RUNNER_SOCKET=/run/millm-rules/runner.sock" in compose["services"][service]["environment"]


def test_compose_volume_names():
    for rel, expected in COMPOSE_VOLUMES.items():
        volumes = yaml.safe_load((REPO / rel).read_text()).get("volumes") or {}
        assert {v["name"] for v in volumes.values() if v and "name" in v} == expected, rel


def test_backup_dump_prefix_and_retention(tmp_path):
    script = _cronjob_container()["command"][-1]
    assert 'target="$BACKUP_DIR/millm-$stamp.dump"' in script
    start = script.index("# ── Retention")
    end = script.index('cat > "$BACKUP_DIR/RESTORE.md"')
    retention = script[start:end]
    for i in range(4):
        (tmp_path / f"millm-2026010{i}T000000Z.dump").write_text("x")
        subprocess.run(["touch", "-d", f"2026-01-0{i + 1}", str(tmp_path / f"millm-2026010{i}T000000Z.dump")],
                       check=True)
    (tmp_path / "other-20260101T000000Z.dump").write_text("keep me")
    out = subprocess.run(["bash", "-euo", "pipefail", "-c", "say() { echo \"$*\"; }\n" + retention],
                         env={"BACKUP_DIR": str(tmp_path), "KEEP": "2", "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["millm-20260102T000000Z.dump", "millm-20260103T000000Z.dump",
                    "other-20260101T000000Z.dump"], left


def test_dashboard_uid():
    dashboard = json.loads((REPO / "k8s/components/otel-reference/dashboard.json").read_text())
    assert dashboard["uid"] == "millm-probes"
