"""Feature 26's settings load with the FTID §9 defaults; `.env.example`, k8s and compose name them."""

from pathlib import Path

import pytest

from millm.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "BATCH_FILES_DIR": "/app/batch_files",
    "BATCH_MAX_ROWS": 50000,
    "BATCH_MAX_FILE_BYTES": 209715200,
    "BATCH_MAX_LINE_BYTES": 1048576,
    "BATCH_PACK_DEFAULT": True,
    "BATCH_CHUNK_ROWS": 8,
    "BATCH_PACK_MAX_ROWS": 16,
    "BATCH_PACK_MAX_TOKENS": 16384,
    "BATCH_MAX_COMPLETION_WINDOW_HOURS": 168,
    "BATCH_FILE_RETENTION_DAYS": 30,
    "BATCH_LEASE_TTL_S": 900,
    "BATCH_WAIT_POLL_S": 10,
    "BATCH_PROGRESS_MIN_INTERVAL_S": 1,
    "BATCH_ERRORS_SHOWN": 100,
    "BATCH_RETENTION_INTERVAL_S": 3600,
    "PROBE_MAX_BATCH_EVENTS_PER_PROBE": 50000,
    "SENSING_MAX_BATCH_EVENTS_PER_CLUSTER": 50000,
    "CIRCUIT_SENSING_MAX_BATCH_EVENTS_PER_CIRCUIT": 50000,
}


def test_defaults_load(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert {name: getattr(settings, name) for name in DEFAULTS} == DEFAULTS


def test_env_example_lists_each_setting():
    text = (ROOT / ".env.example").read_text()
    for name, value in DEFAULTS.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        assert f"#{name}={rendered}" in text, name


def test_the_batch_lease_ttl_must_be_one_feature_29_grants():
    with pytest.raises(ValueError):
        Settings(BATCH_LEASE_TTL_S=7201)


def test_k8s_puts_batch_files_on_the_data_volume_and_creates_the_directory():
    yaml = pytest.importorskip("yaml")
    docs = list(yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    env = {e["name"]: e.get("value") for e in pod["containers"][0]["env"]}
    assert env["BATCH_FILES_DIR"] == "/data/batch_files"
    init = " ".join(pod["initContainers"][0]["command"])
    assert "/data/batch_files" in init


def test_compose_mounts_a_named_batch_files_volume():
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    api = compose["services"]["api"]
    assert "BATCH_FILES_DIR=/app/batch_files" in api["environment"]
    assert "batch_files:/app/batch_files" in api["volumes"]
    assert "batch_files" in compose["volumes"]
