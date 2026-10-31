"""Features 31-33 settings (addendum §2.4): defaults, bounds, and every place that names them.

`WEIGHTS_TOKENS` is a raw string that pydantic never validates (a pydantic error echoes its
input, D-05), hidden from `repr`, and in Kubernetes comes from an OPTIONAL secret key, so its
absence never blocks the pod (031 review D13).
"""

from pathlib import Path

import pytest

from millm.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "WEIGHTS_TOKENS": "",
    "WEIGHTS_HOLD_TIMEOUT_S": 5.0,
    # D45: 10x the real-trainer p99.9 (71.0 ms, miForge f11w2-r2) = 710, next 100 ms = 800.
    "WEIGHTS_COMMIT_WATCHDOG_MS": 800,
    "WEIGHTS_HEARTBEAT_TIMEOUT_S": 30.0,
    "WEIGHTS_HEARTBEAT_INTERVAL_HINT_S": 5.0,
    "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S": 60.0,
    "RETRY_AFTER_WEIGHTS_TORN_S": 30,
    "WEIGHTS_CHECKPOINT_DIR": "/app/checkpoints",
    "WEIGHTS_CHECKPOINT_READ_ROOTS": "",
    "WEIGHTS_LINEAGE_MAX_DEPTH": 10000,
    "WEIGHTS_SAVE_FREE_SPACE_FACTOR": 1.10,
    "WEIGHTS_SAVE_OVERHEAD_MB": 64,
}


@pytest.fixture
def clean_env(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_load(clean_env):
    settings = Settings(_env_file=None)
    assert {name: getattr(settings, name) for name in DEFAULTS} == DEFAULTS
    assert settings.weights_checkpoint_read_roots == []


def test_a_malformed_token_value_never_fails_settings(clean_env, monkeypatch):
    """D-05: the gate fails closed at call time; startup is never the place it fails."""
    for raw in ("{not json", "[1, 2", '[{"sha256": "plaintext-token"}]'):
        monkeypatch.setenv("WEIGHTS_TOKENS", raw)
        assert Settings(_env_file=None).WEIGHTS_TOKENS == raw


def test_the_token_value_is_not_in_repr(clean_env, monkeypatch):
    monkeypatch.setenv("WEIGHTS_TOKENS", "a-secret-looking-value")
    assert "a-secret-looking-value" not in repr(Settings(_env_file=None))


@pytest.mark.parametrize("overrides", [
    {"WEIGHTS_HOLD_TIMEOUT_S": 0},
    {"WEIGHTS_COMMIT_WATCHDOG_MS": 0},
    {"WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S": 0},
    {"WEIGHTS_HEARTBEAT_INTERVAL_HINT_S": 0},
    {"WEIGHTS_HEARTBEAT_INTERVAL_HINT_S": 30.0, "WEIGHTS_HEARTBEAT_TIMEOUT_S": 30.0},
    {"RETRY_AFTER_WEIGHTS_TORN_S": 0},
    {"WEIGHTS_LINEAGE_MAX_DEPTH": 0},
    {"WEIGHTS_SAVE_FREE_SPACE_FACTOR": 0.99},
    {"WEIGHTS_SAVE_OVERHEAD_MB": -1},
    {"WEIGHTS_CHECKPOINT_READ_ROOTS": "/data/a,relative/b"},
])
def test_bounds_are_refused(clean_env, overrides):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)


def test_read_roots_parse_as_a_list(clean_env):
    settings = Settings(_env_file=None, WEIGHTS_CHECKPOINT_READ_ROOTS=" /data/a , ,/mnt/b ")
    assert settings.weights_checkpoint_read_roots == ["/data/a", "/mnt/b"]


def test_env_example_lists_each_setting():
    text = (ROOT / ".env.example").read_text()
    for name in DEFAULTS:
        assert f"#{name}=" in text, name
    assert "#WEIGHTS_TOKENS=[]" in text


def test_the_configuration_page_documents_each_setting():
    text = (ROOT / "manual/docs/reference/configuration.md").read_text()
    for name in DEFAULTS:
        assert f"| `{name}` |" in text, name


def test_k8s_reads_tokens_from_an_optional_secret_key():
    yaml = pytest.importorskip("yaml")
    docs = list(yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    env = {e["name"]: e for e in deployment["spec"]["template"]["spec"]["containers"][0]["env"]}
    ref = env["WEIGHTS_TOKENS"]["valueFrom"]["secretKeyRef"]
    assert ref["optional"] is True
    assert ref["key"] == "WEIGHTS_TOKENS"
    assert "value" not in env["WEIGHTS_TOKENS"]


def _backend_pod():
    yaml = pytest.importorskip("yaml")
    docs = list(yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    return deployment["spec"]["template"]["spec"]


def test_k8s_puts_checkpoints_on_the_data_volume_and_creates_the_directory():
    """032 FTASKS 10.2: saves survive a pod restart because they live on the mounted volume."""
    pod = _backend_pod()
    container = next(c for c in pod["containers"] if c["name"] == "backend")
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["WEIGHTS_CHECKPOINT_DIR"] == "/data/checkpoints"
    mounts = {m["mountPath"] for m in container["volumeMounts"]}
    assert any(env["WEIGHTS_CHECKPOINT_DIR"].startswith(path.rstrip("/") + "/") for path in mounts)
    init = " ".join(pod["initContainers"][0]["command"])
    assert "/data/checkpoints" in init


def test_k8s_mounts_miforges_checkpoints_read_only_as_a_read_root():
    """032 FTASKS 10.2 (operator 2026-10-16): miForge's checkpoints are a READ root (D-21).

    The read root must be exactly the mount path of a read-only mount, backed by the node path
    miForge's local PersistentVolume uses, and never the directory miLLM writes saves to.
    """
    pod = _backend_pod()
    container = next(c for c in pod["containers"] if c["name"] == "backend")
    env = {e["name"]: e.get("value") for e in container["env"]}
    roots = [r.strip() for r in env["WEIGHTS_CHECKPOINT_READ_ROOTS"].split(",") if r.strip()]
    assert roots == ["/mnt/miforge-checkpoints"]
    assert not env["WEIGHTS_CHECKPOINT_DIR"].startswith(roots[0])
    mount = next(m for m in container["volumeMounts"] if m["mountPath"] == roots[0])
    assert mount.get("readOnly") is True
    volume = next(v for v in pod["volumes"] if v["name"] == mount["name"])
    assert volume["hostPath"]["path"] == "/data/miforge/checkpoints"
    assert volume["hostPath"]["type"] == "DirectoryOrCreate"


def test_compose_mounts_a_named_checkpoints_volume():
    """032 FTASKS 10.3."""
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    api = compose["services"]["api"]
    assert "WEIGHTS_CHECKPOINT_DIR=/app/checkpoints" in api["environment"]
    assert "checkpoints:/app/checkpoints" in api["volumes"]
    assert "checkpoints" in compose["volumes"]


def test_the_image_creates_the_checkpoint_directory_for_the_service_user():
    """A named volume over a missing directory is root-owned, so the millm user could not save."""
    text = (ROOT / "Dockerfile").read_text()
    run = next(block for block in text.split("\nRUN ") if "/app/checkpoints" in block)
    assert "mkdir -p" in run and "chown -R millm:millm" in run
    assert run.count("/app/checkpoints") == 2


def test_k8s_reads_the_hf_token_from_an_optional_secret_key():
    """HF_TOKEN reaches the backend from millm-secrets, optional, and never as a literal value.

    Wired 2026-10-17 (operator): both downloaders fall back to HF_TOKEN, but the deployment never
    passed it, so the documented server-wide token could not be configured.
    """
    yaml = pytest.importorskip("yaml")
    docs = list(yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    env = {e["name"]: e for e in deployment["spec"]["template"]["spec"]["containers"][0]["env"]}
    ref = env["HF_TOKEN"]["valueFrom"]["secretKeyRef"]
    assert ref == {"name": "millm-secrets", "key": "HF_TOKEN", "optional": True}
    assert "value" not in env["HF_TOKEN"]


def test_k8s_reads_the_settings_encryption_key_from_an_optional_secret_key():
    """Feature 34: the key that encrypts the saved HF token. Optional (absent, the pod starts and a
    save is refused), from millm-secrets, never a literal value in git."""
    yaml = pytest.importorskip("yaml")
    docs = list(yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    env = {e["name"]: e for e in deployment["spec"]["template"]["spec"]["containers"][0]["env"]}
    ref = env["SETTINGS_ENCRYPTION_KEY"]["valueFrom"]["secretKeyRef"]
    assert ref == {"name": "millm-secrets", "key": "SETTINGS_ENCRYPTION_KEY", "optional": True}
    assert "value" not in env["SETTINGS_ENCRYPTION_KEY"]


def test_every_document_states_the_watchdog_default_the_code_runs():
    """D45 moved the watchdog 200 -> 800. A document still saying 200 tells an operator the
    trainer is cut off at a quarter of the real limit. miForge mirrors this value and pins
    it to this file's field default from its side."""
    default = Settings.model_fields["WEIGHTS_COMMIT_WATCHDOG_MS"].default
    env = (ROOT / ".env.example").read_text()
    assert f"#WEIGHTS_COMMIT_WATCHDOG_MS={default}\n" in env
    config_page = (ROOT / "manual/docs/reference/configuration.md").read_text()
    assert f"| `WEIGHTS_COMMIT_WATCHDOG_MS` | `{default}` |" in config_page
    api_page = (ROOT / "manual/docs/api/weights.md").read_text()
    assert f"| `WEIGHTS_COMMIT_WATCHDOG_MS` | {default} ms |" in api_page
