"""The `rule-runner` sidecar and its Compose twin, as deployed (039 FTASKS 17.5, 17.6, 17.8; TD39-33).

Parsed from `k8s/base/backend.yaml` (what ArgoCD deploys) and `docker-compose.yml`: the container
exists, runs the worker (not the image entrypoint, which would migrate a database), shares ONLY the
socket directory with the backend, holds no secret and no service-account token, and carries every
security key FTDD §6.8 names. The seccomp profile it names refuses inet sockets and allows Unix ones.

Control (039 controls record): delete any one of these keys from the manifest → red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from millm.core.config import Settings

REPO = Path(__file__).resolve().parents[2]
SOCKET_DIR = "/run/millm-rules"
WORKER = ["python", "-m", "millm.rules.runner_worker"]
#: The ONLY environment names the runner may be given. Anything else could be a secret.
RUNNER_ENV = {"RULE_RUNNER_SOCKET", "RULE_RUNNER_BUNDLE_DIR", "NVIDIA_VISIBLE_DEVICES"}
SECRETS = {"DATABASE_URL", "HF_TOKEN", "SETTINGS_ENCRYPTION_KEY", "WEIGHTS_TOKENS",
           "TELEMETRY_HEADERS", "REDIS_URL", "SECRET_KEY"}
AF_UNIX, AF_INET, AF_INET6 = 1, 2, 10


def _pod() -> dict:
    docs = [d for d in yaml.safe_load_all((REPO / "k8s/base/backend.yaml").read_text()) if d]
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    return deployment["spec"]["template"]["spec"]


def _container(pod: dict, name: str) -> dict:
    return next(c for c in pod["containers"] if c["name"] == name)


def test_the_backend_stays_first_and_the_runner_is_beside_it():
    names = [c["name"] for c in _pod()["containers"]]
    assert names == ["backend", "rule-runner"]


def test_the_runner_runs_the_worker_from_the_same_image():
    pod = _pod()
    runner, backend = _container(pod, "rule-runner"), _container(pod, "backend")
    assert runner["image"] == backend["image"]
    assert runner["command"] == WORKER
    assert "args" not in runner


def test_the_runner_holds_no_secret():
    runner = _container(_pod(), "rule-runner")
    env = runner.get("env") or []
    assert {e["name"] for e in env} <= RUNNER_ENV
    assert not ({e["name"] for e in env} & SECRETS)
    assert all("valueFrom" not in e for e in env)
    assert "envFrom" not in runner
    assert {e["name"]: e["value"] for e in env}["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert "nvidia.com/gpu" not in json.dumps(runner.get("resources") or {})


def test_no_service_account_token_is_mounted():
    assert _pod()["automountServiceAccountToken"] is False


@pytest.mark.parametrize("key,value", [
    ("runAsNonRoot", True),
    ("readOnlyRootFilesystem", True),
    ("allowPrivilegeEscalation", False),
])
def test_the_runner_security_context(key, value):
    assert _container(_pod(), "rule-runner")["securityContext"][key] is value


def test_every_capability_is_dropped():
    assert _container(_pod(), "rule-runner")["securityContext"]["capabilities"] == {
        "drop": ["ALL"]}


def test_the_seccomp_profile_is_the_repositorys():
    profile = _container(_pod(), "rule-runner")["securityContext"]["seccompProfile"]
    assert profile == {"type": "Localhost", "localhostProfile": "profiles/millm-rule-runner.json"}
    assert (REPO / "k8s" / "seccomp" / "millm-rule-runner.json").is_file()


def _denied_families(profile: dict) -> set[int]:
    out = set()
    for rule in profile["syscalls"]:
        if "socket" in rule["names"] and rule["action"] == "SCMP_ACT_ERRNO":
            for arg in rule.get("args") or []:
                if arg["index"] == 0 and arg["op"] == "SCMP_CMP_EQ":
                    out.add(arg["value"])
    return out


def test_the_profile_refuses_inet_sockets_and_allows_unix_ones():
    profile = json.loads((REPO / "k8s/seccomp/millm-rule-runner.json").read_text())
    denied = _denied_families(profile)
    assert {AF_INET, AF_INET6} <= denied
    assert AF_UNIX not in denied
    unconditional = {n for r in profile["syscalls"] if not r.get("args")
                     and r["action"] == "SCMP_ACT_ERRNO" for n in r["names"]}
    assert "socket" not in unconditional, "AF_UNIX must stay possible: the worker listens on one"


def test_only_the_socket_directory_is_shared():
    pod = _pod()
    runner, backend = _container(pod, "rule-runner"), _container(pod, "backend")
    assert runner["volumeMounts"] == [{"name": "rule-runner-socket", "mountPath": SOCKET_DIR}]
    assert {"name": "rule-runner-socket", "mountPath": SOCKET_DIR} in backend["volumeMounts"]
    volume = next(v for v in pod["volumes"] if v["name"] == "rule-runner-socket")
    assert "emptyDir" in volume and "hostPath" not in volume


def test_both_containers_name_the_same_socket_and_it_is_the_default():
    pod = _pod()
    sockets = {c["name"]: {e["name"]: e.get("value") for e in c.get("env") or []}
               ["RULE_RUNNER_SOCKET"] for c in pod["containers"]}
    default = Settings.model_fields["RULE_RUNNER_SOCKET"].default
    assert sockets == {"backend": default, "rule-runner": default}
    assert default.startswith(SOCKET_DIR + "/")


def test_the_hostipc_component_note_is_recorded():
    assert "hostipc" in (REPO / "k8s/base/backend.yaml").read_text()


# ── Docker Compose ────────────────────────────────────────────────────────────────────────────


def _compose() -> dict:
    return yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]


def test_compose_runs_the_worker_as_its_own_service():
    service = _compose()["rule-runner"]
    assert service["entrypoint"] == WORKER
    assert service["read_only"] is True and service["cap_drop"] == ["ALL"]
    assert service["network_mode"] == "none"
    assert service["user"] == "1001:1000"
    assert "no-new-privileges:true" in service["security_opt"]
    seccomp = next(o for o in service["security_opt"] if o.startswith("seccomp="))
    assert (REPO / seccomp.split("=", 1)[1]).resolve() == (
        REPO / "k8s/seccomp/millm-rule-runner.json").resolve()
    assert "deploy" not in service, "the runner must not reserve a GPU"


def test_compose_runner_env_holds_no_secret_and_shares_only_the_socket():
    services = _compose()
    runner, api = services["rule-runner"], services["api"]
    names = {e.split("=", 1)[0] for e in runner["environment"]}
    assert names <= RUNNER_ENV and not (names & SECRETS)
    assert runner["volumes"] == [f"rule_runner_socket:{SOCKET_DIR}"]
    assert f"rule_runner_socket:{SOCKET_DIR}" in api["volumes"]
    api_env = dict(e.split("=", 1) for e in api["environment"])
    assert api_env["RULE_RUNNER_SOCKET"] == Settings.model_fields["RULE_RUNNER_SOCKET"].default


# ── Review R1 (S3) ──────────────────────────────────────────────────────────────────────────────


def _service() -> dict:
    docs = [d for d in yaml.safe_load_all((REPO / "k8s/base/backend.yaml").read_text()) if d]
    return next(d for d in docs if d["kind"] == "Service" and d["metadata"]["name"]
                == "millm-backend")


def test_a_runner_that_cannot_start_never_takes_the_backend_out_of_service():
    """R1-H1: a pod is Ready only when EVERY container is. A missing node seccomp profile
    (CreateContainerError), a crash-looping or OOM-killed worker would otherwise drop miLLM's only
    endpoint — the whole API down for a sidecar D41 says a chat never waits on. The backend
    container has no readiness probe, so readiness never said more than "every container runs";
    publishing not-ready addresses keeps chats served while code rules stay pending."""
    assert _service()["spec"].get("publishNotReadyAddresses") is True
    backend = _container(_pod(), "backend")
    assert "readinessProbe" not in backend, (
        "a backend readiness probe is ignored by the Service now: gate traffic another way")


def _profile() -> dict:
    return json.loads((REPO / "k8s/seccomp/millm-rule-runner.json").read_text())


def _unconditional_errno(profile: dict) -> set[str]:
    return {n for r in profile["syscalls"] if not r.get("args")
            and r["action"] == "SCMP_ACT_ERRNO" for n in r["names"]}


def test_the_profile_closes_the_other_ways_to_a_socket_and_a_namespace():
    """R1-H2: the AF_INET refusal is only as good as the paths around it. io_uring creates sockets
    without the `socket` system call (IORING_OP_SOCKET, kernel 5.19+; the node runs 6.8); 32-bit
    and x32 entry points are not needed by anything in the image; a new user namespace (clone
    flags, clone3, the new mount API) widens the kernel surface. Checked live under Docker: the
    worker serves under this profile, AF_INET and io_uring_setup are refused (review record)."""
    profile = _profile()
    assert profile["architectures"] == ["SCMP_ARCH_X86_64"]
    denied = _unconditional_errno(profile)
    assert {"io_uring_setup", "io_uring_enter", "io_uring_register"} <= denied
    assert {"unshare", "setns", "mount", "fsopen", "fsmount", "move_mount", "open_tree",
            "fsconfig", "fspick", "mount_setattr"} <= denied
    clone3 = [r for r in profile["syscalls"] if r["names"] == ["clone3"]]
    assert clone3 and clone3[0]["errnoRet"] == 38, "clone3 must be ENOSYS so libc falls back"
    masked = {a["value"] for r in profile["syscalls"] if r["names"] == ["clone"]
              for a in r.get("args") or [] if a["op"] == "SCMP_CMP_MASKED_EQ"
              and a["value"] == a["valueTwo"] and r["action"] == "SCMP_ACT_ERRNO"}
    namespaces = {0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000,
                  0x40000000}
    assert namespaces <= masked


def test_the_runner_is_not_the_backends_uid():
    """R1-M1: containers of one pod share its IPC namespace and /dev/shm (and the host's, with the
    hostipc component). CUDA and torch create their files there owner-only (0600) as uid 1000; a
    worker under the same uid could open them after an escape. The runner keeps miLLM's GROUP so
    the backend can connect to its 0660 socket."""
    pod = _pod()
    runner = _container(pod, "rule-runner")["securityContext"]
    assert runner["runAsUser"] != 1000 and runner["runAsUser"] >= 1000
    assert runner["runAsGroup"] == 1000
    service = _compose()["rule-runner"]
    uid, gid = service["user"].split(":")
    assert int(uid) == runner["runAsUser"] and int(gid) == 1000


def _dockerfile() -> str:
    return (REPO / "Dockerfile").read_text()


def test_the_image_carries_the_verified_snapshot_where_the_runner_reads_it():
    """R1-M2: without the snapshot every code rule stays pending for good, quietly (the worker
    starts `bundle_mismatch`). The image fetches it BY DIGEST and precompiles it as root, before
    `USER millm`, so the process that runs guests cannot rewrite the native `.cwasm` (018 R1-L5),
    into the directory both manifests give the runner."""
    text = _dockerfile()
    bundle = {e["name"]: e["value"] for e in _container(_pod(), "rule-runner")["env"]}[
        "RULE_RUNNER_BUNDLE_DIR"]
    compose = dict(e.split("=", 1) for e in _compose()["rule-runner"]["environment"])
    assert compose["RULE_RUNNER_BUNDLE_DIR"] == bundle
    fetch = f"python -m millm.rules.runner_worker fetch {bundle}"
    precompile = f"python -m millm.rules.runner_worker precompile {bundle}"
    assert fetch in text and precompile in text
    run_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(run_lines)
    assert body.index(fetch) < body.index(precompile) < body.index("USER millm")
    user = body.index("USER millm")
    assert "chown -R millm:millm /app " in body
    assert body.rindex("chown", 0, user) < body.index(fetch), (
        "a chown after the fetch would hand the bundle (and its native .cwasm) to a runtime uid")


def test_the_image_builds_the_workers_own_seccomp_filter():
    """D55: the worker installs its own filter through libseccomp; the image names the library
    and the build runs `seccomp-check` (builds the filter, never loads it), so an image whose
    libseccomp cannot build it fails to build instead of shipping a worker that refuses to serve."""
    body = "\n".join(ln for ln in _dockerfile().splitlines() if not ln.lstrip().startswith("#"))
    runtime = body.split("FROM runtime AS development")[0]
    assert "libseccomp2" in runtime.split("rm -rf /var/lib/apt/lists/*")[0]
    check = "python -m millm.rules.runner_worker seccomp-check"
    assert check in runtime and runtime.index(check) < runtime.index("USER millm")


def _ci_job() -> dict:
    doc = yaml.safe_load((REPO / ".github/workflows/backend-tests.yml").read_text())
    return doc["jobs"]["rule-runner"]


def test_the_ci_job_requires_the_runner_and_runs_the_real_worker_tests():
    """R1-M3: the job is what proves operator code against the published guest; with
    MILLM_REQUIRE_RUNNER off, or a file dropped from its list, every runner test would skip
    green (the unit shards have no snapshot)."""
    job = _ci_job()
    assert job["env"]["MILLM_REQUIRE_RUNNER"] == "1"
    assert "continue-on-error" not in job
    steps = job["steps"]
    run = "\n".join(s.get("run", "") for s in steps)
    assert "python -m millm.rules.runner_worker fetch" in run
    assert "RULE_RUNNER_BUNDLE_DIR=" in run
    assert all("continue-on-error" not in s for s in steps)
    for name in ("test_runner_vectors.py", "test_runner_wasi_table.py", "test_runner_hostile.py",
                 "test_runner_bundle.py", "test_runner_client.py", "test_rule_code_followup.py",
                 "test_rule_privacy_sentinel.py", "test_runner_r1_pins.py",
                 "test_runner_seccomp.py", "test_rule_code_stateless.py",
                 "test_runner_sigterm.py"):
        assert f"/{name}" in run, name
    assert "|| true" not in run
