"""The `rule-runner` worker's OWN seccomp filter (suite decision D55, next step; 039 follow-up).

- The filter, as data, refuses everything the hardened Localhost profile
  (`k8s/seccomp/millm-rule-runner.json`, reviewed in 039 S3 R1) refuses, and is stricter on sockets
  (every family but `AF_UNIX`).
- A CHILD process under the filter is refused an `AF_INET` socket, `io_uring_setup`, a new user
  namespace and `clone3` (ENOSYS), while `AF_UNIX` sockets and threads still work; a system call
  through another ABI (x32) kills it. These are HOST-side refusals of system calls — no guest
  escape vector is written here (D48).
- Wiring: `runner_worker.start` installs the filter BEFORE it loads the guest (order and count),
  and the REAL worker process runs under one more filter than its parent (`/proc/<pid>/status`).
  Remove the install call → red.
- Fail closed: when the filter cannot be installed the worker loads no guest, says
  `seccomp_unavailable` and answers every call `rule_worker_unavailable`.

⚠ The filter is never installed in THIS process: it cannot be removed and would apply to every
thread of the test run. Each install happens in a child interpreter.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from millm.rules import runner_contract as rc
from millm.rules import runner_seccomp as sc
from tests.support import rule_runner as rr

REPO = Path(__file__).resolve().parents[3]
PROFILE = REPO / "k8s" / "seccomp" / "millm-rule-runner.json"
EPERM, ENOSYS = 1, 38


def _profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


# ── the filter as data, against the Localhost profile ──────────────────────────────────────────


def test_the_filter_refuses_everything_the_localhost_profile_refuses():
    profile = _profile()
    rules = sc.rules()
    assert profile["architectures"] == ["SCMP_ARCH_X86_64"]
    assert profile["defaultAction"] == "SCMP_ACT_ALLOW"
    # Unconditional refusals: the same names, the same errno.
    unconditional = {(n, e["errnoRet"]) for e in profile["syscalls"] if "args" not in e
                     and e["action"] == "SCMP_ACT_ERRNO" for n in e["names"]}
    mine = {(r.syscall, r.action & 0xFFFF) for r in rules if not r.args}
    assert unconditional == mine
    assert all(r.action >> 16 == 0x0005 for r in rules), "every rule is an ERRNO action"
    # `clone` with a CLONE_NEW* flag: the same masks.
    clone_masks = {a["value"] for e in profile["syscalls"] if e["names"] == ["clone"]
                   for a in e["args"] if a["op"] == "SCMP_CMP_MASKED_EQ" and
                   a["valueTwo"] == a["value"]}
    assert clone_masks == {r.args[0][2] for r in rules if r.syscall == "clone"}
    assert all(r.args == ((0, sc.SCMP_CMP_MASKED_EQ, r.args[0][2], r.args[0][2]),)
               and r.action == sc.act_errno(EPERM) for r in rules if r.syscall == "clone")
    # Sockets: the profile refuses four families; the filter refuses every family but AF_UNIX.
    families = {a["value"] for e in profile["syscalls"] if "socket" in e["names"]
                for a in e["args"]}
    assert families == {2, 10, 16, 17} and sc.AF_UNIX not in families
    sockets = {r.syscall: r for r in rules if r.syscall in ("socket", "socketpair")}
    assert set(sockets) == {"socket", "socketpair"}
    assert all(r.args == ((0, sc.SCMP_CMP_NE, sc.AF_UNIX, 0),) and r.action == sc.act_errno(EPERM)
               for r in sockets.values())


def test_the_filter_builds_on_this_hosts_libseccomp_without_loading():
    before = sc.installed_filters()
    sc.check()
    assert sc.installed_filters() == before


# ── a child process under the filter ───────────────────────────────────────────────────────────

_CHILD = textwrap.dedent("""
    import ctypes, errno, json, os, socket, threading
    from millm.rules import runner_seccomp as sc
    out = {"before": sc.installed_filters()}
    # A thread started BEFORE the install is filtered too (TSYNC).
    go, early = threading.Event(), {}
    def early_thread():
        go.wait(30)
        try:
            socket.socket(socket.AF_INET).close()
            early["af_inet"] = "allowed"
        except OSError as exc:
            early["af_inet"] = errno.errorcode.get(exc.errno, str(exc.errno))
    pre = threading.Thread(target=early_thread); pre.start()
    sc.install()
    go.set(); pre.join()
    out["early_thread_af_inet"] = early["af_inet"]
    out["after"] = sc.installed_filters()
    def attempt(fn):
        try:
            fn()
            return "allowed"
        except OSError as exc:
            return errno.errorcode.get(exc.errno, str(exc.errno))
    out["af_inet"] = attempt(lambda: socket.socket(socket.AF_INET).close())
    out["af_inet6"] = attempt(lambda: socket.socket(socket.AF_INET6).close())
    out["af_unix"] = attempt(lambda: socket.socket(socket.AF_UNIX).close())
    out["socketpair_unix"] = attempt(lambda: [s.close() for s in socket.socketpair()])
    out["unshare_newuser"] = attempt(lambda: os.unshare(os.CLONE_NEWUSER))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(425, 1, ctypes.create_string_buffer(120))   # io_uring_setup
    out["io_uring_setup"] = errno.errorcode.get(ctypes.get_errno()) if rc < 0 else "allowed"
    rc = libc.syscall(435, None, 0)                               # clone3
    out["clone3"] = errno.errorcode.get(ctypes.get_errno()) if rc < 0 else "allowed"
    ran = []
    t = threading.Thread(target=lambda: ran.append(1)); t.start(); t.join()
    out["thread"] = bool(ran)
    print(json.dumps(out))
""")


def _child(code: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                          capture_output=True, text=True, timeout=60)


def test_a_process_under_the_filter_is_refused_sockets_io_uring_and_namespaces():
    done = _child(_CHILD)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["after"] == (out["before"] or 0) + 1
    assert out["af_inet"] == "EPERM" and out["af_inet6"] == "EPERM"
    assert out["early_thread_af_inet"] == "EPERM"
    assert out["io_uring_setup"] == "EPERM"
    assert out["unshare_newuser"] == "EPERM"
    assert out["clone3"] == "ENOSYS"
    # What the worker needs still works.
    assert out["af_unix"] == "allowed" and out["socketpair_unix"] == "allowed"
    assert out["thread"] is True


def test_another_system_call_abi_kills_the_process():
    """x32 (the x86-64 ABI with bit 30 set) is not the filter's architecture: SIGSYS, no core."""
    code = textwrap.dedent("""
        import ctypes, resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        from millm.rules import runner_seccomp as sc
        sc.install()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall(0x40000000 | 39)   # getpid through the x32 ABI
        print("survived")
    """)
    done = _child(code)
    assert done.returncode == -signal.SIGSYS, (done.returncode, done.stdout, done.stderr[-500:])
    assert "survived" not in done.stdout


# ── wiring: installed before the guest loads; the real worker runs under it ───────────────────


def _config(tmp_path: Path):
    from millm.rules.runner_worker import WorkerConfig

    return WorkerConfig.from_env({"RULE_RUNNER_SOCKET": str(tmp_path / "r.sock"),
                                  "RULE_RUNNER_BUNDLE_DIR": str(tmp_path / "absent")})


def test_start_installs_the_filter_once_before_it_loads_the_guest(monkeypatch, tmp_path):
    rr.require_wasmtime()
    from millm.rules import runner_host, runner_worker

    order: list[str] = []
    real_load = runner_host.load_bundle
    monkeypatch.setattr(sc, "install", lambda *a, **k: order.append("install"))

    def load(*args, **kwargs):
        order.append("load_bundle")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(runner_host, "load_bundle", load)
    before = sc.installed_filters()
    worker = runner_worker.start(_config(tmp_path))
    try:
        assert order == ["install", "load_bundle"]
    finally:
        worker.close()
    assert sc.installed_filters() == before, "the filter must never reach the test process"


def test_the_real_worker_process_runs_under_its_own_filter():
    """The REAL `python -m millm.rules.runner_worker` (no bundle needed): one more seccomp filter
    than this test process — whatever the host or container already applies."""
    rr.require_wasmtime()
    ours = sc.installed_filters()
    with rr.RunnerProcess(None) as runner:
        status = Path(f"/proc/{runner.proc.pid}/status").read_text(encoding="ascii")
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    assert fields["Seccomp"].strip() == "2"
    assert int(fields["Seccomp_filters"]) == (ours or 0) + 1


# ── fail closed ───────────────────────────────────────────────────────────────────────────────


def test_without_its_filter_the_worker_loads_no_guest_and_serves_nothing(monkeypatch, tmp_path,
                                                                         caplog):
    rr.require_wasmtime()
    from millm.rules import runner_host, runner_worker

    def refuse(*_a, **_k):
        raise sc.SeccompUnavailable("libseccomp_missing")

    loads: list[int] = []
    monkeypatch.setattr(sc, "install", refuse)
    monkeypatch.setattr(runner_host, "load_bundle", lambda *a, **k: loads.append(1))
    monkeypatch.setattr(runner_host.Host, "engine_for", lambda *a, **k: loads.append(2))
    caplog.set_level(logging.INFO, logger="millm.rule_runner")
    worker = runner_worker.start(_config(tmp_path))
    assert loads == [], "no engine and no guest without the filter"
    assert worker.status == "seccomp_unavailable" and worker.host is None
    assert worker.health()["status"] == "seccomp_unavailable"
    code = json.loads((REPO / "docs/schemas/runner-cases-v1.json").read_text())["documents"][
        "count_tokens"]
    sha = rc.code_sha256(code)
    assert worker.handle({"op": "load", "code_sha256": sha,
                          "canonical_utf8": rc.canonical_json(code).decode()}) == {"ok": True}
    assert worker.handle({"op": "call", "code_sha256": sha, "function": "f", "params": {},
                          "text": "a b c", "limits": rr.floor_limits()}) == {
        "reason": "rule_worker_unavailable"}
    assert "rule_runner_seccomp_refused code=libseccomp_missing" in caplog.text


@pytest.mark.parametrize("arch", [0x40000003])  # AUDIT_ARCH_I386: any non-x86-64 host
def test_a_host_that_is_not_x86_64_gets_no_filter_and_refuses(arch):
    class Lib:
        def seccomp_arch_native(self):
            return arch

    with pytest.raises(sc.SeccompUnavailable) as caught:
        sc.install(Lib())
    assert caught.value.code == "not_x86_64"


# ── review R2 (039 follow-ups) ─────────────────────────────────────────────────────────────────


def test_a_clone_with_any_new_namespace_flag_is_refused_by_the_filter_itself():
    """`clone` itself (not `unshare`) with each CLONE_NEW* flag. HOST-side, and harmless either
    way: every attempt also carries CLONE_THREAD without CLONE_SIGHAND, which the kernel rejects
    with EINVAL AFTER seccomp — so EPERM proves the filter refused it, EINVAL that it let it
    through, and no process is ever created. A mask compared with `==` instead of `&` (only the
    lone flag refused) is caught here; the data test alone could not see it (R2 mutation M4)."""
    code = textwrap.dedent("""
        import ctypes, errno, json
        from millm.rules import runner_seccomp as sc
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        CLONE_THREAD, SIGCHLD = 0x00010000, 17
        def clone(flags):
            rc = libc.syscall(56, ctypes.c_ulong(flags | CLONE_THREAD | SIGCHLD), None, None,
                              None, ctypes.c_ulong(0))
            return "created" if rc >= 0 else errno.errorcode.get(ctypes.get_errno())
        sc.install()
        print(json.dumps({"plain": clone(0),
                          "new": {hex(f): clone(f) for f in sc.CLONE_NEW_FLAGS}}))
    """)
    done = _child(code)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["plain"] == "EINVAL", "control: the kernel's own refusal, not the filter's"
    assert out["new"] == {hex(f): "EPERM" for f in sc.CLONE_NEW_FLAGS}


class _FakeLib:
    """libseccomp's surface, recording each call; `load_rc` is what `seccomp_load` returns."""

    def __init__(self, load_rc: int = 0) -> None:
        self.calls: list[tuple] = []
        self.load_rc = load_rc

    def seccomp_arch_native(self):
        return sc.SCMP_ARCH_X86_64

    def seccomp_init(self, action):
        self.calls.append(("init", action))
        return 1

    def seccomp_attr_set(self, ctx, attr, value):
        self.calls.append(("attr", attr, value))
        return 0

    def seccomp_syscall_resolve_name(self, name):
        return 7

    def seccomp_rule_add_exact_array(self, ctx, action, nr, count, args):
        self.calls.append(("rule", action, count))
        return 0

    def seccomp_load(self, ctx):
        self.calls.append(("load",))
        return self.load_rc

    def seccomp_release(self, ctx):
        self.calls.append(("release",))


def test_a_refused_load_fails_closed_and_releases_the_context():
    """A kernel that refuses the filter is not a silent success (R2 mutation M6 survived: no test
    made `seccomp_load` fail)."""
    lib = _FakeLib(load_rc=-13)
    with pytest.raises(sc.SeccompUnavailable) as caught:
        sc.install(lib)
    assert caught.value.code == "seccomp_load_failed"
    assert lib.calls[-1] == ("release",)


def test_install_sets_the_attributes_then_every_rule_then_loads_once():
    """The payload and the count of each call, not only that they happened: default ALLOW,
    bad architecture KILLS, NNP and TSYNC on, one rule per `rules()` entry, one load."""
    lib = _FakeLib()
    sc.install(lib)
    assert lib.calls[0] == ("init", sc.SCMP_ACT_ALLOW)
    attrs = [c[1:] for c in lib.calls if c[0] == "attr"]
    assert attrs == [(sc.SCMP_FLTATR_ACT_BADARCH, sc.SCMP_ACT_KILL_PROCESS),
                     (sc.SCMP_FLTATR_CTL_NNP, 1), (sc.SCMP_FLTATR_CTL_TSYNC, 1)]
    rules = [c[1:] for c in lib.calls if c[0] == "rule"]
    assert rules == [(r.action, len(r.args)) for r in sc.rules()]
    assert [c for c in lib.calls if c[0] == "load"] == [("load",)]
    assert lib.calls[-1] == ("release",)


def test_a_library_missing_a_symbol_fails_closed_instead_of_crashing(monkeypatch):
    class Bare:
        pass

    monkeypatch.setattr(sc.ctypes, "CDLL", lambda *a, **k: Bare())
    with pytest.raises(sc.SeccompUnavailable) as caught:
        sc.install()
    assert caught.value.code == "libseccomp_symbol_missing"


def test_the_image_builds_seccomp_check_really_builds_the_filter(monkeypatch):
    """`python -m millm.rules.runner_worker seccomp-check` (the Dockerfile's build step) calls
    `check()` once and lets its failure out, so an image whose libseccomp cannot build the filter
    fails its build (R2 mutation W3 survived: the command printing without checking)."""
    from millm.rules import runner_worker

    calls: list[int] = []
    monkeypatch.setattr(sc, "check", lambda: calls.append(1))
    runner_worker.main(["seccomp-check"])
    assert calls == [1]

    def refuse():
        raise sc.SeccompUnavailable("libseccomp_missing")

    monkeypatch.setattr(sc, "check", refuse)
    with pytest.raises(sc.SeccompUnavailable):
        runner_worker.main(["seccomp-check"])
