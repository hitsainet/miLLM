"""The `rule-runner` worker's OWN seccomp filter (suite decision D55, next step; 039 follow-up).

The worker installs this filter at start-up, BEFORE it loads the guest snapshot or any operator
code (`runner_worker.start`), so the kernel barrier no longer depends on a file installed on the
node. It stacks with whatever the node applies: the hardened `Localhost` profile
(`k8s/seccomp/millm-rule-runner.json`) stays in the manifest as defence in depth, and the kernel
enforces the stricter of the two for every system call.

What it refuses (each one the Localhost profile refuses, `test_runner_seccomp.py` compares them):

- **x86-64 only.** Any other system-call ABI (i386 `int 0x80`, x32) KILLS the process; on a host
  that is not x86-64 the filter is not installed at all and the worker refuses to serve.
- **Sockets: `AF_UNIX` only.** `socket`/`socketpair` with any other family → `EPERM` (the profile
  names AF_INET, AF_INET6, AF_NETLINK and AF_PACKET; this is stricter).
- **`io_uring_*`** (`IORING_OP_SOCKET` makes sockets without `socket`), the **mount API**
  (`fsopen`, `fsconfig`, `fsmount`, `fspick`, `move_mount`, `open_tree`, `mount_setattr`, `mount`,
  `umount2`, `pivot_root`), **namespaces** (`unshare`, `setns`, and `clone` with any `CLONE_NEW*`
  flag), and the privileged list (`ptrace`, `bpf`, `kexec_*`, modules, keys, `userfaultfd`, …) →
  `EPERM`.
- **`clone3` → `ENOSYS`**: its flags live in memory a filter cannot read, so it is refused as
  "not implemented" and glibc falls back to `clone`, whose flags the filter does read.

How: libseccomp (the maintained C library; `libseccomp2` from the image's Debian, 2.6.0 on
`python:3.13-slim`) through a few `ctypes` calls — no Python dependency. Anything that goes wrong
(no library, an unknown system-call name, a refused load) raises :class:`SeccompUnavailable`; the
worker then FAILS CLOSED: it loads no guest and answers every call `rule_worker_unavailable`, so
code rules stay pending and the backend is unaffected.

⚠ Never call :func:`install` in the API process or a test process: a seccomp filter cannot be
removed, and it applies to every thread (TSYNC). Tests install it in a child process only.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
from dataclasses import dataclass
from typing import Any, Optional

#: libseccomp's ABI (seccomp.h; stable across the 2.x library).
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_KILL_PROCESS = 0x80000000
SCMP_ARCH_X86_64 = 0xC000003E  # AUDIT_ARCH_X86_64
SCMP_FLTATR_ACT_BADARCH = 2
SCMP_FLTATR_CTL_NNP = 3
SCMP_FLTATR_CTL_TSYNC = 4
SCMP_CMP_NE = 1
SCMP_CMP_EQ = 4
SCMP_CMP_MASKED_EQ = 7
NR_SCMP_ERROR = -1

AF_UNIX = 1


def act_errno(code: int) -> int:
    return 0x00050000 | (code & 0x0000FFFF)


#: Refused outright with EPERM (the Localhost profile's list, unchanged).
DENIED: tuple[str, ...] = (
    "ptrace", "process_vm_readv", "process_vm_writev", "mount", "umount2", "pivot_root",
    "unshare", "setns", "kexec_load", "kexec_file_load", "init_module", "finit_module",
    "delete_module", "bpf", "perf_event_open", "keyctl", "add_key", "request_key",
    "userfaultfd", "reboot", "swapon", "swapoff", "acct", "quotactl", "open_by_handle_at",
    "name_to_handle_at", "lookup_dcookie", "iopl", "ioperm", "syslog", "personality",
    "io_uring_setup", "io_uring_enter", "io_uring_register",
    "fsopen", "fsconfig", "fsmount", "fspick", "move_mount", "open_tree", "mount_setattr",
)
#: `socket` / `socketpair`: refused (EPERM) for every family but AF_UNIX.
SOCKET_CALLS: tuple[str, ...] = ("socket", "socketpair")
#: `clone` refused (EPERM) when its flags carry any of these (`CLONE_NEW*`).
CLONE_NEW_FLAGS: tuple[int, ...] = (
    0x00020000,  # CLONE_NEWNS
    0x02000000,  # CLONE_NEWCGROUP
    0x04000000,  # CLONE_NEWUTS
    0x08000000,  # CLONE_NEWIPC
    0x10000000,  # CLONE_NEWUSER
    0x20000000,  # CLONE_NEWPID
    0x40000000,  # CLONE_NEWNET
)
#: `clone3` → ENOSYS (glibc falls back to `clone`).
ENOSYS_CALLS: tuple[str, ...] = ("clone3",)


class SeccompUnavailable(Exception):
    """The filter could not be built or installed. `code` is a short fixed string."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


@dataclass(frozen=True)
class Rule:
    """One libseccomp rule: `action` on `syscall` when every comparison holds."""

    syscall: str
    action: int
    #: `(arg index, op, datum_a, datum_b)`.
    args: tuple[tuple[int, int, int, int], ...] = ()


def rules() -> tuple[Rule, ...]:
    """The whole filter, as data (what `install` loads and what the tests compare)."""
    eperm = act_errno(errno.EPERM)
    out: list[Rule] = [Rule(name, eperm) for name in DENIED]
    out += [Rule(name, eperm, ((0, SCMP_CMP_NE, AF_UNIX, 0),)) for name in SOCKET_CALLS]
    out += [Rule("clone", eperm, ((0, SCMP_CMP_MASKED_EQ, flag, flag),))
            for flag in CLONE_NEW_FLAGS]
    out += [Rule(name, act_errno(errno.ENOSYS)) for name in ENOSYS_CALLS]
    return tuple(out)


def _library() -> Any:
    name = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    try:
        lib = ctypes.CDLL(name, use_errno=True)
    except OSError as exc:
        raise SeccompUnavailable("libseccomp_missing") from exc
    try:
        return _bind(lib)
    except AttributeError as exc:
        # Review R2: a library without one of these symbols crashed the worker (an uncaught
        # AttributeError) instead of failing closed with `seccomp_unavailable`.
        raise SeccompUnavailable("libseccomp_symbol_missing") from exc


def _bind(lib: Any) -> Any:
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_arch_native.restype = ctypes.c_uint32
    lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_exact_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(_ArgCmp)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    return lib


def _check(rc: int, code: str) -> None:
    if rc < 0:
        raise SeccompUnavailable(code)


def _build(lib: Any) -> Any:
    """A filter context holding every rule (not loaded). The caller releases it."""
    if lib.seccomp_arch_native() != SCMP_ARCH_X86_64:
        raise SeccompUnavailable("not_x86_64")
    ctx = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not ctx:
        raise SeccompUnavailable("seccomp_init_failed")
    try:
        # A new filter holds the native ABI only (x86-64); every other ABI, x32 included,
        # takes the bad-architecture action.
        _check(lib.seccomp_attr_set(ctx, SCMP_FLTATR_ACT_BADARCH, SCMP_ACT_KILL_PROCESS),
               "attr_badarch_failed")
        _check(lib.seccomp_attr_set(ctx, SCMP_FLTATR_CTL_NNP, 1), "attr_nnp_failed")
        _check(lib.seccomp_attr_set(ctx, SCMP_FLTATR_CTL_TSYNC, 1), "attr_tsync_failed")
        for rule in rules():
            nr = lib.seccomp_syscall_resolve_name(rule.syscall.encode("ascii"))
            if nr == NR_SCMP_ERROR:
                raise SeccompUnavailable(f"unknown_syscall_{rule.syscall}")
            args = (_ArgCmp * max(1, len(rule.args)))(*[_ArgCmp(*a) for a in rule.args])
            _check(lib.seccomp_rule_add_exact_array(ctx, rule.action, nr, len(rule.args), args),
                   f"rule_add_failed_{rule.syscall}")
    except BaseException:
        lib.seccomp_release(ctx)
        raise
    return ctx


def check() -> None:
    """Build the whole filter WITHOUT loading it (the image build runs this, so an image whose
    libseccomp is missing or too old never ships). Raises :class:`SeccompUnavailable`."""
    lib = _library()
    lib.seccomp_release(_build(lib))


def install(library: Optional[Any] = None) -> None:
    """Build and LOAD the filter into this process, every thread (irreversible). Raises
    :class:`SeccompUnavailable` when anything fails; nothing is half-installed (the kernel
    applies a filter whole or not at all)."""
    lib = library if library is not None else _library()
    ctx = _build(lib)
    try:
        _check(lib.seccomp_load(ctx), "seccomp_load_failed")
    finally:
        lib.seccomp_release(ctx)


def installed_filters() -> Optional[int]:
    """`Seccomp_filters` of this process from `/proc/self/status` (None when unreadable)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("Seccomp_filters:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None
