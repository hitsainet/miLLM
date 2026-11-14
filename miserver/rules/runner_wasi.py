"""Table W: the host's OWN WebAssembly System Interface preview-1 imports (039 S3; FR-39.26).

A port of miDataworks 018's `backend/src/runner/wasi.py` (main 6c71e7e), so both hosts serve the
published guest the same table. Every import the snapshot declares is linked here with
``Linker.define_func`` — never through the runtime's WASI context — so nothing is inherited from the
`rule-runner` process: no preopened directory, an empty environment, a fixed clock and a fixed
random stream. Each row's behaviour is the vendored contract's ``wasi`` table
(`tests/unit/rules/test_runner_wasi_table.py` holds the two equal, both directions).

The phase markers on descriptor 3 are accepted ONCE AND IN ORDER (ENH-006 D41, S0 correction): a
marker sets the store's fuel to the next phase's budget, so a replayed marker would let a rule refill
its own fuel. Anything else on descriptor 3 ends the instance with ``rule_error``.

⚠ Imported by the `rule-runner` worker ONLY (`runner_host`). The API process never imports this
module or `wasmtime` (`test_runner_hostile.py` holds it).
"""

from __future__ import annotations

import hashlib
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import wasmtime
from wasmtime import FuncType, ValType


I32, I64 = ValType.i32(), ValType.i64()
ESUCCESS, EBADF, ENOTCAPABLE = 0, 8, 76
WASI = "wasi_snapshot_preview1"
RULE_ERROR = "rule_error"
RULE_OUTPUT_INVALID = "rule_output_invalid"

#: Preview-1 signatures for every import Table W names.
SIGNATURES: dict[str, tuple[list[ValType], list[ValType]]] = {
    "args_get": ([I32, I32], [I32]),
    "args_sizes_get": ([I32, I32], [I32]),
    "clock_res_get": ([I32, I32], [I32]),
    "clock_time_get": ([I32, I64, I32], [I32]),
    "environ_get": ([I32, I32], [I32]),
    "environ_sizes_get": ([I32, I32], [I32]),
    "fd_advise": ([I32, I64, I64, I32], [I32]),
    "fd_allocate": ([I32, I64, I64], [I32]),
    "fd_close": ([I32], [I32]),
    "fd_datasync": ([I32], [I32]),
    "fd_fdstat_get": ([I32, I32], [I32]),
    "fd_fdstat_set_flags": ([I32, I32], [I32]),
    "fd_fdstat_set_rights": ([I32, I64, I64], [I32]),
    "fd_filestat_get": ([I32, I32], [I32]),
    "fd_filestat_set_size": ([I32, I64], [I32]),
    "fd_filestat_set_times": ([I32, I64, I64, I32], [I32]),
    "fd_pread": ([I32, I32, I32, I64, I32], [I32]),
    "fd_prestat_dir_name": ([I32, I32, I32], [I32]),
    "fd_prestat_get": ([I32, I32], [I32]),
    "fd_pwrite": ([I32, I32, I32, I64, I32], [I32]),
    "fd_read": ([I32, I32, I32, I32], [I32]),
    "fd_readdir": ([I32, I32, I32, I64, I32], [I32]),
    "fd_renumber": ([I32, I32], [I32]),
    "fd_seek": ([I32, I64, I32, I32], [I32]),
    "fd_sync": ([I32], [I32]),
    "fd_tell": ([I32, I32], [I32]),
    "fd_write": ([I32, I32, I32, I32], [I32]),
    "path_create_directory": ([I32, I32, I32], [I32]),
    "path_filestat_get": ([I32, I32, I32, I32, I32], [I32]),
    "path_filestat_set_times": ([I32, I32, I32, I32, I64, I64, I32], [I32]),
    "path_link": ([I32, I32, I32, I32, I32, I32, I32], [I32]),
    "path_open": ([I32, I32, I32, I32, I32, I64, I64, I32, I32], [I32]),
    "path_readlink": ([I32, I32, I32, I32, I32, I32], [I32]),
    "path_remove_directory": ([I32, I32, I32], [I32]),
    "path_rename": ([I32, I32, I32, I32, I32, I32], [I32]),
    "path_symlink": ([I32, I32, I32, I32, I32], [I32]),
    "path_unlink_file": ([I32, I32, I32], [I32]),
    "poll_oneoff": ([I32, I32, I32, I32], [I32]),
    "proc_exit": ([I32], []),
    "proc_raise": ([I32], [I32]),
    "random_get": ([I32, I32], [I32]),
    "sched_yield": ([], [I32]),
    "sock_accept": ([I32, I32, I32], [I32]),
    "sock_recv": ([I32, I32, I32, I32, I32, I32], [I32]),
    "sock_send": ([I32, I32, I32, I32, I32], [I32]),
    "sock_shutdown": ([I32, I32], [I32]),
}

#: Imports that end the instance with rule_error when called (Table W row "sock_*, proc_raise").
TRAPPING = frozenset({"sock_accept", "sock_recv", "sock_send", "sock_shutdown", "proc_raise"})

#: The marker each phase may be followed by, and nothing else (S0 correction).
NEXT_MARKER: dict[str, bytes] = {"prelude": b"load", "load": b"call"}


class GuestEnd(Exception):
    """Raised inside a host function to end the instance; ``reason`` is None for proc_exit."""

    def __init__(self, reason: str | None, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass
class CallState:
    """The per-instance host state Table W reads and writes (one call, one thread)."""

    stdin: bytes
    budgets: dict[str, int]
    output_limit: int
    random_seed: bytes
    final_phase: str
    pos: int = 0
    reply: bytearray = field(default_factory=bytearray)
    stderr_bytes: int = 0
    phase: str = "prelude"
    used: dict[str, int] = field(default_factory=dict)
    fault: str | None = None
    random_counter: int = 0
    random_buffer: bytes = b""


_LOCAL = threading.local()


def bind(state: CallState | None) -> None:
    """Make ``state`` the state Table W serves on this thread (None after the call)."""
    _LOCAL.state = state


def _state() -> CallState:
    state: CallState | None = getattr(_LOCAL, "state", None)
    if state is None:
        raise GuestEnd(RULE_ERROR, "no call state bound")
    return state


def _memory(caller: wasmtime.Caller) -> wasmtime.Memory:
    memory = caller["memory"]
    assert isinstance(memory, wasmtime.Memory)
    return memory


def _span(caller: wasmtime.Caller, ptr: int, length: int) -> tuple[wasmtime.Memory, int, int]:
    """A guest pointer and length checked against the guest's memory (both are u32 in preview 1).

    A span outside the memory ends the instance with ``rule_error``; it never raises anything else
    out of a host function (Review R1: an out-of-range iovec made ``run_one`` raise, failing every
    text of the request with a 500 instead of this one call with a reason).
    """
    memory = _memory(caller)
    start, length = ptr & 0xFFFFFFFF, length & 0xFFFFFFFF
    if start + length > memory.data_len(caller):
        raise GuestEnd(RULE_ERROR, "a pointer outside the guest's memory")
    return memory, start, start + length


def _read(caller: wasmtime.Caller, ptr: int, length: int) -> bytes:
    memory, start, stop = _span(caller, ptr, length)
    return bytes(memory.read(caller, start, stop))


def _write(caller: wasmtime.Caller, data: bytes, ptr: int) -> None:
    memory, start, _ = _span(caller, ptr, len(data))
    memory.write(caller, data, start)


def _iovecs(caller: wasmtime.Caller, ptr: int, count: int) -> list[tuple[int, int]]:
    raw = _read(caller, ptr, 8 * (count & 0xFFFFFFFF))
    return [struct.unpack_from("<II", raw, 8 * i) for i in range(len(raw) // 8)]


def _put_u32(caller: wasmtime.Caller, ptr: int, value: int) -> None:
    _write(caller, struct.pack("<I", value), ptr)


def _random_bytes(state: CallState, n: int) -> bytes:
    while len(state.random_buffer) < n:
        block = hashlib.sha256(state.random_seed + struct.pack("<Q", state.random_counter)).digest()
        state.random_counter += 1
        state.random_buffer += block
    out, state.random_buffer = state.random_buffer[:n], state.random_buffer[n:]
    return out


def _marker(caller: wasmtime.Caller, state: CallState, data: bytes) -> None:
    expected = NEXT_MARKER.get(state.phase)
    if expected is None or data != expected or state.phase == state.final_phase:
        state.fault = RULE_ERROR
        raise GuestEnd(RULE_ERROR, "phase marker refused")
    left = wasmtime.Store.get_fuel(caller)
    state.used[state.phase] = state.budgets[state.phase] - left
    state.phase = data.decode("ascii")
    wasmtime.Store.set_fuel(caller, state.budgets[state.phase])


def fd_read(caller: wasmtime.Caller, fd: int, iov: int, count: int, nread: int) -> int:
    state = _state()
    if fd != 0:
        return EBADF
    total = 0
    for base, length in _iovecs(caller, iov, count):
        _span(caller, base, length)
        chunk = state.stdin[state.pos : state.pos + length]
        if not chunk:
            break
        _write(caller, chunk, base)
        state.pos += len(chunk)
        total += len(chunk)
        if len(chunk) < length:
            break
    _put_u32(caller, nread, total)
    return ESUCCESS


def fd_write(caller: wasmtime.Caller, fd: int, iov: int, count: int, nwritten: int) -> int:
    state = _state()
    if fd not in (1, 2, 3):
        return EBADF
    data = b"".join(_read(caller, b, n) for b, n in _iovecs(caller, iov, count))
    if fd == 1:
        state.reply += data
        if len(state.reply) > state.output_limit:
            state.fault = RULE_OUTPUT_INVALID
            raise GuestEnd(RULE_OUTPUT_INVALID, "reply over the output limit")
    elif fd == 2:
        state.stderr_bytes += len(data)
    else:
        _marker(caller, state, data)
    _put_u32(caller, nwritten, len(data))
    return ESUCCESS


def clock_time_get(caller: wasmtime.Caller, clock: int, precision: int, ptr: int) -> int:
    _state()
    _write(caller, struct.pack("<Q", 0), ptr)
    return ESUCCESS


def clock_res_get(caller: wasmtime.Caller, clock: int, ptr: int) -> int:
    _state()
    _write(caller, struct.pack("<Q", 1), ptr)
    return ESUCCESS


def random_get(caller: wasmtime.Caller, ptr: int, length: int) -> int:
    state = _state()
    _span(caller, ptr, length)
    _write(caller, _random_bytes(state, length & 0xFFFFFFFF), ptr)
    return ESUCCESS


def environ_sizes_get(caller: wasmtime.Caller, count_ptr: int, size_ptr: int) -> int:
    _state()
    _put_u32(caller, count_ptr, 0)
    _put_u32(caller, size_ptr, 0)
    return ESUCCESS


def environ_get(caller: wasmtime.Caller, environ: int, buffer: int) -> int:
    _state()
    return ESUCCESS


ARGV0 = b"python\x00"


def args_sizes_get(caller: wasmtime.Caller, count_ptr: int, size_ptr: int) -> int:
    _state()
    _put_u32(caller, count_ptr, 1)
    _put_u32(caller, size_ptr, len(ARGV0))
    return ESUCCESS


def args_get(caller: wasmtime.Caller, argv: int, buffer: int) -> int:
    _state()
    _write(caller, ARGV0, buffer)
    _put_u32(caller, argv, buffer)
    return ESUCCESS


def fd_fdstat_get(caller: wasmtime.Caller, fd: int, ptr: int) -> int:
    _state()
    if fd not in (0, 1, 2):
        return EBADF
    # filetype CHARACTER_DEVICE (2), no flags, rights fd_read | fd_write.
    _write(caller, struct.pack("<BxHxxxxQQ", 2, 0, 0x42, 0), ptr)
    return ESUCCESS


def proc_exit(caller: wasmtime.Caller, code: int) -> None:
    _state()
    raise GuestEnd(None, f"proc_exit({code})")


def sched_yield(caller: wasmtime.Caller) -> int:
    _state()
    return ESUCCESS


def poll_oneoff(caller: wasmtime.Caller, inp: int, out: int, n: int, ptr: int) -> int:
    _state()
    _put_u32(caller, ptr, 0)
    return ESUCCESS


def _constant(result: int) -> Callable[..., int]:
    def answer(caller: wasmtime.Caller, *args: Any) -> int:
        _state()
        return result

    return answer


def _trapping(name: str) -> Callable[..., int]:
    def trap(caller: wasmtime.Caller, *args: Any) -> int:
        state = _state()
        state.fault = RULE_ERROR
        raise GuestEnd(RULE_ERROR, f"{name} is not available")

    return trap


_EXPLICIT: dict[str, Callable[..., Any]] = {
    "fd_read": fd_read,
    "fd_write": fd_write,
    "clock_time_get": clock_time_get,
    "clock_res_get": clock_res_get,
    "random_get": random_get,
    "environ_sizes_get": environ_sizes_get,
    "environ_get": environ_get,
    "args_sizes_get": args_sizes_get,
    "args_get": args_get,
    "fd_fdstat_get": fd_fdstat_get,
    "proc_exit": proc_exit,
    "sched_yield": sched_yield,
    "poll_oneoff": poll_oneoff,
}


def implementation(name: str) -> Callable[..., Any]:
    """The Table W function for one import name."""
    if name in TRAPPING:
        return _trapping(name)
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    if name.startswith("path_"):
        return _constant(ENOTCAPABLE)
    if name in SIGNATURES:
        return _constant(EBADF)  # every other fd_* (incl. fd_prestat_*): no such descriptor
    raise KeyError(name)


def build_linker(engine: wasmtime.Engine, module: wasmtime.Module) -> wasmtime.Linker:
    """A linker holding Table W for exactly the imports ``module`` declares; an import outside the
    table (or from another module namespace) is refused (the bundle is not the published one)."""
    linker = wasmtime.Linker(engine)
    linker.allow_shadowing = True
    for imp in module.imports:
        if imp.module != WASI or imp.name not in SIGNATURES:
            raise KeyError(f"{imp.module}.{imp.name} is not in Table W")
        params, results = SIGNATURES[imp.name]
        linker.define_func(
            WASI, imp.name, FuncType(params, results), implementation(imp.name), access_caller=True
        )
    return linker
