"""Table W at the host level: every row, through the production linker (039 FTASKS 18.2; D48).

A port of miDataworks 018's `runner/tests/test_wasi_table.py` (main 6c71e7e) onto miLLM's host
(`millm/rules/runner_wasi.py`, `runner_host.py`). The hostile ESCAPE-form vectors (guest source
reaching files, the environment, a socket, the clock or a random source past the prelude: classes
1-4, 9) are the operator's (ENH-006 D48) and are NOT written here. The defences those vectors will
exercise live in the HOST and are pinned without guest source: small WebAssembly modules, in the
text format, import the Table W functions and call them through the worker's own ``Linker``
(``runner_wasi.build_linker``) and ``Host.run_one`` — the code the worker serves. Nothing is
mocked: fuel, the memory cap, the epoch deadline and the wasm stack are the real store's and
engine's.

Also pinned, end to end on a real store: the phase markers on descriptor 3 (once, in order, the
fuel switch itself; classes 17-18 at the host), the one-line reply barrier (class 19 at the host),
the memory and stack limits AT their edge, and that a guest pointer outside its memory is a
``rule_error`` for that call, never an exception out of ``run_one``.

Skips loudly without `wasmtime` (and, for the one snapshot test, without the bundle); fails instead
under `MILLM_REQUIRE_RUNNER=1`.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.support import rule_runner as rr

rr.require_wasmtime()

import wasmtime  # noqa: E402

from millm.rules import runner_contract as rc  # noqa: E402
from millm.rules import runner_host as bundle_mod  # noqa: E402
from millm.rules import runner_wasi as wasi  # noqa: E402
from millm.rules.runner_host import Host, Limits  # noqa: E402

CONTRACT: dict[str, Any] = rc.contract()
#: The contract's published Table W rows (the producer's `WASI_TABLE`).
WASI_TABLE: list[dict[str, Any]] = CONTRACT["wasi"]


class Reason:
    """The contract's reason codes, as the host reports them (plain strings here)."""

    RULE_ERROR = "rule_error"
    RULE_RESOURCE_LIMIT = "rule_resource_limit"
    RULE_OUTPUT_INVALID = "rule_output_invalid"
    RULE_TIMEOUT = "rule_timeout"
W = wasi.WASI
PAGE = 65_536
REPLY_7 = b'{"phase":"call","result":{"t":"int","v":"7"}}\n'


def _ty(types: list[wasmtime.ValType]) -> str:
    return " ".join(str(t) for t in types)


def _import(name: str) -> str:
    params, results = wasi.SIGNATURES[name]
    result = f" (result {_ty(results)})" if results else ""
    return f'(import "{W}" "{name}" (func ${name} (param {_ty(params)}){result}))'


def _host(body: str, imports: list[str], memory_pages: int = 1) -> Host:
    """A ``Host`` over a module importing ``imports`` from Table W; ``body`` is the module rest."""
    wat = "(module\n" + "\n".join(_import(n) for n in imports)
    wat += f'\n(memory (export "memory") {memory_pages})\n{body}\n)'
    engine = Host.engine_for(CONTRACT)
    return Host(engine, wasmtime.Module(engine, wat), CONTRACT)


def _limits(**over: int) -> Limits:
    f = CONTRACT["limits_floor"]
    base = {
        "call_fuel": f["call_fuel"],
        "load_fuel": f["load_fuel"],
        "prelude_fuel": f["prelude_fuel"],
        "memory_bytes": f["memory_bytes"],
        "input_code_points": f["input_code_points"],
        "output_bytes": f["output_bytes"],
        "wall_ms": 10_000,
    }
    base.update(over)
    return Limits(**base)


# --- a guest that writes byte strings to descriptors -------------------------------------------

#: ``$w fd ptr len``: one fd_write of [ptr, ptr+len) with the iovec at 0 and nwritten at 8.
WRITE_FN = """
(func $w (param $fd i32) (param $ptr i32) (param $len i32)
  (i32.store (i32.const 0) (local.get $ptr))
  (i32.store (i32.const 4) (local.get $len))
  (drop (call $fd_write (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 8))))
(func $burn (param $n i32)
  (block $out (loop $l
    (br_if $out (i32.eqz (local.get $n)))
    (local.set $n (i32.sub (local.get $n) (i32.const 1)))
    (br $l))))
"""


def _guest(steps: list[tuple[int, bytes] | int], **kw: Any) -> Host:
    """A ``run`` that performs ``steps`` in order: ``(fd, bytes)`` writes them, an int burns that
    many loop iterations of fuel."""
    data, calls, at = [], [], 64
    for step in steps:
        if isinstance(step, int):
            calls.append(f"(call $burn (i32.const {step}))")
            continue
        fd, raw = step
        escaped = "".join(f"\\{b:02x}" for b in raw)
        data.append(f'(data (i32.const {at}) "{escaped}")')
        calls.append(f"(call $w (i32.const {fd}) (i32.const {at}) (i32.const {len(raw)}))")
        at += len(raw)
    body = WRITE_FN + "\n".join(data) + '\n(func (export "run")\n' + "\n".join(calls) + ")"
    return _host(body, ["fd_write"], **kw)


@pytest.fixture
def made() -> Iterator[list[Host]]:
    hosts: list[Host] = []
    yield hosts
    for h in hosts:
        h.close()


def _run(made: list[Host], host: Host, limits: Limits | None = None) -> Any:
    made.append(host)
    return host.run_one(b"{}", limits or _limits())


# --- the import surface ------------------------------------------------------------------------


def test_the_published_wasi_table_names_exactly_the_linked_imports() -> None:
    """The contract's ``wasi`` rows and the host's ``SIGNATURES`` are one set (both directions)."""
    published = {name for row in WASI_TABLE for name in row["imports"]}
    assert published == set(wasi.SIGNATURES)
    assert wasi.TRAPPING == {n for row in WASI_TABLE for n in row["imports"]
                             if row["behaviour"].startswith("linked as functions that end the instance")}


def test_the_snapshot_imports_only_table_w() -> None:
    bundle_dir = rr.bundle_dir()
    engine = Host.engine_for(CONTRACT)
    module = bundle_mod.load_bundle(engine, bundle_dir, CONTRACT).module
    assert module is not None
    imports = {(i.module, i.name) for i in module.imports}
    assert imports and all(m == W and n in wasi.SIGNATURES for m, n in imports)


@pytest.mark.parametrize(
    ("module", "name"),
    [("env", "getenv"), (W, "sock_open"), (W, "path_open_at"), ("env", "fd_write")],
)
def test_an_import_outside_table_w_is_refused_by_the_linker(module: str, name: str) -> None:
    engine = Host.engine_for(CONTRACT)
    wat = f'(module (import "{module}" "{name}" (func (param i32) (result i32))))'
    with pytest.raises(KeyError):
        wasi.build_linker(engine, wasmtime.Module(engine, wat))


def test_a_bundle_whose_module_imports_outside_table_w_is_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_bundle's own linker check (not only build_linker in isolation) refuses the bundle."""
    bundle_dir = rr.bundle_dir()
    engine = Host.engine_for(CONTRACT)
    bad = wasmtime.Module(engine, '(module (import "env" "getenv" (func)))')
    monkeypatch.setattr(bundle_mod, "_module", lambda *a, **k: bad)
    loaded = bundle_mod.load_bundle(engine, bundle_dir, CONTRACT)
    assert loaded.status == "bundle_mismatch" and loaded.module is None


# --- every row, through the production linker --------------------------------------------------


@pytest.fixture(scope="module")
def table() -> Iterator[tuple[Host, wasmtime.Store, wasmtime.Instance, wasi.CallState]]:
    """One instance importing EVERY Table W function, each re-exported as ``t_<name>``."""
    forwards = []
    for name, (params, results) in wasi.SIGNATURES.items():
        p = " ".join(f"(param {t})" for t in params)
        r = f"(result {_ty(results)})" if results else ""
        gets = " ".join(f"(local.get {i})" for i in range(len(params)))
        forwards.append(f'(func (export "t_{name}") {p} {r} (call ${name} {gets}))')
    host = _host("\n".join(forwards), list(wasi.SIGNATURES))
    store = wasmtime.Store(host.engine)
    store.set_fuel(10**9)
    store.set_epoch_deadline(1 << 40)
    state = wasi.CallState(
        stdin=b"REQUEST",
        budgets={"prelude": 10**9, "load": 10**9, "call": 10**9},
        output_limit=4096,
        random_seed=host.random_seed,
        final_phase="call",
    )
    wasi.bind(state)
    instance = host.pre.instantiate(store)
    yield host, store, instance, state
    wasi.bind(None)
    host.close()


def _call(table: Any, name: str, *args: int) -> Any:
    _, store, instance, _ = table
    return instance.exports(store)["t_" + name](store, *args)


def _mem(table: Any) -> tuple[wasmtime.Memory, wasmtime.Store]:
    _, store, instance, _ = table
    memory = instance.exports(store)["memory"]
    assert isinstance(memory, wasmtime.Memory)
    return memory, store


def _fill(table: Any, at: int = 0, n: int = 64) -> None:
    memory, store = _mem(table)
    memory.write(store, b"\xff" * n, at)


def _peek(table: Any, at: int, n: int) -> bytes:
    memory, store = _mem(table)
    return bytes(memory.read(store, at, at + n))


def test_there_are_no_preopened_directories(table: Any) -> None:
    for fd in range(0, 16):
        assert _call(table, "fd_prestat_get", fd, 0) == wasi.EBADF
        assert _call(table, "fd_prestat_dir_name", fd, 0, 64) == wasi.EBADF


@pytest.mark.parametrize("name", sorted(n for n in wasi.SIGNATURES if n.startswith("path_")))
def test_every_path_call_is_not_capable(table: Any, name: str) -> None:
    params, _ = wasi.SIGNATURES[name]
    assert _call(table, name, *([3] + [0] * (len(params) - 1))) == wasi.ENOTCAPABLE


EBADF_ROWS = sorted(
    {
        n
        for row in WASI_TABLE
        for n in row["imports"]
        if row["behaviour"].startswith("EBADF (8) for every descriptor")
    }
)


def test_the_ebadf_rows_are_the_published_ones() -> None:
    assert "fd_seek" in EBADF_ROWS and "fd_prestat_get" in EBADF_ROWS and len(EBADF_ROWS) >= 16


@pytest.mark.parametrize("name", EBADF_ROWS)
@pytest.mark.parametrize("fd", [0, 1, 2, 3, 4])
def test_every_other_descriptor_call_is_ebadf(table: Any, name: str, fd: int) -> None:
    params, _ = wasi.SIGNATURES[name]
    assert _call(table, name, *([fd] + [0] * (len(params) - 1))) == wasi.EBADF


def test_the_environment_is_empty(table: Any) -> None:
    _fill(table)
    assert _call(table, "environ_sizes_get", 0, 4) == wasi.ESUCCESS
    assert struct.unpack("<II", _peek(table, 0, 8)) == (0, 0)
    _fill(table)
    assert _call(table, "environ_get", 0, 16) == wasi.ESUCCESS
    assert _peek(table, 0, 32) == b"\xff" * 32  # nothing written


def test_the_arguments_are_python_only(table: Any) -> None:
    _fill(table, 0, 128)
    assert _call(table, "args_sizes_get", 0, 4) == wasi.ESUCCESS
    assert struct.unpack("<II", _peek(table, 0, 8)) == (1, len(b"python\x00"))
    assert _call(table, "args_get", 16, 64) == wasi.ESUCCESS
    assert struct.unpack("<I", _peek(table, 16, 4)) == (64,)
    assert _peek(table, 64, 7) == b"python\x00"


@pytest.mark.parametrize("clock", [0, 1, 2, 3])
def test_every_clock_is_the_epoch(table: Any, clock: int) -> None:
    _fill(table)
    assert _call(table, "clock_time_get", clock, 1, 0) == wasi.ESUCCESS
    assert struct.unpack("<Q", _peek(table, 0, 8)) == (0,)
    assert _call(table, "clock_res_get", clock, 8) == wasi.ESUCCESS
    assert struct.unpack("<Q", _peek(table, 8, 8)) == (1,)


def test_random_is_the_contracts_fixed_stream(table: Any) -> None:
    seed = CONTRACT["random"]["seed"].encode("utf-8")
    stream = b"".join(hashlib.sha256(seed + struct.pack("<Q", i)).digest() for i in range(4))
    state: wasi.CallState = table[3]
    state.random_counter, state.random_buffer = 0, b""
    assert _call(table, "random_get", 0, 40) == wasi.ESUCCESS
    assert _peek(table, 0, 40) == stream[:40]
    assert _call(table, "random_get", 100, 50) == wasi.ESUCCESS  # continues, never restarts
    assert _peek(table, 100, 50) == stream[40:90]


def test_stdin_is_the_request_and_only_descriptor_zero_reads(table: Any) -> None:
    state: wasi.CallState = table[3]
    state.pos = 0
    memory, store = _mem(table)
    memory.write(store, struct.pack("<II", 64, 32), 0)
    assert _call(table, "fd_read", 5, 0, 1, 8) == wasi.EBADF
    assert _call(table, "fd_read", 3, 0, 1, 8) == wasi.EBADF
    assert _call(table, "fd_read", 0, 0, 1, 8) == wasi.ESUCCESS
    assert struct.unpack("<I", _peek(table, 8, 4)) == (7,)
    assert _peek(table, 64, 7) == b"REQUEST"
    assert _call(table, "fd_read", 0, 0, 1, 8) == wasi.ESUCCESS  # then end of file
    assert struct.unpack("<I", _peek(table, 8, 4)) == (0,)


def test_writes_go_only_to_one_two_and_three(table: Any) -> None:
    state: wasi.CallState = table[3]
    memory, store = _mem(table)
    memory.write(store, b"secret", 64)
    memory.write(store, struct.pack("<II", 64, 6), 0)
    before = bytes(state.reply)
    for fd in (0, 4, 5, 100):
        assert _call(table, "fd_write", fd, 0, 1, 8) == wasi.EBADF
    assert _call(table, "fd_write", 2, 0, 1, 8) == wasi.ESUCCESS  # stderr: counted, not kept
    assert bytes(state.reply) == before and state.stderr_bytes >= 6


def test_only_the_three_standard_descriptors_stat(table: Any) -> None:
    for fd in (0, 1, 2):
        _fill(table)
        assert _call(table, "fd_fdstat_get", fd, 0) == wasi.ESUCCESS
        assert _peek(table, 0, 1) == b"\x02"  # character device
    for fd in (3, 4, 9):
        assert _call(table, "fd_fdstat_get", fd, 0) == wasi.EBADF


def test_sched_yield_and_poll_report_nothing(table: Any) -> None:
    _fill(table)
    assert _call(table, "sched_yield") == wasi.ESUCCESS
    assert _call(table, "poll_oneoff", 0, 64, 1, 0) == wasi.ESUCCESS
    assert struct.unpack("<I", _peek(table, 0, 4)) == (0,)


@pytest.mark.parametrize("name", sorted(wasi.TRAPPING))
def test_sockets_and_proc_raise_end_the_instance_with_rule_error(
    made: list[Host], name: str
) -> None:
    assert {"sock_accept", "sock_recv", "sock_send", "sock_shutdown", "proc_raise"} <= wasi.TRAPPING
    params, results = wasi.SIGNATURES[name]
    args = " ".join("(i32.const 0)" for _ in params)
    call = f"(call ${name} {args})" if not results else f"(drop (call ${name} {args}))"
    # A reply is written FIRST: the stub, not a missing reply, must decide the reason.
    data = "".join(f"\\{b:02x}" for b in REPLY_7)
    wat_host = _host(
        WRITE_FN
        + f'(data (i32.const 64) "{data}")\n(func (export "run") '
        + f"(call $w (i32.const 1) (i32.const 64) (i32.const {len(REPLY_7)})) {call})",
        ["fd_write", name],
    )
    outcome = _run(made, wat_host)
    assert outcome.reduced.reason == Reason.RULE_ERROR and outcome.reduced.count is None


def test_proc_exit_before_a_reply_is_output_invalid(made: list[Host]) -> None:
    host = _host('(func (export "run") (call $proc_exit (i32.const 0)))', ["proc_exit"])
    assert _run(made, host).reduced.reason == Reason.RULE_OUTPUT_INVALID


# --- the phase markers on descriptor 3 (classes 17, 18 at the host), on a real store ------------


def test_markers_in_order_then_one_reply_line_is_a_count(made: list[Host]) -> None:
    outcome = _run(made, _guest([(3, b"load"), (3, b"call"), (1, REPLY_7)]))
    assert outcome.reduced.count == 7
    assert all(outcome.fuel_used[p] > 0 for p in ("prelude", "load", "call"))


@pytest.mark.parametrize(
    "markers",
    [
        [b"load", b"call", b"call"],  # replay (17)
        [b"load", b"load"],  # replay (17)
        [b"call"],  # out of order (18)
        [b"load", b"call", b"load"],  # backwards (18)
        [b"reset"],  # unknown (18)
        [b"lo", b"ad"],  # split marker
        [b"load\n"],  # not the exact bytes
    ],
)
def test_any_other_marker_sequence_is_rule_error(made: list[Host], markers: list[bytes]) -> None:
    steps: list[tuple[int, bytes] | int] = [(3, m) for m in markers]
    outcome = _run(made, _guest([*steps, (1, REPLY_7)]))
    assert outcome.reduced.reason == Reason.RULE_ERROR and outcome.reduced.count is None



def test_a_marker_switches_the_fuel_to_the_next_phase_budget(made: list[Host]) -> None:
    """Prelude budget too small for the burn, load budget ample: the burn must run on the LOAD
    budget the marker installed (a marker that did not set the fuel would die in the load phase)."""
    limits = _limits(prelude_fuel=5_000, load_fuel=2_000_000, call_fuel=2_000_000)
    outcome = _run(
        made, _guest([(3, b"load"), 100_000, (3, b"call"), 100_000, (1, REPLY_7)]), limits
    )
    assert outcome.reduced.count == 7
    assert outcome.fuel_used["load"] > 100_000 and outcome.fuel_used["call"] > 100_000


@pytest.mark.parametrize(
    ("steps", "reason", "phase"),
    [
        ([500_000, (3, b"load")], Reason.RULE_ERROR, "prelude"),
        ([(3, b"load"), 500_000], Reason.RULE_RESOURCE_LIMIT, "load"),
        ([(3, b"load"), (3, b"call"), 500_000], Reason.RULE_RESOURCE_LIMIT, "call"),
    ],
)
def test_out_of_fuel_is_judged_by_the_phase(
    made: list[Host], steps: list[Any], reason: Reason, phase: str
) -> None:
    limits = _limits(prelude_fuel=100_000, load_fuel=100_000, call_fuel=100_000)
    outcome = _run(made, _guest([*steps, (1, REPLY_7)]), limits)
    assert outcome.reduced.reason == reason
    assert outcome.fuel_used[phase] >= 100_000 - 64  # the whole budget of that phase was spent


def test_a_replayed_marker_cannot_refill_fuel(made: list[Host]) -> None:
    """Class 17 on a real store: a call phase too small for the burn, refilled by a second 'call'."""
    limits = _limits(call_fuel=60_000)
    outcome = _run(
        made,
        _guest([(3, b"load"), (3, b"call"), 5_000, (3, b"call"), 5_000, (1, REPLY_7)]),
        limits,
    )
    assert outcome.reduced.reason == Reason.RULE_ERROR


# --- the one-line reply (class 19 at the host) -------------------------------------------------


@pytest.mark.parametrize(
    "writes",
    [
        [REPLY_7, REPLY_7],  # a forged second line
        [REPLY_7, b"\n"],  # a trailing blank line
        [REPLY_7[:-1]],  # no newline
        [REPLY_7[:20], REPLY_7[20:], b"x"],  # trailing garbage
    ],
)
def test_anything_but_exactly_one_reply_line_is_output_invalid(
    made: list[Host], writes: list[bytes]
) -> None:
    steps: list[tuple[int, bytes] | int] = [(3, b"load"), (3, b"call")]
    steps += [(1, w) for w in writes]
    outcome = _run(made, _guest(steps))
    assert outcome.reduced.reason == Reason.RULE_OUTPUT_INVALID


def test_one_reply_line_in_several_writes_is_a_count(made: list[Host]) -> None:
    steps: list[tuple[int, bytes] | int] = [(3, b"load"), (3, b"call")]
    steps += [(1, REPLY_7[:10]), (1, REPLY_7[10:])]
    assert _run(made, _guest(steps)).reduced.count == 7


# --- limits at their edge ----------------------------------------------------------------------

GROW = """
(func (export "run")
  (if (i32.eq (memory.grow (i32.const {pages})) (i32.const -1))
    (then (call $w (i32.const 1) (i32.const 64) (i32.const {n_fail})))
    (else (call $w (i32.const 1) (i32.const {at_ok}) (i32.const {n_ok})))))
"""


def _grow_guest(pages: int) -> Host:
    fail = b'{"phase":"call","result":{"t":"int","v":"0"}}\n'
    ok = b'{"phase":"call","result":{"t":"int","v":"1"}}\n'
    data = "".join(f"\\{b:02x}" for b in fail + ok)
    body = (
        WRITE_FN
        + f'(data (i32.const 64) "{data}")'
        + GROW.format(pages=pages, n_fail=len(fail), at_ok=64 + len(fail), n_ok=len(ok))
    )
    return _host(body, ["fd_write"])


def test_memory_grows_exactly_to_the_limit_and_no_further(made: list[Host]) -> None:
    limit = 4 * PAGE
    assert _run(made, _grow_guest(3), _limits(memory_bytes=limit)).reduced.count == 1
    assert _run(made, _grow_guest(4), _limits(memory_bytes=limit)).reduced.count == 0


def test_memory_cap_is_the_request_limit_not_the_engine_default(made: list[Host]) -> None:
    floor = CONTRACT["limits_floor"]["memory_bytes"]
    pages = floor // PAGE  # the module has 1 page: growing by floor/PAGE goes one page past
    assert _run(made, _grow_guest(pages - 1), _limits(memory_bytes=floor)).reduced.count == 1
    assert _run(made, _grow_guest(pages), _limits(memory_bytes=floor)).reduced.count == 0


RECURSE = """
(func $rec (param i32) (result i32)
  local.get 0 i32.eqz
  if (result i32) i32.const 0
  else local.get 0 i32.const 1 i32.sub call $rec i32.const 1 i32.add end)
(data (i32.const 64) "{reply}")
(func (export "run")
  (drop (call $rec (i32.const {depth})))
  (call $w (i32.const 1) (i32.const 64) (i32.const {n})))
"""


def _recurse_guest(depth: int) -> Host:
    reply = "".join(f"\\{b:02x}" for b in REPLY_7)
    body = WRITE_FN + RECURSE.format(reply=reply, depth=depth, n=len(REPLY_7))
    return _host(body, ["fd_write"])


def test_the_engine_grants_the_stack_floor(made: list[Host]) -> None:
    """~48 bytes a frame on x86-64 Cranelift (measured, Review R1): 25,000 frames need ~1.2 MiB, so
    they run under the 2 MiB floor and trap under 1 MiB or the 512 KiB engine default."""
    assert CONTRACT["limits_floor"]["wasm_stack_bytes"] == 2 * 1024 * 1024
    assert _run(made, _recurse_guest(25_000)).reduced.count == 7


def test_stack_overflow_is_rule_resource_limit(made: list[Host]) -> None:
    outcome = _run(made, _recurse_guest(400_000))
    assert outcome.reduced.reason == Reason.RULE_RESOURCE_LIMIT


def test_the_wall_time_backstop_interrupts_a_spin(made: list[Host]) -> None:
    marks = "".join(f"\\{b:02x}" for b in b"loadcall")
    spin = (
        WRITE_FN
        + f'(data (i32.const 64) "{marks}")'
        + (
            '(func (export "run") (call $w (i32.const 3) (i32.const 64) (i32.const 4))'
            " (call $w (i32.const 3) (i32.const 68) (i32.const 4)) (loop $l (br $l)))"
        )
    )
    outcome = _run(made, _host(spin, ["fd_write"]), _limits(wall_ms=50, call_fuel=1 << 32))
    assert outcome.reduced.reason == Reason.RULE_TIMEOUT


# --- a guest pointer outside its memory --------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "(drop (call $fd_write (i32.const 1) (i32.const 70000) (i32.const 1) (i32.const 0)))",
        "(drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 70000)))",
        "(i32.store (i32.const 0) (i32.const 65000)) (i32.store (i32.const 4) (i32.const 1000))"
        " (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))",
        "(drop (call $fd_write (i32.const 1) (i32.const -8) (i32.const 1) (i32.const 8)))",
        "(drop (call $random_get (i32.const 65000) (i32.const 1000)))",
        "(drop (call $clock_time_get (i32.const 0) (i64.const 0) (i32.const 65532)))",
        "(i32.store (i32.const 0) (i32.const 65530)) (i32.store (i32.const 4) (i32.const 100))"
        " (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))",
    ],
)
def test_a_pointer_outside_memory_is_rule_error_never_an_exception(
    made: list[Host], body: str
) -> None:
    host = _host(
        f'(func (export "run") {body})', ["fd_write", "fd_read", "random_get", "clock_time_get"]
    )
    outcome = _run(made, host)  # must not raise
    assert outcome.reduced.reason == Reason.RULE_ERROR
