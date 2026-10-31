"""The Wasmtime host inside the `rule-runner` worker (039 S3; FR-39.26; TD39-33).

A port of miDataworks 018's host core (`backend/src/runner/{host,reply,bundle}.py`, main 6c71e7e):
one engine, one compiled snapshot of the PUBLISHED guest (fetched by its sha256, never rebuilt), and
a FRESH instance per call (`instance_rule`). Three fuel budgets on one store (prelude, load, call)
switched by the once-in-order markers of Table W (`runner_wasi`); memory capped by
``Store.set_limits``; wall time by the epoch deadline of an engine-wide 10 ms ticker. Traps map to
contract reasons here; the reply is reduced by :func:`reduce` — a count or a reason, never the
guest's bytes.

⚠ Imported by `runner_worker` ONLY. The API process never imports it (FR-39.26).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import wasmtime

from millm.rules import runner_wasi as wasi
from millm.rules.runner_contract import bundle_digest, canonical_json

EPOCH_TICK_S = 0.010

RULE_TIMEOUT = "rule_timeout"
RULE_RESOURCE_LIMIT = "rule_resource_limit"
RULE_ERROR = "rule_error"
RULE_LOAD_REFUSED = "rule_load_refused"
RULE_OUTPUT_INVALID = "rule_output_invalid"

#: Trap code -> reason (the contract's `guest_interface.traps`). Out of fuel depends on the phase.
TRAP_REASONS: dict[Any, str] = {
    wasmtime.TrapCode.INTERRUPT: RULE_TIMEOUT,
    wasmtime.TrapCode.STACK_OVERFLOW: RULE_RESOURCE_LIMIT,
    wasmtime.TrapCode.ALLOCATION_TOO_LARGE: RULE_RESOURCE_LIMIT,
}
OUT_OF_FUEL_REASONS: dict[str, str] = {
    "prelude": RULE_ERROR,
    "load": RULE_RESOURCE_LIMIT,
    "call": RULE_RESOURCE_LIMIT,
}
OTHER_TRAP = RULE_ERROR


@dataclass(frozen=True)
class Limits:
    call_fuel: int
    load_fuel: int
    prelude_fuel: int
    memory_bytes: int
    input_code_points: int
    output_bytes: int
    wall_ms: int


# ── the reply (the output barrier) ───────────────────────────────────────────────────────

_DECIMAL = re.compile(r"(0|[1-9][0-9]{0,18})")
_TYPED = frozenset({"bool", "float", "str", "bytes", "none", "other"})
_FAIL_REASON: dict[str, str] = {
    "load_refused": RULE_LOAD_REFUSED,
    "error": RULE_ERROR,
    "recursion": RULE_RESOURCE_LIMIT,
    "memory": RULE_RESOURCE_LIMIT,
}


@dataclass
class Reduced:
    """One call's outcome: ``count`` or ``reason``. Nothing of the guest's bytes survives."""

    count: Optional[int] = None
    reason: Optional[str] = None


def _one_line(raw: bytes, output_limit: int) -> Any:
    if not raw or len(raw) > output_limit or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        return None
    try:
        return json.loads(raw[:-1].decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None


def reduce(raw: bytes, *, output_limit: int, count_cap: int) -> Reduced:
    """The host's reading of one `run` reply buffer (the contract's `guest_interface.reply` and
    `host_reduction`). ``bool`` is not an ``int``; the clamp is load-bearing; ``diag`` is never
    kept (miLLM has no dry-run mode)."""
    reply = _one_line(raw, output_limit)
    if not isinstance(reply, dict) or set(reply) != {"phase", "result"}:
        return Reduced(reason=RULE_OUTPUT_INVALID)
    phase, result = reply["phase"], reply["result"]
    if phase not in ("load", "call") or not isinstance(result, dict):
        return Reduced(reason=RULE_OUTPUT_INVALID)
    if "fail" in result:
        kind = result.get("fail")
        reason = _FAIL_REASON.get(kind) if isinstance(kind, str) else None
        if reason is None:
            return Reduced(reason=RULE_OUTPUT_INVALID)
        if kind == "load_refused":
            ok = set(result) == {"fail", "refused"} and isinstance(result.get("refused"), str)
        else:
            ok = set(result) <= {"fail", "diag"}
        return Reduced(reason=reason if ok else RULE_OUTPUT_INVALID)
    if phase != "call":
        return Reduced(reason=RULE_OUTPUT_INVALID)
    if result.get("t") == "int" and set(result) == {"t", "v"}:
        value = result["v"]
        if value == "overflow":
            return Reduced(count=count_cap)
        if isinstance(value, str) and _DECIMAL.fullmatch(value):
            return Reduced(count=min(int(value), count_cap))
    return Reduced(reason=RULE_OUTPUT_INVALID)


def guest_request(*, source: str, function: str, params: dict[str, Any], text: str) -> bytes:
    """The exact request bytes (the contract's `guest_interface.request`): fuel depends on them."""
    return canonical_json({"mode": "run", "source": source, "function": function,
                           "params": params, "text": text, "diagnostics": False})


@dataclass
class Outcome:
    reduced: Reduced
    fuel_used: dict[str, int] = field(default_factory=dict)


class Host:
    """One per worker process: the engine, the snapshot module and the epoch ticker."""

    def __init__(self, engine: wasmtime.Engine, module: wasmtime.Module,
                 contract: dict[str, Any]) -> None:
        self.engine = engine
        self.module = module
        self.count_cap = int(contract["count_cap"])
        self.random_seed = str(contract["random"]["seed"]).encode("utf-8")
        self.linker = wasi.build_linker(engine, module)
        self.pre = self.linker.instantiate_pre(module)
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick, name="epoch-ticker", daemon=True)
        self._ticker.start()

    @staticmethod
    def engine_for(contract: dict[str, Any]) -> wasmtime.Engine:
        config = wasmtime.Config()
        config.consume_fuel = True
        config.epoch_interruption = True
        config.max_wasm_stack = int(contract["limits_floor"]["wasm_stack_bytes"])
        return wasmtime.Engine(config)

    def _tick(self) -> None:
        while not self._stop.wait(EPOCH_TICK_S):
            self.engine.increment_epoch()

    def close(self) -> None:
        self._stop.set()

    def run_one(self, request: bytes, limits: Limits) -> Outcome:
        """Run one guest call on a FRESH instance; never raises for anything the guest does."""
        store = wasmtime.Store(self.engine)
        store.set_limits(memory_size=limits.memory_bytes)
        store.set_fuel(limits.prelude_fuel)
        store.set_epoch_deadline(max(1, math.ceil(limits.wall_ms / (EPOCH_TICK_S * 1000))))
        state = wasi.CallState(
            stdin=request,
            budgets={"prelude": limits.prelude_fuel, "load": limits.load_fuel,
                     "call": limits.call_fuel},
            output_limit=limits.output_bytes,
            random_seed=self.random_seed,
            final_phase="call",
        )
        wasi.bind(state)
        ended: Optional[str] = None
        try:
            instance = self.pre.instantiate(store)
            run = instance.exports(store)["run"]
            assert isinstance(run, wasmtime.Func)
            run(store)
        except wasi.GuestEnd as end:
            ended = end.reason  # None: proc_exit, read what was replied
        except wasmtime.Trap as trap:
            if trap.trap_code == wasmtime.TrapCode.OUT_OF_FUEL:
                ended = OUT_OF_FUEL_REASONS[state.phase]
            else:
                ended = TRAP_REASONS.get(trap.trap_code, OTHER_TRAP)
        except wasmtime.WasmtimeError:
            ended = OTHER_TRAP
        finally:
            wasi.bind(None)
        try:
            left = store.get_fuel()
        except wasmtime.WasmtimeError:
            left = 0
        state.used[state.phase] = state.budgets[state.phase] - left
        fuel = {phase: state.used.get(phase, 0) for phase in ("prelude", "load", "call")}
        if state.fault is not None:
            return Outcome(Reduced(reason=state.fault), fuel)
        if ended is not None:
            return Outcome(Reduced(reason=ended), fuel)
        return Outcome(reduce(bytes(state.reply), output_limit=limits.output_bytes,
                              count_cap=self.count_cap), fuel)


# ── the bundle (the published snapshot, verified) ────────────────────────────────────────


@dataclass
class LoadedBundle:
    status: str  # "ok" | "bundle_mismatch"
    detail: str
    module: Optional[wasmtime.Module]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cwasm_meta(snapshot_sha: str, stack: int) -> dict[str, Any]:
    return {"snapshot_sha256": snapshot_sha, "max_wasm_stack": stack,
            "wasmtime": version("wasmtime")}


def _module(engine: wasmtime.Engine, bundle_dir: Path, snapshot_sha: str, stack: int) -> Any:
    """The precompiled `.cwasm` when its sidecar names THIS snapshot, Wasmtime and stack size;
    otherwise the verified snapshot compiled in process."""
    cwasm, sidecar = bundle_dir / "snapshot.cwasm", bundle_dir / "snapshot.cwasm.json"
    if cwasm.exists() and sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            meta = None
        if meta == _cwasm_meta(snapshot_sha, stack):
            try:
                return wasmtime.Module.deserialize_file(engine, str(cwasm))
            except wasmtime.WasmtimeError:
                pass  # built for another engine or CPU: compile the verified snapshot
    return wasmtime.Module.from_file(engine, str(bundle_dir / "snapshot.wasm"))


def load_bundle(engine: wasmtime.Engine, bundle_dir: Path, contract: dict[str, Any]) -> LoadedBundle:
    """Verify the contract's bundle digest and the snapshot's sha256, then link Table W. Any
    mismatch is `bundle_mismatch`: the worker stays up and answers every call
    `rule_worker_unavailable` — it never runs an unverified guest."""
    block = contract["guest"]["bundle"]
    if bundle_digest(block) != block.get("bundle_digest"):
        return LoadedBundle("bundle_mismatch", "contract bundle_digest", None)
    snapshot = bundle_dir / "snapshot.wasm"
    if not snapshot.is_file():
        return LoadedBundle("bundle_mismatch", "snapshot.wasm absent", None)
    actual = file_sha256(snapshot)
    if actual != block["snapshot"]["sha256"]:
        return LoadedBundle("bundle_mismatch", "snapshot.wasm digest", None)
    stack = int(contract["limits_floor"]["wasm_stack_bytes"])
    module = _module(engine, bundle_dir, actual, stack)
    try:
        wasi.build_linker(engine, module)
    except KeyError:
        return LoadedBundle("bundle_mismatch", "import outside Table W", None)
    return LoadedBundle("ok", "", module)


def precompile(bundle_dir: Path, contract: dict[str, Any]) -> Path:
    """Write `snapshot.cwasm` and its sidecar for the verified snapshot (the image build)."""
    snapshot = bundle_dir / "snapshot.wasm"
    sha = file_sha256(snapshot)
    if sha != contract["guest"]["bundle"]["snapshot"]["sha256"]:
        raise ValueError("snapshot.wasm is not the published guest")
    engine = Host.engine_for(contract)
    module = wasmtime.Module.from_file(engine, str(snapshot))
    out = bundle_dir / "snapshot.cwasm"
    out.write_bytes(module.serialize())
    stack = int(contract["limits_floor"]["wasm_stack_bytes"])
    (bundle_dir / "snapshot.cwasm.json").write_text(json.dumps(_cwasm_meta(sha, stack)),
                                                    encoding="utf-8")
    return out
