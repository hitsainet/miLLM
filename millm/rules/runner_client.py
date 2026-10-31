"""miLLM's side of the `rule-runner` socket (039 S3; FR-39.26; FTID §3.6).

The API process asks; it never runs operator code itself and never imports `wasmtime`
(`test_runner_hostile.py::test_the_api_process_never_loads_the_engine`). One :class:`RunnerSession` per follow-up job: it opens the socket,
sends each code document once per connection (``load``), then ``call`` frames carrying only the
hash, the window text and the parameters; a worker that restarted answers ``unknown_module`` and
the document is re-sent ONCE.

The reply is reduced HERE to ``CodeResult(count)`` or ``CodeResult(reason)`` — a non-negative
integer within the contract's count cap, or one of the contract's reasons — and everything else is
discarded. Nothing the worker says is logged (FR-39.28). Faults, per FTDD §6.8:

- the socket cannot be reached → ``rule_worker_unavailable``;
- the worker dies or breaks the protocol mid-call → ``rule_error``;
- the text is over the input limit → ``rule_input_too_large`` (never sent);
- no reply within the wall time plus a margin → ``rule_timeout`` with ``answered=False``, which
  every caller keeps PENDING (D57, D63): only the worker's own ``rule_timeout`` is final.

⚠ Standard library only, plus `runner_contract` / `runner_protocol`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from millm.rules import runner_contract as rc
from millm.rules import runner_protocol as proto

logger = logging.getLogger(__name__)

WORKER_UNAVAILABLE = "rule_worker_unavailable"
RULE_ERROR = "rule_error"
RULE_TIMEOUT = "rule_timeout"
INPUT_TOO_LARGE = "rule_input_too_large"
#: Seconds the client waits past the call's own wall time before it gives up on a reply.
REPLY_MARGIN_S = 5.0


@dataclass(frozen=True)
class CodeResult:
    """One code rule on one text: exactly one of `count` and `reason`."""

    count: Optional[int] = None
    reason: Optional[str] = None
    #: False only when the worker never answered within the client's wait (`rule_timeout` from
    #: the CLIENT's side: the call may never have run). Both callers keep the result pending
    #: (D57 stateless routes, D63 after the response). Not part of equality.
    answered: bool = field(default=True, compare=False)


def call_limits(settings: Any) -> dict[str, int]:
    """The per-call limits miLLM sends: the deployment's settings, the contract floor for the
    rest (fuel, input). Names are the contract's `request_limits`."""
    floor = rc.floor()
    limits = {
        "call_fuel": floor["call_fuel"],
        "load_fuel": floor["load_fuel"],
        "prelude_fuel": floor["prelude_fuel"],
        "input_code_points": floor["input_code_points"],
        "memory_bytes": int(settings.RULE_CODE_CALL_MEMORY_MIB) << 20,
        "output_bytes": int(settings.RULE_CODE_CALL_OUTPUT_BYTES),
        "wall_ms": int(settings.RULE_CODE_CALL_WALL_MS),
    }
    return {name: limits[name] for name in rc.contract()["request_limits"]}


def floor_violations(settings: Any) -> list[str]:
    """Every per-call limit below the contract floor (FTASKS 17.4). Non-empty → code rules are
    NOT run: the worker would refuse each call, and a host may never lower a floor."""
    floor = rc.floor()
    limits = call_limits(settings)
    out = [name for name, value in sorted(limits.items())
           if name in floor and value < floor[name]]
    if limits["wall_ms"] < 1:
        out.append("wall_ms")
    return out


def reduce_reply(reply: Any) -> CodeResult:
    """The host's reading of one worker reply: a count within the cap, or a contract reason.
    Anything else — text, a structure, a bool, an unknown code — is `rule_error`."""
    if isinstance(reply, dict) and set(reply) == {"count"}:
        count = reply["count"]
        if type(count) is int and 0 <= count <= rc.count_cap():
            return CodeResult(count=count)
    if isinstance(reply, dict) and set(reply) == {"reason"} and reply["reason"] in rc.reasons():
        return CodeResult(reason=reply["reason"])
    return CodeResult(reason=RULE_ERROR)


class RunnerSession:
    """One connection to the worker. Use as ``async with RunnerSession(path) as s``."""

    def __init__(self, socket_path: str, limits: dict[str, int], *,
                 deadline: Optional[float] = None) -> None:
        self.socket_path = socket_path
        self.limits = limits
        #: A `time.monotonic()` instant (the stateless routes' total budget, D57 review R2): no
        #: call starts after it, and no reply is waited for past it — the rest stay pending.
        #: None after the response (each call is bounded on its own).
        self.deadline = deadline
        #: Calls not started because the deadline had passed.
        self.past_deadline = 0
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._loaded: set[str] = set()
        self.unavailable = False
        self.calls = 0

    async def __aenter__(self) -> "RunnerSession":
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path), timeout=REPLY_MARGIN_S)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.info("rule_runner_unreachable error=%s", type(exc).__name__)
            self.unavailable = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                # Review R2: bounded — a peer that stopped reading never lets unsent bytes drain.
                await asyncio.wait_for(self._writer.wait_closed(), timeout=REPLY_MARGIN_S)
            except Exception:  # noqa: BLE001 - closing a dead peer
                pass

    async def _ask(self, message: dict[str, Any]) -> dict[str, Any]:
        assert self._reader is not None and self._writer is not None
        # Review R1 (L1): the worker may wait up to the wall time for a slot and then run up to
        # the wall time; waiting less reported `rule_timeout` (final) for a call that never ran
        # (at RULE_CODE_CALL_WALL_MS above ~5 s). Busy is the worker's own
        # `rule_worker_unavailable`, which stays pending.
        timeout = 2 * self.limits["wall_ms"] / 1000 + REPLY_MARGIN_S
        if self.deadline is not None:
            timeout = min(timeout, max(0.0, self.deadline - time.monotonic()))
        # Review R2: the WRITE is inside the wait too — `drain()` on a worker that stopped reading
        # (wedged, frozen) blocked forever with no timeout, holding the request or the job.
        writer, reader = self._writer, self._reader

        async def exchange() -> dict[str, Any]:
            await proto.write_frame_async(writer, message)
            return await proto.read_frame_async(reader)

        return await asyncio.wait_for(exchange(), timeout=timeout)

    async def _load(self, code_sha256: str, canonical_utf8: str) -> bool:
        reply = await self._ask({"op": proto.OP_LOAD, "code_sha256": code_sha256,
                                 "canonical_utf8": canonical_utf8})
        if reply == {"ok": True}:
            self._loaded.add(code_sha256)
            return True
        logger.warning("rule_runner_load_refused code_sha256=%s error=%s", code_sha256,
                       reply.get("error") if isinstance(reply, dict) else "malformed")
        return False

    async def call(self, code_sha256: str, canonical_utf8: str, function: str,
                   params: dict[str, Any], text: str) -> CodeResult:
        """One code rule on one window's text. Never raises."""
        text = rc.replace_lone_surrogates(text)
        if len(text) > self.limits["input_code_points"]:
            return CodeResult(reason=INPUT_TOO_LARGE)
        if self.unavailable or self._writer is None:
            return CodeResult(reason=WORKER_UNAVAILABLE)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.past_deadline += 1
            return CodeResult(reason=WORKER_UNAVAILABLE)
        self.calls += 1
        try:
            if code_sha256 not in self._loaded and not await self._load(code_sha256,
                                                                        canonical_utf8):
                return CodeResult(reason=RULE_ERROR)
            message = {"op": proto.OP_CALL, "code_sha256": code_sha256, "function": function,
                       "params": params, "text": text, "limits": self.limits}
            reply = await self._ask(message)
            if reply == {"error": proto.UNKNOWN_MODULE}:
                # A restarted worker lost its cache: re-send the bytes ONCE.
                self._loaded.discard(code_sha256)
                if not await self._load(code_sha256, canonical_utf8):
                    return CodeResult(reason=RULE_ERROR)
                reply = await self._ask(message)
            if isinstance(reply, dict) and "error" in reply:
                logger.warning("rule_runner_call_refused code_sha256=%s error=%s", code_sha256,
                               reply.get("error") if isinstance(reply.get("error"), str)
                               else "malformed")
                return CodeResult(reason=RULE_ERROR)
            return reduce_reply(reply)
        except asyncio.TimeoutError:
            self._broken()
            return CodeResult(reason=RULE_TIMEOUT, answered=False)
        except (proto.ProtocolError, OSError) as exc:
            # The worker died or broke the protocol mid-call: this call is `rule_error`, and the
            # connection is gone for the rest of the job.
            logger.warning("rule_runner_call_failed error=%s", type(exc).__name__)
            self._broken()
            return CodeResult(reason=RULE_ERROR)

    def _broken(self) -> None:
        self.unavailable = True
        if self._writer is not None:
            self._writer.close()
            self._writer = None


async def health(socket_path: str) -> Optional[dict[str, Any]]:
    """The worker's health frame, or None when it cannot be reached. Never raises."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(socket_path),
                                                timeout=REPLY_MARGIN_S)
    except (OSError, asyncio.TimeoutError):
        return None
    try:
        await proto.write_frame_async(writer, {"op": proto.OP_HEALTH})
        return await asyncio.wait_for(proto.read_frame_async(reader), timeout=REPLY_MARGIN_S)
    except (OSError, proto.ProtocolError, asyncio.TimeoutError):
        return None
    finally:
        writer.close()
