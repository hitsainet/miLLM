"""miLLM's runner client: one load per hash, reply reduced, faults named (039 FTASKS 17.2).

Against a scripted worker (`tests/support/stub_runner.py`) speaking the real framed protocol, so each
fault can be produced on demand: an unreachable socket, a worker that forgot its cache, a worker
that dies mid-call, a reply that is not a count, a reply that never comes. The real worker's
behaviour is `test_runner_vectors.py`.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from millm.core.config import settings
from millm.rules import runner_client as client
from millm.rules import runner_contract as rc
from tests.support.stub_runner import StubRunner

DOC = {"derived_from": {"sources": [], "statement": "constructed"},
       "functions": [{"name": "f", "output": {"type": "count"}}],
       "params_schema": {"type": "object"}, "runner_contract": "midataworks.runner/v1",
       "source": "def f(text, params):\n    return 1\n"}
SHA = rc.code_sha256(DOC)
TEXT = rc.canonical_json(DOC).decode("utf-8")


def _limits(**over):
    limits = client.call_limits(settings)
    limits.update(over)
    return limits


def _call(stub, text="abc", limits=None, socket_path=None):
    async def go():
        async with client.RunnerSession(str(socket_path or stub.socket),
                                        limits or _limits()) as session:
            return await session.call(SHA, TEXT, "f", {}, text)

    return asyncio.run(go())


def test_one_load_then_the_call_carries_hash_text_and_params_only():
    with StubRunner() as stub:
        assert _call(stub) == client.CodeResult(count=2)
        ops = [r["op"] for r in stub.received]
        assert ops == ["load", "call"]
        call = stub.calls()[-1]
        assert (call["code_sha256"], call["function"], call["text_len"]) == (SHA, "f", 3)
        assert call["limits"] == client.call_limits(settings)


def test_a_worker_that_forgot_its_cache_gets_the_bytes_once_more():
    with StubRunner() as stub:

        async def go():
            async with client.RunnerSession(str(stub.socket), _limits()) as session:
                first = await session.call(SHA, TEXT, "f", {}, "a")
                stub.forget()  # the worker restarted between two calls
                second = await session.call(SHA, TEXT, "f", {}, "b")
                return first, second

        assert asyncio.run(go()) == (client.CodeResult(count=2), client.CodeResult(count=2))
        assert [r["op"] for r in stub.received] == ["load", "call", "call", "load", "call"]


def test_an_unreachable_worker_is_rule_worker_unavailable(tmp_path):
    with StubRunner() as stub:
        missing = stub.socket.with_name("absent.sock")
        assert _call(stub, socket_path=missing) == client.CodeResult(
            reason="rule_worker_unavailable")


def test_a_worker_dying_mid_call_is_rule_error():
    with StubRunner(reply=lambda m: None) as stub:
        assert _call(stub) == client.CodeResult(reason="rule_error")


def test_a_text_over_the_input_limit_is_never_sent():
    with StubRunner() as stub:
        over = "x" * (rc.floor()["input_code_points"] + 1)
        assert _call(stub, text=over) == client.CodeResult(reason="rule_input_too_large")
        assert stub.calls() == []


def test_a_lone_surrogate_is_replaced_before_sending():
    with StubRunner() as stub:
        assert _call(stub, text="a\ud800b") == client.CodeResult(count=2)


def test_no_reply_within_the_wall_time_is_rule_timeout(monkeypatch):
    monkeypatch.setattr(client, "REPLY_MARGIN_S", 0.2)
    with StubRunner(hold=True) as stub:
        assert _call(stub, limits=_limits(wall_ms=50)) == client.CodeResult(reason="rule_timeout")


@pytest.mark.parametrize("reply,expected", [
    ({"count": 3}, client.CodeResult(count=3)),
    ({"count": 0}, client.CodeResult(count=0)),
    ({"count": rc.count_cap()}, client.CodeResult(count=rc.count_cap())),
    ({"count": rc.count_cap() + 1}, client.CodeResult(reason="rule_error")),
    ({"count": -1}, client.CodeResult(reason="rule_error")),
    ({"count": True}, client.CodeResult(reason="rule_error")),
    ({"count": 1.0}, client.CodeResult(reason="rule_error")),
    ({"count": "3"}, client.CodeResult(reason="rule_error")),
    ({"count": 1, "text": "SECRETZZ"}, client.CodeResult(reason="rule_error")),
    ({"reason": "rule_resource_limit"}, client.CodeResult(reason="rule_resource_limit")),
    ({"reason": "SECRETZZ"}, client.CodeResult(reason="rule_error")),
    ({"error": "function_not_declared"}, client.CodeResult(reason="rule_error")),
    ({}, client.CodeResult(reason="rule_error")),
])
def test_the_reply_is_reduced_to_a_count_or_a_contract_reason(reply, expected):
    with StubRunner(reply=lambda m: reply) as stub:
        assert _call(stub) == expected


def test_nothing_the_worker_says_is_logged(caplog):
    caplog.set_level(logging.DEBUG)
    with StubRunner(reply=lambda m: {"count": 1, "leak": "SECRETZZ"}) as stub:
        _call(stub, text="SECRETZZ in the window")
    assert "SECRETZZ" not in caplog.text


def test_the_limits_sent_are_the_floor_or_the_settings(monkeypatch):
    monkeypatch.setattr(settings, "RULE_CODE_CALL_MEMORY_MIB", 128)
    monkeypatch.setattr(settings, "RULE_CODE_CALL_WALL_MS", 777)
    limits = client.call_limits(settings)
    floor = rc.floor()
    assert set(limits) == set(rc.contract()["request_limits"])
    assert limits["memory_bytes"] == 128 << 20 and limits["wall_ms"] == 777
    assert limits["output_bytes"] == settings.RULE_CODE_CALL_OUTPUT_BYTES
    for name in ("call_fuel", "load_fuel", "prelude_fuel", "input_code_points"):
        assert limits[name] == floor[name]


# ── review R2 (039 follow-ups): the write is bounded, and a session can carry a deadline ──────


class _DeafWorker:
    """Answers the first frame (the load) and then never reads again: a wedged or frozen worker.
    A call frame larger than the socket's buffers then blocks the client's `drain()`."""

    def __init__(self, tmp_path):
        import socket
        import threading

        self.socket = tmp_path / "deaf.sock"
        self._server = socket.socket(socket.AF_UNIX)
        self._server.bind(str(self.socket))
        self._server.listen(1)
        self._conns: list = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        from millm.rules import runner_protocol as proto

        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        self._conns.append(conn)
        proto.read_frame(conn)
        proto.write_frame(conn, {"ok": True})  # then never reads again

    def close(self):
        for conn in self._conns:
            conn.close()
        self._server.close()


def test_a_worker_that_stops_reading_cannot_hold_the_call_past_its_wait(monkeypatch, tmp_path):
    """The call frame (the largest text plus 4 MB of parameters — well over any socket buffer) is
    written inside the client's wait: a worker that stopped reading ends the call as an UNANSWERED
    `rule_timeout` instead of blocking `drain()` forever (pre-R2 the write had no timeout)."""
    import time

    monkeypatch.setattr(client, "REPLY_MARGIN_S", 0.3)
    deaf = _DeafWorker(tmp_path)
    try:
        text = "\U0001F600" * rc.floor()["input_code_points"]

        async def go():
            async with client.RunnerSession(str(deaf.socket), _limits(wall_ms=50)) as session:
                return await asyncio.wait_for(
                    session.call(SHA, TEXT, "f", {"pad": "x" * (4 << 20)}, text), 30)

        started = time.monotonic()
        result = asyncio.run(go())
        assert time.monotonic() - started < 5
        assert result == client.CodeResult(reason="rule_timeout") and result.answered is False
    finally:
        deaf.close()


def test_past_the_sessions_deadline_no_call_starts_and_no_reply_is_awaited(monkeypatch):
    """`deadline` (the stateless routes' budget): a call before it waits at most until it; a call
    after it is not sent (`rule_worker_unavailable`, not counted)."""
    import time

    with StubRunner(hold=True) as stub:
        async def go():
            async with client.RunnerSession(str(stub.socket), _limits(wall_ms=10_000),
                                            deadline=time.monotonic() + 0.4) as session:
                started = time.monotonic()
                first = await session.call(SHA, TEXT, "f", {}, "abc")
                waited = time.monotonic() - started
                second = await session.call(SHA, TEXT, "f", {}, "abc")
                return first, waited, second, session.calls

        first, waited, second, calls = asyncio.run(go())
        stub.release.set()
    # Held past the deadline: unanswered (the stateless routes keep it pending), bounded by it.
    assert first == client.CodeResult(reason="rule_timeout") and first.answered is False
    assert waited < 2.0
    assert second == client.CodeResult(reason="rule_worker_unavailable")
    assert calls == 1
    assert len(stub.calls()) == 1


def test_a_call_after_the_deadline_is_not_sent_even_on_a_healthy_connection():
    import time

    with StubRunner() as stub:
        async def go():
            async with client.RunnerSession(str(stub.socket), _limits(),
                                            deadline=time.monotonic() + 0.3) as session:
                first = await session.call(SHA, TEXT, "f", {}, "abc")
                await asyncio.sleep(0.4)
                second = await session.call(SHA, TEXT, "f", {}, "abc")
                return first, second, session.calls, session.past_deadline

        first, second, calls, past = asyncio.run(go())
    assert first == client.CodeResult(count=2)
    assert second == client.CodeResult(reason="rule_worker_unavailable")
    assert (calls, past, len(stub.calls())) == (1, 1, 1)
