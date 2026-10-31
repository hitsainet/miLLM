"""The framed protocol between miLLM's API process and its `rule-runner` worker (039 S3; FTID §3.6).

One Unix-domain socket in a shared `emptyDir` (`/run/millm-rules/runner.sock`). Each message is a
4-byte big-endian length followed by that many bytes of canonical JSON (one object). A connection
carries any number of request/reply pairs, in order.

Requests:

- ``{"op": "load", "code_sha256", "canonical_utf8"}`` → ``{"ok": true}`` or ``{"error": code}``.
  The worker checks the bytes against the hash (the contract's hash rule) and caches the parsed
  document by hash in memory. It never reads a database or a file for it (TD39-34).
- ``{"op": "call", "code_sha256", "function", "params", "text", "limits"}`` → ``{"count": n}``,
  ``{"reason": code}`` (a contract reason) or ``{"error": code}``. ``unknown_module`` means a
  restarted worker lost its cache: the client re-sends ``load`` once.
- ``{"op": "health"}`` → ``{"status", "pid", "contract", "bundle_digest"}``.

⚠ Standard library only; both processes import it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from typing import Any

from millm.rules.runner_contract import canonical_json

#: The largest frame either side accepts: a 1 MiB source and a 65,536-code-point text (at most
#: 4 bytes each) fit with room; anything larger is a protocol fault, not a request.
MAX_FRAME = 8 << 20
_HEADER = struct.Struct(">I")

OP_LOAD, OP_CALL, OP_HEALTH = "load", "call", "health"
UNKNOWN_MODULE = "unknown_module"


class ProtocolError(Exception):
    """A frame that is not one canonical JSON object within `MAX_FRAME`, or a closed peer."""


def encode(message: dict[str, Any]) -> bytes:
    body = canonical_json(message)
    if len(body) > MAX_FRAME:
        raise ProtocolError("frame over MAX_FRAME")
    return _HEADER.pack(len(body)) + body


def _decode(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ProtocolError("frame is not JSON") from None
    if not isinstance(value, dict):
        raise ProtocolError("frame is not an object")
    return value


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ProtocolError("peer closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> dict[str, Any]:
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if length > MAX_FRAME:
        raise ProtocolError("frame over MAX_FRAME")
    return _decode(_recv_exact(sock, length))


def write_frame(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(encode(message))


async def read_frame_async(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        (length,) = _HEADER.unpack(await reader.readexactly(_HEADER.size))
        if length > MAX_FRAME:
            raise ProtocolError("frame over MAX_FRAME")
        return _decode(await reader.readexactly(length))
    except asyncio.IncompleteReadError:
        raise ProtocolError("peer closed") from None


async def write_frame_async(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write(encode(message))
    await writer.drain()
