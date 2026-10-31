"""ENH-008 L10: the two approval events reach Socket.IO once each, with ids and status only.

The service-side half (which call emits which event, once) is in
`tests/unit/services/test_approval_service.py::test_events_are_emitted_once_each_*`; this file
proves the emitter itself sends them, and sends nothing but the ids.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from millm.sockets.progress import ProgressEmitter


async def test_created_and_resolved_reach_the_socket_once_each():
    sio = MagicMock()
    sio.emit = AsyncMock()
    emitter = ProgressEmitter(sio=sio)
    emitter.emit_approval_created({"id": "apr_1", "action": "model_load", "summary": "x"})
    emitter.emit_approval_resolved({"id": "apr_1", "status": "rejected", "reason": "y"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert sio.emit.await_args_list == [
        (("approval:created", {"id": "apr_1", "action": "model_load"}),),
        (("approval:resolved", {"id": "apr_1", "status": "rejected"}),),
    ]


def test_no_socket_is_a_no_op():
    emitter = ProgressEmitter(sio=None)
    emitter.emit_approval_created({"id": "apr_1", "action": "model_load"})
    emitter.emit_approval_resolved({"id": "apr_1", "status": "expired"})
