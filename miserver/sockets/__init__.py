"""
Socket.IO module for miServer.

Handles real-time WebSocket events for progress updates and monitoring.
"""

from miserver.sockets.progress import (
    ProgressEmitter,
    create_socket_io,
    progress_emitter,
    register_handlers,
)

__all__ = [
    "ProgressEmitter",
    "create_socket_io",
    "progress_emitter",
    "register_handlers",
]
