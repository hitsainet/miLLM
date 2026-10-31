"""A failed batch names the exception's class and an error id, never its text (2026-10-20).

`BatchRunner._fail` stored `f"{type(exc).__name__}: {exc}"` in `batch.errors`, which
`GET /v1/batches/{id}` returns, and logged only the class — so the text went to the client and
not to the log. Now the reverse. Through the real app, runner and file store.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from tests.unit.batch_fixtures import (  # noqa: F401
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    harness,
)

SECRET = "CUDA error at /opt/secret/kernels.cu:42 password=hunter2"


async def test_the_stored_error_has_the_id_and_the_log_has_the_text(harness, monkeypatch):
    from millm.core.config import settings
    from millm.services.batch.executors import RowExecutor

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)
    async with client_for(harness.app()) as client:
        created = await harness.create(client, [completion_line(f"c{i}") for i in range(3)])
    assert created.status_code == 200, created.text
    await harness.settle_validation()
    batch_id = created.json()["id"]

    async def boom(self, row):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(RowExecutor, "run_one", boom)
    with capture_logs() as logs:
        await harness.drain()

    async with client_for(harness.app()) as client:
        served = (await client.get(f"/v1/batches/{batch_id}")).json()
    assert served["status"] == "failed"
    assert SECRET not in str(served) and "hunter2" not in str(served)
    message = served["errors"]["data"][-1]["message"]
    assert message.startswith("RuntimeError"), "FTID §12: the failure names the class"

    logged = [e for e in logs if e.get("event") == "batch_failed"]
    assert len(logged) == 1 and SECRET in logged[0]["error_message"]
    assert logged[0]["error_id"] in message
