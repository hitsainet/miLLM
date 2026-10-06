"""Shared fixtures for Feature 26 (Batch API) tests.

* `batch_db` — a FILE-backed SQLite database with foreign keys enforced and every table created,
  yielding an `async_sessionmaker`. File-backed, not `:memory:`, because the runner, the
  validator and the routes each open their own sessions — as production does — and must see one
  database.
* `batch_dir` — a private BATCH_FILES_DIR under `tmp_path` (FTID §8 "Isolation").
* `app_client(factory, **overrides)` — the REAL `create_app()` over that database, called through
  an in-loop `httpx.AsyncClient` (an `ASGITransport`), so routes, exception handlers and the
  OpenAPI document are the production ones.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from millm.db.base import Base


@pytest_asyncio.fixture
async def batch_db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch.db'}", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def batch_dir(tmp_path, monkeypatch):
    from millm.core.config import settings

    root = tmp_path / "batch_files"
    monkeypatch.setattr(settings, "BATCH_FILES_DIR", str(root))
    return root


def app_with_db(factory: Any, overrides: Optional[dict] = None):
    """The real app with `get_db` bound to the test database (and any other overrides)."""
    from millm.api.dependencies import get_db
    from millm.main import create_app

    app = create_app()

    async def _db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    for dep, value in (overrides or {}).items():
        app.dependency_overrides[dep] = value
    return app


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


def jsonl(lines: list[Any]) -> bytes:
    """Lines as JSONL: dicts are dumped, bytes/str are written as given."""
    out = []
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            out.append(bytes(line))
        elif isinstance(line, str):
            out.append(line.encode())
        else:
            out.append(json.dumps(line).encode())
    return b"\n".join(out) + b"\n"


def completion_line(custom_id: str, prompt: str = "w1 w2", *, model: str = "tiny", **body) -> dict:
    """A /v1/completions scoring line (logprobs + max_tokens=1)."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/completions",
        "body": {"model": model, "prompt": prompt, "max_tokens": 1, "logprobs": 2, **body},
    }


async def upload(client: httpx.AsyncClient, data: bytes, *, purpose: str = "batch",
                 filename: str = "in.jsonl", headers: Optional[dict] = None) -> httpx.Response:
    return await client.post(
        "/v1/files",
        data={"purpose": purpose},
        files={"file": (filename, data, "application/jsonl")},
        headers=headers or {},
    )
