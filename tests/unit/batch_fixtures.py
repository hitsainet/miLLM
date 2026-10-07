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


# --------------------------------------------------------------------------- the full harness


class SpyEmitter:
    """Records every `batch:progress` payload (wiring assertions: payload AND count)."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def emit_batch_progress(self, payload: dict) -> None:
        self.payloads.append(payload)


class BatchHarness:
    """A tiny REAL model served by a real InferenceService, a real ModelService (leases), a real
    BatchRunner and the real app — over one file-backed SQLite database."""

    def __init__(self, factory: Any, *, model_name: str = "tiny") -> None:
        from unittest.mock import MagicMock

        from millm.ml.model_loader import ModelLoader
        from millm.services.batch.runner import BatchRunner
        from millm.services.model_service import ModelService
        from tests.unit.f25_fixtures import make_service, word_model, word_tokenizer

        self.factory = factory
        self.inference = make_service(word_model(), word_tokenizer(), name=model_name)
        self.emitter = SpyEmitter()
        self.loader = ModelLoader()

        def model_service(session: Any) -> Any:
            from millm.db.repositories.model_repository import ModelRepository

            return ModelService(
                repository=ModelRepository(session), downloader=MagicMock(), loader=self.loader,
                emitter=None, inference_service=self.inference,
            )

        self.model_service = model_service
        self.runner = BatchRunner(
            factory, inference_provider=lambda: self.inference,
            model_service_factory=model_service, emitter=self.emitter,
        )

    async def seed_model(self, *, model_id: int = 1, name: str = "tiny", status: Any = None,
                         **fields: Any) -> None:
        from millm.db.models.model import ModelStatus
        from tests.support.factories import make_model

        async with self.factory() as session:
            session.add(make_model(id=model_id, name=name, status=status or ModelStatus.LOADED,
                                   **fields))
            await session.commit()

    def app(self):
        from fastapi import Depends

        from millm.api.dependencies import get_db, get_inference_service, get_model_service
        from millm.api.routes.openai.batches import runner_dependency

        # ⚠ `Depends` as a DEFAULT, not an annotation: this module has postponed annotations,
        # and FastAPI would read an unresolvable string annotation as a query parameter.
        async def _model_service(session=Depends(get_db)):  # noqa: B008
            return self.model_service(session)

        return app_with_db(self.factory, {
            get_inference_service: lambda: self.inference,
            get_model_service: _model_service,
            runner_dependency: lambda: self.runner,
        })

    async def settle_validation(self) -> None:
        import asyncio

        while self.runner._validations:
            await asyncio.gather(*list(self.runner._validations), return_exceptions=False)

    async def drain(self, max_steps: int = 500) -> int:
        """Run `step()` until nothing progresses; returns the number of steps taken."""
        await self.settle_validation()
        for n in range(max_steps):
            if not await self.runner.step():
                return n
        raise AssertionError("the runner never went idle")

    async def batch(self, batch_id: str) -> Any:
        from millm.db.models.batch import Batch

        async with self.factory() as session:
            return await session.get(Batch, batch_id)

    async def lines(self, client: httpx.AsyncClient, file_id: Optional[str]) -> list[dict]:
        if file_id is None:
            return []
        response = await client.get(f"/v1/files/{file_id}/content")
        assert response.status_code == 200, response.text
        return [json.loads(line) for line in response.text.splitlines() if line]

    async def create(self, client: httpx.AsyncClient, lines: list[Any], *,
                     endpoint: str = "/v1/completions", headers: Optional[dict] = None,
                     **body: Any) -> httpx.Response:
        file_id = (await upload(client, jsonl(lines))).json()["id"]
        return await client.post(
            "/v1/batches",
            json={"input_file_id": file_id, "endpoint": endpoint, "completion_window": "24h",
                  **body},
            headers=headers or {},
        )


@pytest_asyncio.fixture
async def harness(batch_db, batch_dir):
    from tests.unit.f25_fixtures import clear_loaded

    h = BatchHarness(batch_db)
    await h.seed_model()
    yield h
    await h.runner.stop()
    clear_loaded()
