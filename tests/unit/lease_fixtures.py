"""Shared harness for Feature 29 tests: a REAL ModelService over a fake repository.

What is real: `ModelService` (every rule under test), `ModelLoader` / `LoadedModelState` (the
residency the lease reads), and the `LeaseRegistry`. What is faked: the database (an in-memory
row store whose `update`s really move rows), the background load worker (a counting stand-in,
so "no load started" is a call count of 0), the placement pre-check (it reads nvidia-smi) and
the probe-row bookkeeping an unload writes.

Fixtures disagree with the defect: the resident model is id 1 ("m1"), the requested one id 2
("m2"), and two holders (`midataworks`, `mistudio-agent`) are used.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from millm.db.models.model import ModelStatus
from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader
from millm.services.model_lease import LeaseRegistry, set_lease_registry
from tests.support.factories import make_model


class FakeRepo:
    """Rows by id; `update`/`update_status` move them, so 'no row moved' is checkable."""

    def __init__(self, *rows: Any) -> None:
        self.rows = {row.id: row for row in rows}
        self.status_writes: list[tuple[int, Any]] = []

    async def get_by_id(self, model_id: int) -> Any:
        return self.rows.get(model_id)

    async def get_all(self) -> list[Any]:
        return list(self.rows.values())

    async def find_by_name(self, name: str) -> Any:
        return next((r for r in self.rows.values() if r.name == name), None)

    async def get_by_name(self, name: str) -> Any:
        return next((r for r in self.rows.values() if r.name == name), None)

    async def get_locked_model(self) -> Any:
        return next((r for r in self.rows.values() if r.locked), None)

    async def update_status(self, model_id: int, status: Any, error_message: Any = None) -> Any:
        self.status_writes.append((model_id, status))
        row = self.rows[model_id]
        row.status = status
        row.error_message = error_message
        return row

    async def update(self, model_id: int, **fields: Any) -> Any:
        row = self.rows[model_id]
        if "status" in fields:
            self.status_writes.append((model_id, fields["status"]))
        for key, value in fields.items():
            setattr(row, key, value)
        return row


class Clock:
    def __init__(self) -> None:
        self.mono = 5000.0
        self.wall = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


def install_registry(clock: Clock | None = None) -> tuple[LeaseRegistry, Clock]:
    """A process registry with a controllable clock, reading the REAL loader's residency."""
    clock = clock or Clock()
    registry = LeaseRegistry(monotonic=lambda: clock.mono, now=lambda: clock.wall)
    set_lease_registry(registry)
    return registry, clock


def set_resident(model_id: int = 1, name: str = "m1") -> None:
    LoadedModelState().set(
        LoadedModel(
            model_id=model_id,
            model_name=name,
            model=SimpleNamespace(),
            tokenizer=None,
            loaded_at=datetime(2026, 10, 6),
        )
    )


def clear_resident() -> None:
    LoadedModelState()._loaded = None
    LoadedModelState()._unloading = False


def build_service(*, locked_resident: bool = False) -> tuple[Any, FakeRepo]:
    """Model 1 resident and LOADED, model 2 READY; a real ModelService over them."""
    from millm.services.model_service import ModelService

    m1 = make_model(id=1, name="m1", status=ModelStatus.LOADED, locked=locked_resident)
    m2 = make_model(id=2, name="m2", status=ModelStatus.READY)
    repo = FakeRepo(m1, m2)
    inference = MagicMock()
    inference.request_queue.pending_count = 0
    service = ModelService(
        repository=repo,  # type: ignore[arg-type]
        downloader=MagicMock(),
        loader=ModelLoader(),
        inference_service=inference,
    )
    # The placement pre-check reads nvidia-smi; the background load needs a GPU.
    service._precheck_placement = MagicMock()  # type: ignore[method-assign]
    service._load_worker = MagicMock()  # type: ignore[method-assign]
    set_resident(1, "m1")
    return service, repo


def no_probe_rows():
    """The probe-row write an unload makes goes to the database; replace it with a no-op."""
    return patch("millm.services.probe_arming.mark_armed_rows_disarmed", AsyncMock(return_value=[]))


def drain_executor(service: Any) -> None:
    """Wait for the background worker `load_model` submitted (it is not awaited)."""
    service._executor.shutdown(wait=True)


def resident_inference() -> MagicMock:
    """An inference stand-in whose loaded-model view is the REAL loader's."""
    inference = MagicMock()
    inference.backend_name = "serial"
    inference.cbm_enabled = lambda: False

    def loaded_info() -> Any:
        current = LoadedModelState().current
        if current is None:
            return None
        return SimpleNamespace(name=current.model_name, model_id=current.model_id)

    inference.get_loaded_model_info = loaded_info
    return inference
