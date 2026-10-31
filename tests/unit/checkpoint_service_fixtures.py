"""Harness for the Feature 32 service tests: save, load and lineage over REAL objects.

What is real: the `WeightBarrier` (every state transition), the `WeightService` (holder,
staleness, refusals), `LoadedModelState` and `resident_identity` (the chokepoints and the
compare-and-set), tiny transformers models on the CPU, the checkpoint store on `tmp_path`, and
`ModelRepository` on the SQLite test session (carrier rows).

What is faked, and why:
* the admission slot (`FakeInference`): the barrier takes THE slot through `hold_admission()`;
  a one-permit semaphore is that slot. Its `on_release` hook runs the moment the slot is given
  back, which is exactly "a request admitted right after the barrier exits".
* NOTHING in the `ModelService`: it is W2-F's real service (every §8 method, the placement
  pre-check, the background worker, `wait_until_loaded`, the auto-unload). Only the loader's
  CUDA `load` is replaced (`CpuLoader`: `from_pretrained` on the CPU, so a full reload proves
  the row points at the verified files), `list_gpus` answers one test card, and the probe-row
  write an unload makes is a no-op. `spy_on` records the calls and then runs the real method.
* the attachment: `attach()` exports CUDA IPC handles, so `attach_holder` installs the
  `Attachment` record directly; every holder decision still runs through the real service.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.core.config import settings
from millm.core.weights_auth import WeightsIdentity
from millm.db.models.model import ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.gpu_placement import GpuInfo
from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader
from millm.services import resident_identity
from millm.services import model_service as model_service_module
from millm.services.model_service import ModelService
from millm.services.weight_attachment import (
    Attachment,
    WeightService,
    _digest,
    _ref,
    set_weight_service,
)
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.support.factories import make_model
from tests.unit.weights_fixtures import tiny_llama, tiny_tokenizer

TRAINER = WeightsIdentity("forge", "trainer")
OTHER_TRAINER = WeightsIdentity("forge-2", "trainer")
OPERATOR = WeightsIdentity("ops", "operator")
ATTACHMENT_ID = "a" * 43

RESIDENT_ID = 1
RESIDENT_NAME = "tiny-llama"
RESIDENT_REPO = "org/tiny-llama"


class FakeInference:
    """The admission slot the barrier holds, plus the reads it makes."""

    def __init__(self) -> None:
        self.request_queue = SimpleNamespace(max_concurrent=1, pending_count=0, occupied_count=0)
        self.slot = asyncio.Semaphore(1)
        self.on_take: Optional[Callable[[], None]] = None
        self.on_release: Optional[Callable[[], None]] = None

    def cbm_enabled(self) -> bool:
        return False

    def invalidate_request_caches(self) -> None:
        pass

    def on_model_loaded(self) -> None:
        pass

    def on_model_unloading(self) -> None:
        pass

    @contextlib.asynccontextmanager
    async def hold_admission(self):
        async with self.slot:
            if self.on_take is not None:
                self.on_take()
            yield
        if self.on_release is not None:
            self.on_release()


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class CpuLoader(ModelLoader):
    """The REAL `ModelLoader` (state, unload, residency) with only its CUDA `load` replaced.

    `ModelLoader.load` refuses without CUDA; this one really loads the row's directory with
    `from_pretrained`, so a full reload proves the row points at the verified files. It records
    every call's arguments (including the `attachable` decision `_load_worker` made), and
    `fail` makes the load raise in the worker thread, as a CUDA OOM would.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loads: list[dict[str, Any]] = []
        self.fail: Optional[BaseException] = None
        #: Called (in the worker thread) after the load is installed: a test's way to have
        #: something else replace it before `wait_until_loaded` returns.
        self.after_load: Optional[Callable[[], None]] = None

    def load(self, model_id, model_name, cache_path, quantization, estimated_memory_mb,  # type: ignore[override]
             trust_remote_code=False, torch_compile=False, torch_compile_mode="default",
             is_pre_quantized=False, gguf_file=None, gpu=None, attachable=False):
        self.loads.append({"model_id": model_id, "cache_path": cache_path,
                           "quantization": quantization, "gpu": gpu,
                           "trust_remote_code": trust_remote_code, "attachable": attachable})
        if self.fail is not None:
            raise self.fail
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            cache_path, trust_remote_code=trust_remote_code).eval()
        try:
            tokenizer = AutoTokenizer.from_pretrained(cache_path)
        except Exception:  # noqa: BLE001 - a directory saved without a tokenizer
            tokenizer = None
        loaded = LoadedModel(model_id=model_id, model_name=model_name, model=model,
                             tokenizer=tokenizer, loaded_at=datetime(2026, 10, 14))
        self.state.set(loaded)
        if self.after_load is not None:
            self.after_load()
        return loaded


#: One card with room for any tiny model: the placement pre-check runs for real against it.
TEST_GPUS = [GpuInfo(index=0, name="test-card", uuid=None, total_mb=24_576, free_mb=23_000)]


def spy_on(service: ModelService, calls: list[tuple[str, dict[str, Any]]]) -> None:
    """Record every call to the §8 load-path methods and then run the REAL method.

    Bound onto the instance, so the service's own internal calls (`load_model`'s auto-unload)
    are recorded too."""
    for name in ("load_model", "wait_until_loaded", "unload_model"):
        real = getattr(service, name)
        signature = inspect.signature(real)

        async def wrapper(*args, _real=real, _name=name, _sig=signature, **kwargs):
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop("timeout", None)
            calls.append((_name, arguments))
            return await _real(*args, **kwargs)

        setattr(service, name, wrapper)


def install_resident(model: Any, *, model_id: int = RESIDENT_ID, name: str = RESIDENT_NAME,
                     tokenizer: Any = "default") -> LoadedModel:
    loaded = LoadedModel(
        model_id=model_id, model_name=name, model=model,
        tokenizer=tiny_tokenizer() if tokenizer == "default" else tokenizer,
        loaded_at=datetime(2026, 10, 14),
    )
    LoadedModelState().set(loaded)
    return loaded


def clear_resident() -> None:
    LoadedModelState()._loaded = None
    LoadedModelState()._unloading = False


def attach_holder(service: WeightService, clock: Clock, identity: WeightsIdentity = TRAINER,
                  *, params: Any = None) -> Attachment:
    """An active attachment held by `identity` presenting `ATTACHMENT_ID`."""
    digest = _digest(ATTACHMENT_ID)
    attachment = Attachment(
        digest=digest, ref=_ref(digest), identity=identity,
        attached_at=datetime(2026, 10, 14, tzinfo=timezone.utc), model_id=RESIDENT_ID,
        last_heartbeat_mono=clock(), last_heartbeat_at=datetime(2026, 10, 14, tzinfo=timezone.utc),
        pending=False, keep_alive=list(params or []),
    )
    service._attachment = attachment
    return attachment


def manifest_fields(run_id: str = "run", step: int = 5, parent: Optional[str] = None,
                    **over: Any) -> dict[str, Any]:
    fields = {
        "run_id": run_id, "step": step, "domain": "sql style", "signal_source": "corpus",
        "parent": parent,
        "optimizer": {"name": "adamw", "state": "fresh", "config_hash": "c" * 8},
    }
    fields.update(over)
    return fields


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logits(model: Any) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor([[1, 2, 3, 4, 5]])).logits


class Harness(SimpleNamespace):
    barrier: WeightBarrier
    weights: WeightService
    inference: FakeInference
    clock: Clock
    root: Path
    models: ModelService
    loader: CpuLoader
    calls: list[tuple[str, dict[str, Any]]]
    repo: ModelRepository

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
async def harness(tmp_path, test_session, monkeypatch):
    """Real barrier and weight service, a checkpoint root, a resident row in the database."""
    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(root))
    monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
    inference = FakeInference()
    clock = Clock()
    barrier = WeightBarrier(inference=lambda: inference)
    weights = WeightService(barrier=lambda: barrier, inference=lambda: inference,
                            load_claimed=lambda: False, clock=clock)
    set_weight_barrier(barrier)
    set_weight_service(weights)
    repo = ModelRepository(test_session)
    row = make_model(id=RESIDENT_ID, name=RESIDENT_NAME, repo_id=RESIDENT_REPO,
                     status=ModelStatus.LOADED, quantization=QuantizationType.FP16,
                     cache_path=str(tmp_path / "base"))
    test_session.add(row)
    await test_session.commit()
    callbacks_before = list(resident_identity._base_replaced)
    clear_resident()
    # The placement pre-check reads nvidia-smi; it runs for real against one test card.
    monkeypatch.setattr(model_service_module, "list_gpus", lambda: list(TEST_GPUS))
    # An unload records disarmed probe rows through the app's own session factory.
    monkeypatch.setattr("millm.services.probe_arming.mark_armed_rows_disarmed",
                        AsyncMock(return_value=[]))
    loader = CpuLoader()
    models = ModelService(repository=repo, downloader=MagicMock(), loader=loader,
                          inference_service=inference)
    calls: list[tuple[str, dict[str, Any]]] = []
    spy_on(models, calls)
    yield Harness(barrier=barrier, weights=weights, inference=inference, clock=clock,
                  root=root, models=models, loader=loader, calls=calls, repo=repo)
    models._executor.shutdown(wait=True)
    model_service_module._LOAD_SLOT["model_id"] = None
    clear_resident()
    resident_identity._base_replaced[:] = callbacks_before
    set_weight_barrier(None)
    set_weight_service(None)


def resident_llama(**kwargs: Any) -> LoadedModel:
    return install_resident(tiny_llama(**kwargs))


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


def write_foreign_checkpoint(root: Path, run_id: str, step: int, model: Any, *,
                             base_model: str = RESIDENT_REPO, parent: Optional[str] = None,
                             with_constants: bool = False, tokenizer: Any = "default",
                             name: Optional[str] = None, latest: bool = True) -> Path:
    """A checkpoint written the way miForge writes one (review 6.3): transformers'
    `save_pretrained` (tied key dropped, `{"format": "pt"}`), a hand-built manifest that by
    default omits `kind`/`schema_version` (D-30), a `.tmp-save-*` sibling and a RELATIVE
    `latest`."""
    import json

    from tests.unit.weights_fixtures import save_pretrained_dir

    run_dir = Path(root) / run_id
    directory = run_dir / f"step-{step:06d}"
    save_pretrained_dir(model, directory,
                        tokenizer=tiny_tokenizer() if tokenizer == "default" else tokenizer)
    manifest: dict[str, Any] = {
        "run_id": run_id, "step": step, "domain": None, "signal_source": "corpus",
        "parent": parent,
        "optimizer": {"name": "adamw", "state": "fresh", "config_hash": "c" * 8},
        "weights_sha256": sha256_of(directory / "model.safetensors"),
        "base_model": base_model,
        "created_at": "2026-10-13T12:00:00Z",
        "name": name or f"{base_model}:{run_id}:step-{step}",
    }
    if with_constants:
        manifest.update(kind="miforge.checkpoint-manifest", schema_version="1")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / f".tmp-save-{'0' * 8}").mkdir(exist_ok=True)
    if latest:
        link = run_dir / "latest"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(directory.name)
    return directory
