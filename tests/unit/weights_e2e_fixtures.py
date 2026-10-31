"""End-to-end harness for Features 31, 32 and 33 TOGETHER, through the REAL app (review RX).

Every route is the real `create_app()` route, driven over ASGI on the test's own event loop (so
the barrier's holder task outlives one request, as in production). Real objects:

* the `WeightBarrier`, the `WeightService` (attach, holder checks, staleness, the address
  check), the `WeightsCheckpointService` (save, in-place load, D-01 row reload, lineage), the
  `AdapterService` (033), `resident_identity` and its chokepoints;
* a real `InferenceService` serving a tiny real bfloat16 Llama on the CPU (its `_admit`,
  `hold_admission`, request queue, served-name capture and `X-miLLM-Model` header);
* W2-F's real `ModelService` over the SQLite test session (the `/v1` routes resolve names
  through it; the D-01 recovery reloads through it).

Faked, and why (the only fakes):

* **CUDA IPC export.** `export_handles` runs for real (grouping by storage, the address record)
  with `share=` returning a fixed tuple, because `_share_cuda_` needs a CUDA tensor.
* **CUDA device check in `check_attachable`.** The real function runs against a view of the
  resident whose parameters report `device.type == "cuda"` (dtype, quantization, allocator,
  concurrency and continuous-batching checks all stay real).
* **The loader's CUDA `load`** (`CpuLoader`, shared with the 032 harness): `from_pretrained` on
  the CPU, so a D-01 reload really reads the row's directory.
* **`list_gpus`** answers one test card; the probe-row disarm an unload records is a no-op.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import torch

import millm.services.weight_barrier as barrier_module
from millm.core.config import settings
from millm.db.models.model import ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services import model_service as model_service_module
from millm.services import resident_identity, weight_handles
from millm.services.model_lease import LeaseRegistry, set_lease_registry
from millm.services.model_service import ModelService
from millm.services.weight_attachment import WeightService, set_weight_service
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.support.factories import make_model
from tests.unit.checkpoint_service_fixtures import TEST_GPUS, CpuLoader
from tests.unit.f25_fixtures import word_model, word_tokenizer
from tests.unit.weights_fixtures import save_pretrained_dir

TRAINER_TOKEN = "trainer-e2e-token-0123456789"
OPERATOR_TOKEN = "operator-e2e-token-9876543210"
HOLDER_HEADER = "X-miLLM-Attachment"
ROW_ID = 1
ROW_NAME = "tiny"
ROW_REPO = "org/tiny"
CHAT = {"messages": [{"role": "user", "content": "w1 w2 w3 w4"}], "max_tokens": 4,
        "temperature": 0.0}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def tokens_json() -> str:
    return json.dumps([
        {"label": "forge", "role": "trainer", "sha256": _digest(TRAINER_TOKEN)},
        {"label": "ops", "role": "operator", "sha256": _digest(OPERATOR_TOKEN)},
    ])


def auth(token: str = TRAINER_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


TRAINER = auth(TRAINER_TOKEN)
OPS = auth(OPERATOR_TOKEN)


def cpu_export(model: Any) -> weight_handles.ExportedHandles:
    """The REAL `export_handles` with only `_share_cuda_` replaced (it needs CUDA)."""
    return weight_handles.export_handles(
        model,
        uuid_for=lambda _index: None,
        share=lambda p: (0, b"handle", p.untyped_storage().nbytes(), 0, b"ref", 0, None, False),
    )


def install_cpu_as_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """`check_attachable` runs for real over a view whose parameters say `device.type ==
    'cuda'`; every other attachability check reads the real resident."""
    real = weight_handles.check_attachable

    def check(loaded: Any, **kwargs: Any) -> None:
        if loaded is None:
            return real(loaded, **kwargs)
        model = loaded.model
        params = [
            (name, SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype=p.dtype))
            for name, p in model.named_parameters(remove_duplicate=False)
        ]
        view_model = SimpleNamespace(
            named_parameters=lambda **_: list(params),
            _keep_in_fp32_modules=getattr(model, "_keep_in_fp32_modules", None),
        )
        return real(dataclasses.replace(loaded, model=view_model), **kwargs)

    monkeypatch.setattr(weight_handles, "check_attachable", check)


def resident_loaded(model: Any, tokenizer: Any) -> LoadedModel:
    return LoadedModel(
        model_id=ROW_ID, model_name=ROW_NAME, model=model, tokenizer=tokenizer,
        loaded_at=datetime(2026, 10, 14), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype="bfloat16", quantization_method="none", attachable_allocation=True,
    )


def logits(model: Any) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor([[0, 6, 7, 8, 9]])).logits.float().clone()


def params(model: Any) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def addresses(model: Any) -> dict[str, int]:
    return weight_handles.address_fingerprint(model)


class Flow(SimpleNamespace):
    """What a flow test reads: the app, the live objects and a few helpers."""

    app: Any
    model: Any
    inference: Any
    barrier: WeightBarrier
    weights: WeightService
    models: ModelService
    loader: CpuLoader
    root: Path
    base_dir: Path

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
            base_url="http://test",
        )

    @property
    def resident(self) -> Any:
        return LoadedModelState().current


async def chat(client: httpx.AsyncClient, model: str = ROW_NAME, **over: Any) -> httpx.Response:
    return await client.post("/v1/chat/completions", json={"model": model, **CHAT, **over})


async def attach(client: httpx.AsyncClient) -> str:
    r = await client.post("/api/weights/attach", headers=TRAINER)
    assert r.status_code == 200, r.text
    return r.json()["data"]["attachment_id"]


def holder(attachment_id: str) -> dict[str, str]:
    return {**TRAINER, HOLDER_HEADER: attachment_id}


def manifest(run_id: str = "run", step: int = 1, parent: str | None = None) -> dict[str, Any]:
    return {
        "run_id": run_id, "step": step, "domain": "e2e", "signal_source": "corpus",
        "parent": parent,
        "optimizer": {"name": "adamw", "state": "fresh", "config_hash": "c" * 8},
    }


@pytest.fixture
async def flow(tmp_path, test_session, monkeypatch):
    """The whole stack with one tiny bfloat16 Llama resident as row 1 (`tiny`, `org/tiny`)."""
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app
    from millm.services import adapter_service as adapter_module
    from millm.services.inference_service import InferenceService

    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", tokens_json())
    monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(root))
    monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
    monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 2.0)
    monkeypatch.setattr(model_service_module, "list_gpus", lambda: list(TEST_GPUS))
    monkeypatch.setattr("millm.services.probe_arming.mark_armed_rows_disarmed",
                        AsyncMock(return_value=[]))
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: None)
    install_cpu_as_cuda(monkeypatch)

    # The adapter module registers its drop callback and overlay at import (the live app has
    # both); a test that pruned the registry must not leave this flow without them.
    callbacks_before = list(resident_identity._base_replaced)
    overlays_before = list(resident_identity._overlays)
    resident_identity.register_on_base_replaced(adapter_module.drop_adapter_for_base_change)
    resident_identity.register_served_name_overlay(adapter_module.applied_adapter_name)
    adapter_module.get_adapter_state().take()

    model = word_model().to(torch.bfloat16).eval()
    tokenizer = word_tokenizer()
    base_dir = save_pretrained_dir(word_model().to(torch.bfloat16), tmp_path / "base",
                                   tokenizer=tokenizer)
    test_session.add(make_model(
        id=ROW_ID, name=ROW_NAME, repo_id=ROW_REPO, status=ModelStatus.LOADED,
        quantization=QuantizationType.FP16, cache_path=str(base_dir),
    ))
    await test_session.commit()
    LoadedModelState().set(resident_loaded(model, tokenizer))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        inference = InferenceService(model_service=None)
    inference._device = "cpu"

    barrier = WeightBarrier(inference=lambda: inference)
    set_weight_barrier(barrier)
    weights = WeightService(barrier=lambda: barrier, inference=lambda: inference,
                            export=cpu_export,
                            producer=lambda: {"torch": torch.__version__, "cuda": None,
                                              "driver": None})
    set_weight_service(weights)
    set_lease_registry(LeaseRegistry())
    loader = CpuLoader()
    models = ModelService(repository=ModelRepository(test_session), downloader=MagicMock(),
                          loader=loader, inference_service=inference)
    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: inference
    app.dependency_overrides[get_model_service] = lambda: models

    yield Flow(app=app, model=model, inference=inference, barrier=barrier, weights=weights,
               models=models, loader=loader, root=root, base_dir=base_dir)

    adapter_module.get_adapter_state().take()
    adapter_module.get_adapter_state().release_claim()
    barrier.reset("test teardown")
    models._executor.shutdown(wait=True)
    model_service_module._LOAD_SLOT["model_id"] = None
    LoadedModelState()._loaded = None
    LoadedModelState()._unloading = False
    resident_identity._base_replaced[:] = callbacks_before
    resident_identity._overlays[:] = overlays_before
    set_weight_barrier(None)
    set_weight_service(None)
    set_lease_registry(None)
