"""A refused load must not cost the operator the model they are serving.

The load runs in the background and unloads the resident model first, so a
placement refusal found there arrived as an ERROR status after the served model
was already gone. ModelService.load_model now checks placement BEFORE unloading,
against what each card will have once the resident model is gone: live free
memory plus that model's recorded usage on the card.

The node here: RTX 3080 Ti (index 0, 11 GB free) and RTX 3090 (index 1, 5 GB
free because the resident model holds 16 GB of it).

MUTATION CONTROLS (each must turn this file red):
  * drop the resident-usage add-back   -> "fits once the resident memory is counted back" fails
  * remove the pre-check call          -> both refusal tests see the model unloaded and a 202
"""

from concurrent.futures import Future
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.db.models.model import ModelStatus, QuantizationType
from millm.main import create_app
from millm.ml import model_loader
from millm.ml.model_loader import LoadedModel
from millm.services.model_service import ModelService
from tests.support.factories import make_model
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 5_000, 24_576))


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        future = Future()
        future.set_result(None)
        return future


def _service(model):
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=model)
    repo.update_status = AsyncMock(return_value=model)
    loader = MagicMock()
    loader.is_loaded = True
    loader.loaded_model_id = 9
    loader.state.current = LoadedModel(
        9, "resident", MagicMock(), MagicMock(), datetime.utcnow(),
        memory_used_mb=16_000, device="cuda:1", gpu_indices=[1],
        memory_by_device_mb={"cuda:1": 16_000},
    )
    svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
    svc.unload_model = AsyncMock()
    svc._executor = _RecordingExecutor()
    return svc, repo


def _post(svc, body):
    from millm.api.dependencies import get_model_service

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    return TestClient(app).post("/api/models/3/load", json=body)


def _fp16(estimated_mb, **extra):
    return make_model(
        id=3, status=ModelStatus.READY, quantization=QuantizationType.FP16,
        estimated_memory_mb=estimated_mb, **extra,
    )


class TestARefusalKeepsTheServedModel:
    def test_an_explicit_card_too_small_is_refused_synchronously(self):
        svc, repo = _service(_fp16(18_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": 0})

        assert response.status_code == 507, response.text
        assert "INSUFFICIENT_MEMORY" in response.text
        assert not svc.unload_model.called, "the served model was unloaded for a load that was refused"
        assert not repo.update_status.called
        assert svc._executor.calls == []

    def test_nothing_fits_even_summed_is_refused_synchronously(self):
        # 11 GB + (5 + 16) GB projected = 32 GB, for a 60 GB FP16 model.
        svc, _ = _service(_fp16(60_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": "auto"})

        assert response.status_code == 507, response.text
        assert not svc.unload_model.called
        assert svc._executor.calls == []

    def test_the_refusal_carries_the_same_details_as_the_loader(self):
        svc, _ = _service(_fp16(18_000))
        with fake_gpus(*NODE):
            body = _post(svc, {"gpu": 0}).json()
        details = str(body)
        assert "18000" in details and "11000" in details

    def test_an_explicit_gguf_card_too_small_is_refused_synchronously(self, tmp_path):
        gguf = tmp_path / "big-Q4_K_M.gguf"
        with open(gguf, "wb") as handle:
            handle.truncate(12 * 1024 ** 3)  # sparse: 12 GB of weights, no disk used
        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q4,
            gguf_label="Q4_K_M", gguf_files=["big-Q4_K_M.gguf"], cache_path=str(tmp_path),
        )
        svc, _ = _service(model)
        with fake_gpus(*NODE), patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
            response = _post(svc, {"gpu": 0})

        assert response.status_code == 507, response.text
        assert not svc.unload_model.called


class TestAValidSwitchStillHappens:
    def test_a_card_that_fits_once_the_resident_memory_is_counted_back_is_accepted(self):
        # Card 1 has 5 GB free now, 21 GB once the resident model is gone.
        svc, _ = _service(_fp16(18_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": 1})

        assert response.status_code == 202, response.text
        svc.unload_model.assert_awaited_once_with(9)
        [(_, args)] = svc._executor.calls
        assert args[-1] == 1

    def test_auto_still_loads(self):
        svc, _ = _service(_fp16(8_000))
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 202, response.text
        svc.unload_model.assert_awaited_once_with(9)
        assert svc._executor.calls[0][1][-1] is None

    def test_the_precheck_creates_no_cuda_context(self):
        svc, _ = _service(_fp16(8_000))
        with fake_gpus(*NODE) as fake:
            fake.forbid(0, 1)
            _post(svc, {"gpu": 1})
        assert fake.calls == []
