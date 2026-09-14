"""The load API takes a GPU, the service honours it, and status reports placement.

The request travels route -> ModelService.load_model -> executor ->
_load_worker -> ModelLoader.load. Each hop is asserted here with its PAYLOAD,
because a hop that drops the argument leaves every load on Auto and every test
that only checks "was called" green.

MUTATION CONTROLS (each must turn this file red):
  * route calls service.load_model(model_id) without gpu -> route tests fail
  * load_model drops `wanted` from the executor args      -> executor test fails
  * remove the find_gpu pre-check                         -> unknown-card test fails
  * list_models stops injecting placement                 -> status test fails
  * get_model returns without _with_runtime (2026-09-13)  -> single-route test fails
  * _with_runtime assigns the raw placement dict           -> no-warning test fails
Phase 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  M24 drop planned_mb_by_device from the ModelPlacement schema
      -> test_a_split_reports_its_cards_plan_and_per_card_memory
"""

from concurrent.futures import Future
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from millm.api.schemas.model import ModelLoadRequest
from millm.core.errors import GpuNotFoundError
from millm.db.models.model import ModelStatus
from millm.main import create_app
from millm.services.model_service import ModelService
from tests.support.factories import make_model
from tests.support.fake_gpus import NODE_UUIDS, RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))

PLACEMENT = {
    "mode": "single",
    "reason": "most_free_card_fits",
    "requested": None,
    "required_mb": 8_000,
    "capacity_mb": 23_000,
    "devices": ["cuda:1"],
    "gpu_indices": [1],
    "memory_by_device_mb": {"cuda:1": 9_000},
}


class TestTheRequestSchema:
    @pytest.mark.parametrize("value, expected", [
        (None, None),
        ("auto", None),
        (1, 1),
        ("1", 1),
        (NODE_UUIDS[1][4:].upper(), NODE_UUIDS[1]),
        ("all", "all"),
        (" ALL ", "all"),
    ])
    def test_accepted_forms(self, value, expected):
        assert ModelLoadRequest(gpu=value).gpu == expected

    @pytest.mark.parametrize("value", [True, -1, "the big one"])
    def test_refused_forms(self, value):
        with pytest.raises(ValidationError):
            ModelLoadRequest(gpu=value)


def _client(service):
    from millm.api.dependencies import get_model_service

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    return TestClient(app)


class TestTheRoute:
    def _service(self):
        svc = MagicMock()
        svc.load_model = AsyncMock(return_value=make_model(id=3, status=ModelStatus.LOADING))
        return svc

    def test_a_named_card_reaches_the_service(self):
        svc = self._service()
        response = _client(svc).post("/api/models/3/load", json={"gpu": 1})
        assert response.status_code == 202, response.text
        svc.load_model.assert_awaited_once_with(3, gpu=1)

    def test_a_uuid_reaches_the_service_normalised(self):
        svc = self._service()
        _client(svc).post("/api/models/3/load", json={"gpu": NODE_UUIDS[1][4:]})
        svc.load_model.assert_awaited_once_with(3, gpu=NODE_UUIDS[1])

    def test_no_body_is_auto(self):
        svc = self._service()
        response = _client(svc).post("/api/models/3/load")
        assert response.status_code == 202, response.text
        svc.load_model.assert_awaited_once_with(3, gpu=None)

    def test_auto_is_auto(self):
        svc = self._service()
        _client(svc).post("/api/models/3/load", json={"gpu": "auto"})
        svc.load_model.assert_awaited_once_with(3, gpu=None)

    def test_all_reaches_the_service(self):
        svc = self._service()
        response = _client(svc).post("/api/models/3/load", json={"gpu": "all"})
        assert response.status_code == 202, response.text
        svc.load_model.assert_awaited_once_with(3, gpu="all")

    def test_a_malformed_card_is_422_and_nothing_loads(self):
        svc = self._service()
        response = _client(svc).post("/api/models/3/load", json={"gpu": True})
        assert response.status_code == 422
        assert not svc.load_model.called

    def test_an_unknown_card_is_404(self):
        svc = MagicMock()
        svc.load_model = AsyncMock(side_effect=GpuNotFoundError("No visible GPU matches 7"))
        response = _client(svc).post("/api/models/3/load", json={"gpu": 7})
        assert response.status_code == 404


class TestStatusReportsPlacement:
    def test_the_loaded_model_carries_its_cards_and_per_card_memory(self):
        svc = MagicMock()
        svc.list_models = AsyncMock(return_value=[make_model(id=3, status=ModelStatus.LOADED)])
        svc.get_download_progress = MagicMock(return_value=None)
        svc.get_loaded_model_info = MagicMock(return_value={
            "model_id": 3,
            "num_parameters": 1,
            "memory_footprint": 9_000 * 1024 * 1024,
            "device": "cuda:1",
            "dtype": "bfloat16",
            "gpu_indices": [1],
            "memory_by_device_mb": {"cuda:1": 9_000},
            "placement": PLACEMENT,
        })
        response = _client(svc).get("/api/models")
        assert response.status_code == 200, response.text
        [model] = response.json()["data"]
        assert model["device"] == "cuda:1"
        assert model["placement"]["devices"] == ["cuda:1"]
        assert model["placement"]["gpu_indices"] == [1]
        assert model["placement"]["memory_by_device_mb"] == {"cuda:1": 9_000}

    def test_a_split_reports_its_cards_plan_and_per_card_memory(self):
        """The dict comes from a real Placement, so the report and the producer
        cannot agree only by construction."""
        from millm.ml.gpu_placement import choose_gpu

        with fake_gpus(*NODE):
            placement = choose_gpu(30_000)
        svc = MagicMock()
        svc.list_models = AsyncMock(return_value=[make_model(id=3, status=ModelStatus.LOADED)])
        svc.get_download_progress = MagicMock(return_value=None)
        svc.get_loaded_model_info = MagicMock(return_value={
            "model_id": 3,
            "num_parameters": 1,
            "memory_footprint": 26_000 * 1024 * 1024,
            "device": "cuda:0, cuda:1",
            "dtype": "bfloat16",
            "gpu_indices": [0, 1],
            "memory_by_device_mb": {"cuda:0": 7_000, "cuda:1": 19_000},
            "placement": {
                **placement.to_dict(),
                "memory_by_device_mb": {"cuda:0": 7_000, "cuda:1": 19_000},
            },
        })
        response = _client(svc).get("/api/models")
        assert response.status_code == 200, response.text
        [model] = response.json()["data"]
        assert model["placement"]["mode"] == "shard"
        assert model["placement"]["devices"] == ["cuda:0", "cuda:1"]
        assert model["placement"]["gpu_indices"] == [0, 1]
        assert model["placement"]["planned_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 20_024}
        assert model["placement"]["budget_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 21_976}
        assert model["placement"]["memory_by_device_mb"] == {"cuda:0": 7_000, "cuda:1": 19_000}

    def _single(self, loaded_model_id):
        svc = MagicMock()
        svc.get_model = AsyncMock(return_value=make_model(id=3, status=ModelStatus.LOADED))
        svc.get_download_progress = MagicMock(return_value=None)
        svc.get_loaded_model_info = MagicMock(return_value={
            "model_id": loaded_model_id,
            "num_parameters": 1,
            "memory_footprint": 9_000 * 1024 * 1024,
            "device": "cuda:1",
            "dtype": "bfloat16",
            "placement": PLACEMENT,
        })
        response = _client(svc).get("/api/models/3")
        assert response.status_code == 200, response.text
        return response.json()["data"]

    def test_the_single_model_route_reports_the_same_placement(self):
        """Found on the node (2026-09-13): the list said cuda:1 while
        GET /api/models/{id} said placement null for the same loaded model."""
        model = self._single(loaded_model_id=3)
        assert model["device"] == "cuda:1"
        assert model["placement"]["devices"] == ["cuda:1"]
        assert model["placement"]["memory_by_device_mb"] == {"cuda:1": 9_000}

    def test_placement_serialises_without_a_pydantic_warning(self):
        """Assigning the raw dict to the typed field serialised correctly but
        logged PydanticSerializationUnexpectedValue on every listing (node log,
        2026-09-13). The field must hold a ModelPlacement."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self._single(loaded_model_id=3)
        assert model["placement"]["gpu_indices"] == [1]
        assert not [w for w in caught if "PydanticSerializationUnexpectedValue" in str(w.message)]

    def test_a_model_that_is_not_the_loaded_one_has_no_placement(self):
        model = self._single(loaded_model_id=99)
        assert model["placement"] is None

    def test_service_loaded_info_includes_placement(self):
        from millm.ml.model_loader import LoadedModel

        loader = MagicMock(is_loaded=True)
        loader.state.current = LoadedModel(
            3, "m", MagicMock(), MagicMock(), datetime.utcnow(),
            memory_used_mb=9_000, device="cuda:1", gpu_indices=[1],
            memory_by_device_mb={"cuda:1": 9_000},
            placement={k: v for k, v in PLACEMENT.items() if k != "memory_by_device_mb"},
        )
        svc = ModelService(repository=MagicMock(), downloader=MagicMock(), loader=loader)
        info = svc.get_loaded_model_info()
        assert info["gpu_indices"] == [1]
        assert info["placement"]["devices"] == ["cuda:1"]
        assert info["placement"]["memory_by_device_mb"] == {"cuda:1": 9_000}


class _RecordingExecutor:
    """Stands in for the thread pool: records what the load would run with."""

    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        future = Future()
        future.set_result(None)
        return future


def _model_service():
    repo = MagicMock()
    model = make_model(id=3, status=ModelStatus.READY)
    repo.get_by_id = AsyncMock(return_value=model)
    repo.update_status = AsyncMock(return_value=model)
    loader = MagicMock(is_loaded=False)
    svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
    svc._executor = _RecordingExecutor()
    return svc, repo


class TestTheService:
    async def test_a_named_card_reaches_the_background_load(self):
        svc, _ = _model_service()
        with fake_gpus(*NODE):
            await svc.load_model(3, gpu=1)
        [(fn, args)] = svc._executor.calls
        assert fn == svc._load_worker
        assert args[-1] == 1, f"the load worker was given {args[-1]!r}, not card 1"

    async def test_auto_reaches_the_background_load_as_none(self):
        svc, _ = _model_service()
        with fake_gpus(*NODE):
            await svc.load_model(3)
        assert svc._executor.calls[0][1][-1] is None

    async def test_all_reaches_the_background_load(self):
        svc, _ = _model_service()
        with fake_gpus(*NODE):
            await svc.load_model(3, gpu="all")
        assert svc._executor.calls[0][1][-1] == "all"

    async def test_an_unknown_card_is_refused_before_anything_changes(self):
        svc, repo = _model_service()
        with fake_gpus(*NODE):
            with pytest.raises(GpuNotFoundError):
                await svc.load_model(3, gpu=5)
        assert not repo.update_status.called, "the row was marked LOADING for a card that does not exist"
        assert svc._executor.calls == []

    async def test_garbage_is_refused_as_not_found(self):
        svc, _ = _model_service()
        with fake_gpus(*NODE):
            with pytest.raises(GpuNotFoundError):
                await svc.load_model(3, gpu="the big one")

    def test_the_worker_hands_the_card_to_the_loader(self):
        svc, _ = _model_service()
        svc.loader.load.return_value = MagicMock(memory_used_mb=1)
        svc._run_async_from_thread = MagicMock()
        with patch("millm.core.config.settings") as settings, \
                patch("millm.api.dependencies.get_inference_service"):
            settings.TORCH_COMPILE = False
            settings.TORCH_COMPILE_MODE = "default"
            settings.MODEL_CACHE_DIR = "/tmp"
            svc._load_worker(3, "m", "/tmp/m", "FP16", 8_000, False, None, 1)
        assert svc.loader.load.call_args.kwargs["gpu"] == 1


class TestDetailedHealthReportsPlacement:
    def _get(self, current):
        from millm.api.dependencies import get_inference_service, get_model_loader
        from millm.api.routes.system.health import router

        app = FastAPI()
        app.include_router(router)
        loader = MagicMock(is_loaded=True, model_name="m")
        loader.current_model = current
        inference = MagicMock()
        inference.get_backend_info = MagicMock(return_value={})
        app.dependency_overrides[get_model_loader] = lambda: loader
        app.dependency_overrides[get_inference_service] = lambda: inference

        class _Session:
            async def __aenter__(self):
                return MagicMock()

            async def __aexit__(self, *args):
                return False

        async def _none(*args, **kwargs):
            return None

        with patch("millm.db.base.async_session_factory", return_value=_Session()), \
                patch("millm.db.repositories.profile_repository.ProfileRepository") as repo:
            repo.return_value.get_active = MagicMock(side_effect=_none)
            response = TestClient(app).get("/api/health/detailed")
        assert response.status_code == 200, response.text
        return response.json()

    def test_placement_and_per_card_memory(self):
        body = self._get(SimpleNamespace(
            placement={k: v for k, v in PLACEMENT.items() if k != "memory_by_device_mb"},
            memory_by_device_mb={"cuda:1": 9_000},
        ))
        assert body["model_placement"]["devices"] == ["cuda:1"]
        assert body["model_placement"]["memory_by_device_mb"] == {"cuda:1": 9_000}

    def test_a_split_reports_every_card_and_its_plan(self):
        from millm.ml.gpu_placement import choose_gpu

        with fake_gpus(*NODE):
            placement = choose_gpu(30_000)
        body = self._get(SimpleNamespace(
            placement=placement.to_dict(),
            memory_by_device_mb={"cuda:0": 7_000, "cuda:1": 19_000},
        ))
        assert body["model_placement"]["mode"] == "shard"
        assert body["model_placement"]["gpu_indices"] == [0, 1]
        assert body["model_placement"]["planned_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 20_024}
        assert body["model_placement"]["memory_by_device_mb"] == {"cuda:0": 7_000, "cuda:1": 19_000}

    def test_an_unreadable_placement_never_fails_health(self):
        body = self._get(MagicMock())
        assert body["model_placement"] is None
