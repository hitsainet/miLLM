"""`InferenceService.run_model_work`: non-generation model work, ONE slot, unsteered (FR-27.6).

Before Feature 27 the parity route and the arm route ran the parity forward with no admission slot
and no suppression: beside a generation on the same model, and through whatever steering an earlier
layer applied. These tests pin the seam (one `_admit` entry per call, suppression entered in the
WORKER thread, an unloading model refused before `fn` runs), the two routes' wiring (payload and
call count), the parity report's two new keys, and the steering fix itself, on a real model.

Mutations this catches (FTID §8): M15 (drop `_admit` from `run_model_work`), M16 (drop `executor=`
from the arm route), M17 (enter `_unsteered` outside the worker thread).
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.core.errors import ModelBusyError
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.inference_service import InferenceService
from millm.services.probe_arm_bridge import build_parity_forward, build_probe_forward
from millm.services.probe_parity import ParityReport, ProbeParityEngine
from millm.services.probe_runtime import ProbeRequestContext, ProbeRuntimeState
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.services.test_probe_paths_discovered import _probe


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


def _counting_admit(svc):
    calls = {"n": 0}
    real = svc._admit

    @asynccontextmanager
    async def admit(*args, **kwargs):
        calls["n"] += 1
        async with real(*args, **kwargs) as refusal:
            yield refusal

    svc._admit = admit
    return calls


class _ThreadRecordingSAE:
    """An attached SAE whose `suppressed()` records the thread it was entered in."""

    def __init__(self):
        self.entered_in: list[int] = []

    @contextmanager
    def suppressed(self):
        self.entered_in.append(threading.get_ident())
        yield


class TestTheSeam:
    async def test_one_admission_entry_per_call(self):
        svc = InferenceService()
        calls = _counting_admit(svc)
        assert await svc.run_model_work(lambda: 7) == 7
        assert await svc.run_model_work(lambda: 8) == 8
        assert calls["n"] == 2

    async def test_suppression_is_entered_IN_THE_WORKER_THREAD(self, monkeypatch):
        """Per-thread suppression entered around the await would suppress nothing where the
        forward runs (M17)."""
        sae = _ThreadRecordingSAE()
        monkeypatch.setattr(
            "millm.services.sae_service.AttachedSAEState.entries",
            lambda self: [SimpleNamespace(sae=sae, sae_id="s", layer=0)],
        )
        svc = InferenceService()
        ran_in: list[int] = []
        await svc.run_model_work(lambda: ran_in.append(threading.get_ident()))
        assert ran_in and ran_in[0] != threading.get_ident(), "fn must run in a worker thread"
        assert sae.entered_in == ran_in, "suppression was not entered in the thread running fn"

    async def test_an_unloading_model_refuses_before_fn_runs(self):
        svc = InferenceService()
        svc._model_state = SimpleNamespace(
            is_unloading=True, current=SimpleNamespace(model_name="m", model_id=1)
        )
        ran = []
        with pytest.raises(ModelBusyError):
            await svc.run_model_work(lambda: ran.append(1))
        assert ran == []


class TestBuildProbeForward:
    def test_parity_forward_is_the_one_layer_case(self):
        """Same scores from `build_parity_forward(m, L)` and `build_probe_forward(m, [L])`."""
        model = word_model()
        ids = torch.tensor([[2, 6, 7, 8]])
        out = []
        for fwd in (build_parity_forward(model, 0), build_probe_forward(model, [0])):
            ctx = ProbeRequestContext("p", [_probe()])
            fwd(ids, ctx)
            out.append(ctx.token_scores_for("pr_guard"))
        assert out[0] == out[1] and len(out[0]) == 4

    def test_several_layers_in_one_pass_and_every_hook_removed(self):
        model = word_model()
        probes = [_probe(), _probe_at(1, "pr_b")]
        ctx = ProbeRequestContext("p", probes)
        build_probe_forward(model, [1, 0, 1])(torch.tensor([[2, 6, 7]]), ctx)
        assert len(ctx.token_scores_for("pr_guard")) == 3
        assert len(ctx.token_scores_for("pr_b")) == 3
        for layer in model.model.layers:
            assert not layer._forward_hooks and not layer._forward_pre_hooks


def _probe_at(layer: int, probe_id: str):
    from millm.ml.probe_head import ProbeHead
    from millm.services.probe_runtime import ArmedProbe

    return ArmedProbe(
        probe_id=probe_id, name=probe_id, head=ProbeHead(weight=torch.ones(16), bias=0.0,
                                                         layer=layer),
        rule="mean", scope="all", layer=layer, rung=2, rung_language="x", threshold=1.0,
        windows=("all",),
    )


class TestParityReportKeys:
    def test_model_and_checked_at_are_recorded(self):
        report = ProbeParityEngine(lambda ids, ctx: None).run(
            _probe(), {"test_vectors": {"vectors": []}}, tolerance=1e-3,
            model={"hf_id": "x/y", "revision": "abc", "dtype": "bfloat16", "quantization": "FP16"},
        )
        details = report.as_details()
        assert details["model"] == {"hf_id": "x/y", "revision": "abc", "dtype": "bfloat16",
                                    "quantization": "FP16"}
        assert isinstance(details["checked_at"], str) and details["checked_at"]

    def test_a_report_without_them_still_serialises(self):
        details = ParityReport(tolerance=1e-3).as_details()
        assert details["model"] is None and details["checked_at"] is None


class TestArmingAwaitsTheExecutorOnce:
    async def test_executor_is_required(self):
        import inspect

        from millm.services.probe_arming import ProbeArmingService

        param = inspect.signature(ProbeArmingService.arm).parameters["executor"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


class TestTheRoutesTakeTheSeam:
    def _client(self, inference, probe):
        from fastapi.testclient import TestClient

        from millm.api.dependencies import (
            get_inference_service,
            get_probe_arming_service,
            get_probe_repository,
        )
        from millm.main import create_app

        repo = MagicMock()
        repo.get = AsyncMock(return_value=probe)
        repo.update = AsyncMock()
        arming = MagicMock()
        arming.arm = AsyncMock(return_value=_probe())
        app = create_app()
        app.dependency_overrides[get_inference_service] = lambda: inference
        app.dependency_overrides[get_probe_repository] = lambda: repo
        app.dependency_overrides[get_probe_arming_service] = lambda: arming
        return TestClient(app, raise_server_exceptions=False), repo, arming, None

    @pytest.fixture
    def probe(self, monkeypatch):
        import millm.api.routes.management.probes as routes
        from millm.services.probe_identity import LoadedIdentity

        identity = LoadedIdentity(hf_id="x/y", d_model=16, n_layers=2, chat_template="t",
                                  dtype="bfloat16", quantization="FP16")
        monkeypatch.setattr(routes, "loaded_identity",
                            AsyncMock(return_value=(identity, word_model(), word_tokenizer())))
        monkeypatch.setattr(routes, "build_probe_encoder", AsyncMock(return_value=None))
        from tests.unit.probe_fixtures import probe_definition

        row = MagicMock()
        row.id, row.name, row.layer = "pr_1", "p", 0
        row.rule, row.scope, row.rung = "mean", "all", 2
        row.threshold, row.threshold_revision, row.basis = 1.0, 1, "residual"
        definition = probe_definition()
        definition["head"]["weights"] = [1.0] * 16
        definition["head"]["norm_mean"] = [0.0] * 16
        definition["head"]["norm_std"] = [1.0] * 16
        definition["read"]["layer"] = 0
        row.definition = definition
        return row

    def test_the_arm_route_passes_THIS_services_run_model_work(self, probe):
        inference = InferenceService()
        client, _repo, arming, _ = self._client(inference, probe)
        response = client.post("/api/probes/pr_1/arm", json={})
        assert response.status_code == 200, response.text
        assert arming.arm.await_count == 1
        executor = arming.arm.await_args.kwargs["executor"]
        assert executor == inference.run_model_work, "the parity forward must take the seam"

    def test_the_parity_route_runs_through_run_model_work_once(self, probe):
        inference = make_service(word_model(), word_tokenizer())
        calls = {"n": 0}
        real = inference.run_model_work

        async def counting(fn):
            calls["n"] += 1
            return await real(fn)

        inference.run_model_work = counting
        client, repo, _arming, _ = self._client(inference, probe)
        response = client.post("/api/probes/pr_1/parity", json={})
        assert response.status_code == 200, response.text
        assert calls["n"] == 1
        stored = repo.update.await_args.kwargs["parity"]
        assert stored["model"] == {"hf_id": "x/y", "revision": None, "dtype": "bfloat16",
                                   "quantization": "FP16"}
        assert stored["checked_at"]


class TestArmingCallsTheExecutorOnce:
    async def test_once_per_arm(self):
        from millm.services.probe_arming import ProbeArmingService
        from millm.services.probe_identity import LoadedIdentity

        definition = {
            "model": {}, "head": {"weights": [1.0] * 16, "norm_mean": [0.0] * 16,
                                  "norm_std": [1.0] * 16, "bias": 0.0},
            "read": {"layer": 0}, "aggregation": {"rule": "mean"},
            "test_vectors": {"vectors": []},
        }
        probe = SimpleNamespace(id="pr_1", name="p", rule="mean", scope="all", layer=0, rung=2,
                                threshold=1.0, threshold_revision=1, armed=False,
                                definition=definition)
        repo = MagicMock()
        repo.count_armed = AsyncMock(return_value=0)
        repo.update = AsyncMock()
        calls = []

        async def executor(fn):
            calls.append(fn)
            return fn()

        with pytest.raises(Exception):  # noqa: B017 - parity refuses an empty vector set
            await ProbeArmingService(repo).arm(
                probe, model=word_model(), loaded=LoadedIdentity(
                    hf_id="", d_model=16, n_layers=2, chat_template=None),
                forward=lambda ids, ctx: None, executor=executor,
            )
        assert len(calls) == 1, "parity must run through the executor exactly once"


class TestSteeringNoLongerMovesParity:
    """T-73: a profile steering an EARLIER layer moved the residual parity compared."""

    async def test_parity_scores_equal_the_unsteered_scores(self, monkeypatch):
        model = word_model()
        svc = make_service(model, word_tokenizer())
        torch.manual_seed(3)
        sae = LoadedSAE(W_enc=torch.randn(16, 8), b_enc=torch.zeros(8), W_dec=torch.randn(8, 16),
                        b_dec=torch.zeros(16), config=SAEConfig(d_in=16, d_sae=8, model_name="t",
                                                                hook_name="t", hook_layer=0))
        probe = _probe_at(1, "pr_l1")
        ids = torch.tensor([[2, 6, 7, 8]])
        forward = build_probe_forward(model, [1])

        def score():
            ctx = ProbeRequestContext("p", [probe])
            forward(ids, ctx)
            return ctx.token_scores_for("pr_l1")

        unsteered = score()  # no SAE hooked at all
        handle = SAEHooker().install(model, 0, sae)
        try:
            sae.set_steering(2, 40.0)
            sae.enable_steering(True)
            monkeypatch.setattr(
                "millm.services.sae_service.AttachedSAEState.entries",
                lambda self: [SimpleNamespace(sae=sae, sae_id="s", layer=0)],
            )
            steered_direct = score()            # the pre-fix path: no seam, no suppression
            through_seam = await svc.run_model_work(score)
        finally:
            handle.remove()
        assert steered_direct != unsteered, "precondition: layer-0 steering moves layer-1 scores"
        assert through_seam == unsteered
        # Recorded in the controls file: the pre-fix difference on this fixture.
        diff = max(abs(a - b) for a, b in zip(steered_direct, unsteered, strict=True))
        print(f"pre_fix_max_diff={diff:.6f}")
