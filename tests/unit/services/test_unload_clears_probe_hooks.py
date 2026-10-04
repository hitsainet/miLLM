"""Unloading a model clears its probe hooks, or no probe ever scores again (2026-10-04).

Found on hardware: after `POST /api/models/{id}/unload` and a reload, every probe — including ones
armed AFTER the reload — recorded `no_scored_tokens` on every request, while
`/api/probes/status` reported `hook_installed: true`. `ProbeRuntimeState` keys its forward-hook
handles by layer number alone; the unload destroyed the modules they hung on and left the handles,
so arming on the same layer of the new model saw "already hooked" and installed nothing.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from torch import nn

from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader
from millm.services.probe_runtime import ProbeRuntimeState


class _Layer(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return (self.linear(x), None)


class _Model(nn.Module):
    """`model.model.layers[i]`, the first pattern `get_layer` tries."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_Layer() for _ in range(3)])
        self.model = inner


def _load(model: nn.Module, model_id: int = 9) -> None:
    LoadedModelState().set(
        LoadedModel(
            model_id=model_id, model_name="m", model=model, tokenizer=None,
            loaded_at=datetime(2026, 10, 4), memory_used_mb=1, num_parameters=1,
            device="cpu", dtype="bfloat16",
        )
    )


@pytest.fixture(autouse=True)
def _clean():
    ProbeRuntimeState.reset_for_tests()
    state = LoadedModelState()
    state._loaded = None
    state._unloading = False
    yield
    ProbeRuntimeState.reset_for_tests()
    state._loaded = None
    state._unloading = False


def _probe(probe_id: str, layer: int = 1):
    return SimpleNamespace(probe_id=probe_id, layer=layer)


class TestTheLoaderClearsTheHooks:
    def test_a_probe_armed_after_a_reload_is_hooked_on_the_new_model(self):
        """The defect exactly: unload, reload, arm — the NEW model's layer must carry a hook."""
        first, second = _Model(), _Model()
        _load(first)
        ProbeRuntimeState().arm(_probe("pr_old"), first)
        assert first.model.layers[1]._forward_hooks, "precondition: the first model was hooked"

        ModelLoader().unload()

        assert ProbeRuntimeState().armed() == [], "the registry outlived the model"
        _load(second)
        ProbeRuntimeState().arm(_probe("pr_new"), second)
        assert second.model.layers[1]._forward_hooks, (
            "arming on the reloaded model installed nothing — every verdict would be "
            "no_scored_tokens while status reported the hook installed"
        )

    def test_the_new_hook_actually_reads_the_new_model(self):
        """A hook present is not a hook firing: run a forward pass and see the activations land."""
        seen = []
        first, second = _Model(), _Model()
        _load(first)
        ProbeRuntimeState().arm(_probe("pr_old"), first)
        ModelLoader().unload()
        _load(second)
        state = ProbeRuntimeState()
        state.arm(_probe("pr_new"), second)
        with patch.object(state, "_on_activations", lambda layer, hidden: seen.append(layer)):
            # The hook closes over the bound method at install time, so re-arm under the patch.
            state.disarm("pr_new")
            state.arm(_probe("pr_new"), second)
            second.model.layers[1](torch.ones(1, 2, 4))
        assert seen == [1]

    def test_an_unload_with_nothing_armed_is_harmless(self):
        _load(_Model())
        assert ModelLoader().unload() is True
        assert ProbeRuntimeState().armed() == []


def _factory(session):
    @asynccontextmanager
    async def _cm():
        yield session

    return _cm


class TestTheRowsSaySo:
    async def test_the_helper_disarms_armed_rows_with_the_reason(self, test_session):
        from millm.db.models.probe import Probe
        from millm.services.probe_arming import mark_armed_rows_disarmed
        from tests.unit.services.test_probe_survives_no_restart import _probe_row

        test_session.add_all([
            _probe_row("pr_hot", armed=True), _probe_row("pr_idle", armed=False, paused_reason="operator"),
        ])
        await test_session.commit()

        got = await mark_armed_rows_disarmed(
            _factory(test_session), "disarmed because the model was unloaded", event="t"
        )
        assert got == ["pr_hot"]
        hot = await test_session.get(Probe, "pr_hot")
        idle = await test_session.get(Probe, "pr_idle")
        await test_session.refresh(hot)
        await test_session.refresh(idle)
        assert hot.armed is False and "unloaded" in hot.paused_reason
        assert idle.paused_reason == "operator", "an operator's own reason was overwritten"

    async def test_unload_model_records_the_rows(self):
        """The service CALLS it, once, with an unload reason — after the loader ran."""
        from millm.services import model_service as ms

        loader = MagicMock()
        loader.loaded_model_id = 9
        loader.is_unloading = False
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=SimpleNamespace(id=9))
        repo.update = AsyncMock(return_value=SimpleNamespace(id=9))
        inference = MagicMock()
        inference.request_queue.pending_count = 0
        service = ms.ModelService(repository=repo, downloader=MagicMock(), loader=loader,
                                  inference_service=inference)
        service.get_model = AsyncMock(return_value=SimpleNamespace(id=9))
        calls = []

        async def record(factory, reason, *, event):
            calls.append((reason, event, loader.unload.call_count))
            return []

        with patch("millm.services.probe_arming.mark_armed_rows_disarmed", record):
            await service.unload_model(9)
        assert len(calls) == 1
        reason, event, unloads_before = calls[0]
        assert "unloaded" in reason and event == "probes_disarmed_by_unload"
        assert unloads_before == 1, "the rows were marked before the hooks were cleared"


class TestTheHangGuardWritesTheRowsNow:
    def test_it_calls_the_helper_and_not_the_undefined_global(self):
        """It read `dependencies._probe_arming_service`, which never existed: the rows were never
        marked after a hang."""
        import millm.api.dependencies as deps
        from millm.services.inference_service import InferenceService

        assert not hasattr(deps, "_probe_arming_service")
        tree = ast.parse(textwrap.dedent(inspect.getsource(InferenceService.stream_chat_completion)))
        names = {
            getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(tree) if isinstance(n, ast.Call)
        }
        assert "mark_armed_rows_disarmed" in names
        source = inspect.getsource(InferenceService.stream_chat_completion)
        assert '"_probe_arming_service"' not in source


class TestEveryReasonFitsTheColumn:
    """⚠ `probes.paused_reason` is VARCHAR(64). The first unload reason was 70 characters: Postgres
    refused the write on hardware, the rows stayed `armed`, and this suite was green because its
    database is SQLite, which does not enforce VARCHAR lengths. So the length is checked against
    the column's DECLARED length, not against whatever the test database accepts."""

    def test_the_reasons_fit(self):
        from millm.services.probe_arming import HANG_REASON, UNLOAD_REASON, paused_reason_limit

        assert paused_reason_limit() == 64
        for reason in (UNLOAD_REASON, HANG_REASON):
            assert len(reason) <= paused_reason_limit(), reason

    def test_the_startup_reason_fits_too(self):
        import millm.main as main

        source = inspect.getsource(main.disarm_probes_on_startup)
        tree = ast.parse(textwrap.dedent(source))
        reasons = [
            kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg == "paused_reason"
        ]
        assert reasons, "the startup write's reason was not found"
        from millm.services.probe_arming import paused_reason_limit

        for value in reasons:
            text = ast.literal_eval(value)
            assert len(text) <= paused_reason_limit(), text

    async def test_an_over_long_reason_is_refused_before_any_write(self):
        from millm.services.probe_arming import mark_armed_rows_disarmed

        touched = []

        @asynccontextmanager
        async def factory():
            touched.append(True)
            yield None

        with pytest.raises(ValueError, match="column holds 64"):
            await mark_armed_rows_disarmed(factory, "x" * 65, event="t")
        assert touched == [], "a session was opened for a write the column would refuse"
