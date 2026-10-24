"""A load that fails AFTER the adapter's tensors were placed lets them go (033 review R1-C).

`AdapterService._load` places A and B on each module's device before the pause (FR-33.1.8), so a
refusal from the pause itself (`QUEUE_FULL`, `MODEL_BUSY`), a base change seen inside it
(`ADAPTER_BASE_CHANGED`) or an install failure leaves placed tensors behind unless the `finally`
releases them. On a GPU that is the adapter's whole footprint held by a request that failed.
Removing `applied.release_tensors()` from that `finally` left all 112 adapter service, route,
integration and probe tests green (R1-C mutation control); these tests turn it red.

Also FR-33.1.8's other half: a placement that runs out of memory is `507 INSUFFICIENT_MEMORY`
through the real route, with nothing installed, the slot released and the refusal audited.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from millm.core.config import settings
from millm.core.errors import AdapterBaseChangedError, ModelBusyError
from millm.ml.adapter_hooks import LoraHookSet
from millm.services import adapter_service, resident_identity
from millm.services.adapter_service import get_adapter_service, get_adapter_state
from millm.services.request_queue import QueueFullError
from tests.unit.adapter_service_fixtures import (
    OPERATOR_TOKEN,
    build_env,
    identity,
    weights_tokens,
)


@pytest.fixture
def env(tmp_path):
    e = build_env(tmp_path)
    yield e
    e.close()


def _capture_artifacts():
    """Wrap `prepare_adapter` and keep every artifact it returned."""
    seen = []
    real = adapter_service.prepare_adapter

    def wrapped(*args, **kwargs):
        artifact = real(*args, **kwargs)
        seen.append(artifact)
        return artifact

    return seen, patch.object(adapter_service, "prepare_adapter", side_effect=wrapped)


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


@pytest.mark.parametrize("error", [
    QueueFullError("queue full", details={}),
    ModelBusyError("unloading", details={}),
], ids=["queue_full", "model_busy"])
async def test_a_refusal_from_the_pause_releases_the_placed_tensors(env, error):
    rel = env.write()
    seen, wrap = _capture_artifacts()

    def refusing():
        raise error

    with wrap, patch.object(env.inference, "hold_admission", side_effect=refusing), \
            pytest.raises(type(error)):
        await env.service.load(identity(), rel)
    assert len(seen) == 1
    prepared = seen[0].prepared
    assert prepared.weights == {} and prepared.modules == {}, "placed tensors must be released"
    assert _hooks(env.model) == 0 and get_adapter_state().current is None


async def test_a_base_change_inside_the_pause_releases_the_placed_tensors(env):
    rel = env.write()
    seen, wrap = _capture_artifacts()
    real_epoch = resident_identity.base_epoch

    def moving_epoch():
        # Moved once the artifact has been prepared, i.e. while it is being verified; every read
        # before that (the capture and the pre-prepare check, R2-C) sees the captured epoch.
        return real_epoch() + (1 if seen else 0)

    with wrap, patch.object(resident_identity, "base_epoch", side_effect=moving_epoch), \
            pytest.raises(AdapterBaseChangedError):
        await env.service.load(identity(), rel)
    assert seen[0].prepared.weights == {}
    assert _hooks(env.model) == 0 and get_adapter_state().current is None


async def test_an_install_failure_releases_the_placed_tensors(env):
    rel = env.write()
    seen, wrap = _capture_artifacts()
    with wrap, patch.object(LoraHookSet, "install", side_effect=RuntimeError("boom")), \
            pytest.raises(RuntimeError):
        await env.service.load(identity(), rel)
    assert seen[0].prepared.weights == {}
    assert get_adapter_state().current is None and not get_adapter_state().loading


def test_running_out_of_memory_while_placing_is_507_over_http(env, monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", weights_tokens())
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_adapter_service] = lambda: env.service
    rel = env.write()
    with patch.object(LoraHookSet, "build", side_effect=torch.OutOfMemoryError("CUDA OOM")), \
            patch.object(adapter_service, "weights_audit") as audit, \
            patch.object(env.barrier, "serving_pause", wraps=env.barrier.serving_pause) as pause:
        r = TestClient(app).post("/api/weights/adapter/load", json={"path": rel},
                                 headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"})
    assert r.status_code == 507, r.text
    assert r.json()["error"]["code"] == "INSUFFICIENT_MEMORY"
    assert pause.call_count == 0
    assert _hooks(env.model) == 0
    assert get_adapter_state().current is None and not get_adapter_state().loading
    assert audit.call_args.args[:2] == ("adapter/load", "refused")
    assert audit.call_args.kwargs["code"] == "INSUFFICIENT_MEMORY"
