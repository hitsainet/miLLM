"""The adapter refusal ORDER, pinned pairwise (033 review R1-C; FR-33.1.3 as built, manual table).

`test_adapter_service.TestRefusalOrder` proves each refusal on its own and that nothing after it
ran. It cannot tell the order of two cheap checks apart: moving the lease check after the slot
claim, or the clean check after the applied check, leaves every one of those tests green while the
manual's numbered table becomes false. Here each ADJACENT pair of conditions holds at once, and
the earlier one in the documented order must be the refusal returned:

load:   model -> engine/quantization -> continuous batching -> one slot -> lease -> clean -> applied
unload: applied -> continuous batching -> one slot -> lease -> clean

Real objects throughout (`adapter_service_fixtures`).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from millm.core.errors import (
    AdapterNotServableError,
    ModelLeasedError,
    ModelNotLoadedError,
    NoAdapterLoadedError,
    WeightsBarrierUnavailableError,
    WeightsNotCleanError,
)
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModelState
from millm.services.adapter_service import get_adapter_state
from tests.unit.adapter_service_fixtures import build_env, identity


@pytest.fixture
def env(tmp_path):
    e = build_env(tmp_path)
    yield e
    e.inference._cbm_backend = None
    e.inference._request_queue._max_concurrent = 1
    e.barrier._weights_state = "clean"
    e.close()


def _no_model(env):
    LoadedModelState()._loaded = None


def _gguf(env):
    LoadedModelState().current.engine = ENGINE_LLAMACPP


def _quantized(env):
    LoadedModelState().current.quantization_method = "bitsandbytes"


def _cbm(env):
    env.inference._cbm_backend = MagicMock(is_running=True)


def _two_slots(env):
    env.inference._request_queue._max_concurrent = 2


def _leased(env):
    env.leases.grant(1, "tiny", "midataworks", "label run", 120)


def _torn(env):
    env.barrier._weights_state = "torn"


def _check(error, reason=None):
    return error, reason


LOAD_PAIRS = [
    ("model_before_cbm", [_no_model, _cbm], _check(ModelNotLoadedError)),
    ("engine_before_cbm", [_gguf, _cbm], _check(AdapterNotServableError, "engine")),
    ("quantization_before_cbm", [_quantized, _cbm], _check(AdapterNotServableError, "quantization")),
    ("cbm_before_slots", [_cbm, _two_slots], _check(AdapterNotServableError, "cbm")),
    ("slots_before_lease", [_two_slots, _leased], _check(WeightsBarrierUnavailableError,
                                                         "concurrency")),
    ("lease_before_clean", [_leased, _torn], _check(ModelLeasedError)),
]


@pytest.mark.parametrize("conditions, expected", [p[1:] for p in LOAD_PAIRS],
                         ids=[p[0] for p in LOAD_PAIRS])
async def test_load_refuses_with_the_earlier_check(env, conditions, expected):
    rel = env.write()
    for condition in conditions:
        condition(env)
    error, reason = expected
    with pytest.raises(error) as exc:
        await env.service.load(identity(), rel)
    if reason is not None:
        assert exc.value.details["reason"] == reason
    assert get_adapter_state().current is None and not get_adapter_state().loading


async def test_load_checks_clean_before_the_applied_adapter(env):
    rel = env.write()
    await env.service.load(identity(), rel)
    _torn(env)
    with pytest.raises(WeightsNotCleanError):
        await env.service.load(identity(), rel)


UNLOAD_PAIRS = [
    ("applied_before_cbm", False, [_cbm], _check(NoAdapterLoadedError)),
    ("cbm_before_slots", True, [_cbm, _two_slots], _check(AdapterNotServableError, "cbm")),
    ("slots_before_lease", True, [_two_slots, _leased], _check(WeightsBarrierUnavailableError,
                                                               "concurrency")),
    ("lease_before_clean", True, [_leased, _torn], _check(ModelLeasedError)),
]


@pytest.mark.parametrize("applied, conditions, expected", [p[1:] for p in UNLOAD_PAIRS],
                         ids=[p[0] for p in UNLOAD_PAIRS])
async def test_unload_refuses_with_the_earlier_check(env, applied, conditions, expected):
    if applied:
        await env.service.load(identity(), env.write())
    for condition in conditions:
        condition(env)
    error, reason = expected
    with pytest.raises(error) as exc:
        await env.service.unload(identity())
    if reason is not None:
        assert exc.value.details["reason"] == reason
    if applied:
        assert get_adapter_state().current is not None  # nothing was removed
