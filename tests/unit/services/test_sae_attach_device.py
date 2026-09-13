"""SAEs attach on their hooked layer's device, and memory is gated per device.

A model spread across GPUs keeps later layers on another card. attach_sae and
attach_set loaded every SAE on bare "cuda" (GPU 0) and compared the whole set
against GPU 0's free memory.
"""

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from millm.core.errors import InsufficientMemoryError
from millm.services.sae_service import (
    AttachedSAEState,
    CompatibilityResult,
    SAEService,
)

MB = 1024 * 1024

# Layer 10 lives on the first card, layer 13 on the second. torch.device
# objects need no real GPU to exist.
LAYER_DEVICES = {10: torch.device("cuda", 0), 13: torch.device("cuda", 1)}


@pytest.fixture(autouse=True)
def reset_registry():
    state = AttachedSAEState()
    state.reset_for_tests()
    yield
    state.reset_for_tests()


def _row(sae_id: str):
    row = MagicMock()
    row.id = sae_id
    row.cache_path = f"/tmp/{sae_id}"
    row.d_in = 2048
    row.d_sae = 8192
    row.file_size_bytes = 128 * MB  # projects to ~84.5 MB per SAE
    return row


def _loaded(device: str):
    sae = MagicMock()
    sae.device = device
    sae.estimate_memory_mb.return_value = 64.0
    return sae


def _service():
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    svc._loader = MagicMock()
    svc._loader.load.side_effect = lambda **kw: _loaded(kw["device"])
    svc._hooker = MagicMock()
    svc._hooker.layer_device.side_effect = lambda model, layer: LAYER_DEVICES[layer]
    svc._hooker.install.return_value = MagicMock()
    svc.get_sae = AsyncMock(side_effect=_row)
    svc.check_compatibility = AsyncMock(
        return_value=CompatibilityResult(compatible=True, errors=[], warnings=[])
    )
    svc._reset_dynamo_for_hook_change = MagicMock()
    return svc


def _model_loaded():
    mock_state = MagicMock()
    mock_state.is_loaded = True
    mock_state.current.model = MagicMock()
    return patch("millm.services.sae_service.LoadedModelState", return_value=mock_state)


def _gpus(free_mb_by_index: dict[int, int]):
    return patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        mem_get_info=lambda device: (free_mb_by_index[device.index] * MB, 0),
    )


def _load_devices(svc) -> list[str]:
    return [call.kwargs["device"] for call in svc._loader.load.call_args_list]


class TestAttachSetPlacement:
    async def test_each_sae_loads_on_its_layers_device(self):
        svc = _service()
        with _model_loaded(), _gpus({0: 10_000, 1: 20_000}):
            result = await svc.attach_set([("sae-a", 10), ("sae-b", 13)])

        assert result["attached_count"] == 2
        assert _load_devices(svc) == ["cuda:0", "cuda:1"]

    async def test_cpu_when_cuda_is_unavailable(self):
        svc = _service()
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            await svc.attach_set([("sae-a", 10), ("sae-b", 13)])

        assert _load_devices(svc) == ["cpu", "cpu"]
        svc._hooker.layer_device.assert_not_called()


class TestAttachSetGatePerDevice:
    async def test_a_set_that_fits_each_card_is_admitted(self):
        """~84.5 MB per SAE, 100 MB free on each card: each card fits its SAE.
        Summed against one card (169 MB > 100 MB) it was refused."""
        svc = _service()
        with _model_loaded(), _gpus({0: 100, 1: 100}):
            result = await svc.attach_set([("sae-a", 10), ("sae-b", 13)])

        assert result["attached_count"] == 2

    async def test_a_card_without_room_refuses_the_whole_set_before_loading(self):
        svc = _service()
        with _model_loaded(), _gpus({0: 10_000, 1: 10}):
            with pytest.raises(InsufficientMemoryError) as raised:
                await svc.attach_set([("sae-a", 10), ("sae-b", 13)])

        assert raised.value.details["device"] == "cuda:1"
        assert svc._loader.load.call_count == 0
        assert svc._sae_state.count == 0

    async def test_two_saes_on_one_card_are_summed_for_that_card(self):
        svc = _service()
        svc._hooker.layer_device.side_effect = lambda model, layer: torch.device("cuda", 1)
        with _model_loaded(), _gpus({0: 10_000, 1: 100}):
            with pytest.raises(InsufficientMemoryError):
                await svc.attach_set([("sae-a", 10), ("sae-b", 13)])


def _calls(fn) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


@pytest.mark.parametrize("fn", [SAEService.attach_sae, SAEService.attach_set])
def test_both_attach_paths_load_on_the_resolved_layer_device(fn):
    """Wiring: the device passed to the SAE loader is the resolved layer device,
    never a literal "cuda" or a cuda-or-cpu conditional."""
    calls = _calls(fn)

    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "_sae_device_for_layer"
        for call in calls
    ), f"{fn.__name__} no longer resolves the layer's device"

    loads = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "_loader"
    ]
    assert loads, f"{fn.__name__} no longer loads an SAE"
    for call in loads:
        device = next(kw.value for kw in call.keywords if kw.arg == "device")
        assert isinstance(device, ast.Call) and getattr(device.func, "id", None) == "str", (
            f"{fn.__name__} loads the SAE on {ast.unparse(device)}, not the resolved device"
        )
