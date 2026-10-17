"""An SAE lives on the device of the layer it hooks.

A model spread across GPUs keeps later layers on another card. An SAE left on
GPU 0 made monitoring raise a device mismatch inside the forward pass and made
sensing fail silently on every pass.
"""

import torch
from torch import nn

from millm.ml.sae_hooker import SAEHooker


class _Decoder(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(n_layers))


class _Model(nn.Module):
    """Llama/Gemma shape: model.model.layers[i]."""

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.model = _Decoder(n_layers)


class _FakeSAE:
    is_monitoring_enabled = False
    is_sensing_armed = False
    is_edge_sensing_armed = False

    def __init__(self, device: str) -> None:
        self.W_enc = torch.zeros(1, device=device)
        self.moves: list[str] = []

    def to_device(self, device: str) -> None:
        self.moves.append(device)
        self.W_enc = self.W_enc.to(device)

    def apply_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TestLayerDevice:
    def test_reads_the_hooked_layers_own_weights(self):
        model = _Model(3)
        model.model.layers[2].to("meta")  # stands in for "this layer is on another card"

        hooker = SAEHooker()

        assert hooker.layer_device(model, 0) == torch.device("cpu")
        assert hooker.layer_device(model, 2) == torch.device("meta")


class TestHookPlacesTheSae:
    def test_a_misplaced_sae_moves_to_the_layer_device_once(self):
        sae = _FakeSAE("cpu")
        hook = SAEHooker()._create_hook_fn(sae)
        hidden = torch.zeros(1, 2, 4, device="meta")

        hook(None, (), (hidden,))
        hook(None, (), (hidden,))

        assert sae.moves == ["meta"]

    def test_a_correctly_placed_sae_is_left_alone(self):
        sae = _FakeSAE("cpu")
        hook = SAEHooker()._create_hook_fn(sae)

        hook(None, (), (torch.zeros(1, 2, 4),))

        assert sae.moves == []
