"""Review round 1 (R1-B) additions for `millm/ml/adapter_hooks.py`.

- The input is moved to A's device inside the hook. A forward hook sees the module's CALL-TIME
  arguments, not the ones accelerate has moved, and with a model split over two cards the hidden
  state can arrive on the other card. On a CPU-only suite every tensor shares one device, so
  dropping the device from the cast (`x.to(torch.float32)`) left every test green (mutation
  M31b). The cast is asserted structurally here, by recording the `Tensor.to` call itself.
- transformers' own loader keeps a checkpoint key whose renamed form does not exist in the model
  while its original form does. The planner refused such a key (`missing_module`) — an adapter
  trained on this transformers version, keyed on live module names, would be refused if any
  renaming pattern happened to match its keys.
"""

from __future__ import annotations

import pytest
import torch
from torch.overrides import TorchFunctionMode

from millm.ml import adapter_hooks
from millm.ml.adapter_hooks import AdapterPlanError, adapter_shapes, plan_targets
from tests.unit.adapter_fixtures import RANK, make_tensors, tiny_llama


class _RecordTo(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.to:
            self.calls.append((args[1:], dict(kwargs)))
        return func(*args, **kwargs)


def test_the_hook_moves_the_input_to_the_adapter_device_and_float32():
    a = torch.randn(RANK, 8)
    b = torch.randn(6, RANK)
    hook = adapter_hooks._make_hook(a, b, 1.5)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    out = torch.zeros(2, 6, dtype=torch.bfloat16)
    with _RecordTo() as mode:
        hook(None, (x,), {}, out)
    first = mode.calls[0]
    positional, keywords = first
    assert a.device in positional or keywords.get("device") == a.device, first
    assert torch.float32 in positional or keywords.get("dtype") == torch.float32, first


class TestRenamingKeepsAKeyThatAlreadyNamesALiveModule:
    PATH = "model.layers.0.self_attn.q_proj"

    def _plan(self, monkeypatch, renamer, tensors=None):
        monkeypatch.setattr(adapter_hooks, "_conversion_renamer", lambda m: renamer)
        model = tiny_llama()
        tensors = tensors or make_tensors(model, [self.PATH])
        return plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])

    def test_a_renaming_onto_nothing_keeps_the_live_original(self, monkeypatch):
        def renamer(key):  # e.g. a multimodal prefix rule matching an already-live key
            return "model.language_model." + key[len("model."):], None

        (target,) = self._plan(monkeypatch, renamer)
        assert target.module_path == self.PATH

    def test_when_neither_exists_the_refusal_names_the_renamed_module(self, monkeypatch):
        model = tiny_llama()
        path = "model.layers.0.self_attn.q_proj_old"
        shapes = {f"base_model.model.{path}.lora_A.weight": (RANK, 48),
                  f"base_model.model.{path}.lora_B.weight": (48, RANK)}

        def renamer(key):
            return key.replace(".q_proj_old.", ".q_proj_new."), None

        monkeypatch.setattr(adapter_hooks, "_conversion_renamer", lambda m: renamer)
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, shapes, rank=RANK, target_modules=["q_proj_old"])
        assert e.value.reason == "missing_module"
        assert e.value.offenders[0]["module"] == "model.layers.0.self_attn.q_proj_new"

    def test_a_renaming_onto_a_live_module_still_wins(self, monkeypatch):
        """The fallback applies only when the renamed module is absent: transformers prefers
        the renamed key whenever it exists."""
        model = tiny_llama()
        old = "model.layers.0.self_attn.k_proj"

        def renamer(key):
            return key.replace(".k_proj.", ".q_proj."), None

        tensors = make_tensors(model, [self.PATH])
        shapes = {k.replace(".q_proj.", ".k_proj."): v for k, v in adapter_shapes(tensors).items()}
        monkeypatch.setattr(adapter_hooks, "_conversion_renamer", lambda m: renamer)
        (target,) = plan_targets(model, shapes, rank=RANK, target_modules=["k_proj"])
        assert target.module_path == self.PATH
        assert target.a_key.endswith(f"{old}.lora_A.weight")
