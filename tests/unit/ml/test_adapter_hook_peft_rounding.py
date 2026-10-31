"""The hook's final add, against PEFT's (033 review R1-C; addendum D-12).

D-12 adopts fp32 adapter compute "matching PEFT's default, so miForge evaluation and miLLM serving
agree". PEFT's `lora.Linear.forward` (`autocast_adapter_dtype=True`) does

    result = self.base_layer(x)                      # bf16
    result = result + lora_B(lora_A(x_fp32)) * s      # bf16 + fp32 -> promoted to fp32
    result = result.to(torch_result_dtype)            # ONE rounding to bf16

while `adapter_hooks._make_hook` returns `output + delta.to(output.dtype)`: the delta is rounded to
bf16 first and the sum is rounded again. On a bf16 module about a fifth of the output elements then
differ from PEFT's by one bf16 ulp (measured: 6,576 of 32,768, max 0.0156 on a 256->512 linear).
`test_adapter_hooks.test_module_output_is_base_plus_the_fp32_delta` asserts the double-rounded form,
so it pins the difference rather than preventing it.

Fix (millm/ml/adapter_hooks.py, `_make_hook`): `return (output + delta.to(output.device)).to(
output.dtype)` — a bf16 output upcasts exactly, so this is PEFT's single rounding. Then update the
pinning test's `expected` to the same form and drop the xfail below.

Fixed in R1-B `d6cbdfe` (and proven equal to real PEFT by `test_adapter_peft_golden.py`); this is
now an ordinary regression test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from millm.ml.adapter_hooks import LoraHookSet, plan_targets
from tests.unit.adapter_fixtures import ALPHA, RANK, make_tensors, peft_key, tiny_llama


def _peft_reference(base: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                    scaling: float) -> torch.Tensor:
    return (base + F.linear(F.linear(x.float(), a), b) * scaling).to(base.dtype)


def test_the_bf16_fixture_separates_one_rounding_from_two():
    """The xfail below means something only if the two forms differ on this fixture."""
    model = tiny_llama(torch.bfloat16)
    path = "model.layers.0.mlp.gate_proj"
    tensors = make_tensors(model, [path])
    a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
    x = torch.randn(4, 9, 48, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    with torch.no_grad():
        base = model.get_submodule(path)(x)
    delta = F.linear(F.linear(x.float(), a), b) * (ALPHA / RANK)
    assert not torch.equal(base + delta.to(base.dtype), _peft_reference(base, x, a, b, ALPHA / RANK))


def test_a_bf16_module_output_equals_peft_single_rounding():
    model = tiny_llama(torch.bfloat16)
    path = "model.layers.0.mlp.gate_proj"
    tensors = make_tensors(model, [path])
    module = model.get_submodule(path)
    x = torch.randn(4, 9, 48, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    with torch.no_grad():
        base = module(x)
        shapes = {k: tuple(v.shape) for k, v in tensors.items()}
        targets = plan_targets(model, shapes, rank=RANK, target_modules=["gate_proj"])
        prepared = LoraHookSet.build(model, targets, tensors, ALPHA / RANK)
        handles = LoraHookSet.install(model, prepared)
        try:
            hooked = module(x)
        finally:
            LoraHookSet.remove(handles)
    a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
    assert torch.equal(hooked, _peft_reference(base, x, a, b, ALPHA / RANK))
