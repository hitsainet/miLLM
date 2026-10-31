"""A split the plan accepts is a split transformers can actually map onto the GPUs.

Placement plans in MB. transformers 5.15.1 then maps modules with its own copy of
accelerate's `infer_auto_device_map`, which

  1. fills the cards it is given in INDEX order, whatever order they were chosen in;
  2. holds back room for the model's largest layer on the LOWEST-index card
     (`main_devices = [gpus[0], "cpu"]`, integrations/accelerate.py:578 and 722);
  3. strands the tail of every card but the last when the next layer does not fit
     whole.

A plan that caps a card at a planned share below its limit gives that room away.
On the node (3080 Ti at index 0 with less free memory, 3090 at index 1) Phase 2's
first plan did exactly that: the 3090 was planned whole, the 3080 Ti was capped
at the remainder, and (2) + (3) came out of the 3080 Ti's cap with nowhere to go.
Simulated on the Qwen2.5-14B shape (the Phase 2 acceptance model) with an
estimate 5% over its real weights, lm_head (1,485 MiB) was mapped to "disk" with
4.3 GB of planned budget unused — refused after the resident model was unloaded.

These tests run the REAL map inference over a meta-device model (no weights, no
GPU) with exactly the `device_map` and `max_memory` the placement hands
`from_pretrained`. Card sizes are derived from the model's measured size, so the
fixture cannot agree with the plan by construction: the largest layer is an
untied lm_head, as on Qwen2.5 and gemma, and the smaller free card is index 0.
The model is big enough (~3.7 GB) that no card's free memory, reserve included,
holds it alone — otherwise the placement never splits and nothing is tested.

MUTATION CONTROLS (review round 1, 2026-09-14; mutate.py, file restored, sha256 verified):
  R1-M1  transformers splits fill cards most-free-first again (fill_in_index_order=False)
         -> test_an_auto_split_the_plan_accepts_maps_nothing_off_the_gpu,
            test_an_auto_split_uses_only_the_cards_the_plan_chose
  R1-M1b plan_shard ignores the rule's fill order (always most-free-first)
         -> the same two tests
"""

from __future__ import annotations

import pytest
import torch

from millm.ml.gpu_placement import SHARD_RESERVE_MB, GpuInfo
from millm.ml.model_loader import decide_transformers_placement

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, Qwen2Config  # noqa: E402
from transformers.integrations.accelerate import (  # noqa: E402
    _get_device_map,
    compute_module_sizes,
)

MIB = 1024 * 1024


@pytest.fixture(scope="module")
def model():
    """Eight ~338 MiB decoder layers; untied embed_tokens and lm_head of ~488 MiB each."""
    config = Qwen2Config(
        vocab_size=64_000, hidden_size=4_096, intermediate_size=11_008, num_hidden_layers=8,
        num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=False,
    )
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)


def _mib(model, name: str) -> int:
    sizes, _ = compute_module_sizes(model, None)
    return int(sizes[name] / MIB)


def _card(index: int, limit_mb: int) -> GpuInfo:
    free = limit_mb + SHARD_RESERVE_MB
    return GpuInfo(index=index, name=f"card{index}", uuid=None, total_mb=free + 2_000, free_mb=free)


def _mapped_mb(model, placement) -> dict[str, int]:
    """Run transformers' own map inference on what the placement passes; MiB per device."""
    device_map = _get_device_map(
        model, placement.transformers_device_map(), dict(placement.transformers_max_memory()), None
    )
    sizes, _ = compute_module_sizes(model, None)
    per_device: dict[str, int] = {}
    for name, device in device_map.items():
        per_device[str(device)] = per_device.get(str(device), 0) + int(sizes[name] / MIB)
    return per_device


def test_the_fixture_is_in_the_region_where_the_bug_bites(model):
    """Guards the guard: lm_head is the largest layer, and a reservation plus one
    stranded layer is more than the 5% the estimate below carries over the weights."""
    total, lm_head, layer = _mib(model, ""), _mib(model, "lm_head"), _mib(model, "model.layers.0")
    assert lm_head > layer
    assert lm_head + layer > 0.15 * total
    assert int(total * 0.7) + SHARD_RESERVE_MB < int(total * 1.05), "one card would hold it"


def test_an_auto_split_the_plan_accepts_maps_nothing_off_the_gpu(model):
    total = _mib(model, "")
    need = int(total * 1.05)  # an estimate 5% over the real weights
    cards = [_card(0, int(total * 0.6)), _card(1, int(total * 0.7))]

    placement = decide_transformers_placement(need, "FP16", requested=None, gpus=cards)

    assert placement.is_shard and placement.gpu_indices == [0, 1]
    assert placement.budget_mb >= need, "the plan accepted this split"
    mapped = _mapped_mb(model, placement)
    assert set(mapped) == {"0", "1"}, f"a module left the GPUs: {mapped}"
    assert sum(mapped.values()) == pytest.approx(total, abs=8)


def test_an_auto_split_uses_only_the_cards_the_plan_chose(model):
    """Filling in index order must not reach a card the plan left out."""
    total = _mib(model, "")
    need = int(total * 1.05)
    cards = [
        _card(0, int(total * 0.3)),  # least free: not needed, must not be used
        _card(1, int(total * 0.6)),
        _card(2, int(total * 0.7)),
    ]

    placement = decide_transformers_placement(need, "FP16", requested=None, gpus=cards)

    assert placement.gpu_indices == [1, 2]
    assert set(placement.transformers_max_memory()) == {1, 2}
    mapped = _mapped_mb(model, placement)
    assert set(mapped) == {"1", "2"}, f"a module left the chosen GPUs: {mapped}"


def test_all_divides_a_model_one_card_could_hold(model):
    """"all" exists to force a split: the real map must put layers on both cards."""
    total, layer = _mib(model, ""), _mib(model, "model.layers.0")
    cards = [_card(0, total * 2), _card(1, total * 3)]

    placement = decide_transformers_placement(total + 50, "FP16", requested="all", gpus=cards)

    mapped = _mapped_mb(model, placement)
    assert set(mapped) == {"0", "1"}, mapped
    assert min(mapped.values()) >= layer, f"one card holds less than a layer: {mapped}"
