"""A split's layout gives each card all of its weights and the KV cache of every layer it touches.

Review round 5, 2026-09-14. TransformersFit.layout and preflight_split turn
transformers' own device map into MiB per card, and the fit then asks each card
for its weights plus its layers' KV cache. Two ways a card was under-asked:

  * Weights were floored to MiB PER MAPPED MODULE and then summed. A map of many
    small entries loses up to a MiB each: on gemma-3-12b-it (text and vision
    towers) the cards' weights summed to 23,248 MiB of a 23,274 MiB model — 26 MiB
    the check never asked any card for. Now bytes are summed per device and
    rounded up once.
  * A decoder layer whose submodules transformers puts on two cards — any class
    with no `_no_split_modules`, BioGPT among them — resolved to no device at all
    (`_mapped_device` looks for the layer or an ancestor in the map, and only its
    children are there). Its KV cache was filed under "None" and counted on no
    card. It is now counted on every card holding part of it: keys and values
    follow their projections, so either card may hold the cache.

Shapes are the real configs' fields (google/gemma-3-12b-it, microsoft/BioGPT-Large),
built on the meta device only. The placement is plan_shard's with the slack's
rule, so these figures do not move with the fit's own budgeting.

MUTATION CONTROLS (mutate.py, millm-p2-review5; each restored, sha256 verified,
git diff clean):
  R5-M10 layout floors MiB per module again             -> test_every_mib_of_the_weights_is_on_a_card
  R5-M11 preflight_split floors MiB per module again    -> test_the_preflight_counts_every_mib_too
  R5-M12 a straddling layer resolves to its nearest mapped ancestor only
         -> test_a_layer_on_two_cards_is_counted_on_both
  R5-M13 transformers_fit floors the model's weights    -> test_the_models_weights_round_up
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("transformers")
from transformers import BioGptConfig, Gemma3Config  # noqa: E402

from millm.ml.gpu_placement import (  # noqa: E402
    REASON_NO_SINGLE_CARD,
    GpuInfo,
    plan_shard,
    transformers_shard_rule,
)
from millm.ml.model_loader import preflight_split, transformers_fit  # noqa: E402

MIB = 1024 * 1024

GEMMA3_12B = dict(
    text_config=dict(
        vocab_size=262_208, hidden_size=3_840, intermediate_size=15_360, num_hidden_layers=48,
        num_attention_heads=16, num_key_value_heads=8, head_dim=256, sliding_window=1_024,
    ),
    vision_config=dict(
        hidden_size=1_152, intermediate_size=4_304, num_hidden_layers=27, num_attention_heads=16,
        image_size=896, patch_size=14,
    ),
)
#: microsoft/BioGPT-Large config.json.
BIOGPT_LARGE = dict(
    vocab_size=57_717, hidden_size=1_600, intermediate_size=6_400, num_hidden_layers=48,
    num_attention_heads=25, max_position_embeddings=2_048,
)


def _save(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    return str(directory)


def _split(fit, free):
    gpus = [GpuInfo(index, f"card{index}", None, 24_576, mb) for index, mb in enumerate(free)]
    return plan_shard(gpus, transformers_shard_rule(fit.weights_mb), REASON_NO_SINGLE_CARD, fit.single_card_mb)


@pytest.fixture
def gemma3(tmp_path):
    config = Gemma3Config(**GEMMA3_12B)
    config.architectures = ["Gemma3ForConditionalGeneration"]
    return _save(tmp_path / "gemma3", config)


class TestWeights:
    def test_the_models_weights_round_up(self, gemma3):
        fit = transformers_fit(gemma3, "FP16", False)
        assert fit.weights_mb == math.ceil(fit.sizes[""] / MIB)

    def test_every_mib_of_the_weights_is_on_a_card(self, gemma3):
        fit = transformers_fit(gemma3, "FP16", False)
        layout = fit.layout(_split(fit, (11_500, 23_500)))

        assert set(layout.weights_mb) == {"cuda:0", "cuda:1"}
        placed = sum(layout.weights_mb.values())
        assert fit.weights_mb <= placed <= fit.weights_mb + len(layout.weights_mb), (
            f"the cards hold {placed} MiB of a {fit.weights_mb} MiB model"
        )

    def test_the_preflight_counts_every_mib_too(self, gemma3):
        fit = transformers_fit(gemma3, "FP16", False)

        mapped = preflight_split("gemma-3-12b-it", gemma3, "FP16", _split(fit, (11_500, 23_500)))

        assert fit.weights_mb <= sum(mapped.values()) <= fit.weights_mb + len(mapped)


class TestLayers:
    def test_a_layer_on_two_cards_is_counted_on_both(self, tmp_path):
        """BioGPT has no _no_split_modules: on 2,000 / 3,000 MB free (limits 976 /
        1,976) transformers splits one decoder layer's attention across the cards."""
        config = BioGptConfig(**BIOGPT_LARGE)
        config.architectures = ["BioGptForCausalLM"]
        fit = transformers_fit(_save(tmp_path, config), "FP16", False)
        placement = _split(fit, (2_000, 3_000))

        layout = fit.layout(placement)

        assert set(layout.layers) <= {"cuda:0", "cuda:1", "disk", "cpu"}, (
            f"a layer attributed to no device: {sorted(layout.layers)}"
        )
        on_gpus = [set(layout.layers.get(label, ())) for label in ("cuda:0", "cuda:1")]
        straddling = on_gpus[0] & on_gpus[1]
        assert straddling, "the fixture must split a layer across the two cards"
        attributed = set().union(*layout.layers.values())
        assert attributed == set(range(fit.kv.num_layers)), "every decoder layer lands somewhere"
        for index in straddling:
            per_layer = fit.kv_mb([index])
            cards = [fit.card(gpu, layout.weights_mb.get(gpu.device_label, 0), layout.layers.get(gpu.device_label, ()))
                     for gpu in placement.gpus]
            assert all(card.kv_mb >= per_layer for card in cards)
