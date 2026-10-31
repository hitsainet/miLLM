"""The placement resolver: which card a job goes on, or how it is split.

Operator decisions this pins (2026-09-13):
  1. Auto picks the card with the MOST free memory that fits.
  2. Nothing is reserved for other apps; live free memory is the budget.
  3. A requested card is honoured if it fits and REFUSED if it does not — never
     swapped for another card. "all" (every card) is an explicit request too.
  4. No single card fits -> a split over GPUs only: the most-free cards first,
     as few as cover the model, each less SHARD_RESERVE_MB (Phase 2, 2026-09-14).

Fixtures deliberately put the most free memory on a card that is NOT index 0,
and give cards uneven sizes, so "pick index 0", "pick the biggest total" and
"choose cards in index order" all produce wrong answers here. The expected split
figures were worked out by hand (free - 1024, x0.9 for bitsandbytes), not read
back from the implementation.

Review round 1 (2026-09-14): a transformers split CHOOSES its cards most free
first but plans their shares in INDEX order, each card but the last to its
whole budget — the order accelerate fills them, so its held-back largest layer
lands on real memory (tests/unit/ml/test_shard_plan_against_accelerate.py runs
the real map inference). GGUF shares stay most-free-first.

MUTATION CONTROLS (each must turn this file red):
  * `max(..., key=free_mb)` -> `min(...)`             -> most-free tests fail
  * drop the explicit card's free < required refusal  -> refusal tests fail
  * return best card when best.free < required        -> no-single-card test fails
Phase 2, 2026-09-14 (mutate.py: one line changed, suite run, file restored and
its sha256 verified):
  M1  a "cpu" entry added to a split's max_memory
      -> test_the_split_is_planned_on_gpus_only, test_max_memory_names_gpus_only (and 8 more)
  M4  plan_shard orders cards by index, not most free first
      -> test_the_split_is_planned_on_gpus_only, test_three_cards_takes_only_the_most_free_two,
         test_a_split_reports_its_plan, test_the_plan_counts_transformers_09_...
  M5  the Auto split takes every usable card whatever the need
      -> test_three_cards_takes_only_the_most_free_two, test_the_most_free_card_at_index_0_is_filled_first
  M8  max_memory applies bitsandbytes' 0.9 itself    -> test_the_plan_counts_transformers_09_and_the_map_does_not_repeat_it
  M14 drop the embedding leaf-name match             -> test_a_nested_multimodal_embedding_in_the_map
  M15 "all" parsed as auto                           -> test_all_is_every_card, TestAllCardsOnRequest (12 failed)
  M16 the highest-index card capped at its share too -> test_three_cards_takes_only_the_most_free_two,
         test_the_most_free_card_at_index_0_is_filled_first, test_a_small_model_is_divided_across_every_card
  M17 an unmeasured split uses "sequential"         -> test_every_card_gpu_only_balanced_by_accelerate,
         test_an_unmeasured_model_on_request_gets_every_card
  M30 "all" never refused                            -> test_refused_when_every_card_together_lacks_room,
         test_no_card_with_room_is_refused
Review round 1, 2026-09-14 (mutate.py; restored and sha256-verified):
  R1-M1  transformers_shard_rule back to most-free-first shares (fill_in_index_order=False)
         -> 9 red: test_the_split_is_planned_on_gpus_only, test_a_split_reports_its_plan,
            test_the_plan_counts_transformers_09_..., the two auto-split tests in
            test_shard_plan_against_accelerate.py, and 4 load/fit-check tests
  R1-M1b plan_shard ignores the rule's fill order                  -> the same 9 red
  M4 re-run against the new plan (cards chosen by index)           -> 6 red, incl.
         test_three_cards_takes_only_the_most_free_two and 4 GGUF tensor-split tests
  M16 re-run (the highest-index card capped at its share too)      -> 10 red, incl.
         test_all_divides_a_model_one_card_could_hold on real map inference
Review round 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  R2-M1  plan_shard's "all" index-order rule disabled
         -> test_a_model_the_lower_cards_cannot_hold_is_filled_like_auto
            (and both real-map tests in test_split_preflight.py)
  R2-M1b its boundary `>` -> `>=`
         -> test_a_model_the_lower_cards_could_hold_exactly_stays_proportional
  R1-M1 re-run -> 17 red, 4 here; M16 re-run -> 20 red, 6 here
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from millm.core.errors import GpuNotFoundError, InsufficientMemoryError
from millm.ml import gpu_placement
from millm.ml.gpu_placement import (
    ALL,
    MODE_SHARD,
    MODE_SINGLE,
    REASON_MOST_FREE,
    REASON_NO_SINGLE_CARD,
    REASON_REQUESTED,
    REASON_REQUESTED_ALL,
    REASON_SIZE_UNKNOWN,
    GpuInfo,
    choose_gpu,
    layer_share_by_index,
    list_gpus,
    memory_used_by_device,
    model_device_labels,
    model_input_device,
    normalize_gpu_uuid,
    parse_gpu_request,
    transformers_shard_rule,
)
from tests.support.fake_gpus import NODE_UUIDS, RTX_3090, TI_3080, fake_gpus


def _cards(*free_total: tuple[int, int]) -> list[GpuInfo]:
    return [
        GpuInfo(index=i, name=f"card{i}", uuid=NODE_UUIDS[i], total_mb=t, free_mb=f)
        for i, (f, t) in enumerate(free_total)
    ]


class TestInventory:
    def test_zero_cards(self):
        with fake_gpus():
            assert list_gpus() == []

    def test_one_card(self):
        with fake_gpus((RTX_3090, 23_000, 24_576)):
            [gpu] = list_gpus()
        assert (gpu.index, gpu.name, gpu.free_mb, gpu.total_mb) == (0, RTX_3090, 23_000, 24_576)

    def test_two_cards_carry_name_uuid_and_memory(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_500, 24_576)):
            gpus = list_gpus()
        assert [g.to_dict() for g in gpus] == [
            {"index": 0, "name": TI_3080, "uuid": NODE_UUIDS[0], "total_mb": 12_288, "free_mb": 11_000},
            {"index": 1, "name": RTX_3090, "uuid": NODE_UUIDS[1], "total_mb": 24_576, "free_mb": 23_500},
        ]

    def test_three_cards(self):
        with fake_gpus((TI_3080, 1_000, 12_288), (RTX_3090, 2_000, 24_576), ("A", 3_000, 48_000)):
            assert [g.index for g in list_gpus()] == [0, 1, 2]

    def test_a_card_torch_cannot_identify_is_left_out(self):
        """Without its UUID a card cannot be mapped to a torch index, and a
        guessed index could name another card."""
        with fake_gpus((TI_3080, 11_000, 12_288)), patch(
            "torch.cuda.get_device_properties", side_effect=RuntimeError("nvml")
        ):
            assert list_gpus() == []


class TestUuidNormalisation:
    def test_torch_bare_uuid_gets_the_nvidia_smi_prefix(self):
        assert normalize_gpu_uuid("AAAAAAAA-bbbb-cccc-dddd-eeeeeeeeeeee") == NODE_UUIDS[1]

    def test_prefixed_uuid_in_any_case(self):
        assert normalize_gpu_uuid("gpu-AAAAAAAA-BBBB-cccc-dddd-eeeeeeeeeeee") == NODE_UUIDS[1]

    def test_sixteen_raw_bytes(self):
        raw = bytes.fromhex("aaaaaaaabbbbccccddddeeeeeeeeeeee")
        assert normalize_gpu_uuid(raw) == NODE_UUIDS[1]

    def test_garbage_is_not_a_uuid(self):
        assert normalize_gpu_uuid("not-a-card") is None
        assert normalize_gpu_uuid(None) is None


class TestRequestParsing:
    @pytest.mark.parametrize("value", [None, "auto", "AUTO", "", "  "])
    def test_auto(self, value):
        assert parse_gpu_request(value) is None

    @pytest.mark.parametrize("value", ["all", "ALL", " All "])
    def test_all_is_every_card(self, value):
        assert parse_gpu_request(value) == ALL

    def test_all_parses_to_itself(self):
        """The service parses the request, then the loader parses it again."""
        assert parse_gpu_request(parse_gpu_request("all")) == ALL

    def test_index_as_int_or_digit_string(self):
        assert parse_gpu_request(1) == 1
        assert parse_gpu_request("1") == 1

    def test_uuid_is_normalised(self):
        assert parse_gpu_request(NODE_UUIDS[1][4:]) == NODE_UUIDS[1]

    @pytest.mark.parametrize("value", [True, -1, "gpu-one", 1.5, ["1"], "every"])
    def test_anything_else_is_refused_not_read_as_auto(self, value):
        with pytest.raises(ValueError):
            parse_gpu_request(value)


class TestAutoPicksTheMostFreeCardThatFits:
    def test_two_cards_most_free_is_index_1(self):
        placement = choose_gpu(8_000, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert (placement.mode, placement.index, placement.reason) == (MODE_SINGLE, 1, REASON_MOST_FREE)
        assert placement.device_label == "cuda:1"
        assert placement.gpu_indices == [1]
        assert placement.capacity_mb == 23_500

    def test_most_free_not_largest_total(self):
        # Card 1 is bigger but busier; card 0 has more FREE memory right now.
        placement = choose_gpu(4_000, gpus=_cards((11_000, 12_288), (6_000, 24_576)))
        assert placement.index == 0

    def test_three_cards_uneven(self):
        placement = choose_gpu(
            10_000, gpus=_cards((11_000, 12_288), (15_000, 24_576), (40_000, 48_000))
        )
        assert placement.index == 2

    def test_one_card(self):
        placement = choose_gpu(8_000, gpus=_cards((23_000, 24_576)))
        assert (placement.mode, placement.index) == (MODE_SINGLE, 0)

    def test_a_tie_goes_to_the_lower_index_so_the_choice_is_stable(self):
        assert choose_gpu(1_000, gpus=_cards((9_000, 12_288), (9_000, 24_576))).index == 0

    def test_a_model_that_fits_a_card_exactly_is_not_split(self):
        """A single card carries no reserve: the estimate has its own overhead."""
        placement = choose_gpu(23_500, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert placement.mode == MODE_SINGLE

    def test_live_inventory_is_read_when_none_is_given(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_500, 24_576)) as fake:
            placement = choose_gpu(18_000)
        assert placement.index == 1
        # Read from nvidia-smi: deciding creates no CUDA context on any card.
        assert fake.smi_calls >= 1
        assert fake.calls == []


class TestASplitWhenNoSingleCardFits:
    def test_the_split_is_planned_on_gpus_only(self):
        placement = choose_gpu(30_000, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert (placement.mode, placement.reason) == (MODE_SHARD, REASON_NO_SINGLE_CARD)
        assert placement.gpu_indices == [0, 1]
        # Both cards are needed. accelerate fills index 0 first, so the 3080 Ti
        # is planned whole and the 3090 carries the remainder with slack for
        # the layer accelerate holds back on card 0.
        assert placement.planned_mb_by_index == {0: 9_976, 1: 20_024}
        assert placement.budget_mb_by_index == {1: 22_476, 0: 9_976}
        assert placement.budget_mb == 32_452
        assert placement.capacity_mb == 34_500
        assert placement.transformers_device_map() == "sequential"
        assert placement.transformers_max_memory() == {0: "9976MiB", 1: "22476MiB"}

    def test_max_memory_names_gpus_only(self):
        placement = choose_gpu(30_000, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert all(isinstance(key, int) for key in placement.transformers_max_memory())

    def test_three_cards_takes_only_the_most_free_two(self):
        placement = choose_gpu(
            50_000, gpus=_cards((11_000, 12_288), (40_000, 48_000), (15_000, 24_576))
        )
        assert placement.gpu_indices == [1, 2], "card 0 is not needed and must not be claimed"
        assert placement.planned_mb_by_index == {1: 38_976, 2: 11_024}
        assert placement.transformers_max_memory() == {1: "38976MiB", 2: "13976MiB"}

    def test_the_most_free_card_at_index_0_is_filled_first(self):
        placement = choose_gpu(30_000, gpus=_cards((23_500, 24_576), (11_000, 12_288)))
        assert placement.planned_mb_by_index == {0: 22_476, 1: 7_524}
        # Index 1 is filled last by accelerate, so it keeps its whole limit.
        assert placement.transformers_max_memory() == {0: "22476MiB", 1: "9976MiB"}

    def test_when_every_card_together_is_short_the_split_names_every_usable_card(self):
        placement = choose_gpu(
            40_000, gpus=_cards((900, 12_288), (23_500, 24_576), (11_000, 12_288))
        )
        assert placement.gpu_indices == [1, 2], "a card with no room past its reserve is left out"
        assert placement.planned_mb_by_index == {1: 22_476, 2: 9_976}
        assert placement.budget_mb == 32_452

    def test_a_split_reports_its_plan(self):
        report = choose_gpu(30_000, gpus=_cards((11_000, 12_288), (23_500, 24_576))).to_dict()
        assert report["mode"] == "shard"
        assert report["devices"] == ["cuda:0", "cuda:1"]
        assert report["planned_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 20_024}
        assert report["budget_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 22_476}


class TestAnUnmeasuredModel:
    def test_every_card_gpu_only_balanced_by_accelerate(self):
        placement = choose_gpu(0, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert (placement.mode, placement.reason) == (MODE_SHARD, REASON_SIZE_UNKNOWN)
        assert placement.gpu_indices == [0, 1]
        assert placement.transformers_device_map() == "auto"
        assert placement.transformers_max_memory() == {0: "9976MiB", 1: "22476MiB"}


class TestAllCardsOnRequest:
    NODE = ((11_000, 12_288), (23_000, 24_576))

    def test_a_small_model_is_divided_across_every_card(self):
        placement = choose_gpu(2_870, requested="all", gpus=_cards(*self.NODE))
        assert (placement.mode, placement.reason, placement.requested) == (
            MODE_SHARD, REASON_REQUESTED_ALL, ALL,
        )
        # In proportion to each card's budget (9,976 : 21,976), rounded up.
        assert placement.planned_mb_by_index == {1: 1_974, 0: 897}
        assert placement.transformers_device_map() == "sequential"
        assert placement.transformers_max_memory() == {0: "897MiB", 1: "21976MiB"}

    def test_a_model_the_lower_cards_cannot_hold_is_filled_like_auto(self):
        """Review round 2. 30,000 MB is more than card 0's 9,976 budget, so filling
        in index order reaches card 1 anyway: card 0 whole, card 1 the remainder.
        The proportional plan capped card 0 at ceil(30,000 x 9,976 / 31,952) =
        9,367, and the layer accelerate holds back on card 0 had nowhere to go."""
        placement = choose_gpu(30_000, requested="all", gpus=_cards(*self.NODE))
        assert (placement.reason, placement.gpu_indices) == (REASON_REQUESTED_ALL, [0, 1])
        assert placement.planned_mb_by_index == {0: 9_976, 1: 20_024}
        assert placement.transformers_max_memory() == {0: "9976MiB", 1: "21976MiB"}

    def test_a_model_the_lower_cards_could_hold_exactly_stays_proportional(self):
        """At exactly card 0's budget (9,976) card 0 could hold it alone, so it
        is divided: ceil(9,976 x 9,976 / 31,952) = 3,115 and
        ceil(9,976 x 21,976 / 31,952) = 6,862. Filled in index order it would
        leave card 1 with nothing — not a split across every card."""
        placement = choose_gpu(9_976, requested="all", gpus=_cards(*self.NODE))
        assert placement.planned_mb_by_index == {0: 3_115, 1: 6_862}

    def test_refused_when_every_card_together_lacks_room(self):
        with pytest.raises(InsufficientMemoryError) as raised:
            choose_gpu(40_000, requested="all", gpus=_cards(*self.NODE))
        details = raised.value.details
        assert details["required_mb"] == 40_000
        assert details["available_mb"] == 31_952
        assert details["requested"] == ALL
        assert details["budget_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 21_976}

    def test_an_unmeasured_model_on_request_gets_every_card(self):
        placement = choose_gpu(0, requested="all", gpus=_cards(*self.NODE))
        assert (placement.reason, placement.gpu_indices) == (REASON_REQUESTED_ALL, [0, 1])
        assert placement.transformers_device_map() == "auto"

    def test_no_card_with_room_is_refused(self):
        with pytest.raises(InsufficientMemoryError):
            choose_gpu(0, requested="all", gpus=_cards((500, 12_288), (800, 24_576)))


class TestBitsandbytesFactorIsCountedOnce:
    def test_the_plan_counts_transformers_09_and_the_map_does_not_repeat_it(self):
        placement = choose_gpu(
            25_000,
            gpus=_cards((11_000, 12_288), (23_000, 24_576)),
            shard=transformers_shard_rule(25_000, bitsandbytes=True),
        )
        assert placement.budget_mb_by_index == {1: 19_778, 0: 8_978}
        assert placement.planned_mb_by_index == {0: 8_978, 1: 16_022}
        # transformers multiplies these by 0.9: 9976 -> 8978 and 21976 -> 19778.
        assert placement.transformers_max_memory() == {0: "9976MiB", 1: "21976MiB"}


class TestAnExplicitCardIsHonouredOrRefused:
    def test_a_requested_card_that_fits_is_used_even_when_another_has_more(self):
        placement = choose_gpu(8_000, requested=0, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert (placement.mode, placement.index, placement.reason) == (MODE_SINGLE, 0, REASON_REQUESTED)
        assert placement.requested == 0

    def test_by_uuid(self):
        placement = choose_gpu(
            8_000, requested=NODE_UUIDS[1][4:], gpus=_cards((11_000, 12_288), (23_500, 24_576))
        )
        assert placement.index == 1

    def test_a_requested_card_without_room_is_refused_not_swapped(self):
        with pytest.raises(InsufficientMemoryError) as raised:
            choose_gpu(18_000, requested=0, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        details = raised.value.details
        assert details["required_mb"] == 18_000
        assert details["available_mb"] == 11_000
        assert details["gpu"]["index"] == 0
        assert [g["index"] for g in details["gpus"]] == [0, 1]
        assert "GPU 0" in str(raised.value) and "11000 MB free" in str(raised.value)

    def test_an_unknown_card_is_not_found(self):
        with pytest.raises(GpuNotFoundError):
            choose_gpu(1_000, requested=5, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        with pytest.raises(GpuNotFoundError):
            choose_gpu(1_000, requested=NODE_UUIDS[2], gpus=_cards((11_000, 12_288)))


class TestNoCards:
    def test_auto_with_no_gpu_is_refused(self):
        with pytest.raises(InsufficientMemoryError) as raised:
            choose_gpu(1_000, gpus=[])
        assert raised.value.details["gpus"] == []

    def test_all_with_no_gpu_is_refused(self):
        with pytest.raises(InsufficientMemoryError):
            choose_gpu(1_000, requested="all", gpus=[])


class TestDeviceMaps:
    def test_single_card_is_whole_model_on_that_card(self):
        placement = choose_gpu(8_000, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert placement.transformers_device_map() == {"": "cuda:1"}
        assert placement.transformers_max_memory() is None

    def test_a_single_card_reports_no_plan(self):
        report = choose_gpu(8_000, gpus=_cards((11_000, 12_288), (23_500, 24_576))).to_dict()
        assert report["planned_mb_by_device"] == {}
        assert report["budget_mb_by_device"] == {}


class TestMeasurement:
    def test_per_device_deltas_not_card_usage(self):
        assert memory_used_by_device({0: 11_000, 1: 23_000}, {0: 10_900, 1: 7_000}, [1]) == {
            "cuda:1": 16_000
        }

    def test_a_card_that_gained_memory_counts_zero(self):
        assert memory_used_by_device({1: 5_000}, {1: 6_000}) == {"cuda:1": 0}

    def test_model_devices_come_from_the_map_and_the_tensors(self):
        tensor = SimpleNamespace(device=torch.device("cuda", 1))
        model = SimpleNamespace(
            hf_device_map={"model.embed_tokens": 0, "lm_head": "cpu"},
            parameters=lambda: iter([tensor]),
            buffers=lambda: iter([]),
        )
        assert model_device_labels(model) == ["cpu", "cuda:0", "cuda:1"]

    def test_the_inventory_comes_from_nvidia_smi(self):
        with fake_gpus((TI_3080, 11_000, 12_288)), patch.object(
            gpu_placement.nvidia_smi, "query_gpus", return_value=[]
        ) as reader:
            assert list_gpus() == []
        reader.assert_called_once_with()


class TestTheInputDevice:
    def test_the_model_answers_first(self):
        weight = SimpleNamespace(device=torch.device("cuda", 1))
        model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=weight),
            hf_device_map={"": 0},
        )
        assert model_input_device(model) == "cuda:1"

    def test_a_nested_multimodal_embedding_in_the_map(self):
        """gemma-4 keeps its text stack under model.language_model, after the
        vision tower in the map."""
        model = SimpleNamespace(hf_device_map={
            "model.vision_tower": 0,
            "model.language_model.embed_tokens": 1,
            "model.language_model.layers.0": 1,
            "lm_head": 0,
        })
        assert model_input_device(model) == "cuda:1"

    def test_flat_layouts_by_name_not_by_position(self):
        assert model_input_device(SimpleNamespace(hf_device_map={"lm_head": 1, "model.embed_tokens": 0})) == "cuda:0"
        assert model_input_device(SimpleNamespace(hf_device_map={"transformer.h.0": 1, "transformer.wte": 0})) == "cuda:0"

    def test_a_map_with_no_embedding_uses_its_first_card(self):
        assert model_input_device(SimpleNamespace(hf_device_map={"a": "cpu", "b": 1})) == "cuda:1"

    def test_a_model_on_the_cpu(self):
        assert model_input_device(SimpleNamespace(hf_device_map={"": "cpu"})) == "cpu"


class TestLayerShares:
    def test_the_share_follows_the_layers_on_each_card(self):
        model = SimpleNamespace(hf_device_map={
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.layers.2": 1,
            "model.layers.3": 1,
            "model.norm": 1,
            "lm_head": 1,
        })
        assert layer_share_by_index(model, [0, 1]) == {0: 0.25, 1: 0.75}

    def test_one_card_holds_all_of_it(self):
        assert layer_share_by_index(SimpleNamespace(), [1]) == {1: 1.0}

    def test_a_map_naming_no_layers_is_shared_evenly(self):
        model = SimpleNamespace(hf_device_map={"model": 0, "lm_head": 1})
        assert layer_share_by_index(model, [0, 1]) == {0: 0.5, 1: 0.5}
