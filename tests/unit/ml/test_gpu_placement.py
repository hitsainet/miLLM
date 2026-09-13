"""The placement resolver: which card a job goes on.

Operator decisions this pins (2026-09-13):
  1. Auto picks the card with the MOST free memory that fits.
  2. Nothing is reserved; live free memory is the budget.
  3. A requested card is honoured if it fits and REFUSED if it does not — never
     swapped for another card.
  4. No single card fits -> every card (Phase 1's stand-in for sharding).

Fixtures deliberately put the most free memory on a card that is NOT index 0,
and give cards uneven sizes, so "pick index 0" and "pick the biggest total"
both produce wrong answers here.

MUTATION CONTROLS (each must turn this file red):
  * `max(..., key=free_mb)` -> `min(...)`             -> most-free tests fail
  * drop the explicit card's free < required refusal  -> refusal tests fail
  * return best card when best.free < required        -> no-single-card test fails
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from millm.core.errors import GpuNotFoundError, InsufficientMemoryError
from millm.ml import gpu_placement
from millm.ml.gpu_placement import (
    MODE_ALL,
    MODE_SINGLE,
    REASON_MOST_FREE,
    REASON_NO_SINGLE_CARD,
    REASON_REQUESTED,
    REASON_SIZE_UNKNOWN,
    GpuInfo,
    choose_gpu,
    list_gpus,
    memory_used_by_device,
    model_device_labels,
    normalize_gpu_uuid,
    parse_gpu_request,
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

    def test_index_as_int_or_digit_string(self):
        assert parse_gpu_request(1) == 1
        assert parse_gpu_request("1") == 1

    def test_uuid_is_normalised(self):
        assert parse_gpu_request(NODE_UUIDS[1][4:]) == NODE_UUIDS[1]

    @pytest.mark.parametrize("value", [True, -1, "gpu-one", 1.5, ["1"]])
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

    def test_live_inventory_is_read_when_none_is_given(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_500, 24_576)) as fake:
            placement = choose_gpu(18_000)
        assert placement.index == 1
        # Read from nvidia-smi: deciding creates no CUDA context on any card.
        assert fake.smi_calls >= 1
        assert fake.calls == []


class TestNoSingleCardFits:
    def test_falls_back_to_every_card(self):
        placement = choose_gpu(30_000, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert (placement.mode, placement.reason) == (MODE_ALL, REASON_NO_SINGLE_CARD)
        assert placement.gpu_indices == [0, 1]
        assert placement.capacity_mb == 34_500
        assert placement.transformers_device_map(bitsandbytes=False) == "auto"

    def test_unknown_size_keeps_the_spread_it_always_had(self):
        placement = choose_gpu(0, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert (placement.mode, placement.reason) == (MODE_ALL, REASON_SIZE_UNKNOWN)


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


class TestDeviceMaps:
    def test_single_card_is_whole_model_on_that_card(self):
        placement = choose_gpu(8_000, gpus=_cards((11_000, 12_288), (23_500, 24_576)))
        assert placement.transformers_device_map(bitsandbytes=False) == {"": "cuda:1"}

    def test_bitsandbytes_max_memory_is_keyed_to_the_chosen_card_not_0(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)):
            placement = choose_gpu(8_000)
            budget = placement.bitsandbytes_max_memory("60GiB")
        assert placement.transformers_device_map(bitsandbytes=True) == "auto"
        assert set(budget) == {1, "cpu"}
        assert budget[1] == f"{int(23_000 * 1024 * 1024 * 0.9) // 1024 ** 3}GiB"
        assert budget["cpu"] == "60GiB"

    def test_bitsandbytes_fallback_budgets_every_card(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)):
            budget = choose_gpu(30_000).bitsandbytes_max_memory("8GiB")
        assert set(budget) == {0, 1, "cpu"}


class TestMeasurement:
    def test_per_device_deltas_not_card_usage(self):
        assert memory_used_by_device({0: 11_000, 1: 23_000}, {0: 10_900, 1: 7_000}, [1]) == {
            "cuda:1": 16_000
        }

    def test_a_card_that_gained_memory_counts_zero(self):
        assert memory_used_by_device({1: 5_000}, {1: 6_000}) == {"cuda:1": 0}

    def test_model_devices_come_from_the_map_and_the_tensors(self):
        import torch

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
