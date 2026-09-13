"""GPU memory helpers see every card, not only device 0."""

from unittest.mock import patch

from millm.ml import memory_utils

MB = 1024 * 1024


def _cards(*free_total_mb: tuple[int, int]):
    """Patch torch.cuda to report one card per (free_mb, total_mb) pair."""
    return patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        device_count=lambda: len(free_total_mb),
        mem_get_info=lambda index: (
            free_total_mb[index][0] * MB,
            free_total_mb[index][1] * MB,
        ),
    )


class TestListGpuMemory:
    def test_lists_every_card_in_index_order(self):
        with _cards((11_000, 12_288), (23_500, 24_576)):
            assert memory_utils.list_gpu_memory() == [
                {"index": 0, "free_mb": 11_000, "total_mb": 12_288},
                {"index": 1, "free_mb": 23_500, "total_mb": 24_576},
            ]

    def test_no_cuda_lists_nothing(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert memory_utils.list_gpu_memory() == []

    def test_a_driver_error_lists_nothing_instead_of_raising(self):
        with patch.multiple(
            "torch.cuda",
            is_available=lambda: True,
            device_count=lambda: 2,
            mem_get_info=lambda index: (_ for _ in ()).throw(RuntimeError("driver")),
        ):
            assert memory_utils.list_gpu_memory() == []


class TestAcrossCards:
    def test_total_free_sums_every_card(self):
        with _cards((11_000, 12_288), (23_500, 24_576)):
            assert memory_utils.get_total_free_memory_mb() == 34_500

    def test_largest_free_is_the_single_best_card(self):
        with _cards((11_000, 12_288), (23_500, 24_576)):
            assert memory_utils.get_largest_free_memory_mb() == 23_500

    def test_no_cards_means_zero(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert memory_utils.get_total_free_memory_mb() == 0
            assert memory_utils.get_largest_free_memory_mb() == 0
