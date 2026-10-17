"""GPU memory helpers see every card, not only device 0 — without touching any.

Phase 0 read `torch.cuda.mem_get_info(i)` for every card, which creates a CUDA
context on each. The helpers now read the placement inventory (nvidia-smi).
"""

from millm.ml import memory_utils
from tests.support.fake_gpus import fake_gpus


class TestListGpuMemory:
    def test_lists_every_card_in_index_order(self):
        with fake_gpus(("a", 11_000, 12_288), ("b", 23_500, 24_576)):
            assert memory_utils.list_gpu_memory() == [
                {"index": 0, "free_mb": 11_000, "total_mb": 12_288},
                {"index": 1, "free_mb": 23_500, "total_mb": 24_576},
            ]

    def test_creates_no_cuda_context(self):
        with fake_gpus(("a", 11_000, 12_288), ("b", 23_500, 24_576)) as fake:
            fake.forbid(0, 1)
            memory_utils.list_gpu_memory()
        assert fake.calls == []

    def test_no_cuda_lists_nothing(self):
        with fake_gpus():
            assert memory_utils.list_gpu_memory() == []

    def test_no_nvidia_smi_lists_nothing_instead_of_raising(self):
        with fake_gpus(("a", 11_000, 12_288), smi_available=False):
            assert memory_utils.list_gpu_memory() == []


class TestAcrossCards:
    def test_total_free_sums_every_card(self):
        with fake_gpus(("a", 11_000, 12_288), ("b", 23_500, 24_576)):
            assert memory_utils.get_total_free_memory_mb() == 34_500

    def test_largest_free_is_the_single_best_card(self):
        with fake_gpus(("a", 11_000, 12_288), ("b", 23_500, 24_576)):
            assert memory_utils.get_largest_free_memory_mb() == 23_500

    def test_no_cards_means_zero(self):
        with fake_gpus():
            assert memory_utils.get_total_free_memory_mb() == 0
            assert memory_utils.get_largest_free_memory_mb() == 0
