"""Fake GPU inventories for CPU-only test runs.

Fakes both sources placement reads:
  * nvidia-smi (`millm.ml.nvidia_smi._run_query`) — the inventory: name, UUID,
    free and total memory, with no CUDA context;
  * torch.cuda — `is_available`, `device_count`, `get_device_properties` (for
    the UUID -> torch index mapping) and `mem_get_info(index)`.

`mem_get_info` REQUIRES an index: a call without one is the implicit GPU 0 read
these tests exist to rule out. And a test can FORBID it per card
(`fake.forbid(0)`): each real call creates a CUDA context on its card, so asking
a card the model is not on costs that card memory.

`smi_order` lets nvidia-smi list cards in a different order from torch, so a
mapping by position instead of UUID gives a wrong answer.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator, Optional
from unittest.mock import patch

MB = 1024 * 1024

#: The node as of 2026-09-13: RTX 3080 Ti at index 0, RTX 3090 at index 1.
NODE_UUIDS = [
    "GPU-11111111-2222-3333-4444-555555555555",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "GPU-99999999-8888-7777-6666-555555555555",
    "GPU-12345678-1234-1234-1234-123456789abc",
]


class FakeCards:
    """Mutable per-card free memory, so a test can model a load consuming it."""

    def __init__(
        self,
        cards: list[tuple[str, int, int]],
        smi_order: Optional[list[int]] = None,
        smi_available: bool = True,
    ):
        # (name, free_mb, total_mb), indexed by TORCH index.
        self.names = [card[0] for card in cards]
        self.free_mb = [card[1] for card in cards]
        self.total_mb = [card[2] for card in cards]
        self.uuids = NODE_UUIDS[: len(cards)]
        self.smi_order = smi_order if smi_order is not None else list(range(len(cards)))
        self.smi_available = smi_available
        self.forbidden: set[int] = set()
        self.calls: list[tuple[str, object]] = []
        self.smi_calls = 0

    def forbid(self, *indices: int) -> None:
        """Make mem_get_info raise for these cards."""
        self.forbidden.update(indices)

    def smi_stdout(self) -> str:
        lines = []
        for smi_index, torch_index in enumerate(self.smi_order):
            free, total = self.free_mb[torch_index], self.total_mb[torch_index]
            lines.append(
                f"{smi_index}, {self.uuids[torch_index]}, 5, {total - free}, {total}, "
                f"{free}, 45, {self.names[torch_index]}"
            )
        return "\n".join(lines) + "\n"

    def run_query(self):
        self.smi_calls += 1
        if not self.smi_available or not self.names:
            return None
        return self.smi_stdout()

    def mem_get_info(self, *args, **kwargs):
        if not args and "device" not in kwargs:
            raise AssertionError("mem_get_info() without a device reads GPU 0 implicitly")
        device = args[0] if args else kwargs["device"]
        index = device if isinstance(device, int) else getattr(device, "index", None)
        if index is None:
            index = int(str(device).split(":")[1])
        if index in self.forbidden:
            raise AssertionError(
                f"mem_get_info({index}) creates a CUDA context on card {index}, "
                "which this load does not use"
            )
        self.calls.append(("mem_get_info", index))
        return (self.free_mb[index] * MB, self.total_mb[index] * MB)

    def get_device_properties(self, index):
        return SimpleNamespace(
            name=self.names[index],
            # torch reports the UUID without nvidia-smi's "GPU-" prefix.
            uuid=self.uuids[index][len("GPU-"):],
            total_memory=self.total_mb[index] * MB,
        )


@contextmanager
def fake_gpus(
    *cards: tuple[str, int, int],
    smi_order: Optional[list[int]] = None,
    smi_available: bool = True,
) -> Iterator[FakeCards]:
    """Patch nvidia-smi and torch.cuda with these cards: each is (name, free_mb, total_mb)."""
    fake = FakeCards(list(cards), smi_order=smi_order, smi_available=smi_available)
    with patch.multiple(
        "torch.cuda",
        is_available=lambda: bool(cards),
        device_count=lambda: len(cards),
        mem_get_info=fake.mem_get_info,
        get_device_properties=fake.get_device_properties,
    ), patch("millm.ml.nvidia_smi._run_query", fake.run_query):
        yield fake


TI_3080 = "NVIDIA GeForce RTX 3080 Ti"
RTX_3090 = "NVIDIA GeForce RTX 3090"
