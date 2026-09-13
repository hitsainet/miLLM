"""Fake CUDA inventories for CPU-only test runs.

Patches the four torch.cuda entry points placement reads — `is_available`,
`device_count`, `mem_get_info(index)` and `get_device_properties(index)` — so a
test can describe 0, 1, 2 or 3 cards with uneven memory. `mem_get_info` REQUIRES
an index: a call without one is exactly the hard-coded GPU 0 read these tests
exist to rule out, so it raises instead of quietly answering for card 0.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

MB = 1024 * 1024

#: The node as of 2026-09-13: RTX 3080 Ti at index 0, RTX 3090 at index 1.
NODE_UUIDS = [
    "GPU-11111111-2222-3333-4444-555555555555",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "GPU-99999999-8888-7777-6666-555555555555",
]


class FakeCards:
    """Mutable per-card free memory, so a test can model a load consuming it."""

    def __init__(self, cards: list[tuple[str, int, int]]):
        # (name, free_mb, total_mb)
        self.names = [card[0] for card in cards]
        self.free_mb = [card[1] for card in cards]
        self.total_mb = [card[2] for card in cards]
        self.calls: list[tuple[str, object]] = []

    def mem_get_info(self, *args, **kwargs):
        if not args and "device" not in kwargs:
            raise AssertionError("mem_get_info() without a device reads GPU 0 implicitly")
        device = args[0] if args else kwargs["device"]
        index = device if isinstance(device, int) else getattr(device, "index", None)
        if index is None:
            index = int(str(device).split(":")[1])
        self.calls.append(("mem_get_info", index))
        return (self.free_mb[index] * MB, self.total_mb[index] * MB)

    def get_device_properties(self, index):
        return SimpleNamespace(
            name=self.names[index],
            uuid=NODE_UUIDS[index][len("GPU-"):],
            total_memory=self.total_mb[index] * MB,
        )


@contextmanager
def fake_gpus(*cards: tuple[str, int, int]) -> Iterator[FakeCards]:
    """Patch torch.cuda with the given cards: each is (name, free_mb, total_mb)."""
    fake = FakeCards(list(cards))
    with patch.multiple(
        "torch.cuda",
        is_available=lambda: bool(cards),
        device_count=lambda: len(cards),
        mem_get_info=fake.mem_get_info,
        get_device_properties=fake.get_device_properties,
    ):
        yield fake


TI_3080 = "NVIDIA GeForce RTX 3080 Ti"
RTX_3090 = "NVIDIA GeForce RTX 3090"
