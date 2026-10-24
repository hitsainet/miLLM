"""A model of PyTorch's CUDA caching allocator, for replaying a traced request's allocations on a CPU-only machine.

The default policy of c10/cuda/CUDACachingAllocator.cpp (torch 2.10): no max_split_size, no
expandable segments, no garbage-collection threshold. Sizes round up to 512 bytes; a request
of at most 1 MiB is served from 2 MiB segments, one under 10 MiB from 20 MiB segments, and a
larger one from a segment rounded up to 2 MiB. The smallest free block that fits is used
(ties to the lower address) and split when what is left is at least 512 bytes (small pool)
or more than 1 MiB (large pool); a freed block merges with free neighbours in its segment.
When a new segment would pass the card's capacity, every wholly free segment is released and
the request is retried before it fails.

Checked against the node (hardware acceptance, 2026-09-14): replaying OLMo-2-13B's cuda:0
reproduces its out-of-memory error to within 7 MiB allocated and 25 MiB of reserved-but-
unallocated memory, and the next three requests' peaks to within 1%
(millm/ml/working_memory.py).
"""

from __future__ import annotations

import bisect
from typing import Callable, Iterable, Optional

MIB = 1024 * 1024
MIN_BLOCK = 512
SMALL_SIZE = MIB
SMALL_BUFFER = 2 * MIB
MIN_LARGE_ALLOC = 10 * MIB
ROUND_LARGE = 2 * MIB
LARGE_BUFFER = 20 * MIB


class OutOfMemory(Exception):
    def __init__(self, size: int, allocated: int, reserved: int):
        super().__init__(f"tried to allocate {size} bytes")
        self.size, self.allocated, self.reserved = size, allocated, reserved


class _Block:
    __slots__ = ("size", "addr", "small", "used", "prev", "next")

    def __init__(self, size: int, addr: int, small: bool):
        self.size, self.addr, self.small = size, addr, small
        self.used = False
        self.prev: Optional[_Block] = None
        self.next: Optional[_Block] = None


class CachingAllocator:
    """Allocated and reserved bytes, as the allocator counts them."""

    def __init__(self, capacity: int, reserved: int = 0):
        self.capacity = capacity
        self.reserved = reserved
        self.allocated = reserved  # memory already held (weights) counts as allocated
        self.peak_reserved = reserved
        self._pools: dict[bool, list] = {True: [], False: []}
        self._blocks: dict[int, _Block] = {}
        self._next_addr = 0

    @staticmethod
    def _round(size: int) -> int:
        return MIN_BLOCK if size < MIN_BLOCK else MIN_BLOCK * -(-size // MIN_BLOCK)

    @staticmethod
    def _segment(size: int) -> int:
        if size <= SMALL_SIZE:
            return SMALL_BUFFER
        if size < MIN_LARGE_ALLOC:
            return LARGE_BUFFER
        return ROUND_LARGE * -(-size // ROUND_LARGE)

    def _insert(self, block: _Block) -> None:
        bisect.insort(self._pools[block.small], (block.size, block.addr, id(block)))
        self._blocks[id(block)] = block

    def _remove(self, block: _Block) -> None:
        pool = self._pools[block.small]
        pool.pop(bisect.bisect_left(pool, (block.size, block.addr, id(block))))

    def _release_free_segments(self) -> None:
        for pool in self._pools.values():
            for _, _, key in list(pool):
                block = self._blocks[key]
                if block.prev is None and block.next is None:
                    self._remove(block)
                    self.reserved -= block.size

    def malloc(self, nbytes: int) -> _Block:
        size = self._round(nbytes)
        small = size <= SMALL_SIZE
        pool = self._pools[small]
        index = bisect.bisect_left(pool, (size, -1, -1))
        if index < len(pool):
            block = self._blocks[pool[index][2]]
            self._remove(block)
        else:
            segment = self._segment(size)
            if self.reserved + segment > self.capacity:
                self._release_free_segments()
                if self.reserved + segment > self.capacity:
                    raise OutOfMemory(size, self.allocated, self.reserved)
            block = _Block(segment, self._next_addr, small)
            self._next_addr += segment + SMALL_BUFFER
            self.reserved += segment
        rest = block.size - size
        if (small and rest >= MIN_BLOCK) or (not small and rest > SMALL_SIZE):
            tail = _Block(rest, block.addr + size, small)
            tail.prev, tail.next = block, block.next
            if block.next is not None:
                block.next.prev = tail
            block.next = tail
            block.size = size
            self._insert(tail)
        block.used = True
        self.allocated += block.size
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        return block

    def free(self, block: _Block) -> None:
        self.allocated -= block.size
        block.used = False
        for neighbour in (block.prev, block.next):
            if neighbour is None or neighbour.used:
                continue
            self._remove(neighbour)
            if neighbour is block.prev:
                block.addr, block.size, block.prev = neighbour.addr, block.size + neighbour.size, neighbour.prev
                if neighbour.prev is not None:
                    neighbour.prev.next = block
            else:
                block.size, block.next = block.size + neighbour.size, neighbour.next
                if neighbour.next is not None:
                    neighbour.next.prev = block
        self._insert(block)


def replay(
    events: Iterable[tuple[bool, int, int, object]],
    allocator: CachingAllocator,
    on_card: Callable[[object], bool],
) -> Optional[OutOfMemory]:
    """Replay (allocated?, id, bytes, phase) events whose allocation happened in a phase on this card."""
    live: dict[int, _Block] = {}
    for allocated, sid, nbytes, phase in events:
        if allocated:
            if not on_card(phase):
                continue
            try:
                live[sid] = allocator.malloc(nbytes)
            except OutOfMemory as oom:
                return oom
        elif sid in live:
            allocator.free(live.pop(sid))
    return None
