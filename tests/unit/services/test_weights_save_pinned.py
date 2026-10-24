"""The save snapshot copies into page-locked host memory allocated before the pause (finding 4).

Hardware record 2026-10-16: LFM2.5-1.2B's 2.34 GB device-to-host copy took 1.09-1.14 s into fresh
pageable memory (the 1,000 ms save-hold target was missed in 2 of 5 saves) and 348 ms into pinned
memory. Pinning fresh memory is itself ~1.2 s, so `allocate_pinned_snapshot` runs BEFORE
`serving_pause` and only the copy runs inside it; the pinned pages are returned after the write.

MUTATION CONTROLS: allocate inside the pause (the order test goes red); stop passing the buffers to
`snapshot_to_host` (the wiring test goes red); drop `pin_memory=True` (the allocation test goes
red); drop the release after the write (the release test goes red).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import millm.services.weights_checkpoint_service as svc_module
from millm.ml import weights_io
from millm.ml.weights_io import (
    PINNED_SNAPSHOT_MAX_FRACTION,
    allocate_pinned_snapshot,
    canonical_state,
    snapshot_to_host,
)
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    TRAINER,
    attach_holder,
    harness,
    manifest_fields,
    resident_llama,
)
from tests.unit.weights_fixtures import tiny_llama


def _buffers(model):
    return {n: torch.empty(tuple(t.shape), dtype=t.dtype) for n, t in canonical_state(model).items()}


# --- snapshot_to_host with buffers ----------------------------------------------------------


def test_copies_into_the_given_buffers_and_returns_them():
    model = tiny_llama(tied=True)
    into = _buffers(model)
    snap = snapshot_to_host(model, into)
    assert set(snap) == set(into)
    for name, tensor in canonical_state(model).items():
        assert snap[name] is into[name]
        assert torch.equal(snap[name], tensor.detach())


def test_buffers_that_do_not_describe_the_resident_fall_back_to_fresh_copies():
    model = tiny_llama(tied=False)
    into = _buffers(tiny_llama(tied=True))  # a different parameter set
    snap = snapshot_to_host(model, into)
    for name, tensor in canonical_state(model).items():
        assert torch.equal(snap[name], tensor.detach())
        assert all(snap[name] is not b for b in into.values())


def test_without_buffers_the_snapshot_is_a_copy_not_the_live_tensor():
    model = tiny_llama(tied=False)
    snap = snapshot_to_host(model)
    name, live = next(iter(canonical_state(model).items()))
    assert snap[name].data_ptr() != live.data_ptr()


# --- allocate_pinned_snapshot ---------------------------------------------------------------


class _Cudaish:
    """A stand-in for a CUDA parameter: only what the allocator reads."""

    def __init__(self, n: int) -> None:
        self.shape, self.dtype, self.is_cuda, self._n = (n,), torch.bfloat16, True, n

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return 2


@pytest.fixture
def cudaish(monkeypatch):
    state = {"a": _Cudaish(1024 * 1024), "b": _Cudaish(512 * 1024)}  # 3 MB in all
    monkeypatch.setattr(weights_io, "canonical_state", lambda model: state)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def fake_empty(shape, dtype=None, pin_memory=False):
        calls.append((shape, dtype, pin_memory))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(weights_io.torch, "empty", fake_empty)
    return calls


def test_every_buffer_is_page_locked(cudaish):
    buffers, reason = allocate_pinned_snapshot(object(), available_mb=1024)
    assert reason is None and set(buffers) == {"a", "b"}
    assert cudaish and all(pin for _, _, pin in cudaish)


def test_a_snapshot_above_the_host_bound_stays_pageable(cudaish):
    # 3 MB needs available_mb > 3 / PINNED_SNAPSHOT_MAX_FRACTION.
    assert PINNED_SNAPSHOT_MAX_FRACTION == 0.5
    assert allocate_pinned_snapshot(object(), available_mb=5) == (None, "too_large")
    assert allocate_pinned_snapshot(object(), available_mb=0) == (None, "too_large")
    assert cudaish == []


def test_a_failed_pin_stays_pageable(cudaish, monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError("cudaHostAlloc failed")

    monkeypatch.setattr(weights_io.torch, "empty", refuse)
    assert allocate_pinned_snapshot(object(), available_mb=1024) == (None, "pin_failed")


def test_a_cpu_model_is_never_pinned():
    assert allocate_pinned_snapshot(tiny_llama(tied=False), available_mb=10**6) == (None, "no_cuda")


# --- the service: allocate before the pause, copy inside it, release after ------------------


def _service(h) -> WeightsCheckpointService:
    return WeightsCheckpointService(h.models)


async def test_buffers_are_allocated_before_the_pause_and_used_inside_it(harness):
    h = harness
    loaded = resident_llama()
    attach_holder(h.weights, h.clock)
    seen = SimpleNamespace(alloc_slot=None, into=None, releases=0)
    buffers = _buffers(loaded.model)

    def fake_alloc(model, *, available_mb):
        seen.alloc_slot = h.barrier.holds_slot()
        return buffers, None

    real = svc_module.snapshot_to_host

    def spy(model, into=None):
        seen.into = into
        seen.slot_during_copy = h.barrier.holds_slot()
        return real(model, into)

    def fake_release():
        seen.releases += 1

    with patch.object(svc_module, "allocate_pinned_snapshot", fake_alloc), \
            patch.object(svc_module, "snapshot_to_host", spy), \
            patch.object(svc_module, "release_pinned_host_memory", fake_release):
        result = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
    assert seen.alloc_slot is False, "allocated inside the pause"
    assert seen.slot_during_copy is True
    assert seen.into is buffers
    assert seen.releases == 1
    assert result.weights_sha256


async def test_a_pageable_fallback_is_logged_and_releases_nothing(harness, caplog):
    h = harness
    resident_llama()
    attach_holder(h.weights, h.clock)
    releases = []
    with patch.object(svc_module, "allocate_pinned_snapshot",
                      lambda model, *, available_mb: (None, "too_large")), \
            patch.object(svc_module, "release_pinned_host_memory", lambda: releases.append(1)), \
            patch.object(svc_module.logger, "info") as info:
        await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
    assert releases == []
    events = [c.args[0] for c in info.call_args_list if c.args]
    assert "weights_save_snapshot_pageable" in events
    kw = next(c.kwargs for c in info.call_args_list
              if c.args and c.args[0] == "weights_save_snapshot_pageable")
    assert kw["reason"] == "too_large"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NEEDS CUDA: pinned host memory")
def test_on_a_card_the_buffers_are_pinned_and_the_copy_is_exact():  # pragma: no cover
    model = tiny_llama(tied=True).to("cuda")
    buffers, reason = allocate_pinned_snapshot(model, available_mb=10**5)
    assert reason is None and all(b.is_pinned() for b in buffers.values())
    snap = snapshot_to_host(model, buffers)
    for name, tensor in canonical_state(model).items():
        assert torch.equal(snap[name], tensor.detach().cpu())
