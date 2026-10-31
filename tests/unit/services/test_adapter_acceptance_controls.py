"""Pins found by the consolidated 033 mutation-control sweep (FTID §8.4, 2026-10-14).

M37 — "turn the OOM -> `INSUFFICIENT_MEMORY` catch into a no-op" — SURVIVED the whole adapter
suite (233 focused, 1742 broad) once R2-C added a second, generic out-of-memory catch in
`adapter_service.prepare_adapter_mapped`. The 507 contract itself still held through that outer
catch, so the HTTP test stayed green; what the mutation removed is the refusal saying WHERE memory
ran out. `prepare_adapter` has two stage-specific catches — reading the tensors out of the
verified bytes, and placing A and B on the module's device — and an operator sizing a card needs
to know which one it was. Without these tests either catch can be deleted with every test green.

Negative controls (recorded in the sweep's results): M37 (placement catch a no-op) and M37d
(read catch a no-op) each turn exactly one case here red.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from millm.core.errors import InsufficientMemoryError
from millm.ml.adapter_hooks import LoraHookSet
from millm.services.adapter_service import get_adapter_state
from tests.unit.adapter_service_fixtures import build_env, identity


@pytest.fixture
def env(tmp_path):
    e = build_env(tmp_path)
    yield e
    e.close()


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


@pytest.mark.parametrize("stage, target, words", [
    ("place", (LoraHookSet, "build"), "place the adapter's tensors"),
    ("read", ("safetensors.torch", "load"), "read the adapter's tensors"),
], ids=["placement", "read"])
async def test_out_of_memory_names_the_stage_that_ran_out(env, stage, target, words):
    rel = env.write()
    owner, attr = target
    if isinstance(owner, str):
        import importlib

        owner = importlib.import_module(owner)
    with patch.object(owner, attr, side_effect=torch.OutOfMemoryError("CUDA OOM")), \
            pytest.raises(InsufficientMemoryError) as caught:
        await env.service.load(identity(), rel)
    err = caught.value
    assert err.code == "INSUFFICIENT_MEMORY"
    assert words in err.message, err.message
    # The stage-specific catch answered, not the generic fallback in `prepare_adapter_mapped`.
    details = err.details if isinstance(err.details, dict) else {}
    assert details.get("reason") != "prepare_out_of_memory", details
    assert details.get("error") == "OutOfMemoryError"
    assert _hooks(env.model) == 0
    assert get_adapter_state().current is None and not get_adapter_state().loading
