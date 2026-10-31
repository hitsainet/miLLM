"""Regression tests for 032 controls that survived the consolidated sweep (FTID §8.4, 2026-10-14).

M12 — "load accepted while `writing`". Deleting the `writing` refusal in
`WeightsCheckpointService._check_load_permitted` left every suite green, because the barrier's
`internal_write` refuses `writing` too (`WeightBarrier._begin_internal_locked`), so the existing
test still saw `WEIGHTS_NOT_CLEAN`. What only the service check provides is its PLACE in the
documented refusal order (FTDD §5.2: `WEIGHTS_NOT_CLEAN` "checked first and outside the barrier",
before `ALREADY_ATTACHED`, `MODEL_LEASED` and `MODEL_BUSY`): without it a load during an open
commit is answered with whichever later refusal applies, telling the caller to detach or wait for
a lease when the reason is the commit in progress. These tests pin that order on the two paths
where a later refusal would otherwise win.

M43 — "drop the D-25 dtype refusal". Two checks refuse a full reload's dtype: the servable-set check
(F16/BF16 only, one of them) and the R32 F1 config-dtype check. Dropping the second is caught
(`test_weights_r32_review.py`). Dropping the FIRST changes no accept/refuse outcome — measured over
every subset of the six float dtypes × eight config.json dtype spellings (512 cases), the F1 check
refuses every set the servable check refuses — but the quantized-config check sits between them,
so a checkpoint that is both non-16-bit and quantized was then refused as `quantized` rather than
`dtype`, against the documented order (FTDD §5.2: `dtype` before `quantized`). Pinned below.
"""

from __future__ import annotations

import pytest
import torch

from millm.core.errors import CheckpointNotLoadableError, WeightsNotCleanError
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    RESIDENT_ID,
    TRAINER,
    attach_holder,
    harness,
    resident_llama,
    write_foreign_checkpoint,
)
from tests.unit.services.test_weights_load import _load, _same, _saved, _snapshot
from tests.unit.services.test_weights_r32_review import _edit_config, _untouched
from tests.unit.weights_fixtures import tiny_llama


async def _open_commit(h):
    attachment = attach_holder(h.weights, h.clock)
    await h.barrier.quiesce(attachment.digest, TRAINER)
    h.barrier.commit_begin(attachment.digest)
    return attachment


class TestWritingIsRefusedFirst:
    async def test_full_reload_during_a_commit_is_not_clean_not_attached(self, harness):
        """M12: a full-reload checkpoint while the holder's commit is open is WEIGHTS_NOT_CLEAN,
        before the attachment rule (which would answer ALREADY_ATTACHED) and before any carrier
        row is written."""
        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1,
                                 tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        await _open_commit(h)
        before = _snapshot(loaded.model)
        with pytest.raises(WeightsNotCleanError) as info:
            await _load(h, attachment_id=ATTACHMENT_ID, path="big/latest")
        assert info.value.details["weights_state"] == "writing"
        assert _same(loaded.model, before)
        assert h.names() == []
        assert h.barrier.snapshot().weights_state == "writing"

    async def test_in_place_during_a_commit_outranks_a_foreign_lease(self, harness):
        """M12: the open commit is reported, not the lease a third party holds."""
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        await h.models.acquire_lease(RESIDENT_ID, "midataworks", "eval", 60)
        await _open_commit(h)
        before = _snapshot(loaded.model)
        with pytest.raises(WeightsNotCleanError) as info:
            await _load(h, attachment_id=ATTACHMENT_ID, name=saved.name)
        assert info.value.details["weights_state"] == "writing"
        assert _same(loaded.model, before)


class TestDtypeBeforeQuantized:
    async def test_float32_quantized_checkpoint_is_refused_for_its_dtype(self, harness):
        """M43: float32 weights are not servable as stored; that is the reason given, even when
        config.json also declares a quantization_config."""
        h = harness
        loaded = resident_llama()
        directory = write_foreign_checkpoint(h.root, "big", 1,
                                             tiny_llama(vocab_size=320, dtype=torch.float32))
        _edit_config(directory, quantization_config={"quant_method": "gptq", "bits": 4})
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "dtype"
        assert info.value.details["dtypes"] == ["F32"]
        _untouched(h, loaded)
