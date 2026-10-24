"""`WeightsCheckpointService.save` over the REAL barrier and weight service (032 FTASKS 4.7, 4.8).

Fixtures disagree with the defects they guard: the resident is a tied model (one storage, two
names); the counter is non-zero before a save (so "unchanged" is not "zero"); a second save
races a load that really replaces the weights (so a missing compare-and-set mislabels them).

Mutation controls (032 FTID §8.4 and review additions), each red in this file:
  M2  save skips the clean check          -> TestRefusals.test_*_while_writing/torn (barrier)
  M26 save without the serving pause      -> TestPause.test_commit_begin_during_snapshot_refused
  M27 save accepts any trainer token      -> TestRefusals.test_non_holder_refused
  M28 tie canonicalisation removed        -> TestRoundTrip.test_tied_model_saves_once
  CAS removed (expected_epoch dropped)    -> TestRaces.test_save_racing_a_load_does_not_relabel
  save lock removed                       -> TestRaces.test_concurrent_save_refused
  test_miLLM_loader_loads_it is CUDA-only (skips loudly here); its control is on the GPU node
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open

import millm.services.weights_checkpoint_service as svc_module
from millm.core.errors import (
    CheckpointExistsError,
    CheckpointWriteFailedError,
    InsufficientMemoryError,
    NotAttachedError,
    NotAttachmentHolderError,
    PauseInProgressError,
    WeightsNotCleanError,
)
from millm.services.checkpoint_manifest import validate_full
from millm.services.resident_identity import base_epoch, identity_of, served_name
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    OTHER_TRAINER,
    RESIDENT_REPO,
    TRAINER,
    attach_holder,
    harness,
    logits,
    manifest_fields,
    resident_llama,
    sha256_of,
)


def _service(h) -> WeightsCheckpointService:
    return WeightsCheckpointService(h.models)


async def _save(h, **fields):
    return await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields(**fields))


def _step(h, run="run", step=5):
    return h.root / run / f"step-{step:06d}"


class TestRoundTrip:
    async def test_publishes_a_standard_directory_and_renames_the_resident(self, harness):
        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        result = await _save(h)

        name = f"{RESIDENT_REPO}:run:step-5"
        directory = _step(h)
        assert result.name == name and result.path == "run/step-000005"
        assert result.latest == "step-000005" and os.readlink(h.root / "run" / "latest") == "step-000005"
        files = {p.name for p in directory.iterdir()}
        assert {"model.safetensors", "config.json", "tokenizer.json", "manifest.json"} <= files
        assert not (directory / "tokenizer").exists()
        # BRD-01 acceptance 4 (hash half): the recorded hash is of the bytes on disk.
        assert result.weights_sha256 == sha256_of(directory / "model.safetensors")
        assert result.manifest_sha256 == sha256_of(directory / "manifest.json")
        manifest = validate_full(json.loads((directory / "manifest.json").read_text()))
        assert manifest["name"] == name and manifest["base_model"] == RESIDENT_REPO
        assert manifest["kind"] == "miforge.checkpoint-manifest"
        assert result.bytes == sum(p.stat().st_size for p in directory.iterdir())
        assert result.identity_set is True and result.parent_check == "match"
        identity = identity_of(loaded)
        assert identity.name == name and identity.origin == "checkpoint"
        assert identity.manifest_sha256 == result.manifest_sha256
        assert served_name() == name
        snap = h.barrier.snapshot()
        assert snap.last_save["checkpoint_sha256"] == result.weights_sha256
        assert snap.last_save["path"] == "run/step-000005"
        assert (snap.weights_state, snap.barrier, snap.hold_owner) == ("clean", "serving", None)

    async def test_every_published_file_is_readable_by_other_uids(self, harness):
        """A checkpoint is read by other processes (miForge's continue, D47's read-only
        mount, under another uid). safetensors writes its file 0600 regardless of umask,
        so the weights were the one file nobody else could open (found live 2026-10-18)."""
        import stat

        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        await _save(h)
        unreadable = {
            p.name: oct(stat.S_IMODE(p.stat().st_mode))
            for p in _step(h).iterdir()
            if p.is_file() and stat.S_IMODE(p.stat().st_mode) & 0o044 != 0o044
        }
        assert unreadable == {}, f"not group/other-readable: {unreadable}"

    async def test_plain_transformers_loads_it_with_equal_logits(self, harness):
        """5.8 (b): the directory is a standard Hugging Face directory — tokenizer at the root."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        h = harness
        loaded = resident_llama(tied=False, seed=3)
        attach_holder(h.weights, h.clock)
        await _save(h)
        directory = _step(h)
        AutoTokenizer.from_pretrained(directory)
        fresh = AutoModelForCausalLM.from_pretrained(directory).eval()
        assert torch.equal(logits(fresh), logits(loaded.model))

    async def test_miLLM_loader_loads_it(self, harness):
        """5.8 (a) through `ModelLoader.load`, which needs CUDA (hardware: FTASKS 11.2; RX F8).

        The saved step directory loads through miLLM's own loader, as served (no cast: the
        weights stay bfloat16 on the card), and its log-probabilities match the resident's to
        within D-27's tolerance (1e-2; CPU and CUDA kernels differ, so not bit equality)."""
        if not torch.cuda.is_available():
            pytest.skip("ModelLoader.load needs CUDA; 5.8 (a) runs on the GPU node (FTASKS 11.2)")
        from millm.ml.model_loader import ModelLoader

        h = harness
        loaded = resident_llama(tied=False, seed=3, dtype=torch.bfloat16)
        attach_holder(h.weights, h.clock)
        await _save(h)
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        expected = torch.log_softmax(logits(loaded.model).float(), dim=-1)

        loader = ModelLoader()
        fresh = loader.load(model_id=999, model_name="saved-step", cache_path=str(_step(h)),
                            quantization="FP16", estimated_memory_mb=64)
        try:
            params = list(fresh.model.parameters())
            assert params and all(p.device.type == "cuda" for p in params)
            assert {p.dtype for p in params} == {torch.bfloat16}, "the loader cast the weights"
            with torch.no_grad():
                out = fresh.model(tokens.to(params[0].device)).logits.float().cpu()
            got = torch.log_softmax(out, dim=-1)
            assert got.shape == expected.shape
            assert (got - expected).abs().max().item() <= 1e-2
        finally:
            loader.unload()

    async def test_tied_model_saves_once(self, harness):
        """M28: one name per storage, the name save_pretrained writes."""
        from transformers import AutoModelForCausalLM

        h = harness
        loaded = resident_llama(tied=True)
        attach_holder(h.weights, h.clock)
        await _save(h)
        with safe_open(str(_step(h) / "model.safetensors"), framework="pt") as f:
            keys = set(f.keys())
            assert f.metadata() == {"format": "pt"}
        assert "model.embed_tokens.weight" in keys and "lm_head.weight" not in keys
        fresh = AutoModelForCausalLM.from_pretrained(_step(h)).eval()
        assert fresh.lm_head.weight.data_ptr() == fresh.model.embed_tokens.weight.data_ptr()
        assert torch.equal(logits(fresh), logits(loaded.model))

    async def test_counter_and_state_unchanged_and_the_audit_carries_the_hash(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        h.barrier._steps_since_load = 3
        epoch = base_epoch()
        audits = []
        with patch.object(svc_module, "weights_audit",
                          side_effect=lambda *a, **k: audits.append((a, k))):
            result = await _save(h)
        snap = h.barrier.snapshot()
        assert snap.steps_since_load == 3 and snap.weights_state == "clean"
        assert base_epoch() == epoch  # a save never advances the base epoch (9A.1)
        assert identity_of(svc_module.LoadedModelState().current).commits_at_identity == 3
        assert len(audits) == 1
        (operation, outcome, who), kwargs = audits[0]
        assert (operation, outcome, who) == ("weights_save", "accepted", TRAINER)
        assert kwargs["checkpoint_sha256"] == result.weights_sha256
        assert kwargs["path"] == "run/step-000005"


class TestParentCheck:
    async def test_match_and_mismatch_are_reported_not_refused(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        first = await _save(h, step=5)
        good = await _save(h, step=6, parent=first.name)
        assert good.parent_check == "match"
        bad = await _save(h, step=7, parent="someone:else:step-1")
        assert bad.parent_check == "mismatch" and _step(h, step=7).is_dir()
        assert bad.identity_set is True


class TestRefusals:
    async def test_non_holder_refused(self, harness):
        """M27: another trainer token, or the holder's token without the header."""
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        service = _service(h)
        with pytest.raises(NotAttachmentHolderError):
            await service.save(OTHER_TRAINER, ATTACHMENT_ID, manifest_fields())
        with pytest.raises(NotAttachmentHolderError):
            await service.save(TRAINER, None, manifest_fields())
        assert list(h.root.iterdir()) == []

    async def test_no_attachment_refused(self, harness):
        h = harness
        resident_llama()
        with pytest.raises(NotAttachedError):
            await _save(h)
        assert list(h.root.iterdir()) == []

    async def test_low_disk_refused_before_the_pause(self, harness, monkeypatch):
        """FR-32.1.8: the free-space check runs before the pause; a refusal writes nothing and
        never takes the admission slot."""
        import shutil
        from collections import namedtuple

        from millm.core.errors import InsufficientStorageError

        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10, 10, 0))
        takes: list[int] = []
        h.inference.on_take = lambda: takes.append(1)
        with pytest.raises(InsufficientStorageError) as info:
            await _save(h)
        assert info.value.details["free_bytes"] == 0 and info.value.details["need_bytes"] > 0
        assert takes == [] and list(h.root.iterdir()) == []

    async def test_refused_while_torn(self, harness):
        """BRD-01 acceptance 9 (M2)."""
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        h.barrier.mark_torn("test")
        with pytest.raises(WeightsNotCleanError):
            await _save(h)
        assert not (h.root / "run").exists() or list((h.root / "run").iterdir()) == []

    async def test_refused_while_writing_by_the_holder(self, harness):
        """BRD-01 acceptance 9 (M2): the holder's own open commit."""
        h = harness
        resident_llama()
        attachment = attach_holder(h.weights, h.clock)
        await h.barrier.quiesce(attachment.digest, TRAINER)
        h.barrier.commit_begin(attachment.digest)
        with pytest.raises(WeightsNotCleanError):
            await _save(h)
        assert not _step(h).exists()
        assert h.barrier.snapshot().weights_state == "writing"

    async def test_existing_step_refused_before_the_pause(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        await _save(h)
        episode = h.barrier.snapshot().episode
        with pytest.raises(CheckpointExistsError):
            await _save(h)
        assert h.barrier.snapshot().episode == episode

    async def test_low_host_memory_refused_unmeasured_proceeds(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        with patch.object(svc_module, "get_available_cpu_memory_mb", return_value=1):
            with pytest.raises(InsufficientMemoryError) as info:
                await _save(h)
        assert info.value.details["where"] == "host" and info.value.details["free_mb"] == 1
        assert list(h.root.iterdir()) == []
        with patch.object(svc_module, "get_available_cpu_memory_mb", return_value=0):
            assert (await _save(h)).identity_set is True


class TestWriteFailures:
    async def test_a_failure_inside_the_write_leaves_nothing(self, harness):
        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        with patch.object(type(loaded.tokenizer), "save_pretrained",
                          side_effect=RuntimeError("disk gone")):
            with pytest.raises(CheckpointWriteFailedError) as info:
                await _save(h)
        assert info.value.details["stage"] == "write"
        assert [p.name for p in (h.root / "run").iterdir()] == []
        assert identity_of(loaded) is None and h.barrier.snapshot().last_save is None

    async def test_latest_failure_is_reported_honestly(self, harness):
        """The step directory is published; only `latest` failed (C2): a 500 that says so."""
        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        with patch("millm.services.checkpoint_store.replace_latest",
                   side_effect=OSError("read-only")):
            with pytest.raises(CheckpointWriteFailedError) as info:
                await _save(h)
        details = info.value.details
        assert details["stage"] == "latest" and details["published"] is True
        assert details["path"] == "run/step-000005" and details["identity_set"] is True
        assert (_step(h) / "manifest.json").is_file()
        assert not (h.root / "run" / "latest").exists()
        assert identity_of(loaded).name == details["name"]


class TestPause:
    async def test_commit_begin_during_snapshot_refused(self, harness):
        """M26 (FR-32.2.4): the holder's quiesce is adopted, so its commit-begin is refused
        while the snapshot runs; the caller's quiesce is left held afterwards."""
        h = harness
        resident_llama()
        attachment = attach_holder(h.weights, h.clock)
        await h.barrier.quiesce(attachment.digest, TRAINER)
        entered, release = threading.Event(), threading.Event()
        real = svc_module.snapshot_to_host

        def slow(model, into=None):
            entered.set()
            release.wait(5)
            return real(model, into)

        with patch.object(svc_module, "snapshot_to_host", slow):
            task = asyncio.create_task(_save(h))
            await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(PauseInProgressError):
                h.barrier.commit_begin(attachment.digest)
            release.set()
            await task
        snap = h.barrier.snapshot()
        assert (snap.barrier, snap.hold_owner, snap.weights_state) == ("quiesced", "holder", "clean")

    async def test_an_internal_pause_is_released(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        await _save(h)
        assert h.barrier.holds_slot() is False and not h.inference.slot.locked()


class TestRaces:
    async def _blocked_write(self, h, step):
        """A save stopped inside its disk write (after the pause)."""
        entered, release = threading.Event(), threading.Event()
        real = svc_module.write_checkpoint_atomically

        def slow(*args, **kwargs):
            entered.set()
            release.wait(5)
            return real(*args, **kwargs)

        p = patch.object(svc_module, "write_checkpoint_atomically", slow)
        p.start()
        task = asyncio.create_task(_save(h, step=step))
        await asyncio.to_thread(entered.wait, 5)
        return task, release, p

    async def test_concurrent_save_refused(self, harness):
        """D-20: a second save while one writes is PAUSE_IN_PROGRESS, not a second snapshot."""
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        task, release, p = await self._blocked_write(h, 5)
        try:
            with pytest.raises(PauseInProgressError) as info:
                await _save(h, step=6)
            assert info.value.details["operation"] == "save"
        finally:
            release.set()
            await task
            p.stop()
        assert not _step(h, step=6).exists()

    async def test_save_racing_a_load_does_not_relabel(self, harness):
        """Review 5.1: a load completing during the write moves the epoch; the save publishes
        but leaves the identity and last_save alone."""
        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        first = await _save(h, step=1)
        task, release, p = await self._blocked_write(h, 2)
        try:
            from millm.api.schemas.weights_checkpoints import LoadRequest

            await _service(h).load(TRAINER, LoadRequest(name=first.name),
                                   attachment_id=ATTACHMENT_ID)
        finally:
            release.set()
            second = await task
            p.stop()
        assert second.identity_set is False
        assert _step(h, step=2).is_dir()
        assert identity_of(loaded).name == first.name
        assert h.barrier.snapshot().last_save["path"] == "run/step-000001"


def test_the_save_lock_is_process_wide():
    assert isinstance(svc_module._SAVE_LOCK, type(threading.Lock()))
    assert not svc_module._SAVE_LOCK.locked()
    assert hashlib  # imported for parity with the load tests
