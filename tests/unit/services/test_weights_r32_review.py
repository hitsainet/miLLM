"""032 review round 1 (R32): defects found in save, load and lineage, each pinned here.

Every test is the regression for one finding in the R32 report and was run red against the line
it guards (the negative controls are listed in the report):

  F1  full reload over a config that would make the loader CAST the verified bytes (a recorded
      dtype other than the weights', or none at all over float16 weights; a quantized config)
  F2  full reload over a directory the loader cannot open (no config.json, no tokenizer) left the
      weights `torn` instead of being refused before the first write
  F3  full reload over a directory holding a FIFO or a link out of its root (the loader opens
      every file there; a FIFO hangs it while it holds the barrier)
  F4  a save under a run_id that already exists in a read root made the run ambiguous, so the
      checkpoint just written could never be loaded or walked
  F5  a weights file removed between the hash and the barrier answered 500 (FileNotFoundError)
  F6  the weights file replaced WHILE it was being copied or reloaded was served as verified
  F7  a load or save that failed with a non-miLLM error was never audited
  F8  cancelling a load released the barrier while the copy thread was still writing; cancelling
      a save released the save lock while its write thread was still publishing
  F9  `auto_map` module names copied files from outside the source directory into a checkpoint
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import millm.services.weights_checkpoint_service as svc_module
from millm.api.schemas.weights_checkpoints import LoadRequest
from millm.core.errors import (
    CheckpointAmbiguousError,
    CheckpointHashMismatchError,
    CheckpointNotLoadableError,
    WeightsLoadFailedError,
)
from millm.core.config import settings
from millm.ml.model_loader import LoadedModelState
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    TRAINER,
    attach_holder,
    harness,
    manifest_fields,
    resident_llama,
    sha256_of,
    write_foreign_checkpoint,
)
from tests.unit.weights_fixtures import tiny_llama


def _service(h) -> WeightsCheckpointService:
    return WeightsCheckpointService(h.models)


async def _load(h, identity=TRAINER, **request):
    return await _service(h).load(identity, LoadRequest(**request))


def _foreign(h, dtype=torch.bfloat16):
    return write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=dtype))


def _edit_config(directory, **changes):
    path = directory / "config.json"
    config = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    path.write_text(json.dumps(config))


def _untouched(h, loaded):
    assert LoadedModelState().current is loaded and h.names() == []
    assert h.barrier.snapshot().weights_state == "clean"


# ----------------------------------------------------------------------------- F1


class TestFullReloadNeverCasts:
    async def test_config_dtype_disagreeing_with_the_weights_refused(self, harness):
        h = harness
        loaded = resident_llama()
        directory = _foreign(h)  # bfloat16 weights
        _edit_config(directory, dtype="float16", torch_dtype="float16")
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "dtype"
        assert info.value.details["load_dtype"] == "float16"
        assert info.value.details["dtypes"] == ["BF16"]
        _untouched(h, loaded)

    async def test_float16_weights_with_no_recorded_dtype_refused(self, harness):
        """The loader's default for an unrecorded dtype is bfloat16: float16 bytes would be cast."""
        h = harness
        loaded = resident_llama()
        directory = _foreign(h, dtype=torch.float16)
        _edit_config(directory, dtype=None, torch_dtype=None)
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "dtype"
        _untouched(h, loaded)

    async def test_matching_dtype_still_loads(self, harness):
        h = harness
        resident_llama()
        directory = _foreign(h, dtype=torch.float16)
        _edit_config(directory, dtype="float16")
        assert (await _load(h, path="big/latest")).mode == "full_reload"

    async def test_quantized_config_refused(self, harness):
        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        _edit_config(directory, quantization_config={"quant_method": "gptq", "bits": 4})
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "quantized"
        _untouched(h, loaded)


# ----------------------------------------------------------------------------- F2


class TestFullReloadNeedsALoadableDirectory:
    async def test_missing_config_refused_before_the_first_write(self, harness):
        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        (directory / "config.json").unlink()
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "incomplete"
        assert "config.json" in info.value.details["missing"]
        _untouched(h, loaded)

    async def test_unparseable_config_refused(self, harness):
        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        (directory / "config.json").write_text("{not json")
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "incomplete"
        _untouched(h, loaded)

    async def test_missing_tokenizer_refused(self, harness):
        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320,
                                                              dtype=torch.bfloat16),
                                 tokenizer=None)
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "incomplete"
        assert "tokenizer" in info.value.details["missing"]
        _untouched(h, loaded)


# ----------------------------------------------------------------------------- F3


class TestFullReloadRefusesUnsafeFiles:
    async def test_fifo_in_the_directory_refused(self, harness):
        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        os.mkfifo(directory / "special_tokens_map.json")
        with pytest.raises(CheckpointNotLoadableError) as info:
            await asyncio.wait_for(_load(h, path="big/latest"), timeout=30)
        assert info.value.details["reason"] == "unsafe_file"
        _untouched(h, loaded)

    async def test_link_out_of_the_root_refused(self, harness, tmp_path):
        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        outside = tmp_path / "outside-config.json"
        outside.write_text((directory / "config.json").read_text())
        (directory / "config.json").unlink()
        (directory / "config.json").symlink_to(outside)
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "unsafe_file"
        _untouched(h, loaded)


# ----------------------------------------------------------------------------- F4


class TestSaveNeverCreatesAmbiguity:
    async def test_run_id_present_in_a_read_root_refused(self, harness, tmp_path, monkeypatch):
        h = harness
        resident_llama()
        forge = tmp_path / "forge"
        write_foreign_checkpoint(forge, "run", 4, tiny_llama())
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", str(forge))
        attach_holder(h.weights, h.clock)
        with pytest.raises(CheckpointAmbiguousError) as info:
            await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields(step=5))
        assert info.value.details["run_id"] == "run"
        assert not (h.root / "run").exists()


# ----------------------------------------------------------------------------- F5


class TestVanishedWeights:
    async def test_removed_after_the_hash_is_a_mismatch_not_a_500(self, harness):
        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        h.weights._attachment = None
        weights = h.root / "run" / "step-000005" / "model.safetensors"
        h.inference.on_take = lambda: weights.unlink()
        with pytest.raises(CheckpointHashMismatchError) as info:
            await _load(h, name=saved.name)
        assert info.value.details == {"changed_since_hash": True}
        assert LoadedModelState().current is loaded
        assert h.barrier.snapshot().weights_state == "clean"


# ----------------------------------------------------------------------------- F6


def _bump_mtime(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


class TestChangedDuringTheWrite:
    async def test_in_place_copy_over_a_changing_file_is_torn(self, harness):
        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields(step=5))
        later = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields(step=6))
        h.weights._attachment = None
        weights = h.root / "run" / "step-000005" / "model.safetensors"
        real = svc_module.copy_in_place

        def copy_then_change(model, handle):
            count = real(model, handle)
            _bump_mtime(weights)
            return count

        with patch.object(svc_module, "copy_in_place", copy_then_change):
            with pytest.raises(WeightsLoadFailedError) as info:
                await _load(h, name=saved.name)
        assert info.value.details["stage"] == "verify"
        assert h.barrier.snapshot().weights_state == "torn"
        # The bytes copied are not known to be the verified ones: never named after them.
        assert svc_module.identity_of(loaded).name == later.name

    async def test_full_reload_over_a_changing_file_is_torn(self, harness):
        h = harness
        resident_llama()
        directory = _foreign(h)
        h.loader.after_load = lambda: _bump_mtime(directory / "model.safetensors")
        with pytest.raises(WeightsLoadFailedError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["stage"] == "verify"
        assert h.barrier.snapshot().weights_state == "torn"
        assert svc_module.identity_of(LoadedModelState().current) is None


# ----------------------------------------------------------------------------- F7


class TestUnexpectedErrorsAreAudited:
    async def test_load(self, harness):
        h = harness
        resident_llama()
        audits = []
        with patch.object(svc_module, "weights_audit",
                          side_effect=lambda *a, **k: audits.append((a, k))), \
                patch.object(WeightsCheckpointService, "_verify",
                             side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                await _load(h, path="big/latest")
        assert [(a[:2], k["code"]) for a, k in audits] == [(("weights_load", "refused"), "INTERNAL_ERROR")]

    async def test_save(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        audits = []
        with patch.object(svc_module, "weights_audit",
                          side_effect=lambda *a, **k: audits.append((a, k))), \
                patch.object(WeightsCheckpointService, "_save",
                             side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        assert [(a[:2], k["code"]) for a, k in audits] == [(("weights_save", "refused"), "INTERNAL_ERROR")]


# ----------------------------------------------------------------------------- F8


class TestCancellationWaitsForTheThread:
    async def test_cancelled_in_place_load_holds_the_barrier_until_the_copy_ends(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        h.weights._attachment = None
        started, release = threading.Event(), threading.Event()

        def slow_copy(model, handle):
            started.set()
            release.wait(10)
            return 0

        with patch.object(svc_module, "copy_in_place", slow_copy):
            task = asyncio.create_task(_load(h, name=saved.name))
            await asyncio.to_thread(started.wait, 10)
            task.cancel()
            await asyncio.sleep(0.2)
            # The copy thread is still writing: the slot must still be the barrier's.
            assert h.inference.slot.locked()
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not h.inference.slot.locked()
        assert h.barrier.snapshot().weights_state == "torn"

    async def test_cancelled_save_holds_the_lock_until_the_write_ends(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        started, release = threading.Event(), threading.Event()

        def slow_write(*args, **kwargs):
            started.set()
            release.wait(10)
            raise OSError("never published")

        with patch.object(svc_module, "write_checkpoint_atomically", slow_write):
            task = asyncio.create_task(
                _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields()))
            await asyncio.to_thread(started.wait, 10)
            task.cancel()
            await asyncio.sleep(0.2)
            assert svc_module._SAVE_LOCK.locked()
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not svc_module._SAVE_LOCK.locked()


# ----------------------------------------------------------------------------- F9


class TestAutoMapConfinement:
    def test_module_outside_the_source_directory_is_not_copied(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "modeling_x.py").write_text("# inside")
        (tmp_path / "secret.py").write_text("# outside")
        config = SimpleNamespace(auto_map={
            "AutoConfig": "modeling_x.XConfig",
            "AutoModel": "repo--..secret.Model",
            "AutoModelForCausalLM": "x--/" + str(tmp_path / "secret").lstrip("/") + ".M",
        })
        files = WeightsCheckpointService._auto_map_files(config, source)
        assert [p.name for p in files] == ["modeling_x.py"]
        assert all(os.path.realpath(p).startswith(os.path.realpath(source)) for p in files)


# ----------------------------------------------------------------------------- F10


class TestLeaseRecheckedInsideTheBarrier:
    """F10: the in-place path checked the lease only before waiting for the slot, which can take
    as long as the request in flight. A lease granted in that wait was ignored; the full-reload
    path re-checks it in `precheck_load`."""

    async def test_lease_granted_while_waiting_for_the_slot_refuses_in_place(self, harness):
        from millm.core.errors import ModelLeasedError
        from tests.unit.checkpoint_service_fixtures import RESIDENT_ID, RESIDENT_NAME

        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        h.weights._attachment = None
        with torch.no_grad():
            for p in loaded.model.parameters():
                p.add_(1.0)
        before = {n: p.detach().clone() for n, p in loaded.model.named_parameters()}
        h.inference.on_take = lambda: h.models._leases.grant(
            RESIDENT_ID, RESIDENT_NAME, "midataworks", "eval", 60)
        with pytest.raises(ModelLeasedError):
            await _load(h, name=saved.name)
        assert all(torch.equal(p, before[n]) for n, p in loaded.model.named_parameters())
        assert h.barrier.snapshot().weights_state == "clean"

    async def test_load_claimed_while_waiting_for_the_slot_refuses_in_place(self, harness):
        from millm.core.errors import ModelBusyError
        from millm.services import model_service as module

        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        h.weights._attachment = None
        with torch.no_grad():
            for p in loaded.model.parameters():
                p.add_(1.0)
        before = {n: p.detach().clone() for n, p in loaded.model.named_parameters()}
        h.inference.on_take = lambda: module._LOAD_SLOT.__setitem__("model_id", 99)
        try:
            with pytest.raises(ModelBusyError):
                await _load(h, name=saved.name)
        finally:
            module._LOAD_SLOT["model_id"] = None
        assert all(torch.equal(p, before[n]) for n, p in loaded.model.named_parameters())
        assert h.barrier.snapshot().weights_state == "clean"


class TestAutoMapEachGuard:
    """The two F9 guards, each alone: a nested module stays inside the directory (only the
    bare-stem rule refuses it), and a link out of the directory has a bare stem (only the
    confinement refuses it)."""

    def test_nested_module_refused(self, tmp_path):
        source = tmp_path / "source"
        (source / "sub").mkdir(parents=True)
        (source / "sub" / "mod.py").write_text("# nested")
        config = SimpleNamespace(auto_map={"AutoModel": "sub/mod.Model",
                                           "AutoConfig": "sub.mod.Config"})
        assert WeightsCheckpointService._auto_map_files(config, source) == []

    def test_link_out_of_the_directory_refused(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (tmp_path / "secret.py").write_text("# outside")
        (source / "evil.py").symlink_to(tmp_path / "secret.py")
        config = SimpleNamespace(auto_map={"AutoModel": "evil.Model"})
        assert WeightsCheckpointService._auto_map_files(config, source) == []


class TestTheHandlePinsTheHashedInode:
    async def test_rename_replace_during_the_copy_copies_the_hashed_bytes(self, harness):
        """The `safe_open` mapping is taken BEFORE the hash, so a rename-replace of the file's
        name mid-load cannot change what is copied, and the post-copy check (on the pinned
        inode) does not call that torn."""
        from safetensors.torch import load_file, save_file

        h = harness
        loaded = resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        h.weights._attachment = None
        weights = h.root / "run" / "step-000005" / "model.safetensors"
        hashed = load_file(str(weights))
        with torch.no_grad():
            for p in loaded.model.parameters():
                p.add_(1.0)
        real = svc_module.copy_in_place

        def replace_then_copy(model, handle):
            other = {k: v + 7.0 for k, v in hashed.items()}
            tmp = weights.with_name("model.safetensors.new")
            save_file(other, str(tmp))
            os.replace(tmp, weights)
            return real(model, handle)

        with patch.object(svc_module, "copy_in_place", replace_then_copy):
            result = await _load(h, name=saved.name)
        assert result.mode == "in_place" and h.barrier.snapshot().weights_state == "clean"
        state = loaded.model.state_dict()
        assert all(torch.equal(state[k], v) for k, v in hashed.items())

    async def test_descriptor_closed_after_the_load(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
        h.weights._attachment = None
        before = len(os.listdir("/proc/self/fd"))
        for _ in range(3):
            await _load(h, name=saved.name)
        manifest = h.root / "run" / "step-000005" / "manifest.json"
        doc = json.loads(manifest.read_text())
        doc["weights_sha256"] = "0" * 64
        manifest.write_text(json.dumps(doc))
        for _ in range(3):  # refused AFTER the descriptor is opened
            with pytest.raises(CheckpointHashMismatchError):
                await _load(h, name=saved.name)
        assert len(os.listdir("/proc/self/fd")) <= before


# ----------------------------------------------------------------------------- survivors pinned


def _saved_dir(h):
    return h.root / "run" / "step-000005"


async def _save_one(h, **fields):
    attach_holder(h.weights, h.clock)
    saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields(**fields))
    h.weights._attachment = None
    return saved


class TestVerificationEdges:
    async def test_manifest_copied_to_another_step_directory_refused(self, harness):
        """B3: a manifest must name its own directory (run_id and step)."""
        import shutil

        from millm.core.errors import InvalidManifestError

        h = harness
        resident_llama()
        await _save_one(h)
        shutil.copytree(_saved_dir(h), h.root / "run" / "step-000006")
        with pytest.raises(InvalidManifestError) as info:
            await _load(h, path="run/step-000006")
        assert info.value.details["errors"][0]["json_path"] == "$.step"

    async def test_a_name_with_another_base_is_not_this_checkpoint(self, harness):
        """B4: the name resolves to a directory by run and step; the manifest's name decides."""
        from millm.core.errors import CheckpointNotFoundError

        h = harness
        resident_llama()
        saved = await _save_one(h)
        other = "someone/else:run:step-5"
        assert other != saved.name
        with pytest.raises(CheckpointNotFoundError):
            await _load(h, name=other)

    async def test_hash_correct_but_unreadable_safetensors_refused_untouched(self, harness):
        """B5: header offsets inside the file, but too few bytes for the shape: `read_header`
        accepts it, `safe_open` does not, and the load must stop there, not after a write."""
        import struct

        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        header = json.dumps({"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 4]}})
        blob = struct.pack("<Q", len(header)) + header.encode() + b"\0" * 4
        (directory / "model.safetensors").write_bytes(blob)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["weights_sha256"] = sha256_of(directory / "model.safetensors")
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "safetensors_header"
        _untouched(h, loaded)

    async def test_run_id_with_a_colon_reached_by_absolute_path_refused(self, harness):
        """B1: the load rule on the manifest's run_id. A run directory `a:b` reached by an
        absolute path would otherwise yield a name that parses as run `b`."""
        from millm.core.errors import InvalidRunIdError

        h = harness
        resident_llama()
        directory = write_foreign_checkpoint(h.root, "a:b", 1, tiny_llama(), latest=False)
        with pytest.raises(InvalidRunIdError):
            await _load(h, path=str(directory))


class TestNamesAndPathsFitTheRow:
    async def test_full_reload_name_too_long_for_the_row_refused_before_the_insert(self, harness):
        """B8: SQLite does not enforce VARCHAR lengths, so only this check stands between a
        long foreign name and a Postgres 500."""
        from millm.core.errors import InvalidRunIdError

        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320,
                                                              dtype=torch.bfloat16),
                                 base_model="b" * 250)
        with pytest.raises(InvalidRunIdError):
            await _load(h, path="big/latest")
        _untouched(h, loaded)

    async def test_full_reload_path_too_long_for_the_row_refused_before_the_insert(
            self, harness, monkeypatch):
        """F11: `local_path` and `cache_path` are VARCHAR(500)."""
        from millm.db.models.model import Model

        h = harness
        loaded = resident_llama()
        directory = _foreign(h)
        monkeypatch.setattr(Model.__table__.c.local_path.type, "length",
                            len(os.path.realpath(directory)) - 1)
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "path_length"
        _untouched(h, loaded)


class TestAttachmentRuleEdges:
    async def test_torn_lets_an_operator_through_never_another_trainer(self, harness):
        """B17b: the torn/stale exception is the operator's alone."""
        from millm.core.errors import AlreadyAttachedError
        from tests.unit.checkpoint_service_fixtures import OTHER_TRAINER

        h = harness
        resident_llama()
        saved = await _save_one(h)
        attach_holder(h.weights, h.clock)
        h.barrier.mark_torn("test")
        with pytest.raises(AlreadyAttachedError):
            await _load(h, OTHER_TRAINER, name=saved.name)
        h.clock.now += 10_000  # stale too
        with pytest.raises(AlreadyAttachedError):
            await _load(h, OTHER_TRAINER, name=saved.name)

    async def test_resident_changed_dtype_while_waiting_is_refused_not_torn(self, harness):
        """B27: the tensor comparison is made again under the barrier."""
        from millm.core.errors import ModelBusyError

        h = harness
        loaded = resident_llama()
        saved = await _save_one(h)
        h.inference.on_take = lambda: loaded.model.to(torch.bfloat16)
        with pytest.raises(ModelBusyError):
            await _load(h, name=saved.name)
        assert h.barrier.snapshot().weights_state == "clean"


class TestLineageEdges:
    async def test_a_manifest_whose_run_and_step_disagree_with_its_name_is_invalid(self, harness):
        """B15: a parent name matched, but the manifest describes another step."""
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        s1 = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields("run-a", 1))
        await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields("run-a", 2,
                                                                     parent=s1.name))
        path = h.root / "run-a" / "step-000001" / "manifest.json"
        doc = json.loads(path.read_text())
        doc["step"] = 9
        path.write_text(json.dumps(doc))
        body = await _service(h).lineage()
        assert body["break"] == {"at": s1.name, "reason": "manifest_invalid"}

    async def test_commits_since_identity_is_never_negative(self, harness):
        """B21: a counter reset under an unchanged identity reports 0, not a negative count."""
        h = harness
        loaded = resident_llama()
        await _save_one(h)
        identity = svc_module.identity_of(loaded)
        from dataclasses import replace

        loaded.weights_identity = replace(identity, commits_at_identity=7)
        body = await _service(h).lineage()
        assert body["resident"]["commits_since_identity"] == 0


class TestStartupCleanupNeverRaises:
    def test_an_unreadable_run_directory_is_logged_and_skipped(self, tmp_path, monkeypatch):
        """F12: called from the lifespan; a PermissionError on one run directory must neither
        stop serving nor skip the other runs' leftovers."""
        from millm.services import checkpoint_store as cs

        root = tmp_path / "primary"
        (root / "locked").mkdir(parents=True)
        leftover = root / "open" / f".step-000001.tmp-{'a' * 32}"
        leftover.mkdir(parents=True)
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(root))
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
        real_iterdir = type(root).iterdir

        def iterdir(self):
            if self.name == "locked":
                raise PermissionError("denied")
            return real_iterdir(self)

        monkeypatch.setattr(type(root), "iterdir", iterdir)
        assert cs.remove_stale_temp_dirs() == [leftover]

    def test_an_unreadable_root_is_logged_not_raised(self, tmp_path, monkeypatch):
        from millm.services import checkpoint_store as cs

        root = tmp_path / "primary"
        root.mkdir()
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(root))
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
        real_iterdir = type(root).iterdir
        monkeypatch.setattr(type(root), "iterdir", lambda self: (_ for _ in ()).throw(
            PermissionError("denied")) if self == root else real_iterdir(self))
        assert cs.remove_stale_temp_dirs() == []
