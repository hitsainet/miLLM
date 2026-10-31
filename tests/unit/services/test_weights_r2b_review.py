"""032 review round 2 (R2-B): the fixes from round 1 and the fix round, re-reviewed.

Each test pins one finding in the R2-B report and was run red against the line it guards:

  E1  `covered_base_replacement` (P-EPOCH) covered EVERY chokepoint for a model id for the whole
      scope. Nothing pinned that another model's chokepoint still advances, and a second call for
      a covered id (another operation's unload inside the scope: a Feature-1 unload is not
      refused while the barrier holds the slot) silently skipped the epoch and the callbacks.
      Each expected call is now consumed once.
  E2  the scope's cleanup on an exception was pinned only by test-order leakage.
  E3  the reentrancy refusal came AFTER the epoch advanced and the callbacks ran.
  E4  the full reload's callback COUNT was not asserted (only the first call's arguments), and
      the D-01 row reload of the resident row (unload then load of ONE id) had no epoch test.
  C1  a cancelled full reload released the barrier while the model service's load worker was
      still running; the worker's Feature-1 completion then turned `torn` into `clean`, serving
      the checkpoint under its own name without the post-load fingerprint check (R32 F6).
  C2  a cancelled save or load wrote no audit line (031 and 033 audit `CANCELLED`).
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime
from unittest.mock import patch

import pytest
import torch

import millm.services.resident_identity as ri
import millm.services.weights_checkpoint_service as svc_module
from millm.api.schemas.weights_checkpoints import LoadRequest
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services.resident_identity import (
    base_epoch,
    begin_chokepoint_replacement,
    covered_base_replacement,
    identity_of,
    register_on_base_replaced,
)
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    OPERATOR,
    RESIDENT_ID,
    TRAINER,
    attach_holder,
    harness,
    manifest_fields,
    resident_llama,
    write_foreign_checkpoint,
)
from tests.unit.weights_fixtures import tiny_llama


def _service(h) -> WeightsCheckpointService:
    return WeightsCheckpointService(h.models)


async def _load(h, identity=TRAINER, **request):
    return await _service(h).load(identity, LoadRequest(**request))


class _Model:
    def to(self, device):
        return self


def _loaded(model_id: int) -> LoadedModel:
    return LoadedModel(model_id=model_id, model_name=f"row-{model_id}", model=_Model(),
                       tokenizer=None, loaded_at=datetime(2026, 10, 14), device="cpu")


@pytest.fixture
def fired(monkeypatch):
    """A clean callback registry and resident slot; returns the reasons callbacks saw."""
    state = LoadedModelState()
    state._loaded, state._unloading = None, False
    monkeypatch.setattr(ri, "_base_replaced", [])
    seen: list[str] = []
    register_on_base_replaced(seen.append)
    yield seen
    state._loaded, state._unloading = None, False
    assert ri._covered is None, "a covered scope leaked out of the test"


# ----------------------------------------------------------------------------- E1


class TestTheCoverIsPerCall:
    def test_another_models_chokepoint_still_advances_inside_the_scope(self, fired):
        with covered_base_replacement("full", [("model_load", 2)]) as epoch:
            assert fired == ["full"]
            assert begin_chokepoint_replacement("model_load", 3) == epoch + 1
            assert fired == ["full", "model_load"]

    def test_the_other_kind_of_call_for_a_covered_id_still_advances(self, fired):
        """Covering a model's LOAD does not cover its UNLOAD."""
        with covered_base_replacement("full", [("model_load", 2)]) as epoch:
            assert begin_chokepoint_replacement("model_unload", 2) == epoch + 1
            assert fired == ["full", "model_unload"]

    def test_each_expected_call_is_covered_once(self, fired):
        with covered_base_replacement("full", [("model_unload", 1), ("model_load", 2)]) as epoch:
            assert begin_chokepoint_replacement("model_unload", 1) == epoch
            assert begin_chokepoint_replacement("model_load", 2) == epoch
            assert base_epoch() == epoch and fired == ["full"]
            # A second unload of the newly loaded row, by another operation, inside the scope:
            # it is a real replacement and must advance and run the callbacks.
            assert begin_chokepoint_replacement("model_unload", 2) == epoch + 1
            assert begin_chokepoint_replacement("model_load", 2) == epoch + 2
            assert fired == ["full", "model_unload", "model_load"]

    def test_the_real_chokepoints_consume_the_cover(self, fired):
        """Through `LoadedModelState.set`/`.clear`, the call sites the scope exists for."""
        state = LoadedModelState()
        state.set(_loaded(1))
        epoch0 = base_epoch()
        fired.clear()
        with covered_base_replacement("full", [("model_unload", 1), ("model_load", 2)]):
            state.clear()
            state.set(_loaded(2))
            state.clear()  # a foreign unload of the new row inside the scope
        assert base_epoch() == epoch0 + 2
        assert fired == ["full", "model_unload"]

    def test_a_chokepoint_in_another_thread_sees_the_cover(self, fired):
        """`clear` runs in the unload worker's executor thread."""
        state = LoadedModelState()
        state.set(_loaded(1))
        fired.clear()
        with covered_base_replacement("full", [("model_unload", 1)]) as epoch:
            worker = threading.Thread(target=state.clear)
            worker.start()
            worker.join(5)
            assert base_epoch() == epoch and fired == ["full"]


# ----------------------------------------------------------------------------- E2, E3


class TestTheScopeNeverLeaks:
    def test_an_exception_inside_the_scope_removes_the_cover(self, fired):
        with pytest.raises(RuntimeError, match="load failed"):
            with covered_base_replacement("full", [("model_load", 2)]):
                raise RuntimeError("load failed")
        assert ri._covered is None
        epoch = base_epoch()
        assert begin_chokepoint_replacement("model_load", 2) == epoch + 1

    def test_a_raising_callback_leaves_no_cover(self, monkeypatch, fired):
        def refuse(reason):
            raise RuntimeError("adapter drop failed")

        register_on_base_replaced(refuse)
        with pytest.raises(RuntimeError, match="adapter drop failed"):
            with covered_base_replacement("full", [("model_load", 2)]):
                pytest.fail("the scope must not open")
        assert ri._covered is None

    def test_reentry_is_refused_before_any_side_effect(self, fired):
        with covered_base_replacement("full", [("model_load", 2)]) as epoch:
            with pytest.raises(RuntimeError, match="not reentrant"):
                with covered_base_replacement("nested", [("model_load", 3)]):
                    pytest.fail("the nested scope must not open")
            assert base_epoch() == epoch and fired == ["full"]
            # ...and the outer cover is intact.
            assert begin_chokepoint_replacement("model_load", 2) == epoch


# ----------------------------------------------------------------------------- E4


class TestTheFullReloadReplacesOnce:
    async def test_the_callbacks_run_exactly_once(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        seen = []
        register_on_base_replaced(seen.append)
        epoch = base_epoch()
        assert (await _load(h, path="big/latest")).mode == "full_reload"
        assert seen == ["weights_load_full_reload"]
        assert base_epoch() == epoch + 1

    async def test_the_row_reload_of_the_resident_row_advances_once(self, harness, tmp_path):
        """D-01 on the resident row: OUR unload, then the load, of ONE id."""
        from tests.unit.weights_fixtures import save_pretrained_dir

        h = harness
        resident_llama()
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        seen = []
        register_on_base_replaced(seen.append)
        epoch = base_epoch()
        await _load(h, OPERATOR, model_id=RESIDENT_ID)
        assert "unload_model" in h.names() and h.loader.loads
        assert seen == ["weights_load_full_reload"]
        assert base_epoch() == epoch + 1

    async def test_a_failed_reload_leaves_no_cover(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        h.loader.fail = RuntimeError("CUDA out of memory")
        with pytest.raises(Exception):
            await _load(h, path="big/latest")
        for _ in range(200):
            if not h.models.is_loading:
                break
            await asyncio.sleep(0.01)
        assert ri._covered is None
        epoch = base_epoch()
        LoadedModelState().set(_loaded(7))
        assert base_epoch() == epoch + 1


# ----------------------------------------------------------------------------- C1, C2


class TestCancellation:
    async def test_a_cancelled_full_reload_holds_the_barrier_and_stays_torn(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        started, release = threading.Event(), threading.Event()

        def block():
            started.set()
            release.wait(10)

        h.loader.after_load = block
        audits = []
        with patch.object(svc_module, "weights_audit",
                          side_effect=lambda *a, **k: audits.append((a, k))):
            task = asyncio.create_task(_load(h, path="big/latest"))
            await asyncio.to_thread(started.wait, 10)
            task.cancel()
            await asyncio.sleep(0.2)
            # The load worker is still running: the slot must still be the barrier's.
            assert h.inference.slot.locked()
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        for _ in range(200):
            if not h.models.is_loading:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert not h.inference.slot.locked()
        # Torn, never `clean`: the checkpoint was never re-checked after the reload, and the
        # identity was never set.
        assert h.barrier.snapshot().weights_state == "torn"
        assert identity_of(LoadedModelState().current) is None
        assert [(a[:2], k["code"]) for a, k in audits] == [
            (("weights_load", "refused"), "CANCELLED")]

    async def test_a_cancelled_save_is_audited(self, harness):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        started, release = threading.Event(), threading.Event()

        def slow_write(*args, **kwargs):
            started.set()
            release.wait(10)
            raise OSError("never published")

        audits = []
        with patch.object(svc_module, "write_checkpoint_atomically", slow_write), \
                patch.object(svc_module, "weights_audit",
                             side_effect=lambda *a, **k: audits.append((a, k))):
            task = asyncio.create_task(
                _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields()))
            await asyncio.to_thread(started.wait, 10)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert [(a[:2], k["code"]) for a, k in audits] == [
            (("weights_save", "refused"), "CANCELLED")]
        assert audits[0][1]["attachment_ref"] is not None


# ----------------------------------------------------------------------------- S1


class TestStartupCleanupOnARealFilesystem:
    """R32 F12 wrapped the directory listings, and its tests faked `iterdir` raising — exactly
    the call the fix wrapped. On a real filesystem the first call to raise is `root.is_dir()`:
    `Path.is_dir` re-raises PermissionError when a parent of the root cannot be searched."""

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_root_under_an_unsearchable_parent_is_logged_not_raised(self, tmp_path,
                                                                      monkeypatch):
        from millm.core.config import settings
        from millm.services import checkpoint_store as cs

        parent = tmp_path / "mount"
        (parent / "checkpoints").mkdir(parents=True)
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(parent / "checkpoints"))
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
        parent.chmod(0)
        try:
            assert cs.remove_stale_temp_dirs() == []
        finally:
            parent.chmod(0o755)

    def test_a_read_roots_setting_that_raises_is_logged_not_raised(self, tmp_path, monkeypatch):
        from millm.services import checkpoint_store as cs

        def broken():
            raise ValueError("unparseable WEIGHTS_CHECKPOINT_READ_ROOTS")

        monkeypatch.setattr(cs, "read_roots", broken)
        assert cs.remove_stale_temp_dirs(tmp_path) == []


# ----------------------------------------------------------------------------- V1


class TestChangedWhileHashed:
    """The fingerprint re-check right after the hash had no test of its own: the later re-check
    inside the barrier refuses the same file, so deleting this one left the suite green. The
    difference is WHERE: this one refuses before the barrier is taken and before a full
    reload's carrier row is written; the later one only after both."""

    async def test_refused_before_the_barrier_and_before_any_carrier_row(self, harness):
        from sqlalchemy import func, select

        from millm.core.errors import CheckpointHashMismatchError
        from millm.db.models.model import Model

        h = harness
        loaded = resident_llama()
        directory = write_foreign_checkpoint(
            h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        weights = directory / "model.safetensors"
        real_sha = svc_module.sha256_file

        def hash_then_touch(path):
            digest = real_sha(path)
            st = os.stat(weights)
            os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            return digest

        taken = []
        h.inference.on_take = lambda: taken.append(True)
        rows_before = (await h.repo.session.execute(select(func.count(Model.id)))).scalar_one()
        with patch.object(svc_module, "sha256_file", hash_then_touch):
            with pytest.raises(CheckpointHashMismatchError, match="while it was being hashed"):
                await _load(h, path="big/latest")
        rows_after = (await h.repo.session.execute(select(func.count(Model.id)))).scalar_one()
        assert taken == [] and rows_after == rows_before
        assert LoadedModelState().current is loaded
        assert h.barrier.snapshot().weights_state == "clean"


# ----------------------------------------------------------------------------- open (r2a files)


class TestOpenFindingsOutsideTheseFiles:
    """Defects R2-B found in `model_service.py` (O1, O2), recorded here as strict xfails and
    fixed by R2-A (cceca15, 25b50bc, d7b9296); now ordinary regression tests."""

    async def test_a_feature1_unload_during_an_in_place_copy_is_refused(self, harness):
        from millm.core.errors import ModelBusyError

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
            try:
                with pytest.raises(ModelBusyError):
                    await h.models.unload_model(RESIDENT_ID)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_a_load_finishing_after_its_wait_timed_out_leaves_torn(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        started, release = threading.Event(), threading.Event()

        def block():
            started.set()
            release.wait(10)

        h.loader.after_load = block
        real_wait = h.models.wait_until_loaded

        async def short_wait(model_id, timeout=180.0):
            return await real_wait(model_id, timeout=0.5)

        h.models.wait_until_loaded = short_wait
        with pytest.raises(Exception):
            await _load(h, path="big/latest")
        assert h.barrier.snapshot().weights_state == "torn"
        release.set()
        for _ in range(200):
            if not h.models.is_loading:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert h.barrier.snapshot().weights_state == "torn"
