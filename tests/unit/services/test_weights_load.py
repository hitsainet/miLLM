"""`WeightsCheckpointService.load` over the REAL barrier and weight service (032 FTASKS 6.8, 9A).

Fixtures disagree with the defects: the checkpoint and the resident hold DIFFERENT values in the
same shapes (so a copy that did nothing fails); the full-reload checkpoint has a different
vocabulary (so it cannot be copied in place); the foreign checkpoint omits `kind` and
`schema_version` and drops the tied key, as transformers writes it.

Mutation controls, each red in this file:
  M9  hash check after the first copy           -> TestVerification.test_hash_mismatch_touches_nothing
  M10 `dst.data = ...` instead of `copy_`        -> TestInPlace.test_keeps_every_address
  M11 failure after the first copy sets clean    -> TestInPlace.test_interrupted_copy_is_torn
  M12 load accepted while writing                -> TestAttachmentRule.test_refused_while_writing
  M13 full reload allowed while attached         -> TestFullReload.test_refused_while_attached
  M14 `_refuse_if_leased` removed                -> TestLease.*
  identity set after the barrier                 -> TestIdentityInsideTheBarrier.*
  fingerprint re-check removed (either branch)   -> TestVerification.test_file_changed_*
  begin_base_replacement removed from in place   -> TestBaseReplaced.*
  begin_base_replacement removed from full reload -> TestBaseReplaced.test_full_reload_*,
                                                    test_row_reload_* (FX-B)
  weights-origin row reloaded without its hash   -> TestRowReload.test_carrier_row_refused
  LOCAL row conflict accepted (D-26)             -> TestFullReload.test_local_row_on_the_directory_refused
  trust_remote_code refusal removed (D-23)       -> TestFullReload.test_remote_code_*
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
import torch
from sqlalchemy import func, select

import millm.services.weights_checkpoint_service as svc_module
from millm.api.schemas.weights_checkpoints import LoadRequest
from millm.core.errors import (
    AlreadyAttachedError,
    CheckpointAmbiguousError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    CheckpointNotLoadableError,
    InvalidCheckpointPathError,
    InvalidManifestError,
    LocalPathRegisteredError,
    ModelLeasedError,
    WeightsLoadFailedError,
    WeightsNotCleanError,
)
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.ml.model_loader import LoadedModelState
from millm.services import resident_identity
from millm.services.resident_identity import base_epoch, identity_of, served_name
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.support.factories import make_model
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    OPERATOR,
    OTHER_TRAINER,
    RESIDENT_ID,
    RESIDENT_REPO,
    TRAINER,
    attach_holder,
    harness,
    logits,
    manifest_fields,
    resident_llama,
    write_foreign_checkpoint,
)
from tests.unit.weights_fixtures import tiny_llama


def _service(h) -> WeightsCheckpointService:
    return WeightsCheckpointService(h.models)


async def _saved(h, step=5):
    """A checkpoint of the resident's current values, saved through the service."""
    attach_holder(h.weights, h.clock)
    result = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields(step=step))
    h.weights._attachment = None
    return result


def _scramble(model) -> dict[str, torch.Tensor]:
    """Overwrite every parameter in place; return the values before."""
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    return before


def _snapshot(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _same(model, values) -> bool:
    return all(torch.equal(p, values[n]) for n, p in model.named_parameters())


async def _load(h, identity=TRAINER, *, attachment_id=None, lease_id=None, **request):
    return await _service(h).load(identity, LoadRequest(**request),
                                  attachment_id=attachment_id, lease_id=lease_id)


class TestInPlace:
    async def test_round_trip_and_identity(self, harness):
        h = harness
        loaded = resident_llama()
        saved_logits = logits(loaded.model)
        saved = await _saved(h)
        _scramble(loaded.model)
        assert not torch.equal(logits(loaded.model), saved_logits)
        result = await _load(h, name=saved.name)
        assert result.mode == "in_place" and result.weights_state == "clean"
        assert result.weights_sha256 == saved.weights_sha256
        assert result.manifest_sha256 == saved.manifest_sha256
        assert torch.equal(logits(loaded.model), saved_logits)
        assert served_name() == saved.name and identity_of(loaded).origin == "checkpoint"
        assert LoadedModelState().current is loaded  # never unloaded
        assert h.names() == []

    async def test_keeps_every_address_and_the_attached_handle_reads_new_values(self, harness):
        """Acceptance 11 on the CPU (M10): the holder's tensors are the model's own storage."""
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        _scramble(loaded.model)
        params = list(loaded.model.parameters())
        addresses = [p.data_ptr() for p in params]
        attach_holder(h.weights, h.clock, params=params)
        await _load(h, attachment_id=ATTACHMENT_ID, name=saved.name)
        assert [p.data_ptr() for p in loaded.model.parameters()] == addresses
        handle = h.weights._attachment.keep_alive
        from safetensors.torch import load_file

        on_disk = load_file(str(h.root / "run" / "step-000005" / "model.safetensors"))
        assert torch.equal(handle[0], on_disk["model.embed_tokens.weight"])

    async def test_interrupted_copy_is_torn(self, harness):
        """Acceptance 12 (M11): a failure after the first write leaves `torn`, /v1 refused."""
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        _scramble(loaded.model)
        before = _snapshot(loaded.model)
        real = svc_module.copy_in_place
        calls = {"n": 0}

        class Flaky:
            def __init__(self, inner):
                self.inner = inner

            def keys(self):
                return self.inner.keys()

            def get_slice(self, name):
                return self.inner.get_slice(name)

            def get_tensor(self, name):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise OSError("read error")
                return self.inner.get_tensor(name)

        with patch.object(svc_module, "copy_in_place",
                          lambda model, handle: real(model, Flaky(handle))):
            with pytest.raises(WeightsLoadFailedError) as info:
                await _load(h, name=saved.name)
        assert info.value.details["weights_state"] == "torn"
        assert h.barrier.snapshot().weights_state == "torn"
        assert h.barrier.admission_refusal() is not None
        changed = sum(not torch.equal(p, before[n]) for n, p in loaded.model.named_parameters())
        assert changed == 2

    async def test_load_from_torn_recovers_clean(self, harness):
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        h.barrier.mark_torn("test")
        result = await _load(h, name=saved.name)
        assert result.weights_state == "clean" and h.barrier.admission_refusal() is None

    async def test_by_path_and_by_latest(self, harness):
        h = harness
        resident_llama()
        saved = await _saved(h)
        for path in ("run/latest", "run/step-000005", str(h.root / "run" / "step-000005")):
            assert (await _load(h, path=path)).name == saved.name


class TestVerification:
    async def test_hash_mismatch_touches_nothing(self, harness):
        """M9: a byte changed in the data section; the header still matches."""
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        weights = h.root / "run" / "step-000005" / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 0xFF
        weights.write_bytes(bytes(data))
        _scramble(loaded.model)
        before = _snapshot(loaded.model)
        with pytest.raises(CheckpointHashMismatchError) as info:
            await _load(h, name=saved.name)
        assert info.value.details["expected"] == saved.weights_sha256
        assert info.value.details["actual"] != saved.weights_sha256
        assert _same(loaded.model, before)
        assert h.barrier.snapshot().weights_state == "clean"

    async def _change_on_take(self, h, weights):
        def touch():
            st = os.stat(weights)
            os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        h.inference.on_take = touch

    async def test_file_changed_after_hash_in_place(self, harness):
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        _scramble(loaded.model)
        before = _snapshot(loaded.model)
        await self._change_on_take(h, h.root / "run" / "step-000005" / "model.safetensors")
        with pytest.raises(CheckpointHashMismatchError) as info:
            await _load(h, name=saved.name)
        assert info.value.details == {"changed_since_hash": True}
        assert _same(loaded.model, before)
        assert h.barrier.snapshot().weights_state == "clean"  # refused, not torn

    async def test_file_changed_after_hash_full_reload(self, harness):
        h = harness
        loaded = resident_llama()
        directory = write_foreign_checkpoint(h.root, "big", 1,
                                             tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        await self._change_on_take(h, directory / "model.safetensors")
        with pytest.raises(CheckpointHashMismatchError):
            await _load(h, path="big/latest")
        assert LoadedModelState().current is loaded and h.names() == []
        assert h.barrier.snapshot().weights_state == "clean"

    async def test_missing_invalid_and_inconsistent_manifests(self, harness):
        h = harness
        resident_llama()
        saved = await _saved(h)
        directory = h.root / "run" / "step-000005"
        with pytest.raises(CheckpointNotFoundError):
            await _load(h, name=f"{RESIDENT_REPO}:nope:step-5")
        with pytest.raises(InvalidCheckpointPathError):
            await _load(h, path="../outside")
        doc = json.loads((directory / "manifest.json").read_text())
        doc["name"] = f"{RESIDENT_REPO}:run:step-400"
        (directory / "manifest.json").write_text(json.dumps(doc))
        with pytest.raises(InvalidManifestError):
            await _load(h, path="run/step-000005")
        doc["name"] = saved.name
        doc["extra"] = 1
        (directory / "manifest.json").write_text(json.dumps(doc))
        with pytest.raises(InvalidManifestError) as info:
            await _load(h, path="run/step-000005")
        assert any("extra" in e["json_path"] for e in info.value.details["errors"])
        (directory / "manifest.json").unlink()
        with pytest.raises(CheckpointNotFoundError):
            await _load(h, path="run/step-000005")

    async def test_ambiguous_run_across_roots_refused(self, harness, tmp_path, monkeypatch):
        """D-21: the same run_id under two roots is never guessed."""
        h = harness
        resident_llama()
        await _saved(h)
        other = tmp_path / "forge"
        write_foreign_checkpoint(other, "run", 5, tiny_llama())
        from millm.core.config import settings

        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", str(other))
        with pytest.raises(CheckpointAmbiguousError):
            await _load(h, path="run/step-000005")


class TestAttachmentRule:
    async def test_refused_while_writing(self, harness):
        """M12: including the caller's own open commit."""
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        attachment = attach_holder(h.weights, h.clock)
        await h.barrier.quiesce(attachment.digest, TRAINER)
        h.barrier.commit_begin(attachment.digest)
        before = _snapshot(loaded.model)
        with pytest.raises(WeightsNotCleanError):
            await _load(h, attachment_id=ATTACHMENT_ID, name=saved.name)
        assert _same(loaded.model, before)

    async def test_holder_needs_its_header(self, harness):
        """Review 3.3: a same-role token without X-miLLM-Attachment is not the holder."""
        h = harness
        resident_llama()
        saved = await _saved(h)
        attach_holder(h.weights, h.clock)
        with pytest.raises(AlreadyAttachedError):
            await _load(h, TRAINER, name=saved.name)
        with pytest.raises(AlreadyAttachedError):
            await _load(h, OTHER_TRAINER, attachment_id=ATTACHMENT_ID, name=saved.name)
        assert (await _load(h, TRAINER, attachment_id=ATTACHMENT_ID, name=saved.name)).mode \
            == "in_place"

    async def test_operator_only_when_torn_or_stale(self, harness):
        h = harness
        resident_llama()
        saved = await _saved(h)
        attach_holder(h.weights, h.clock)
        with pytest.raises(AlreadyAttachedError):
            await _load(h, OPERATOR, name=saved.name)
        h.barrier.mark_torn("test")
        assert (await _load(h, OPERATOR, name=saved.name)).weights_state == "clean"
        h.clock.now += 10_000  # stale
        assert (await _load(h, OPERATOR, name=saved.name)).mode == "in_place"


class TestLease:
    async def _lease(self, h):
        return await h.models.acquire_lease(RESIDENT_ID, "midataworks", "eval", 60)

    async def test_in_place_refused_and_lifted(self, harness):
        """M14 (in place)."""
        h = harness
        resident_llama()
        saved = await _saved(h)
        grant = await self._lease(h)
        with pytest.raises(ModelLeasedError):
            await _load(h, name=saved.name)
        assert (await _load(h, name=saved.name, lease_id=grant.lease_id)).mode == "in_place"

    async def test_full_reload_refused(self, harness):
        """M14 (full reload)."""
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        await self._lease(h)
        with pytest.raises(ModelLeasedError):
            await _load(h, path="big/latest")
        assert h.names() == []


class TestRowLease:
    async def test_row_reload_refused_under_a_foreign_lease(self, harness, tmp_path):
        """M14 (the D-01 row path)."""
        from tests.unit.weights_fixtures import save_pretrained_dir

        h = harness
        resident_llama()
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        grant = await h.models.acquire_lease(RESIDENT_ID, "midataworks", "eval", 60)
        with pytest.raises(ModelLeasedError):
            await _load(h, model_id=RESIDENT_ID)
        assert h.names() == []
        await _load(h, model_id=RESIDENT_ID, lease_id=grant.lease_id)


class TestFullReload:
    async def test_loads_a_carrier_and_reuses_it(self, harness, test_session):
        """M34: one carrier row per directory, however often it is loaded."""
        h = harness
        resident_llama()
        model = tiny_llama(vocab_size=320, dtype=torch.bfloat16, seed=7)
        expected = logits(model)
        directory = write_foreign_checkpoint(h.root, "big", 1, model)
        result = await _load(h, path="big/latest")
        assert result.mode == "full_reload" and result.weights_state == "clean"
        loaded = LoadedModelState().current
        assert all(torch.equal(t, model.state_dict()[k])
                   for k, t in loaded.model.state_dict().items())
        assert expected is not None
        assert served_name() == f"{RESIDENT_REPO}:big:step-1"
        row = await h.repo.find_by_local_path(os.path.realpath(directory))
        assert row.weights_origin == "checkpoint" and row.cache_path == os.path.realpath(directory)
        assert loaded.model_id == row.id
        load_calls = [c for c in h.calls if c[0] == "load_model"]
        assert load_calls == [("load_model", {"model_id": row.id, "gpu": None, "lease_id": None,
                                              "under_barrier": True,
                                              "allow_weights_origin": True})]
        assert h.names().count("wait_until_loaded") == 1
        # The REAL load path ran: load_model auto-unloaded the resident under the barrier, the
        # background worker opened the verified directory, and the row is LOADED.
        assert h.names() == ["load_model", "unload_model", "wait_until_loaded"]
        assert h.calls[1][1]["model_id"] == RESIDENT_ID and h.calls[1][1]["under_barrier"]
        assert [(c["model_id"], c["cache_path"]) for c in h.loader.loads] == [
            (row.id, os.path.realpath(directory))]
        assert (await h.repo.get_by_id(row.id)).status == ModelStatus.LOADED
        assert (await h.repo.get_by_id(RESIDENT_ID)).status == ModelStatus.READY
        h.calls.clear()
        # Again: the resident IS the carrier now, so it is unloaded first, under the barrier.
        identity = identity_of(loaded)
        resident_identity.set_identity(loaded, resident_identity.ResidentIdentity(
            name="moved", base_model="elsewhere", origin="checkpoint", checkpoint_dir=None,
            manifest_sha256=None, expected_parent=None, commits_at_identity=0))
        await _load(h, path="big/latest")
        assert h.names() == ["unload_model", "load_model", "wait_until_loaded"]
        assert h.calls[0][1]["under_barrier"] is True
        count = await test_session.scalar(select(func.count()).select_from(Model).where(
            Model.local_path == os.path.realpath(directory)))
        assert count == 1
        assert identity is not None

    async def test_refused_while_attached(self, harness):
        """M13."""
        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        attach_holder(h.weights, h.clock)
        with pytest.raises(AlreadyAttachedError):
            await _load(h, attachment_id=ATTACHMENT_ID, path="big/latest")
        assert LoadedModelState().current is loaded and h.names() == []

    async def test_interrupted_reload_is_torn(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        h.loader.fail = RuntimeError("CUDA out of memory")
        with pytest.raises(WeightsLoadFailedError):
            await _load(h, path="big/latest")
        assert h.barrier.snapshot().weights_state == "torn"

    async def test_local_row_on_the_directory_refused(self, harness, test_session):
        """D-26."""
        h = harness
        resident_llama()
        directory = write_foreign_checkpoint(h.root, "big", 1,
                                             tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        test_session.add(make_model(id=9, name="step-000001", source=ModelSource.LOCAL,
                                    repo_id=None, local_path=os.path.realpath(directory),
                                    cache_path=os.path.realpath(directory)))
        await test_session.commit()
        with pytest.raises(LocalPathRegisteredError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["model_id"] == 9 and h.names() == []

    async def test_non_16_bit_refused(self, harness):
        """D-25: never cast silently."""
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320))  # float32
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["reason"] == "dtype" and h.names() == []

    async def _remote(self, h):
        directory = write_foreign_checkpoint(h.root, "big", 1,
                                             tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        config = json.loads((directory / "config.json").read_text())
        config["auto_map"] = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}
        (directory / "config.json").write_text(json.dumps(config))
        return directory

    async def test_remote_code_refused_without_operator_consent(self, harness):
        """D-23."""
        h = harness
        resident_llama()
        await self._remote(h)
        for who, flag in ((TRAINER, True), (OPERATOR, False), (TRAINER, False)):
            with pytest.raises(CheckpointNotLoadableError) as info:
                await _load(h, who, path="big/latest", trust_remote_code=flag)
            assert info.value.details["reason"] == "remote_code"
        assert h.names() == []

    async def test_remote_code_with_operator_consent_records_trust(self, harness):
        h = harness
        resident_llama()
        directory = await self._remote(h)
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                   side_effect=lambda path, **k: tiny_llama(vocab_size=320,
                                                            dtype=torch.bfloat16)):
            await _load(h, OPERATOR, path="big/latest", trust_remote_code=True)
        row = await h.repo.find_by_local_path(os.path.realpath(directory))
        assert row.trust_remote_code is True


class TestPrecheckBeforeTheFirstWrite:
    """W2-G debt: a full reload marked its first write before `load_model`, so a refusal
    `load_model` makes before touching anything (placement, memory, a load in progress) left
    the weights `torn` over a model that was never moved. `precheck_load` now runs inside the
    barrier BEFORE `mark_first_write`: such a refusal leaves the state exactly as it was —
    `clean` stays `clean`, `torn` stays `torn` — and nothing is unloaded. A failure AFTER the
    pre-check passes (the load itself) is still `torn` (`TestFullReload.test_interrupted_*`)."""

    @staticmethod
    def _tiny_card(monkeypatch, free_mb):
        from millm.ml.gpu_placement import GpuInfo
        from millm.services import model_service as module

        monkeypatch.setattr(module, "list_gpus", lambda: [
            GpuInfo(index=0, name="tiny", uuid=None, total_mb=free_mb, free_mb=free_mb)])

    async def _assert_untouched(self, h, loaded, state):
        assert h.barrier.snapshot().weights_state == state
        assert LoadedModelState().current is loaded
        assert h.names() == [] and h.loader.loads == []
        assert (await h.repo.get_by_id(RESIDENT_ID)).status == ModelStatus.LOADED

    @pytest.mark.parametrize("start", ["clean", "torn"])
    async def test_unknown_card_leaves_the_state_unchanged(self, harness, start):
        from millm.core.errors import GpuNotFoundError

        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        if start == "torn":
            h.barrier.mark_torn("test")
        with pytest.raises(GpuNotFoundError):
            await _load(h, OPERATOR, path="big/latest", gpu=7)
        await self._assert_untouched(h, loaded, start)

    @pytest.mark.parametrize("start", ["clean", "torn"])
    async def test_memory_refusal_leaves_the_state_unchanged(self, harness, monkeypatch, start):
        from millm.core.errors import InsufficientMemoryError

        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        # A card the planner's own trace of this model cannot fit (it needs a few MB).
        self._tiny_card(monkeypatch, 1)
        if start == "torn":
            h.barrier.mark_torn("test")
        with pytest.raises(InsufficientMemoryError):
            await _load(h, OPERATOR, path="big/latest", gpu=0)
        await self._assert_untouched(h, loaded, start)

    async def test_load_in_progress_leaves_the_state_unchanged(self, harness):
        """A load claimed elsewhere between `_refuse_if_busy` and the barrier (the slot is
        process-wide): refused before the first write, not torn."""
        from millm.core.errors import ModelBusyError
        from millm.services import model_service as module

        h = harness
        loaded = resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        real_busy = svc_module.WeightsCheckpointService._refuse_if_busy

        def claim_after_check(self):
            real_busy(self)
            module._LOAD_SLOT["model_id"] = 99

        with patch.object(svc_module.WeightsCheckpointService, "_refuse_if_busy",
                          claim_after_check):
            with pytest.raises(ModelBusyError):
                await _load(h, path="big/latest")
        module._LOAD_SLOT["model_id"] = None
        await self._assert_untouched(h, loaded, "clean")

    async def test_row_reload_refusal_keeps_torn(self, harness, tmp_path):
        """The D-01 recovery path: a refused recovery leaves `torn`, the model resident."""
        from millm.core.errors import GpuNotFoundError
        from tests.unit.weights_fixtures import save_pretrained_dir

        h = harness
        loaded = resident_llama()
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        h.barrier.mark_torn("test")
        with pytest.raises(GpuNotFoundError):
            await _load(h, OPERATOR, model_id=RESIDENT_ID, gpu=7)
        await self._assert_untouched(h, loaded, "torn")
        assert (await _load(h, OPERATOR, model_id=RESIDENT_ID)).weights_state == "clean"

    async def test_the_precheck_runs_inside_the_barrier_before_the_first_write(self, harness):
        """The order itself: the pre-check sees the slot held and the state not yet writing."""
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        seen = []
        real = h.models.precheck_load

        async def spy(*args, **kwargs):
            seen.append((h.barrier.holds_slot(), h.barrier.snapshot().weights_state, kwargs))
            return await real(*args, **kwargs)

        h.models.precheck_load = spy
        await _load(h, path="big/latest")
        assert len(seen) == 1
        holds, state, kwargs = seen[0]
        assert holds is True and state == "clean"
        assert kwargs["under_barrier"] is True and kwargs["allow_weights_origin"] is True


class TestAttachableReload:
    """D-03 / D-37 through the REAL worker: a full reload of a 16-bit row is an attachable load
    once a trainer token is configured (so the trainer can attach to what it loaded); an FP32
    row never is, and without a trainer token nothing is."""

    @staticmethod
    def _trainer_token(monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", json.dumps(
            [{"label": "forge", "role": "trainer", "sha256": "0" * 64}]))

    async def test_carrier_reload_is_attachable_with_a_trainer_token(self, harness, monkeypatch):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        self._trainer_token(monkeypatch)
        await _load(h, path="big/latest")
        assert [c["attachable"] for c in h.loader.loads] == [True]

    async def test_not_attachable_without_a_trainer_token(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        await _load(h, path="big/latest")
        assert [c["attachable"] for c in h.loader.loads] == [False]

    async def test_fp32_row_reload_is_never_attachable(self, harness, monkeypatch, tmp_path):
        from tests.unit.weights_fixtures import save_pretrained_dir

        h = harness
        resident_llama()
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        await h.repo.update(RESIDENT_ID, quantization=QuantizationType.FP32)
        self._trainer_token(monkeypatch)
        await _load(h, OPERATOR, model_id=RESIDENT_ID)
        assert [(c["quantization"], c["attachable"]) for c in h.loader.loads] == [
            ("FP32", False)]


class TestRefusalsBeforeThePause:
    """Refusals the 032 load makes BEFORE the barrier takes the admission slot, so a refused load
    never pauses serving. Each check is also made again later (by `precheck_load` inside the
    barrier, or by the barrier itself), which is why removing the early one used to leave the
    suite green (W3-A mutation sweep): the observable difference is the pause."""

    @staticmethod
    def _count_takes(h) -> list[int]:
        takes: list[int] = []
        h.inference.on_take = lambda: takes.append(1)
        return takes

    async def test_row_reload_lease_refusal_never_pauses(self, harness, tmp_path):
        from tests.unit.weights_fixtures import save_pretrained_dir

        h = harness
        resident_llama()
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        await h.models.acquire_lease(RESIDENT_ID, "midataworks", "eval", 60)
        takes = self._count_takes(h)
        with pytest.raises(ModelLeasedError) as info:
            await _load(h, OPERATOR, model_id=RESIDENT_ID)
        assert info.value.details.get("operation") == "weights_load"
        assert takes == [] and h.names() == []

    async def test_row_reload_attachment_refusal_never_pauses(self, harness, tmp_path):
        from tests.unit.weights_fixtures import save_pretrained_dir

        h = harness
        resident_llama()
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        attach_holder(h.weights, h.clock)
        takes = self._count_takes(h)
        with pytest.raises(AlreadyAttachedError) as info:
            await _load(h, OPERATOR, model_id=RESIDENT_ID)
        assert info.value.details["operation"] == "weights_load"
        assert takes == [] and h.names() == []

    async def test_checkpoint_lease_refusal_never_pauses(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        await h.models.acquire_lease(RESIDENT_ID, "midataworks", "eval", 60)
        takes = self._count_takes(h)
        with pytest.raises(ModelLeasedError):
            await _load(h, path="big/latest")
        assert takes == []

    @pytest.mark.parametrize("busy", ["loading", "unloading"])
    async def test_in_place_refused_while_a_load_or_unload_runs(self, harness, busy):
        """`_refuse_if_busy`: the in-place path has no `load_model` to refuse it later."""
        from millm.core.errors import ModelBusyError
        from millm.services import model_service as module

        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        before = _scramble(loaded.model)
        after = _snapshot(loaded.model)
        takes = self._count_takes(h)
        if busy == "loading":
            module._LOAD_SLOT["model_id"] = 99
        else:
            LoadedModelState().begin_unload()
        try:
            with pytest.raises(ModelBusyError) as info:
                await _load(h, name=saved.name)
        finally:
            module._LOAD_SLOT["model_id"] = None
            LoadedModelState().cancel_unload()
        assert info.value.details == {"operation": "weights_load"}
        assert takes == [] and _same(loaded.model, after) and not _same(loaded.model, before)


class TestTheResidentMovedUnderTheLoad:
    async def test_in_place_refused_when_the_resident_changed_before_the_barrier(self, harness):
        """The in-place path re-checks, under the barrier, that the resident is the model it
        verified against: a same-shaped model loaded meanwhile must not receive the copy."""
        from millm.core.errors import ModelBusyError
        from tests.unit.checkpoint_service_fixtures import install_resident

        h = harness
        resident_llama()
        saved = await _saved(h)
        other = tiny_llama(seed=3)  # same shapes: only the identity check can tell
        untouched = _snapshot(other)
        swapped: list[int] = []

        def swap():
            if not swapped:
                swapped.append(1)
                install_resident(other, model_id=77, name="intruder")

        h.inference.on_take = swap
        with pytest.raises(ModelBusyError):
            await _load(h, name=saved.name)
        assert swapped == [1]
        assert LoadedModelState().current.model is other and _same(other, untouched)
        assert h.barrier.snapshot().weights_state == "clean"
        assert identity_of(LoadedModelState().current) is None

    async def test_full_reload_ending_with_another_model_resident_is_torn(self, harness):
        """The carrier loaded, then something else replaced it before the barrier released:
        nothing verified is resident, so the weights are `torn` and no identity is set."""
        from tests.unit.checkpoint_service_fixtures import install_resident

        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        h.loader.after_load = lambda: install_resident(tiny_llama(), model_id=77, name="intruder")
        with pytest.raises(WeightsLoadFailedError) as info:
            await _load(h, path="big/latest")
        assert info.value.details["stage"] == "reload"
        assert h.barrier.snapshot().weights_state == "torn"
        current = LoadedModelState().current
        assert current.model_id == 77 and identity_of(current) is None


class TestRowReload:
    async def test_recovers_from_torn(self, harness, tmp_path):
        """D-01: `{"model_id": N}` reloads a plain row inside the barrier."""
        h = harness
        resident_llama()
        from tests.unit.weights_fixtures import save_pretrained_dir

        base = save_pretrained_dir(tiny_llama(), tmp_path / "base")
        assert base.is_dir()
        h.barrier.mark_torn("test")
        result = await _load(h, model_id=RESIDENT_ID)
        assert result.mode == "full_reload" and result.weights_state == "clean"
        assert result.weights_sha256 is None
        assert h.names() == ["unload_model", "load_model", "wait_until_loaded"]
        assert h.calls[1][1]["allow_weights_origin"] is False
        assert identity_of(LoadedModelState().current) is None

    async def test_carrier_row_refused(self, harness, test_session):
        """A checkpoint row is loaded only by name or path, where its hash is verified."""
        h = harness
        resident_llama()
        test_session.add(make_model(id=5, name="b:r:step-1", source=ModelSource.LOCAL,
                                    repo_id=None, local_path="/ck/r/step-000001",
                                    cache_path="/ck/r/step-000001", weights_origin="checkpoint",
                                    quantization=QuantizationType.FP16, status=ModelStatus.READY))
        await test_session.commit()
        with pytest.raises(CheckpointNotLoadableError) as info:
            await _load(h, model_id=5)
        assert info.value.details["reason"] == "carrier_row" and h.names() == []


class TestIdentityInsideTheBarrier:
    """Review 5.2: the slot is released only after the identity is set, so the first request
    admitted afterwards captures the new name."""

    async def test_in_place(self, harness):
        h = harness
        loaded = resident_llama()
        saved = await _saved(h, step=5)
        await _saved(h, step=6)  # the resident is now named after step 6: the load must rename it
        assert served_name() != saved.name
        seen = []
        h.inference.on_release = lambda: seen.append(served_name())
        await _load(h, name=saved.name)
        assert seen == [saved.name]
        assert loaded is LoadedModelState().current

    async def test_full_reload(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        seen = []
        # The carrier row carries the manifest name too, so the name alone cannot tell; the
        # identity's origin and manifest hash can.
        h.inference.on_release = lambda: seen.append(
            identity_of(LoadedModelState().current))
        result = await _load(h, path="big/latest")
        assert len(seen) == 1 and seen[0] is not None
        assert seen[0].origin == "checkpoint"
        assert seen[0].manifest_sha256 == result.manifest_sha256


class TestBaseReplaced:
    """032 FTASKS 9A: the in-place path advances the epoch once and runs the callbacks before
    its first write; a raising callback aborts with nothing written and the state unchanged."""

    async def test_in_place_advances_once_before_the_first_write(self, harness):
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        _scramble(loaded.model)
        before = _snapshot(loaded.model)
        seen = []

        def callback(reason):
            seen.append((reason, _same(loaded.model, before),
                         h.barrier.snapshot().weights_state))

        resident_identity.register_on_base_replaced(callback)
        epoch = base_epoch()
        await _load(h, name=saved.name)
        assert base_epoch() == epoch + 1
        assert seen == [("weights_load_in_place", True, "clean")]

    async def test_raising_callback_aborts_untouched(self, harness):
        h = harness
        loaded = resident_llama()
        saved = await _saved(h)
        _scramble(loaded.model)
        before = _snapshot(loaded.model)

        def refuse(reason):
            raise RuntimeError("adapter drop failed")

        resident_identity.register_on_base_replaced(refuse)
        with pytest.raises(RuntimeError):
            await _load(h, name=saved.name)
        assert _same(loaded.model, before)
        assert h.barrier.snapshot().weights_state == "clean"
        assert identity_of(loaded).name == saved.name  # the save's identity, unchanged

    async def test_full_reload_runs_the_callbacks_before_the_first_write(self, harness):
        """RX F5: the full reload's own replacement fires first, while the old resident is
        still untouched and the weights are still `clean` (the first write not yet marked)."""
        h = harness
        loaded = resident_llama()
        before = _snapshot(loaded.model)
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        seen = []

        def callback(reason):
            if not seen:  # later calls (the loader's chokepoints) see the old model freed
                seen.append((reason, LoadedModelState().current is loaded,
                             _same(loaded.model, before), h.barrier.snapshot().weights_state))

        resident_identity.register_on_base_replaced(callback)
        result = await _load(h, path="big/latest")
        assert result.mode == "full_reload"
        assert seen[0] == ("weights_load_full_reload", True, True, "clean")

    async def test_full_reload_raising_callback_aborts_with_the_state_unchanged(self, harness):
        """RX F5: a callback that raises aborts the full reload before the first write: the
        weights stay `clean`, nothing is unloaded and nothing is loaded (it used to fire inside
        `LoadedModelState.clear`, after the mark, and leave `torn`)."""
        h = harness
        loaded = resident_llama()
        before = _snapshot(loaded.model)
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))

        def refuse(reason):
            raise RuntimeError("adapter drop failed")

        resident_identity.register_on_base_replaced(refuse)
        with pytest.raises(RuntimeError, match="adapter drop failed"):
            await _load(h, path="big/latest")
        assert h.barrier.snapshot().weights_state == "clean"
        assert LoadedModelState().current is loaded and _same(loaded.model, before)
        assert "unload_model" not in h.names() and "load_model" not in h.names()
        assert h.loader.loads == []

    async def test_row_reload_raising_callback_aborts_with_the_state_unchanged(self, harness):
        """The D-01 row reload shares `_replace_resident`: the same abort, state unchanged."""
        h = harness
        loaded = resident_llama()

        def refuse(reason):
            raise RuntimeError("adapter drop failed")

        resident_identity.register_on_base_replaced(refuse)
        with pytest.raises(RuntimeError, match="adapter drop failed"):
            await _load(h, OPERATOR, model_id=RESIDENT_ID)
        assert h.barrier.snapshot().weights_state == "clean"
        assert LoadedModelState().current is loaded and h.loader.loads == []

    async def test_full_reload_advances_the_epoch_once(self, harness):
        h = harness
        resident_llama()
        write_foreign_checkpoint(h.root, "big", 1, tiny_llama(vocab_size=320, dtype=torch.bfloat16))
        epoch = base_epoch()
        await _load(h, path="big/latest")
        assert base_epoch() == epoch + 1


class TestInboundCompatibility:
    """Review 6.3: a directory miForge writes (save_pretrained, no kind/schema_version, a
    `.tmp-save-*` sibling, a relative `latest`) under a READ root loads in place and by full
    reload."""

    async def test_in_place_from_a_read_root(self, harness, tmp_path, monkeypatch):
        from millm.core.config import settings

        h = harness
        loaded = resident_llama(seed=0)
        forge = tmp_path / "forge"
        source = tiny_llama(seed=11)
        write_foreign_checkpoint(forge, "house-sql", 400, source)
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", str(forge))
        result = await _load(h, path="house-sql/latest")
        assert result.mode == "in_place"
        assert torch.equal(logits(loaded.model), logits(source))
        assert (forge / "house-sql" / ".tmp-save-00000000").is_dir()  # never cleaned

    async def test_full_reload_from_a_read_root(self, harness, tmp_path, monkeypatch):
        from millm.core.config import settings

        h = harness
        resident_llama()
        forge = tmp_path / "forge"
        source = tiny_llama(seed=12, vocab_size=320, dtype=torch.bfloat16)
        write_foreign_checkpoint(forge, "house-sql", 400, source)
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", str(forge))
        result = await _load(h, name=f"{RESIDENT_REPO}:house-sql:step-400")
        assert result.mode == "full_reload"
        assert torch.equal(logits(LoadedModelState().current.model), logits(source))
