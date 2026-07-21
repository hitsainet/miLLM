"""Service-level tests for attach_set (Feature 12, review R3 coverage gap).

R1/R2 added a free-VRAM pre-check and mid-set rollback to attach_set, but the
route tests mock attach_set away entirely and the existing test_attach_set.py
only covers the dtype helper + config constants. These tests drive the real
attach_set body with a fake loader/hooker so the rollback, idempotent skip and
pre-validation guarantees are actually verified.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.core.errors import SAEIncompatibleError, SAENotFoundError
from millm.services.sae_service import (
    AttachedSAEState,
    CompatibilityResult,
    SAEService,
)


@pytest.fixture(autouse=True)
def reset_registry():
    state = AttachedSAEState()
    state.reset_for_tests()
    yield
    state.reset_for_tests()


def _loaded_sae(mb: float = 64.0):
    sae = MagicMock()
    sae.estimate_memory_mb.return_value = mb
    return sae


def _sae_row(sae_id: str, d_in: int = 2048, d_sae: int = 8192):
    row = MagicMock()
    row.id = sae_id
    row.cache_path = f"/tmp/{sae_id}"
    row.d_in = d_in
    row.d_sae = d_sae
    row.file_size_bytes = 128 * 1024 * 1024
    return row


def _service(compatible: bool = True):
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    svc._loader = MagicMock()
    svc._loader.load.side_effect = lambda **kw: _loaded_sae()
    svc._hooker = MagicMock()
    svc._hooker.install.return_value = MagicMock()
    svc.get_sae = AsyncMock(side_effect=lambda sid: _sae_row(sid))
    svc.check_compatibility = AsyncMock(
        return_value=CompatibilityResult(compatible=compatible, errors=[], warnings=[])
    )
    svc._reset_dynamo_for_hook_change = MagicMock()
    return svc


def _model_loaded():
    """Patch LoadedModelState so attach_set sees a loaded model."""
    mock_state = MagicMock()
    mock_state.is_loaded = True
    mock_state.current.model = MagicMock()
    return patch(
        "millm.services.sae_service.LoadedModelState", return_value=mock_state
    )


class TestAttachSetHappyPath:
    async def test_attaches_all_requested_keys(self):
        svc = _service()
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            result = await svc.attach_set([("sae-a", 10), ("sae-b", 13)])
        assert result["attached_count"] == 2
        assert svc._sae_state.get("sae-a", 10) is not None
        assert svc._sae_state.get("sae-b", 13) is not None
        assert result["total_memory_usage_mb"] == 128

    async def test_idempotent_skip_does_not_reload(self):
        svc = _service()
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            await svc.attach_set([("sae-a", 10)])
            svc._loader.load.reset_mock()
            result = await svc.attach_set([("sae-a", 10)])
        assert svc._loader.load.call_count == 0
        statuses = [e["status"] for e in result["entries"]]
        assert statuses == ["already_attached"]

    async def test_dedups_repeated_keys_in_one_request(self):
        svc = _service()
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            await svc.attach_set([("sae-a", 10), ("sae-a", 10)])
        assert svc._sae_state.count == 1
        assert svc._loader.load.call_count == 1


class TestAttachSetPreValidation:
    async def test_incompatible_sae_attaches_nothing(self):
        """Pre-validation runs BEFORE any load — an incompatible key in the set
        must leave the registry completely unchanged (no partial attach)."""
        svc = _service()
        svc.check_compatibility = AsyncMock(
            side_effect=[
                CompatibilityResult(compatible=True, errors=[], warnings=[]),
                CompatibilityResult(
                    compatible=False, errors=["dim mismatch"], warnings=[]
                ),
            ]
        )
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(SAEIncompatibleError):
                await svc.attach_set([("sae-a", 10), ("sae-bad", 13)])
        assert svc._sae_state.count == 0
        assert svc._loader.load.call_count == 0  # nothing was ever loaded

    async def test_missing_sae_attaches_nothing(self):
        svc = _service()
        svc.get_sae = AsyncMock(side_effect=SAENotFoundError("nope"))
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(SAENotFoundError):
                await svc.attach_set([("ghost", 10)])
        assert svc._sae_state.count == 0


class TestAttachSetRollback:
    async def test_hook_install_failure_rolls_back_and_frees(self):
        """If the hook install throws for the 2nd SAE, the 1st must be rolled
        back and the just-loaded SAE freed — no partial attach, no leak."""
        svc = _service()
        loaded = [_loaded_sae(), _loaded_sae()]
        svc._loader.load.side_effect = lambda **kw: loaded.pop(0)
        svc._hooker.install.side_effect = [MagicMock(), RuntimeError("hook boom")]

        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="hook boom"):
                await svc.attach_set([("sae-a", 10), ("sae-b", 13)])

        # Nothing left attached from this call.
        assert svc._sae_state.count == 0

    async def test_rollback_preserves_pre_existing_attachment(self):
        """A pre-existing attachment (from an earlier call) survives a failed
        attach_set — rollback only undoes THIS call's keys."""
        svc = _service()
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            await svc.attach_set([("pre", 5)])
            assert svc._sae_state.count == 1

            svc._hooker.install.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError):
                await svc.attach_set([("sae-a", 10)])

        assert svc._sae_state.count == 1
        assert svc._sae_state.get("pre", 5) is not None

    async def test_load_failure_rolls_back_earlier_keys(self):
        svc = _service()
        svc._loader.load.side_effect = [_loaded_sae(), RuntimeError("load boom")]
        with _model_loaded(), patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="load boom"):
                await svc.attach_set([("sae-a", 10), ("sae-b", 13)])
        assert svc._sae_state.count == 0


class TestAttachSetVramGate:
    async def test_insufficient_vram_refuses_before_loading(self):
        from millm.core.errors import InsufficientMemoryError

        svc = _service()
        with _model_loaded(), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.mem_get_info", return_value=(1 * 1024 * 1024, 0)):
            with pytest.raises(InsufficientMemoryError):
                await svc.attach_set([("sae-a", 10), ("sae-b", 13)])
        assert svc._sae_state.count == 0
        assert svc._loader.load.call_count == 0  # refused BEFORE loading

    async def test_null_file_size_still_gated_by_dim_estimate(self):
        """R2: a NULL file_size_bytes must not collapse the projection to 0."""
        from millm.core.errors import InsufficientMemoryError

        svc = _service()
        row = _sae_row("sae-a")
        row.file_size_bytes = None  # unknown on-disk size
        svc.get_sae = AsyncMock(return_value=row)
        with _model_loaded(), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.mem_get_info", return_value=(1 * 1024 * 1024, 0)):
            with pytest.raises(InsufficientMemoryError):
                await svc.attach_set([("sae-a", 10)])

    async def test_sufficient_vram_proceeds(self):
        svc = _service()
        with _model_loaded(), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.mem_get_info", return_value=(20_000 * 1024 * 1024, 0)):
            result = await svc.attach_set([("sae-a", 10)])
        assert result["attached_count"] == 1
