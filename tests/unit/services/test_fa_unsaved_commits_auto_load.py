"""D-45 (review R2-A F-A): the `/v1` auto-load never discards commits no checkpoint records.

Through the Feature 32 harness (tests/unit/checkpoint_service_fixtures.py): the REAL
`WeightBarrier`, `WeightService`, `ModelService`, `WeightsCheckpointService` and
`LoadedModelState` over a tiny CPU Llama and the SQLite test session. The commits are real
commit-ends through the holder path, the save is the real 032 save, the operator loads are the
real weights-API loads, and "nothing was unloaded" is the same `LoadedModel` object still
resident with no load or unload call recorded by the harness spy.

Fixtures disagree with the defect: the resident is row 1 (`tiny-llama`), the request names row 2
(`other-model`), and the commit counts differ before and after a save (2, then 1), so a guard
that ignored the save, or the counter, answers with the wrong number.

MUTATION CONTROLS (recorded in 0xcc/reviews/attachable_weights_progress.md, D-45 / D-46):
  FA-1 the guard call removed from `load_model_and_wait`
  FA-2 `unsaved_commits` ignores `commits_at_identity` (returns the raw counter)
  FA-3 the save records `commits_at_identity=0` instead of the counter
  FA-4 `unsaved_commits` always returns 0
  FA-5 the reset branch (counter below `commits_at_identity`) returns 0
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from millm.api.schemas.weights_checkpoints import LoadRequest
from millm.core.errors import WeightsUnsavedError
from millm.db.models.model import ModelStatus, QuantizationType
from millm.ml.model_loader import LoadedModelState
from millm.services import resident_identity
from millm.services.model_service import ModelService
from millm.services.resident_identity import ResidentIdentity
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.support.factories import make_model
from tests.unit import checkpoint_service_fixtures as _fixtures
from tests.unit.checkpoint_service_fixtures import (
    ATTACHMENT_ID,
    OPERATOR,
    RESIDENT_ID,
    RESIDENT_NAME,
    TRAINER,
    attach_holder,
    manifest_fields,
    resident_llama,
)
from tests.unit.weights_fixtures import save_pretrained_dir, tiny_llama

#: The Feature 32 harness fixture, bound under its own name so pytest collects it here.
harness = _fixtures.harness

OTHER_ID = 2
OTHER_NAME = "other-model"


class _Reached(Exception):
    """Raised by the stand-in `load_model`: the auto-load got past every refusal."""


async def _add_other_row(h) -> None:
    session = h.repo.session
    session.add(make_model(id=OTHER_ID, name=OTHER_NAME, repo_id="org/other",
                           status=ModelStatus.READY, quantization=QuantizationType.FP16))
    await session.commit()


def _attach(h) -> None:
    """An active attachment whose address check reads the real resident model."""
    from millm.services.weight_handles import address_fingerprint

    attachment = attach_holder(h.weights, h.clock)
    attachment.model = LoadedModelState().current.model
    attachment.addresses = address_fingerprint(attachment.model)


async def _commit(h, n: int) -> None:
    """`n` real write cycles through the holder path: quiesce, commit-begin, commit-end, resume."""
    for _ in range(n):
        await h.weights.quiesce(TRAINER, ATTACHMENT_ID)
        h.weights.commit_begin(TRAINER, ATTACHMENT_ID)
        h.weights.commit_end(TRAINER, ATTACHMENT_ID)
        h.weights.resume(TRAINER, ATTACHMENT_ID)


async def _detach(h) -> None:
    await h.weights.detach(TRAINER, ATTACHMENT_ID)
    assert h.weights.is_attached() is False


async def _save(h, step: int = 5):
    return await WeightsCheckpointService(h.models).save(
        TRAINER, ATTACHMENT_ID, manifest_fields(step=step))


async def _trained_then_detached(h, commits: int = 2):
    """The F-A scenario: a trainer commits, then detaches without saving."""
    loaded = resident_llama()
    await _add_other_row(h)
    _attach(h)
    await _commit(h, commits)
    await _detach(h)
    return loaded


def _stop_at_load(h) -> AsyncMock:
    stub = AsyncMock(side_effect=_Reached("reached load_model"))
    h.models.load_model = stub
    return stub


class TestRefused:
    async def test_auto_load_after_detach_with_unsaved_commits_is_409_and_nothing_moves(
        self, harness
    ):
        h = harness
        loaded = await _trained_then_detached(h, commits=2)
        with pytest.raises(WeightsUnsavedError) as info:
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        exc = info.value
        assert (exc.status_code, exc.code) == (409, "WEIGHTS_UNSAVED")
        assert exc.details == {
            "operation": "auto_load",
            "resident": RESIDENT_NAME,
            "resident_model_id": RESIDENT_ID,
            "unsaved_commits": 2,
            "requested_model": OTHER_NAME,
            "requested_model_id": OTHER_ID,
            "recovery": [
                "POST /api/weights/save (as the attachment holder: re-attach first)",
                f'POST /api/weights/load {{"model_id": {OTHER_ID}}}',
                f"POST /api/models/{OTHER_ID}/load",
            ],
        }
        assert f"'{RESIDENT_NAME}' carries 2 commit(s)" in exc.message
        assert f"'{OTHER_NAME}'" in exc.message and "POST /api/weights/save" in exc.message
        # Nothing was unloaded or loaded: the same resident object, no load-path call at all.
        assert LoadedModelState().current is loaded
        assert h.names() == []
        assert h.barrier.snapshot().steps_since_load == 2
        assert (await h.repo.get_by_id(OTHER_ID)).status == ModelStatus.READY

    async def test_commits_after_a_save_are_counted_from_the_save(self, harness):
        """The count is commits SINCE THE LAST SAVE: 2 before, a save, 1 after -> 1, not 3."""
        h = harness
        resident_llama()
        await _add_other_row(h)
        _attach(h)
        await _commit(h, 2)
        saved = await _save(h)
        await _commit(h, 1)
        await _detach(h)
        with pytest.raises(WeightsUnsavedError) as info:
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        assert info.value.details["unsaved_commits"] == 1
        # The resident is named by the save now, and the refusal names that served name.
        assert info.value.details["resident"] == saved.name
        assert h.names() == []


class TestAllowed:
    async def test_after_a_save_the_auto_load_proceeds(self, harness):
        h = harness
        resident_llama()
        await _add_other_row(h)
        _attach(h)
        await _commit(h, 2)
        await _save(h)
        await _detach(h)
        stub = _stop_at_load(h)
        with pytest.raises(_Reached):
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        assert stub.await_count == 1 and stub.await_args.args[0] == OTHER_ID

    async def test_nothing_committed_the_auto_load_proceeds(self, harness):
        h = harness
        resident_llama()
        await _add_other_row(h)
        stub = _stop_at_load(h)
        with pytest.raises(_Reached):
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        assert stub.await_count == 1

    async def test_naming_the_resident_is_served_without_a_load(self, harness):
        h = harness
        loaded = await _trained_then_detached(h, commits=2)
        row = await h.models.load_model_and_wait(RESIDENT_ID, timeout=0.5)
        assert row.id == RESIDENT_ID
        assert LoadedModelState().current is loaded and h.names() == []

    async def test_an_explicit_weights_api_row_reload_replaces_them(self, harness, tmp_path):
        """`POST /api/weights/load {"model_id": 1}` (operator, D-01) is never refused for unsaved
        commits, and once it has replaced the weights the auto-load is allowed again."""
        h = harness
        await _trained_then_detached(h, commits=2)
        save_pretrained_dir(tiny_llama(), tmp_path / "base")
        result = await WeightsCheckpointService(h.models).load(
            OPERATOR, LoadRequest(model_id=RESIDENT_ID), attachment_id=None, lease_id=None)
        assert result.mode == "full_reload" and result.steps_since_load == 0
        stub = _stop_at_load(h)
        with pytest.raises(_Reached):
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        assert stub.await_count == 1

    async def test_an_explicit_checkpoint_load_replaces_them(self, harness):
        """An operator in-place load of a saved checkpoint replaces 1 unsaved commit; the
        identity it sets records 0 and the counter restarts, so nothing is unsaved."""
        h = harness
        resident_llama()
        await _add_other_row(h)
        _attach(h)
        saved = await _save(h)
        await _commit(h, 1)
        await _detach(h)
        with pytest.raises(WeightsUnsavedError):
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        result = await WeightsCheckpointService(h.models).load(
            OPERATOR, LoadRequest(name=saved.name), attachment_id=None, lease_id=None)
        assert result.mode == "in_place"
        stub = _stop_at_load(h)
        with pytest.raises(_Reached):
            await h.models.load_model_and_wait(OTHER_ID, timeout=0.5)
        assert stub.await_count == 1

    async def test_the_management_load_is_not_refused(self, harness):
        """The Feature-1 management load is the operator's explicit replacement: it never reads
        the unsaved count (the worker is stubbed; the point is that no refusal fires)."""
        h = harness
        await _trained_then_detached(h, commits=2)
        worker_calls: list[tuple] = []
        h.models._load_worker = lambda *a, **_k: worker_calls.append(a)
        await h.models.load_model(OTHER_ID)
        h.models._executor.shutdown(wait=True)
        assert "load_model" in h.names()
        assert worker_calls and worker_calls[0][0] == OTHER_ID


class TestTheCountIsDerived:
    """`resident_identity.unsaved_commits` from the counter and the identity alone."""

    @staticmethod
    def _loaded(at_identity=None):
        identity = None if at_identity is None else ResidentIdentity(
            name="m:run:step-1", base_model="m", origin="checkpoint", checkpoint_dir="/c",
            manifest_sha256="0" * 64, expected_parent=None, commits_at_identity=at_identity)
        return SimpleNamespace(weights_identity=identity, model_name="m")

    @pytest.mark.parametrize("steps, at, expected", [
        (3, None, 3),     # a plain row load: every commit since the load is unsaved
        (0, None, 0),
        (5, 2, 3),        # saved at 2, committed to 5
        (2, 2, 0),        # just saved
        (1, 4, 1),        # the counter restarted after the identity: no save since, all unsaved
    ])
    def test_the_rule(self, steps, at, expected):
        assert resident_identity.unsaved_commits(steps, self._loaded(at)) == expected

    @pytest.mark.parametrize("steps", [None, True, "3", 2.0])
    def test_a_counter_that_is_not_an_int_is_zero(self, steps):
        assert resident_identity.unsaved_commits(steps, self._loaded()) == 0

    def test_nothing_resident_is_zero(self):
        LoadedModelState()._loaded = None
        assert resident_identity.unsaved_commits(7) == 0


class TestWiring:
    """The guard sits in the one auto-load body, after torn, and in no explicit load."""

    @staticmethod
    def _calls(method) -> list[ast.Call]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "_refuse_if_unsaved_commits"]

    def test_load_model_and_wait_calls_it_once_for_auto_load(self):
        calls = self._calls(ModelService.load_model_and_wait)
        assert len(calls) == 1 and calls[0].args[0].value == "auto_load"

    @pytest.mark.parametrize("name", ["load_model", "precheck_load", "unload_model"])
    def test_no_explicit_operation_calls_it(self, name):
        assert self._calls(getattr(ModelService, name)) == []
