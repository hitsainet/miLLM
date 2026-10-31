"""`ModelService.precheck_load` makes every refusal `load_model` makes, changing nothing (W3-A).

032's full reload calls it inside the barrier BEFORE `mark_first_write`: a refusal found only
later, inside `load_model` or the resident's unload, comes after the first write and leaves the
weights `torn` over a model that was never touched. So each guard in `precheck_load` is
load-bearing on its own, and the full-reload tests cover only the placement and the load slot.

Review R2-A found four of its lines that no test caught (each mutation left the 032 load, attach
guard, torn-recovery and model-service suites green): the lease check, the attachment check, the
unloading check, and `reloading_resident` (only the RESIDENT row's LOADED status is excused).

MUTATION CONTROLS (R2-A, each verified red against this file, restore verified):
  P1 `_refuse_if_leased` removed from `precheck_load`          -> the lease test red
  P2 `_refuse_if_attached` removed                              -> the attachment test red
  P3 `_refuse_if_unloading` removed                             -> the unloading test red
  P4 `reloading_resident=True` (every LOADED row excused)       -> the LOADED-row test red
"""

from __future__ import annotations

import pytest

from millm.core.errors import (
    AlreadyAttachedError,
    ModelAlreadyLoadedError,
    ModelBusyError,
    ModelLeasedError,
)
from millm.db.models.model import ModelStatus, QuantizationType
from millm.ml.model_loader import LoadedModelState
from millm.services import model_service as model_service_module
from tests.support.factories import make_model
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    RESIDENT_ID,
    attach_holder,
    harness,
    resident_llama,
)

OTHER_ID = 2


async def _assert_nothing_changed(h, loaded) -> None:
    assert LoadedModelState().current is loaded
    assert model_service_module._LOAD_SLOT["model_id"] is None
    assert h.names() == [] and h.loader.loads == []
    assert (await h.repo.get_by_id(RESIDENT_ID)).status == ModelStatus.LOADED


async def test_the_resident_row_is_not_refused_as_already_loaded(harness):
    """The baseline the refusals below are measured against: reloading the resident passes."""
    loaded = resident_llama()
    model = await harness.models.precheck_load(RESIDENT_ID, under_barrier=False)
    assert model.id == RESIDENT_ID
    await _assert_nothing_changed(harness, loaded)


async def test_a_foreign_lease_on_the_resident_is_refused(harness):
    loaded = resident_llama()
    grant = await harness.models.acquire_lease(RESIDENT_ID, "someone", "eval run")
    try:
        with pytest.raises(ModelLeasedError):
            await harness.models.precheck_load(RESIDENT_ID)
        await _assert_nothing_changed(harness, loaded)
        # The holder's own lease ID lifts it.
        await harness.models.precheck_load(RESIDENT_ID, lease_id=grant.lease_id)
    finally:
        harness.models._leases.release(RESIDENT_ID, grant.lease_id)


async def test_an_attachment_is_refused(harness):
    loaded = resident_llama()
    attach_holder(harness.weights, harness.clock)
    with pytest.raises(AlreadyAttachedError):
        await harness.models.precheck_load(RESIDENT_ID)
    await _assert_nothing_changed(harness, loaded)


async def test_an_unload_in_progress_is_refused(harness):
    loaded = resident_llama()
    LoadedModelState()._unloading = True
    try:
        with pytest.raises(ModelBusyError) as exc:
            await harness.models.precheck_load(RESIDENT_ID)
        assert exc.value.details.get("unloading_model_id") == RESIDENT_ID
    finally:
        LoadedModelState()._unloading = False
    await _assert_nothing_changed(harness, loaded)


async def test_a_loaded_row_that_is_not_the_resident_is_refused(harness, test_session):
    """Only the resident's LOADED status is excused (it is unloaded first). Another row reading
    LOADED (an unload that cleared the loader and has not yet written its row) is refused, as
    `load_model` refuses it."""
    test_session.add(make_model(id=OTHER_ID, name="other", repo_id="org/other",
                                status=ModelStatus.LOADED, quantization=QuantizationType.FP16,
                                cache_path="/nonexistent"))
    await test_session.commit()
    loaded = resident_llama()
    with pytest.raises(ModelAlreadyLoadedError):
        await harness.models.precheck_load(OTHER_ID)
    await _assert_nothing_changed(harness, loaded)
