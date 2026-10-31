"""`ModelRepository.upsert_checkpoint_carrier` (032 FTASKS 5.2; addendum D-26; M34)."""

import pytest
from sqlalchemy import func, select

from millm.core.errors import LocalPathRegisteredError, ModelNameTakenError
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from tests.support.factories import make_model

DIR = "/ck/run/step-000005"


async def test_creates_one_carrier_and_reuses_it(test_session):
    repo = ModelRepository(test_session)
    first = await repo.upsert_checkpoint_carrier(local_path=DIR, name="b:run:step-5")
    assert (first.weights_origin, first.source, first.cache_path) == ("checkpoint",
                                                                      ModelSource.LOCAL, DIR)
    assert first.quantization == QuantizationType.FP16 and first.status == ModelStatus.READY
    again = await repo.upsert_checkpoint_carrier(local_path=DIR, name="b:run:step-5",
                                                 trust_remote_code=True)
    assert again.id == first.id and again.trust_remote_code is True
    count = await test_session.scalar(select(func.count()).select_from(Model))
    assert count == 1


async def test_a_plain_local_row_on_the_directory_is_refused(test_session):
    test_session.add(make_model(id=3, name="step-000005", source=ModelSource.LOCAL, repo_id=None,
                                local_path=DIR, cache_path=DIR))
    await test_session.commit()
    with pytest.raises(LocalPathRegisteredError) as info:
        await ModelRepository(test_session).upsert_checkpoint_carrier(local_path=DIR,
                                                                     name="b:run:step-5")
    assert info.value.details["model_id"] == 3


async def test_a_name_held_by_another_row_is_refused(test_session):
    test_session.add(make_model(id=4, name="b:run:step-5"))
    await test_session.commit()
    with pytest.raises(ModelNameTakenError):
        await ModelRepository(test_session).upsert_checkpoint_carrier(local_path=DIR,
                                                                     name="b:run:step-5")
