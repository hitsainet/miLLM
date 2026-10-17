"""Several SAEs attached at once each get an active attachment row (migration 024).

On the RTX 3090 (2026-10-16, Qwen2.5-7B, SAEs at L20 and L25) the second `POST
/api/saes/{id}/attach` answered 500 `IntegrityError` while the SAE was hooked and steering:
migration 002's `idx_single_active_attachment` allowed ONE active row in the whole table, and
Feature 12's multi-SAE attach never widened it. The ORM declared it with `postgresql_where` only,
so SQLite created no index the suite could hit; the replacement carries `sqlite_where` as well,
and these tests run against what `create_all` builds from the ORM.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from millm.db.models.model import ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.db.repositories.sae_repository import SAERepository


@pytest.fixture
async def ids(test_session):
    model = await ModelRepository(test_session).create(
        name="m", source=ModelSource.HUGGINGFACE, repo_id="o/m", quantization=QuantizationType.FP16,
        cache_path="/tmp/m", params="1B", architecture="causal-lm", disk_size_mb=1,
        estimated_memory_mb=1, trust_remote_code=False, status=ModelStatus.LOADED,
    )
    repo = SAERepository(test_session)
    for sae_id in ("sae-a", "sae-b"):
        await repo.create(id=sae_id, repository_id=f"o/{sae_id}", name=sae_id, d_in=16,
                          d_sae=32, cache_path=f"/tmp/{sae_id}")
    return model.id, repo


async def test_two_saes_attached_at_once_both_get_a_row(ids):
    model_id, repo = ids
    await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=20)
    await repo.create_attachment(sae_id="sae-b", model_id=model_id, layer=25)
    assert await repo.is_sae_attached("sae-a") and await repo.is_sae_attached("sae-b")


async def test_one_sae_at_two_layers_gets_two_rows_and_detach_closes_both(ids):
    model_id, repo = ids
    await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=3)
    await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=5)
    closed = await repo.deactivate_attachment("sae-a")
    assert closed is not None and closed.layer == 5
    assert not await repo.is_sae_attached("sae-a")
    assert all(not a.is_active for a in await repo.get_attachments_for_sae("sae-a"))


async def test_the_same_sae_and_layer_cannot_be_active_twice(ids, test_session):
    model_id, repo = ids
    await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=20)
    with pytest.raises(IntegrityError):
        await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=20)
    await test_session.rollback()


async def test_a_closed_row_does_not_block_reattaching(ids):
    model_id, repo = ids
    await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=20)
    await repo.deactivate_attachment("sae-a")
    await repo.create_attachment(sae_id="sae-a", model_id=model_id, layer=20)
    assert await repo.is_sae_attached("sae-a")
