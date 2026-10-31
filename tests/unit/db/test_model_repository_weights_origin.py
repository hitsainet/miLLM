"""Feature 32 data layer: `models.weights_origin` and what it changes in name resolution.

A checkpoint carrier row is named `<base>:<run_id>:step-<N>`. The bare-name fallback in
`find_by_name` is `name LIKE '<name>:%'`, which matched those names, so a request for the base
model would silently resolve to a trained checkpoint whenever only the checkpoint row existed. And
the pattern was built from the raw name, so `_` and `%` in it acted as wildcards.

FIXTURES THAT DISAGREE WITH THE DEFECT (FTID §8.3): a row named `a_b` beside `axb:run:step-1`
(only a wildcard can join them), and a base name with ONLY a checkpoint row (only the exclusion
can say "not found").

MUTATION CONTROLS (each turns this file red; FTID M18, M19, M32):
  * drop `Model.weights_origin.is_(None)` from the fallback   -> the base-name tests fail
  * drop `escape=` / `escape_like` from the LIKE              -> the wildcard tests fail
  * drop the CHECK from `Model.__table_args__`                -> the constraint test fails
"""

import pytest
from sqlalchemy.exc import IntegrityError

from millm.db.models.model import Model, ModelSource, ModelStatus
from millm.db.repositories.model_repository import ModelRepository, escape_like
from tests.support.factories import make_model


async def _add(session, *, weights_origin: str | None = None, **overrides) -> Model:
    model = make_model(**overrides)
    model.weights_origin = weights_origin
    session.add(model)
    await session.commit()
    return model


def _checkpoint(**overrides) -> dict:
    return {
        "source": ModelSource.LOCAL,
        "repo_id": None,
        "weights_origin": "checkpoint",
        **overrides,
    }


class TestTheColumn:
    @pytest.mark.asyncio
    async def test_existing_rows_default_to_null(self, test_session):
        row = await _add(test_session, id=1, name="plain", cache_path="p1")
        assert (await ModelRepository(test_session).get_by_id(row.id)).weights_origin is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("origin", ["checkpoint", "instantiated"])
    async def test_both_origins_are_accepted(self, test_session, origin):
        row = await _add(
            test_session, id=1, name=f"b:{origin}:step-000001", cache_path="p1",
            **_checkpoint(weights_origin=origin, local_path=f"/ck/{origin}"),
        )
        assert row.weights_origin == origin

    @pytest.mark.asyncio
    async def test_any_other_value_is_refused_by_the_database(self, test_session):
        """The CHECK is declared on the ORM too, so SQLite's create_all enforces it."""
        with pytest.raises(IntegrityError, match="ck_models_weights_origin|CHECK"):
            await _add(
                test_session, id=1, name="b:r:step-000001", cache_path="p1",
                **_checkpoint(weights_origin="adapter", local_path="/ck/x"),
            )

    def test_the_check_is_declared_on_the_model(self):
        names = {c.name for c in Model.__table__.constraints}
        assert "ck_models_weights_origin" in names


class TestBareNameSkipsCheckpointRows:
    @pytest.mark.asyncio
    async def test_a_base_name_with_only_a_checkpoint_row_is_not_found(self, test_session):
        await _add(
            test_session, id=1, name="LFM2-1.2B:run:step-000001", cache_path="p1",
            **_checkpoint(local_path="/ck/run/step-000001"),
        )
        assert await ModelRepository(test_session).find_by_name("LFM2-1.2B") is None

    @pytest.mark.asyncio
    async def test_a_base_name_resolves_to_its_tagged_row_not_the_checkpoint(self, test_session):
        """A checkpoint row must not make an unambiguous GGUF bare name ambiguous."""
        await _add(
            test_session, id=1, name="LFM2-1.2B:Q4_K_M", repo_id="o/LFM2-1.2B",
            gguf_label="Q4_K_M", cache_path="p1",
        )
        await _add(
            test_session, id=2, name="LFM2-1.2B:run:step-000001", cache_path="p2",
            **_checkpoint(local_path="/ck/run/step-000001"),
        )
        assert (await ModelRepository(test_session).find_by_name("LFM2-1.2B")).id == 1

    @pytest.mark.asyncio
    async def test_a_checkpoint_row_is_reachable_by_its_full_name(self, test_session):
        await _add(
            test_session, id=1, name="LFM2-1.2B:run:step-000001", cache_path="p1",
            **_checkpoint(local_path="/ck/run/step-000001"),
        )
        found = await ModelRepository(test_session).find_by_name("LFM2-1.2B:run:step-000001")
        assert found is not None and found.id == 1

    @pytest.mark.asyncio
    async def test_the_exact_base_row_still_wins(self, test_session):
        await _add(test_session, id=1, name="LFM2-1.2B", cache_path="p1")
        await _add(
            test_session, id=2, name="LFM2-1.2B:run:step-000001", cache_path="p2",
            **_checkpoint(local_path="/ck/run/step-000001"),
        )
        assert (await ModelRepository(test_session).find_by_name("LFM2-1.2B")).id == 1


class TestTheFallbackIsLiteral:
    @pytest.mark.asyncio
    async def test_underscore_is_not_a_wildcard(self, test_session):
        await _add(test_session, id=1, name="axb:Q4_K_M", repo_id="o/axb", cache_path="p1")
        assert await ModelRepository(test_session).find_by_name("a_b") is None

    @pytest.mark.asyncio
    async def test_percent_is_not_a_wildcard(self, test_session):
        await _add(test_session, id=1, name="abc:Q4_K_M", repo_id="o/abc", cache_path="p1")
        assert await ModelRepository(test_session).find_by_name("a%") is None

    @pytest.mark.asyncio
    async def test_a_name_containing_wildcards_still_matches_itself(self, test_session):
        await _add(
            test_session, id=1, name="a_b%c\\d:Q4_K_M", repo_id="o/a_b", cache_path="p1"
        )
        found = await ModelRepository(test_session).find_by_name("a_b%c\\d")
        assert found is not None and found.id == 1

    def test_escape_like_escapes_the_escape_first(self):
        assert escape_like("a\\_%") == "a\\\\\\_\\%"


class TestAvailableModels:
    @pytest.mark.asyncio
    async def test_the_listing_includes_every_available_row(self, test_session):
        await _add(test_session, id=1, name="base", cache_path="p1")
        await _add(
            test_session, id=2, name="base:run:step-000001", cache_path="p2",
            status=ModelStatus.READY, **_checkpoint(local_path="/ck/1"),
        )
        await _add(
            test_session, id=3, name="base:run:step-000002", cache_path="p3",
            status=ModelStatus.LOADED, **_checkpoint(local_path="/ck/2"),
        )
        repo = ModelRepository(test_session)

        default = {m.id for m in await repo.get_available_models()}

        assert default == {1, 2, 3}, "the default listing must not change for existing callers"
