"""The model steering lock, and the ways it used to leak.

A locked model is the ONLY model `/v1/models` advertises. So a lock that
outlives the steering session it was taken for does not degrade the service in
some small way — it collapses the catalogue to one entry for every OpenAI
client, silently, until somebody unlocks the row by hand. That is not
hypothetical: `gemma-2-2b-it` held the lock from 2026-05-12 to 2026-08-19 and
hid thirteen ready models the whole time, including one downloaded that morning.

These tests pin the three independent ways it survived: the startup reset that
cleared four kinds of stale state and forgot this one, the read that trusted the
flag without asking whether the model was even loaded, and a detach that
released the model it had in hand instead of the one actually holding the lock.
"""

import pytest
from sqlalchemy import text

from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.main import STALE_STATE_RESETS


@pytest.fixture
def repository(test_session) -> ModelRepository:
    return ModelRepository(test_session)


async def _locked_flag(session, model_id: int) -> bool:
    """Read `locked` straight from the row.

    Not through the repository: these two tests run a raw UPDATE that goes round
    the ORM, so the identity map still holds the pre-reset object and an ORM read
    would report the old value — the assertion would pass or fail on cache state
    rather than on what the statement did.
    """
    result = await session.execute(
        text("SELECT locked FROM models WHERE id = :id"), {"id": model_id}
    )
    return bool(result.scalar_one())


def _model(name: str, status: ModelStatus, locked: bool) -> dict:
    return {
        "name": name,
        "source": ModelSource.HUGGINGFACE,
        "repo_id": f"vendor/{name}",
        "quantization": QuantizationType.Q4,
        "cache_path": f"/data/models/{name}",
        "status": status,
        "locked": locked,
    }


class TestALockOnlyCountsWhileLoaded:
    """`get_locked_model` must ask whether the lock means anything."""

    async def test_a_locked_but_UNLOADED_model_does_not_hold_the_lock(self, repository):
        """The exact production row: locked, but status back to READY.

        `lock_model` refuses anything but a LOADED model, so this state is not
        reachable by any legitimate route — it is debris from a restart. Read as
        a live lock it pins the whole catalogue to one model.
        """
        await repository.create(**_model("stale", ModelStatus.READY, locked=True))
        assert await repository.get_locked_model() is None

    async def test_a_locked_LOADED_model_DOES_hold_the_lock(self, repository):
        """The control. Without this the test above passes on a gutted method."""
        created = await repository.create(**_model("live", ModelStatus.LOADED, locked=True))
        held = await repository.get_locked_model()
        assert held is not None and held.id == created.id

    async def test_a_stale_lock_does_not_MASK_a_live_one(self, repository):
        """Debris on one row must not hide the model that is genuinely steering."""
        await repository.create(**_model("stale", ModelStatus.READY, locked=True))
        live = await repository.create(**_model("live", ModelStatus.LOADED, locked=True))
        held = await repository.get_locked_model()
        assert held is not None and held.id == live.id

    async def test_two_locked_rows_do_not_RAISE(self, repository):
        """A listing degrades to the wrong entry, never to a 500.

        The previous `scalar_one_or_none()` raised MultipleResultsFound here,
        which surfaces as a 500 on `/v1/models` for every client at once.
        """
        await repository.create(**_model("one", ModelStatus.LOADED, locked=True))
        await repository.create(**_model("two", ModelStatus.LOADED, locked=True))
        assert await repository.get_locked_model() is not None


class TestExclusiveLock:
    """"Only one model can be locked at a time" is enforced by the writer."""

    async def test_locking_one_model_RELEASES_every_other_lock(self, repository):
        first = await repository.create(**_model("first", ModelStatus.LOADED, locked=True))
        second = await repository.create(**_model("second", ModelStatus.LOADED, locked=False))

        await repository.set_exclusive_lock(second.id)

        assert (await repository.get_by_id(first.id)).locked is False
        assert (await repository.get_by_id(second.id)).locked is True

    async def test_clear_locks_releases_ALL_of_them_and_counts(self, repository):
        await repository.create(**_model("a", ModelStatus.LOADED, locked=True))
        await repository.create(**_model("b", ModelStatus.READY, locked=True))

        assert await repository.clear_locks() == 2
        assert await repository.get_locked_model() is None

    async def test_clear_locks_can_SPARE_one(self, repository):
        keep = await repository.create(**_model("keep", ModelStatus.LOADED, locked=True))
        await repository.create(**_model("drop", ModelStatus.LOADED, locked=True))

        assert await repository.clear_locks(except_model_id=keep.id) == 1
        assert (await repository.get_by_id(keep.id)).locked is True


class TestTheStartupReset:
    """The statement `main.py` actually runs, against the row it has to catch.

    BOUND TO THE REAL LIST, not to a copy of the SQL. A test holding its own
    string passes unchanged after the entry is deleted from the startup
    sequence — it would prove the statement works, while nothing ran it.
    """

    @property
    def RESET(self) -> str:
        matches = [sql for event, sql, _ in STALE_STATE_RESETS if event == "reset_stale_model_lock"]
        assert matches, (
            "no 'reset_stale_model_lock' entry in STALE_STATE_RESETS — nothing "
            "clears a leaked steering lock at startup, and a locked model is the "
            "only model /v1/models will advertise"
        )
        return matches[0]

    async def test_it_clears_a_lock_whose_status_is_ALREADY_ready(self, test_session, repository):
        """The row the sibling reset cannot see.

        The status reset is scoped `WHERE status IN ('loaded','loading')`. An
        earlier restart already set this row to READY and left `locked` true, so
        that filter drops the one row that still needs clearing — which is why
        this cannot be folded into it as an extra SET column.
        """
        stale = await repository.create(**_model("stale", ModelStatus.READY, locked=True))

        await test_session.execute(text(self.RESET))

        assert await _locked_flag(test_session, stale.id) is False

    async def test_the_sibling_status_reset_would_MISS_that_row(self, test_session, repository):
        """Names why the separate statement exists, so nobody merges them back."""
        stale = await repository.create(**_model("stale", ModelStatus.READY, locked=True))

        await test_session.execute(
            text(
                "UPDATE models SET status = 'ready', loaded_at = NULL, locked = false "
                "WHERE status IN ('loaded', 'loading')"
            )
        )

        assert await _locked_flag(test_session, stale.id) is True
