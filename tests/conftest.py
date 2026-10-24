"""
Shared pytest fixtures for miLLM tests.
"""

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Generator

import pytest
import pytest_asyncio
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from millm.db.base import Base

# 038 (FTID §3.3): every probe result a test builds is checked against the conditional rules of
# FTDD §5.1, and a violation RAISES rather than being logged as it is in production.
os.environ.setdefault("MILLM_STRICT_RESULTS", "1")


# Tests run against in-memory SQLite, which has no JSONB type.  Several models
# (models.config_json, profiles.steering) use PostgreSQL JSONB; without this
# compiler override, Base.metadata.create_all() fails on SQLite with
# "SQLiteTypeCompiler has no attribute visit_JSONB".  Render JSONB as plain
# JSON when compiling for SQLite — production (PostgreSQL) still uses native
# JSONB.  Registered once at import time; harmless if already registered.
@compiles(JSONB, "sqlite")
def _compile_jsonb_as_json_on_sqlite(element, compiler, **kw):  # noqa: ANN001
    return "JSON"


@pytest.fixture(autouse=True)
def _release_model_load_slot() -> Generator[None, None, None]:
    """The one-load-at-a-time slot is process-wide, so a test that leaves a
    load 'in progress' (a stubbed executor never runs the worker's finally)
    must not make every later test's load busy."""
    from millm.services import model_service

    model_service._LOAD_SLOT["model_id"] = None
    yield
    model_service._LOAD_SLOT["model_id"] = None


@pytest.fixture(autouse=True)
def _fresh_lease_registry() -> Generator[None, None, None]:
    """The lease registry is process-wide (Feature 29). A lease one test takes must not
    refuse the next test's load, so every test starts and ends with a fresh registry."""
    from millm.services.model_lease import set_lease_registry

    set_lease_registry(None)
    yield
    set_lease_registry(None)


@pytest.fixture(autouse=True)
def _fresh_weight_barrier_and_service() -> Generator[None, None, None]:
    """The weight barrier and the weight service are process-wide (Feature 31). A barrier one
    test leaves `torn` would refuse every later test's request, and an attachment would refuse
    every later load, so both are dropped after every test (addendum §8)."""
    yield
    from millm.services.weight_attachment import set_weight_service
    from millm.services.weight_barrier import set_weight_barrier

    set_weight_barrier(None)
    set_weight_service(None)


@pytest.fixture(autouse=True)
def _no_saved_hf_token() -> Generator[None, None, None]:
    """The saved Hugging Face token is process memory (Feature 34). One test's save must not
    become the next test's credential, so every test starts and ends with none saved."""
    from millm.services import hf_token

    hf_token._reset_memory()
    hf_token._forget_check()
    yield
    hf_token._reset_memory()
    hf_token._forget_check()


@pytest.fixture(autouse=True)
def _no_runtime_overrides() -> Generator[None, None, None]:
    """Runtime overrides (Feature 35) are assigned onto the process-wide `settings` singleton.
    One test's override must not become the next test's configuration, so every test ends with
    each writable key back at the value it had BEFORE the test (taken here, ahead of any
    monkeypatch the test applies, so a monkeypatched baseline cannot outlive the test), the root
    log level restored, and the override state forgotten."""
    import logging

    from millm.core.config import settings
    from millm.services import runtime_settings

    before = {key: getattr(settings, key) for key in runtime_settings.RUNTIME_KEYS}
    root_level = logging.getLogger().level
    yield
    runtime_settings._forget_for_tests()
    for key, value in before.items():
        setattr(settings, key, value)
    logging.getLogger().setLevel(root_level)


@pytest.fixture(autouse=True)
def _structlog_never_caches_loggers(monkeypatch) -> Generator[None, None, None]:
    """A test that runs the real `main.lifespan` calls `setup_logging()`, which configures structlog
    with `cache_logger_on_first_use=True`. Every module-level logger used after that is frozen for
    the rest of the PROCESS, and `structlog.testing.capture_logs` in any later test then captures
    nothing — so a log-asserting test passed alone and failed whenever the CI shard split happened
    to put a lifespan test before it (`test_weight_barrier_states.py`, `test_weight_attachment.py`,
    `test_runtime_settings.py`; pre-existing on `main` 68a73375, found 2026-10-21 when new files
    re-balanced the shards). Caching is a production speed-up with no behaviour to test, so the suite
    forces it off: configuration still runs for real, it just never freezes a logger."""
    import structlog

    from millm.core import logging as millm_logging

    real = structlog.configure

    def configure(*args, **kwargs):
        kwargs["cache_logger_on_first_use"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(millm_logging.structlog, "configure", configure)
    yield


@pytest.fixture(autouse=True)
def _probe_cost_timer_is_injected(monkeypatch) -> Generator[None, None, None]:
    """Feature 37: arming measures each probe's cost with a wall-clock timer. Unit tests never
    depend on a wall clock (memory `check-ci-not-just-local-green`): every test sees a timer that
    reports 0 ms unless it injects its own figures. The real timer has its own direct test
    (`test_probe_cost.py::test_the_default_timer_really_times`)."""
    from millm.services import probe_cost

    monkeypatch.setattr(probe_cost, "_timer", lambda fn, k, device: 0.0)
    yield


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_engine():
    """Create a test database engine using SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # ⚠ SQLite ignores FOREIGN KEY constraints unless this pragma is set PER CONNECTION, so
    # without it every `ondelete="CASCADE"` in the repo is untested — a broken one passes the
    # whole suite and only fails in production, where PostgreSQL does enforce it. Added
    # 2026-09-27 while building Feature 024, when a probe cascade test passed against a cascade
    # that had never run.
    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_sqlite_foreign_keys(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        await session.rollback()



def pytest_ignore_collect(collection_path, config):
    """CI shards: with `CI_SHARD=<i>/<n>` set, skip every test file another shard owns.

    Unset (every local run, every other job) this ignores nothing. The split and its proof
    live in tests/support/ci_shard.py and tests/unit/test_ci_shards.py.
    """
    from pathlib import Path

    from tests.support import ci_shard

    return ci_shard.ignore(collection_path, config, Path(__file__).resolve().parent)
