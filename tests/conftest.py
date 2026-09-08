"""
Shared pytest fixtures for miLLM tests.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Generator

import pytest
import pytest_asyncio
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from millm.db.base import Base


# Tests run against in-memory SQLite, which has no JSONB type.  Several models
# (models.config_json, profiles.steering) use PostgreSQL JSONB; without this
# compiler override, Base.metadata.create_all() fails on SQLite with
# "SQLiteTypeCompiler has no attribute visit_JSONB".  Render JSONB as plain
# JSON when compiling for SQLite — production (PostgreSQL) still uses native
# JSONB.  Registered once at import time; harmless if already registered.
@compiles(JSONB, "sqlite")
def _compile_jsonb_as_json_on_sqlite(element, compiler, **kw):  # noqa: ANN001
    return "JSON"


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


# ── TEMPORARY DIAGNOSTIC ────────────────────────────────────────────────────
# Route registration silently stops working partway through a CI run: at 5% of
# the suite `create_app()` serves /v1 requests, and by 70% `register_routes()`
# on a bare app adds nothing. It does not reproduce on any developer machine
# after matching CI's pytest, plugins, command, environment and a llama_cpp
# stub, so the only place that can answer is CI itself.
#
# This reports the FIRST test after which the routers are empty, plus enough
# module identity to say why. It never fails a test and never raises — a
# diagnostic that breaks the run it is diagnosing is useless.
#
# REMOVE once the cause is found.

_ROUTE_PROBE_FIRED = []


def pytest_runtest_teardown(item, nextitem):  # noqa: D103
    if _ROUTE_PROBE_FIRED:
        return
    try:
        import sys

        health_mod = sys.modules.get("millm.api.routes.system.health")
        routes_pkg = sys.modules.get("millm.api.routes")
        if health_mod is None or routes_pkg is None:
            return  # not imported yet; nothing to say

        bound = getattr(routes_pkg, "health_router", None)
        live = getattr(health_mod, "router", None)
        n_bound = len(getattr(bound, "routes", []) or [])
        n_live = len(getattr(live, "routes", []) or [])
        if n_bound and n_live and bound is live:
            return  # healthy

        _ROUTE_PROBE_FIRED.append(item.nodeid)
        mocked = [
            name
            for name, mod in list(sys.modules.items())
            if name.startswith("millm.") and type(mod).__name__ in {"MagicMock", "Mock"}
        ]
        print(
            "\n\n===== ROUTE PROBE: routers went empty =====\n"
            f"  after test      : {item.nodeid}\n"
            f"  bound router    : id={id(bound)} routes={n_bound}\n"
            f"  live module rtr : id={id(live)} routes={n_live}\n"
            f"  same object     : {bound is live}\n"
            f"  routes pkg type : {type(routes_pkg).__name__}\n"
            f"  register_routes : {type(getattr(routes_pkg, 'register_routes', None)).__name__}\n"
            f"  mocked millm.*  : {mocked[:10]}\n"
            "===========================================\n",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - a probe must never break the run
        _ROUTE_PROBE_FIRED.append("probe-error")
        print(f"\n[ROUTE PROBE ERROR] {type(exc).__name__}: {exc}\n", flush=True)
