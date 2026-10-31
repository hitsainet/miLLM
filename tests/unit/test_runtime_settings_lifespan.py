"""Feature 35: `lifespan` applies the stored overrides FIRST, before anything reads a setting.

Run through the real `main.lifespan` over a database holding two override rows. Fails when the
load line is deleted (the values stay the environment's) and when it is moved below
`setup_logging()` (logging is configured from the environment's LOG_LEVEL, and anything built
before the load captures the environment's values).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

from millm.db.models.env_override import EnvOverride
from tests.unit.batch_fixtures import batch_db  # noqa: F401


async def test_overrides_are_in_force_before_logging_is_configured(monkeypatch, batch_db):
    import millm.db.base as db_base
    import millm.main as main
    from millm.core.config import settings

    async with batch_db() as session:
        session.add(EnvOverride(key="LOG_LEVEL", value="ERROR", updated_by="operator"))
        session.add(EnvOverride(key="MAX_PENDING_REQUESTS", value=4, updated_by="operator"))
        await session.commit()

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main, "remove_stale_temp_dirs", lambda: [])
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    seen: dict[str, object] = {}
    real_setup = main.setup_logging

    def spying_setup_logging() -> None:
        seen["LOG_LEVEL"] = settings.LOG_LEVEL
        seen["MAX_PENDING_REQUESTS"] = settings.MAX_PENDING_REQUESTS
        real_setup()

    monkeypatch.setattr(main, "setup_logging", spying_setup_logging)

    async with main.lifespan(MagicMock()):
        assert seen == {"LOG_LEVEL": "ERROR", "MAX_PENDING_REQUESTS": 4}, (
            "logging was configured before the overrides were applied"
        )
        assert settings.MAX_PENDING_REQUESTS == 4
        assert logging.getLogger("millm.api").getEffectiveLevel() == logging.ERROR


def test_the_name_lifespan_calls_is_the_real_function():
    """A local stub named `load_runtime_overrides_on_startup` in main.py would satisfy the test."""
    import millm.main as main
    from millm.services import runtime_settings

    assert main.load_runtime_overrides_on_startup is runtime_settings.load_runtime_overrides_on_startup
