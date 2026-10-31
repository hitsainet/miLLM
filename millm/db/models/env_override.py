"""Runtime overrides of environment settings, and their change history (Feature 35, migration 028).

``env_overrides`` holds one row per overridden setting, keyed by its environment variable name. The
row's ``value`` replaces the environment's value while the server runs; deleting the row ("Use the
environment value") returns to the environment. Only the keys in
``millm.services.runtime_settings.RUNTIME_KEYS`` are ever written, and a row for any other key is
ignored (and logged) when the server loads them. Secrets are never writable, so never stored here.

A sibling of ``app_settings`` rather than rows in it: every ``app_settings`` value is an encrypted
secret, and that rule stays true with no exception for plaintext rows.

``env_override_changes`` is APPEND-ONLY: one row per save or reset, with the in-force value before
and after, who made it, and the operator's optional reason. Nothing updates or deletes its rows.
``approval_id`` names the approval an AI agent's telemetry change ran under (ENH-006 D76, actor
``agent:<name>``); it is null for the operator's own changes. A redacted key (the receiver endpoint)
is stored redacted in ``previous_value`` / ``new_value``.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base

#: Longest reason stored with a change.
REASON_MAX_CHARS = 500


class EnvOverride(Base):
    __tablename__ = "env_overrides"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(REASON_MAX_CHARS), nullable=True)


class EnvOverrideChange(Base):
    __tablename__ = "env_override_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: `set` (an override saved) or `reset` (the environment's value restored).
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    previous_value: Mapped[Any] = mapped_column(JSON, nullable=True)
    new_value: Mapped[Any] = mapped_column(JSON, nullable=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    approval_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(REASON_MAX_CHARS), nullable=True)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
