"""Operator settings stored in the database (Feature 34, migration 025).

One row per setting, keyed by name. A secret's ``value`` is ALWAYS the encrypted envelope from
`millm.core.secret_box` — never plaintext; there is no flag that allows otherwise, so no code path
can store a secret in the clear by passing the wrong flag. Today the only row is ``hf_token``.

``verified_at`` / ``verified_user`` record the last successful check of the stored value against
its service (for ``hf_token``, Hugging Face ``whoami``); ``last_check_error`` the reason the last
check FAILED (``invalid`` / ``network``, migration 026). All three clear when the value changes,
unless the new value is the one that was just checked (a Test of the typed token, then Save).
"""

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_user: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_check_error: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
