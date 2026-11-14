"""Old (miLLM) names still in use (040 FR-040.10, migration 040).

One row per `(kind, old_name)`; `kind` is `env`, `host` or `meta` (the single `observing_since`
row). Written once a minute by `miserver/services/legacy_names_store.py`; read by the detailed health
route. Names, times and counts only.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from miserver.db.base import Base


class LegacyNameUse(Base):
    __tablename__ = "legacy_name_uses"

    kind: Mapped[str] = mapped_column(String(8), primary_key=True)
    old_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    new_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default="0")
