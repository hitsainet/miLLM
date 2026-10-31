"""Batch API tables (Feature 26, FTDD §4).

* ``batch_files`` — an uploaded input file or an assembled output/error file. The BYTES live on the
  data volume under ``BATCH_FILES_DIR`` (FTDD TD4: 200 MB files would bloat the nightly dumps);
  this row is the metadata, the sha256 of the stored bytes, and the generated storage path.
* ``batches`` — one job, with OpenAI's eight statuses (CHECK constraint built from
  ``BatchStatus``) and every OpenAI timestamp.
* ``batch_rows`` — one row per input line. ⚠ The composite primary key ``(batch_id, line_no)`` is
  what makes "no recorded row runs twice" (FR-26.3.4) a database fact rather than a hope: a row
  is recorded only by moving it out of ``pending`` (``BatchRepository.record_chunk``), and a row
  that is not ``pending`` can never be recorded again.

⚠ The lease ID is NEVER a column (029 FR-29.1.6): ``lease_mode`` says whose lease the batch runs
under; the ID itself lives only in the runner's memory, and a restart forgets it, as Feature 29
forgets every lease (X-01).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base
from millm.core.batch_values import BatchStatus, FilePurpose, FileStatus, RowKind, RowState

JSONVariant = JSON().with_variant(JSONB(), "postgresql")


def _in(column: str, values: Any) -> str:
    return f"{column} IN ({', '.join(repr(v.value) for v in values)})"


class BatchFile(Base):
    """An uploaded input file, or an assembled output/error file."""

    __tablename__ = "batch_files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    purpose: Mapped[str] = mapped_column(String(16), nullable=False)
    #: As uploaded — METADATA ONLY. Never reaches the filesystem (FTDD §8).
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Relative to BATCH_FILES_DIR, generated. Never returned by the API.
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=FileStatus.PROCESSED.value,
        server_default=FileStatus.PROCESSED.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(_in("purpose", FilePurpose), name="ck_batch_files_purpose"),
        CheckConstraint(_in("status", FileStatus), name="ck_batch_files_status"),
        Index("ix_batch_files_created", "created_at"),
        Index("ix_batch_files_expires", "expires_at"),
    )


class Batch(Base):
    """One batch job (FR-26.1.2)."""

    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    completion_window: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Set at the end of validation, from the lines' single model (FR-26.2.7).
    model_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    input_file_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("batch_files.id", ondelete="SET NULL"), nullable=True
    )
    output_file_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("batch_files.id", ondelete="SET NULL"), nullable=True
    )
    error_file_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("batch_files.id", ondelete="SET NULL"), nullable=True
    )
    #: Resolved at create from the request or BATCH_PACK_DEFAULT (T-63).
    pack: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: OpenAI's `metadata`. The attribute is not `metadata`: that name is the declarative base's.
    batch_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSONVariant, nullable=True
    )
    #: OpenAI's `{object: "list", data: [...]}`, first BATCH_ERRORS_SHOWN entries.
    errors: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    request_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    request_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    request_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    output_expires_after_s: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Why an `in_progress` batch is not running a row; NULL while it runs and once it leaves.
    waiting_reason: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    #: `own` or `caller`. The lease ID itself is never persisted (029 FR-29.1.6).
    lease_mode: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    #: Chunks recorded so far; the next chunk's `chunk_seq`.
    chunks_recorded: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    in_progress_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finalizing_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expired_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelling_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(_in("status", BatchStatus), name="ck_batches_status"),
        Index("ix_batches_status_created", "status", "created_at"),
        Index("ix_batches_created", "created_at"),
    )


class BatchRow(Base):
    """One input line. Deleted with its batch's finalisation (FTDD §7)."""

    __tablename__ = "batch_rows"

    batch_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("batches.id", ondelete="CASCADE"), primary_key=True
    )
    #: 1-based input line number.
    line_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: NULL when the line could not be read far enough to find one.
    custom_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    #: NULL for an invalid line, whose kind was never decided.
    kind: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    byte_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    byte_length: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(12), nullable=False)
    #: The finished output line or error line.
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    packed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    chunk_seq: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint(_in("state", RowState), name="ck_batch_rows_state"),
        CheckConstraint(f"kind IS NULL OR {_in('kind', RowKind)}", name="ck_batch_rows_kind"),
        Index("ix_batch_rows_batch_state_line", "batch_id", "state", "line_no"),
    )
