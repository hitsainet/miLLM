"""The Batch API (Feature 26): batch files, batches, one row per input line; batch-marked probe events.

Additive only. ``batch_rows`` carries the composite primary key ``(batch_id, line_no)`` that makes
"no recorded row runs twice" (FR-26.3.4) a database fact. ``probe_events`` gains ``origin`` (NOT
NULL, server default ``'live'`` — true of every existing row, because no batch API existed before
this migration), ``batch_id`` and ``batch_line``, and a partial unique index so a batch row re-run
after a crash cannot record a second event (FR-26.4.8, T-70).

The lease ID is never a column (029 FR-29.1.6); ``batches.lease_mode`` records only whose lease.

Downgrade drops everything this adds, in reverse order. It deletes all batch state: take a manual
database dump first (FTDD §11 "Rollback").

Revision ID: 019
Revises: 018
Create Date: 2026-10-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

_STATUSES = (
    "validating", "in_progress", "finalizing", "completed", "failed", "cancelling", "cancelled",
    "expired",
)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.create_table(
        "batch_files",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("purpose", sa.String(length=16), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="processed"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(_in("purpose", ("batch", "batch_output")), name="ck_batch_files_purpose"),
        sa.CheckConstraint(
            _in("status", ("processed", "error", "deleted", "expired")),
            name="ck_batch_files_status",
        ),
    )
    op.create_index("ix_batch_files_created", "batch_files", ["created_at"])
    op.create_index("ix_batch_files_expires", "batch_files", ["expires_at"])

    op.create_table(
        "batches",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("completion_window", sa.String(length=16), nullable=False),
        sa.Column(
            "model_id", sa.Integer(), sa.ForeignKey("models.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "input_file_id", sa.String(length=32),
            sa.ForeignKey("batch_files.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column(
            "output_file_id", sa.String(length=32),
            sa.ForeignKey("batch_files.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column(
            "error_file_id", sa.String(length=32),
            sa.ForeignKey("batch_files.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("pack", sa.Boolean(), nullable=False),
        sa.Column("metadata", JSONVariant, nullable=True),
        sa.Column("errors", JSONVariant, nullable=True),
        sa.Column("request_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_expires_after_s", sa.Integer(), nullable=False),
        sa.Column("waiting_reason", sa.String(length=48), nullable=True),
        sa.Column("lease_mode", sa.String(length=8), nullable=True),
        sa.Column("chunks_recorded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("in_progress_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finalizing_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelling_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(_in("status", _STATUSES), name="ck_batches_status"),
    )
    op.create_index("ix_batches_status_created", "batches", ["status", "created_at"])
    op.create_index("ix_batches_created", "batches", ["created_at"])

    op.create_table(
        "batch_rows",
        sa.Column(
            "batch_id", sa.String(length=32), sa.ForeignKey("batches.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("line_no", sa.Integer(), primary_key=True),
        sa.Column("custom_id", sa.String(length=512), nullable=True),
        sa.Column("kind", sa.String(length=12), nullable=True),
        sa.Column("byte_offset", sa.BigInteger(), nullable=False),
        sa.Column("byte_length", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=12), nullable=False),
        sa.Column("result", JSONVariant, nullable=True),
        sa.Column("packed", sa.Boolean(), nullable=True),
        sa.Column("chunk_seq", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            _in("state", ("pending", "invalid", "done", "failed")), name="ck_batch_rows_state"
        ),
        sa.CheckConstraint(
            "kind IS NULL OR " + _in("kind", ("scoring", "generation", "embedding", "probe")),
            name="ck_batch_rows_kind",
        ),
    )
    op.create_index(
        "ix_batch_rows_batch_state_line", "batch_rows", ["batch_id", "state", "line_no"]
    )

    op.add_column(
        "probe_events",
        sa.Column("origin", sa.String(length=8), nullable=False, server_default="live"),
    )
    op.add_column("probe_events", sa.Column("batch_id", sa.String(length=32), nullable=True))
    op.add_column("probe_events", sa.Column("batch_line", sa.Integer(), nullable=True))
    op.create_index(
        "ix_probe_events_probe_origin_created", "probe_events", ["probe_id", "origin", "created_at"]
    )
    op.create_index(
        "uq_probe_events_batch_line",
        "probe_events",
        ["probe_id", "batch_id", "batch_line", "window"],
        unique=True,
        postgresql_where=sa.text("batch_id IS NOT NULL"),
        sqlite_where=sa.text("batch_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_probe_events_batch_line", table_name="probe_events")
    op.drop_index("ix_probe_events_probe_origin_created", table_name="probe_events")
    op.drop_column("probe_events", "batch_line")
    op.drop_column("probe_events", "batch_id")
    op.drop_column("probe_events", "origin")
    op.drop_index("ix_batch_rows_batch_state_line", table_name="batch_rows")
    op.drop_table("batch_rows")
    op.drop_index("ix_batches_created", table_name="batches")
    op.drop_index("ix_batches_status_created", table_name="batches")
    op.drop_table("batches")
    op.drop_index("ix_batch_files_expires", table_name="batch_files")
    op.drop_index("ix_batch_files_created", table_name="batch_files")
    op.drop_table("batch_files")
