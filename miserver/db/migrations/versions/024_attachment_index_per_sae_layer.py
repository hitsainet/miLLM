"""One active attachment per (SAE, layer), not one in total.

Migration 002 created `idx_single_active_attachment`, a partial unique index on `is_active` WHERE
is_active — at most ONE active attachment row in the whole table, from the single-SAE era.
Feature 12 made several SAEs attachable at once and never widened it. On PostgreSQL the second
attach therefore raised `UniqueViolationError` AFTER the hook was installed and the in-memory
registry updated: `POST /api/saes/{id}/attach` answered 500 while the SAE was hooked and steering,
and `attach_set` (circuit serving) swallowed the same error, so every multi-SAE attachment past the
first had no database row. Found on the RTX 3090 on 2026-10-16 (Qwen2.5-7B, SAEs at L20 + L25).
SQLite never enforced it as written (the ORM gave only `postgresql_where`), so no unit test could.

The invariant the runtime actually holds is `AttachedSAEState`'s: one entry per `(sae_id, layer)`.

Revision ID: 024
Revises: 023
Create Date: 2026-10-16
"""

from alembic import op
import sqlalchemy as sa

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None

_OLD = "idx_single_active_attachment"
_NEW = "uq_active_attachment_sae_layer"


def upgrade() -> None:
    op.drop_index(_OLD, table_name="sae_attachments")
    op.create_index(
        _NEW, "sae_attachments", ["sae_id", "layer"], unique=True,
        postgresql_where=sa.text("is_active = true"),
        sqlite_where=sa.text("is_active = 1"),
    )


def downgrade() -> None:
    # The old index admits one active row in total: keep the most recent, close the rest, as a
    # restart would (startup deactivates every attachment anyway).
    op.execute(
        """
        UPDATE sae_attachments SET is_active = false, detached_at = now()
        WHERE is_active = true AND id <> (
            SELECT id FROM sae_attachments WHERE is_active = true
            ORDER BY attached_at DESC, id DESC LIMIT 1
        )
        """
    )
    op.drop_index(_NEW, table_name="sae_attachments")
    op.execute(
        f"CREATE UNIQUE INDEX {_OLD} ON sae_attachments (is_active) WHERE is_active = true"
    )
