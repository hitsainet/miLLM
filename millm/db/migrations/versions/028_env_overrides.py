"""Runtime overrides of environment settings and their history (Feature 35, ENH-003).

`env_overrides`: one row per setting the operator changed on the Admin UI's Settings page while
miLLM runs. The row outranks the environment variable until it is deleted ("Use the environment
value"). `env_override_changes`: the append-only history of those saves and resets, with the
in-force value before and after, who made the change and the optional reason.

No backfill: both tables start empty, which means "no override" — every setting keeps exactly the
value the environment gives it, as before this migration.

Revision ID: 028
Revises: 027
Create Date: 2026-10-20
"""

import sqlalchemy as sa
from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "env_overrides",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_by", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "env_override_changes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("previous_value", sa.JSON(), nullable=True),
        sa.Column("new_value", sa.JSON(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("approval_id", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column(
            "changed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_env_override_changes_key", "env_override_changes", ["key"], unique=False
    )
    op.create_index(
        "ix_env_override_changes_changed_at", "env_override_changes", ["changed_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_env_override_changes_changed_at", table_name="env_override_changes")
    op.drop_index("ix_env_override_changes_key", table_name="env_override_changes")
    op.drop_table("env_override_changes")
    op.drop_table("env_overrides")
