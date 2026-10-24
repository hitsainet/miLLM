"""Carrier rows for checkpoint and instantiated weights (Feature 32, FTASKS 5.1).

A full reload of a checkpoint goes through the existing load path, which loads a ROW
(`LoadedModel.model_id` is a row id). So a checkpoint directory needs a row of its own: a
"carrier" whose `weights_origin` says the bytes came from the weights API, not from a download
or a hand-registered directory. NULL keeps every existing row's meaning, so there is no backfill.

A string with a CHECK constraint rather than a native enum (FTDD T2): a PostgreSQL enum value
added by ALTER TYPE cannot be used in the transaction that adds it. `instantiated` stays in the
constraint although instantiate is deferred (addendum, scope): it is free now and a migration
later.

The same CHECK is declared in `Model.__table_args__`, or the pg_catalog parity guard in
tests/schema/test_schema_guards.py sees the ORM and the migrations diverge.

Revision ID: 021
Revises: 020
Create Date: 2026-10-14
"""

from alembic import op
import sqlalchemy as sa

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None

CONSTRAINT = "ck_models_weights_origin"
CONDITION = "weights_origin IS NULL OR weights_origin IN ('checkpoint', 'instantiated')"


def upgrade() -> None:
    op.add_column("models", sa.Column("weights_origin", sa.String(length=16), nullable=True))
    op.create_check_constraint(CONSTRAINT, "models", CONDITION)


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT, "models", type_="check")
    op.drop_column("models", "weights_origin")
