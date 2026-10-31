"""Record which window a probe verdict read, and whether its threshold was calibrated for it.

A probe scores every token in a window and takes the MEAN. That window was the whole request —
the person's prompt AND the model's reply — so a long reply dragged the average down and the same
conversation scored differently depending on how much the model happened to say. Measured on the
live system 2026-09-30: the same sentence about a brother losing his flat scored **+9.86 (fires)**
against an 8-token reply and **-0.32 (silent)** against a full one.

One probe now reports several windows, because the prompt says something about the USER and the
response says something about the MODEL, and averaging them together answers neither.

⚠ ``window`` IS ``NOT NULL DEFAULT 'all'`` AND THAT BACKFILL IS HONEST. Every event ever recorded
before this migration was scored over the whole request, so ``'all'`` is what those rows actually
were — the default states a fact, it does not invent one.

That distinction is the entire point, and this estate has been on the wrong side of it: a
``chat_format`` column was once added ``NOT NULL DEFAULT 'auto'``, which made every historical row
claim it had used the tokenizer's real template when nothing had. A column added to stop a silent
misattribution introduced one. The test is not "is there a sensible default" but "was the default
true of the rows it is about to be written onto". Here it was; there it was not.

``provisional`` defaults FALSE on the same reasoning: every historical verdict was read under the
probe's own scope, which is the window its threshold was calibrated for.

Revision ID: 017
Revises: 016
Create Date: 2026-09-30
"""

from alembic import op
import sqlalchemy as sa

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_events",
        sa.Column("window", sa.String(length=16), nullable=False, server_default="all"),
    )
    op.add_column(
        "probe_events",
        sa.Column(
            "provisional", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )


def downgrade() -> None:
    op.drop_column("probe_events", "provisional")
    op.drop_column("probe_events", "window")
