"""Probe sessions, turns and opt-in transcripts; session columns on events; behaviour key (Feature 36).

`probe_sessions` groups one conversation's requests; `probe_session_turns` places each request in
it (turn index, salted continuation hashes); `probe_transcripts` holds the newest turn's text only
while the operator has turned recording on (D5, off by default).

`probe_events` gains `session_id` (SET NULL, so a session delete can keep its verdict history) and
`turn_index`. NO BACKFILL of either: NULL means "not recorded", which is true of every earlier row.

`probes.behaviour_key` is a PROJECTION of `definition.behaviour.key` (the definition stays the
source of truth, `millm/db/models/probe.py`), so it is the one column this migration backfills: a
projection that disagreed with the stored document would be a second source of truth. The backfill
reads the document in Python, so it behaves the same on SQLite and PostgreSQL.

Revision ID: 029
Revises: 028
Create Date: 2026-10-21
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def _behaviour_key(definition):  # noqa: ANN001, ANN202 - JSON on one dialect, text on another
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except ValueError:
            return None
    if not isinstance(definition, dict):
        return None
    behaviour = definition.get("behaviour")
    if not isinstance(behaviour, dict):
        return None
    key = behaviour.get("key")
    return key[:64] if isinstance(key, str) and key else None


def upgrade() -> None:
    op.create_table(
        "probe_sessions",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("client_session_id", sa.String(length=128), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("predecessor_id", sa.String(length=24), nullable=True),
        sa.Column("link_ambiguous", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("served_model", sa.String(length=255), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("n_turns", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["predecessor_id"], ["probe_sessions.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_probe_sessions_client_session_id", "probe_sessions",
                    ["client_session_id"], unique=False)
    op.create_index("ix_probe_sessions_last_seen_at", "probe_sessions", ["last_seen_at"],
                    unique=False)

    op.create_table(
        "probe_session_turns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=24), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("prefix_hash", sa.String(length=64), nullable=False),
        sa.Column("continuation_hash", sa.String(length=64), nullable=True),
        sa.Column("weak_hash", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=8), server_default="live", nullable=False),
        sa.Column("served_model", sa.String(length=255), nullable=True),
        sa.Column("has_transcript", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("transcript_incomplete", sa.Boolean(), server_default=sa.false(),
                  nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["probe_sessions.id"], ondelete="CASCADE"),
    )
    op.create_index("uq_probe_session_turns_request_id", "probe_session_turns", ["request_id"],
                    unique=True)
    op.create_index("ix_probe_session_turns_session_turn", "probe_session_turns",
                    ["session_id", "turn_index"], unique=False)
    op.create_index("ix_probe_session_turns_prefix_hash", "probe_session_turns",
                    ["prefix_hash"], unique=False)
    op.create_index("ix_probe_session_turns_continuation_hash", "probe_session_turns",
                    ["continuation_hash"], unique=False)
    op.create_index("ix_probe_session_turns_weak_hash", "probe_session_turns", ["weak_hash"],
                    unique=False)

    op.create_table(
        "probe_transcripts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("turn_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=24), nullable=False),
        sa.Column("user_text", sa.Text(), nullable=True),
        sa.Column("reply_text", sa.Text(), nullable=True),
        sa.Column("n_choices", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["turn_id"], ["probe_session_turns.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["probe_sessions.id"], ondelete="CASCADE"),
    )
    op.create_index("uq_probe_transcripts_turn_id", "probe_transcripts", ["turn_id"],
                    unique=True)
    op.create_index("ix_probe_transcripts_session_id", "probe_transcripts", ["session_id"],
                    unique=False)
    op.create_index("ix_probe_transcripts_created_at", "probe_transcripts", ["created_at"],
                    unique=False)

    op.add_column("probe_events", sa.Column("session_id", sa.String(length=24), nullable=True))
    op.add_column("probe_events", sa.Column("turn_index", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "probe_events_session_id_fkey", "probe_events", "probe_sessions",
        ["session_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_probe_events_session_turn", "probe_events",
                    ["session_id", "turn_index"], unique=False)

    op.add_column("probes", sa.Column("behaviour_key", sa.String(length=64), nullable=True))
    op.create_index("ix_probes_behaviour_key", "probes", ["behaviour_key"], unique=False)

    bind = op.get_bind()
    probes = sa.table(
        "probes",
        sa.column("id", sa.String()),
        sa.column("definition", sa.JSON()),
        sa.column("behaviour_key", sa.String()),
    )
    for probe_id, definition in bind.execute(sa.select(probes.c.id, probes.c.definition)).all():
        key = _behaviour_key(definition)
        if key is not None:
            bind.execute(
                probes.update().where(probes.c.id == probe_id).values(behaviour_key=key)
            )


def downgrade() -> None:
    op.drop_index("ix_probes_behaviour_key", table_name="probes")
    op.drop_column("probes", "behaviour_key")
    op.drop_index("ix_probe_events_session_turn", table_name="probe_events")
    op.drop_constraint("probe_events_session_id_fkey", "probe_events", type_="foreignkey")
    op.drop_column("probe_events", "turn_index")
    op.drop_column("probe_events", "session_id")
    op.drop_index("ix_probe_transcripts_created_at", table_name="probe_transcripts")
    op.drop_index("ix_probe_transcripts_session_id", table_name="probe_transcripts")
    op.drop_index("uq_probe_transcripts_turn_id", table_name="probe_transcripts")
    op.drop_table("probe_transcripts")
    op.drop_index("ix_probe_session_turns_weak_hash", table_name="probe_session_turns")
    op.drop_index("ix_probe_session_turns_continuation_hash", table_name="probe_session_turns")
    op.drop_index("ix_probe_session_turns_prefix_hash", table_name="probe_session_turns")
    op.drop_index("ix_probe_session_turns_session_turn", table_name="probe_session_turns")
    op.drop_index("uq_probe_session_turns_request_id", table_name="probe_session_turns")
    op.drop_table("probe_session_turns")
    op.drop_index("ix_probe_sessions_last_seen_at", table_name="probe_sessions")
    op.drop_index("ix_probe_sessions_client_session_id", table_name="probe_sessions")
    op.drop_table("probe_sessions")
