"""Per-model probe limits and measured probe capacity (Feature 37 follow-up, operator 2026-10-21).

`model_probe_limits`: one row per registered model the operator gave its own limits (NULL field =
the global default). `probe_capacity_measurements`: append-only measured step time and probe cost
lines per (Hub repo, precision, device).

SEEDED with the two models of the 2026-10-21 hardware record (`0xcc/reviews/probe_scaling_2026-10-21.md`,
RTX 3090 `GPU-247aa582-0d1b-e161-8156-983ed1fefc57`): probe cost lines fitted over N = 1..64 (decode
from the product's own per-pass timer, prefill from the arm-time measurement), and an APPROXIMATE decode
step time (disarmed request wall time / forward passes, a 64-token greedy request with a short prompt —
it includes HTTP and sampling, so it slightly overstates the forward). Prefill step time was not
measured; it stays NULL until the measurement route runs.

Revision ID: 031
Revises: 030
Create Date: 2026-10-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True),
                                                      "postgresql")

CARD = "cuda:GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
RECORD = "0xcc/reviews/probe_scaling_2026-10-21.md"
METHOD = (
    "seeded from the 2026-10-21 hardware record: probe cost lines fitted over N=1..64 (decode = the "
    "product's per-pass hook timer, prefill = the arm-time measurement); decode step time APPROXIMATE "
    "(disarmed 64-token request wall time / forward passes, includes HTTP and sampling); prefill "
    "step time not measured"
)
SEEDS = [
    {
        "repo_id": "LiquidAI/LFM2.5-1.2B-Instruct", "quantization": "FP16", "device": CARD,
        "decode_ms_per_token": 6.69, "prefill_ms_per_1k": None,
        "dense": {"decode_own_ms": 0.0253, "decode_fixed_ms": 0.0496,
                  "prefill_own_ms_per_1k": 0.0569, "prefill_fixed_ms_per_1k": 0.0411},
        "sparse": {"decode_own_ms": 0.1149, "decode_fixed_ms": 0.0752,
                   "prefill_own_ms_per_1k": 0.1465, "prefill_fixed_ms_per_1k": 0.0143, "k": 128},
    },
    {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct", "quantization": "FP16", "device": CARD,
        "decode_ms_per_token": 22.68, "prefill_ms_per_1k": None,
        "dense": {"decode_own_ms": 0.0236, "decode_fixed_ms": 0.0757,
                  "prefill_own_ms_per_1k": 0.1063, "prefill_fixed_ms_per_1k": 0.0538},
        "sparse": None,
    },
]


def upgrade() -> None:
    op.create_table(
        "model_probe_limits",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("max_armed", sa.Integer(), nullable=True),
        sa.Column("decode_budget_value", sa.Float(), nullable=True),
        sa.Column("decode_budget_unit", sa.String(length=4), nullable=True),
        sa.Column("prefill_budget_value", sa.Float(), nullable=True),
        sa.Column("prefill_budget_unit", sa.String(length=4), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("model_id"),
    )
    table = op.create_table(
        "probe_capacity_measurements",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("repo_id", sa.String(length=255), nullable=False),
        sa.Column("quantization", sa.String(length=16), nullable=False),
        sa.Column("device", sa.String(length=400), nullable=False),
        sa.Column("decode_ms_per_token", sa.Float(), nullable=True),
        sa.Column("prefill_ms_per_1k", sa.Float(), nullable=True),
        sa.Column("dense", JSONVariant, nullable=True),
        sa.Column("sparse", JSONVariant, nullable=True),
        sa.Column("method", sa.String(length=500), nullable=False),
        sa.Column("source", sa.String(length=200), nullable=True),
        sa.Column("measured_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_probe_capacity_repo_quant", "probe_capacity_measurements",
                    ["repo_id", "quantization"], unique=False)
    for seed in SEEDS:  # one at a time: an absent line must be SQL NULL, not JSON `null`
        row = {k: v for k, v in seed.items() if v is not None}
        op.bulk_insert(table, [{**row, "method": METHOD, "source": RECORD}])


def downgrade() -> None:
    op.drop_index("ix_probe_capacity_repo_quant", table_name="probe_capacity_measurements")
    op.drop_table("probe_capacity_measurements")
    op.drop_table("model_probe_limits")
