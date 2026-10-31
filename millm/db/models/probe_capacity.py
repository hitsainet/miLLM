"""Per-model probe limits and measured probe capacity (Feature 37 follow-up, migration 031).

Operator decision 2026-10-21: "I want to know the actual limits, what drives them, and to be able
to adjust on a per model basis accordingly."

* `model_probe_limits` — what the operator set for ONE registered model: the count backstop and the
  two budget halves, each in milliseconds or as a percentage of that model's measured step time.
  A NULL field means "use the global default" (`PROBE_MAX_ARMED`; for a budget half,
  `PROBE_SLOWDOWN_BUDGET_PCT` of the measured step time, else the absolute
  `PROBE_MAX_OVERHEAD_MS_PER_PASS` / `PROBE_MAX_PREFILL_MS_PER_1K` — ENH-006 D83).
* `probe_capacity_measurements` — what was MEASURED for a model (by Hub repo and precision, so a
  re-registered row keeps its evidence) on a device: the model's own step time (decode ms per token,
  prefill ms per 1,000 prompt tokens) and the cost lines of a probe (own part per probe plus the
  shared part per hooked layer). Append-only; the newest row for a (repo, precision) is used.
  Devices are labelled by GPU UUID, never an index (1 + n GPUs).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base

#: ⚠ `none_as_null`: an absent cost line is SQL NULL, not the JSON value `null`.
JSONVariant = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")

BUDGET_UNITS = ("ms", "pct")


class ModelProbeLimits(Base):
    __tablename__ = "model_probe_limits"

    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), primary_key=True
    )
    max_armed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    decode_budget_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    decode_budget_unit: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    prefill_budget_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prefill_budget_unit: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # ── 039 S1 (migration 034): the count gate in behaviours, and the approval record ────────
    #: Was the count gate, in BEHAVIOURS (D13). ENH-006 D83 (2026-10-29): stored and readable for
    #: back-compat, and binds NOTHING — the slowdown budget decides how many probes fit.
    #: `max_armed` above stays as the per-model armed-PROBE backstop (migration 034 nulled every
    #: stored value, because a probe count is not a behaviour count, TD39-14).
    max_behaviours: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    #: `recommended-approved` (the explained recommendation, approved by its digest) or `operator`.
    source: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    #: `admin-ui`, or the miStudio approval id that carried the change.
    set_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_measurement_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("probe_capacity_measurements.id", ondelete="SET NULL",
                   name="model_probe_limits_approved_measurement_id_fkey"),
        nullable=True,
    )
    #: `{repo_id, quantization, dtype, devices[]}` of the measurement the approval rests on.
    #: ⚠ STALENESS COMPARES THIS, never measurement ids (TD39-15): a newer measurement under the
    #: same fingerprint is "a newer recommendation is available", not "stale".
    fingerprint: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    #: The approved recommendation, verbatim: values, reasons and digest.
    recommendation: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)


class ModelProbeLimitsHistory(Base):
    """Every change of one model's probe limits, old and new (039 S1, migration 034).

    ⚠ APPEND-ONLY BY SERVICE RULE, as `env_override_changes` is (migration 028 has no trigger):
    `probe_limits.set_limits` is the only writer and only ever `session.add`s a row;
    `test_probe_limits_history_append_only.py` walks `millm/` for any UPDATE or DELETE against it.
    """

    __tablename__ = "model_probe_limits_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    old: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    new: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    set_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_model_probe_limits_history_model", "model_id", "changed_at"),
    )


class ProbeCapacityMeasurement(Base):
    __tablename__ = "probe_capacity_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String(255), nullable=False)
    quantization: Mapped[str] = mapped_column(String(16), nullable=False)
    #: `cuda:GPU-…` for one card, the labels joined by `+` for a model split across cards.
    device: Mapped[str] = mapped_column(String(400), nullable=False)
    #: The model's own step time with nothing armed.
    decode_ms_per_token: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prefill_ms_per_1k: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    #: `{decode_own_ms, decode_fixed_ms, prefill_own_ms_per_1k, prefill_fixed_ms_per_1k, k?}`.
    dense: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    sparse: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    #: 039 (TD39-3, migration 034): per-rule cost lines `{mean: {...}, max: {...},
    #: attention: {...}}`, each in `dense`'s shape. `dense` stays the `mean` line for older readers;
    #: NULL for a measurement taken before 039 (the recommendation then says "estimated").
    lines: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    #: How each figure was obtained, in words (a seeded figure says it was derived from a record).
    method: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_probe_capacity_repo_quant", "repo_id", "quantization"),
    )
