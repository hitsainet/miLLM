"""The probe coverage counters (FR-38.18, FR-38.19; FTDD §6.4): in-process, capped, always on.

Prometheus renders them whether or not OpenTelemetry export is on (`render_prometheus`); when
metrics export is on, observable OTel instruments read the same store (`pipeline.py`).

⚠ COUNTED IN THE BUILDER, ONCE PER MINTED RESULT (`probe_result.build_probe_results`), so a result
is counted before any sampling, persistence or export can lose it. `expected` +1 and the outcome
counter +1 with its reason, for every result; threshold crossings ONLY for a completed thresholded
result whose verdict is true — a skip is never a "no", and never a crossing (control M18).

⚠ BOUNDED (D38.40). Each instrument admits at most `TELEMETRY_SERIES_CAP` distinct attribute
tuples; the next tuple folds into the single overflow series `{"otel.metric.overflow": true}`
(the OTel specification's overflow attribute — the SDK 1.45.1 has no cardinality limit of its
own) and `series_overflow` +1. Identifier dimensions (`probe_id`) are admitted only for probes the
builder actually evaluated at that moment (armed for live and batch traffic, the scored set for
the stateless route); anything else folds into overflow. Never a request, result, trace, user id,
free text or token position (FR-18 of the OTel BRD).

No OpenTelemetry import (D38.17).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from millm.telemetry import allowlist

EXPECTED = "millm.probe.evaluations.expected"
EVALUATIONS = "millm.probe.evaluations"
CROSSINGS = "millm.probe.threshold_crossings"
STATELESS = "millm.probe.stateless.evaluations"
OVERFLOW_INSTRUMENT = "millm.probe.series_overflow"

#: Instrument → (Prometheus name, help, dimension keys).
INSTRUMENTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    EXPECTED: (
        "millm_probe_evaluations_expected_total",
        "Probe results minted (one per armed probe and window per request)",
        ("probe_id", "behaviour", "kind", "window", "engine", "execution_mode", "origin"),
    ),
    EVALUATIONS: (
        "millm_probe_evaluations_total",
        "Probe results by outcome and reason",
        ("probe_id", "behaviour", "kind", "window", "engine", "execution_mode", "origin",
         "outcome", "reason"),
    ),
    CROSSINGS: (
        "millm_probe_threshold_crossings_total",
        "Completed thresholded probe results whose verdict was true",
        ("probe_id", "behaviour", "kind", "window", "origin"),
    ),
    STATELESS: (
        "millm_probe_stateless_evaluations_total",
        "Probe results of the stateless score route by outcome and reason",
        ("probe_id", "behaviour", "kind", "window", "outcome", "reason"),
    ),
}
OVERFLOW_ATTRS: tuple[tuple[str, Any], ...] = (("otel.metric.overflow", True),)


class CounterStore:
    """`(instrument, attribute tuple) → int` behind one lock, with a per-instrument series cap."""

    def __init__(self, series_cap: int = 2000) -> None:
        self._lock = threading.Lock()
        self.series_cap = int(series_cap)
        self._values: dict[str, dict[tuple[Any, ...], int]] = {name: {} for name in INSTRUMENTS}
        self.series_overflow = 0

    # ── writing ──────────────────────────────────────────────────────────────────

    def _admit(self, instrument: str, attrs: tuple[Any, ...]) -> tuple[Any, ...]:
        """The tuple itself when it is already a series or there is room; else overflow."""
        series = self._values[instrument]
        if attrs in series or len(series) < self.series_cap:
            return attrs
        self.series_overflow += 1
        return OVERFLOW_ATTRS

    def add(self, instrument: str, attrs: dict[str, Any], value: int = 1,
            admitted: bool = True) -> None:
        names = INSTRUMENTS[instrument][2]
        kept, dropped = allowlist.filter("metric", {k: attrs.get(k) for k in names})
        with self._lock:
            if not admitted or dropped or len(kept) != len(names):
                # An identifier that is not armed, or a value the allowlist refused: never its
                # own series.
                self.series_overflow += 1
                key = OVERFLOW_ATTRS
            else:
                key = self._admit(instrument, tuple((k, kept[k]) for k in names))
            series = self._values[instrument]
            series[key] = series.get(key, 0) + value

    def observe(self, results: Iterable[Any], admitted_ids: set[str] | None = None) -> None:
        """Count each minted result once (module doc). `admitted_ids`: the probe ids the builder
        evaluated; `None` admits every id (tests)."""
        for result in results:
            data = result if isinstance(result, dict) else _fields(result)
            behaviour = (data.get("behaviour") or {}).get("key") or "none"
            base = {
                "probe_id": data.get("probe_id"),
                "behaviour": behaviour,
                "kind": data.get("kind", "probe"),
                "window": data.get("window"),
                "engine": data.get("engine"),
                "execution_mode": data.get("execution_mode"),
                "origin": data.get("origin"),
            }
            outcome = {"outcome": data.get("outcome"), "reason": data.get("reason") or "none"}
            admitted = admitted_ids is None or data.get("probe_id") in admitted_ids
            if data.get("origin") == "stateless":
                self.add(STATELESS, {**base, **outcome}, admitted=admitted)
                continue
            self.add(EXPECTED, base, admitted=admitted)
            self.add(EVALUATIONS, {**base, **outcome}, admitted=admitted)
            # ⚠ ONLY a completed result with a threshold and verdict true (M18): a skip, a failure
            # or an interruption is never a crossing, and neither is a probe that only ranks.
            if (data.get("outcome") == "completed" and data.get("threshold") is not None
                    and data.get("verdict") is True):
                self.add(CROSSINGS, base, admitted=admitted)

    def count_overflow(self, n: int = 1) -> None:
        with self._lock:
            self.series_overflow += n

    # ── reading ──────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, dict[tuple[Any, ...], int]]:
        with self._lock:
            return {name: dict(series) for name, series in self._values.items()}

    def series_count(self, instrument: str) -> int:
        with self._lock:
            return len(self._values[instrument])

    def total(self, instrument: str, **match: Any) -> int:
        with self._lock:
            return sum(v for k, v in self._values[instrument].items()
                       if all(dict(k).get(a) == b for a, b in match.items()))

    def render_prometheus(self) -> list[str]:
        """The store as Prometheus text lines (label values escaped). Present with telemetry off."""
        snap = self.snapshot()
        lines: list[str] = []
        for instrument, (prom, help_text, _names) in INSTRUMENTS.items():
            lines.append(f"# HELP {prom} {help_text}")
            lines.append(f"# TYPE {prom} counter")
            for attrs, value in sorted(snap[instrument].items(), key=lambda kv: repr(kv[0])):
                labels = ",".join(f'{_label(k)}="{_escape(v)}"' for k, v in attrs)
                lines.append(f"{prom}{{{labels}}} {value}" if labels else f"{prom} {value}")
            lines.append("")
        lines.append("# HELP millm_probe_series_overflow_total Probe measurements folded into "
                     "the overflow series (cap or unadmitted identifier)")
        lines.append("# TYPE millm_probe_series_overflow_total counter")
        lines.append(f"millm_probe_series_overflow_total {self.series_overflow}")
        lines.append("")
        return lines

    def reset(self, series_cap: int | None = None) -> None:
        with self._lock:
            if series_cap is not None:
                self.series_cap = int(series_cap)
            self._values = {name: {} for name in INSTRUMENTS}
            self.series_overflow = 0


def _label(name: str) -> str:
    return name.replace(".", "_")


def _escape(value: Any) -> str:
    text = "true" if value is True else "false" if value is False else str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fields(result: Any) -> dict[str, Any]:
    behaviour = getattr(result, "behaviour", None)
    return {
        "probe_id": getattr(result, "probe_id", None),
        "behaviour": {"key": getattr(behaviour, "key", None)} if behaviour is not None else None,
        "kind": getattr(result, "kind", "probe"),
        "window": getattr(result, "window", None),
        "engine": getattr(result, "engine", None),
        "execution_mode": getattr(result, "execution_mode", None),
        "origin": getattr(result, "origin", None),
        "outcome": getattr(result, "outcome", None),
        "reason": getattr(result, "reason", None),
        "threshold": getattr(result, "threshold", None),
        "verdict": getattr(result, "verdict", None),
    }


#: The process store. `configure()` sets its cap from `TELEMETRY_SERIES_CAP`.
STORE = CounterStore()
