"""The only user of `opentelemetry.sdk._logs` (FTDD §2.4 U10, D38.43). Imported lazily, on the
export thread, only when the logs signal is on.

The log SDK is still a PRIVATE module in 1.45.1 (development status). Keeping every use of its
types here means a change of that API breaks one module and its construct-and-export test
(`tests/unit/telemetry/test_result_records.py`), not the pipeline.

⚠ NO BRIDGE FROM PYTHON `logging` (D38.37): nothing here installs a handler; records are built
only from attributes the allowlist already passed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import INVALID_SPAN, NonRecordingSpan, SpanContext, TraceFlags


def build_records(items: Iterable[tuple[Any, ...]], resource: Any, version: str,
                  schema_url: str) -> list[ReadableLogRecord]:
    """`(event_name, attrs, trace_id, span_id, sampled, timestamp_ns, observed_ns)` → records.

    The trace and span ids ride on the record through an explicit, non-recording span context
    built from the RESULT's ids — never `get_current()`, whose context is not this record's.
    """
    scope = InstrumentationScope("millm", version, schema_url)
    out = []
    for event_name, attrs, trace_id, span_id, sampled, timestamp_ns, observed_ns in items:
        if trace_id:
            context = trace_api.set_span_in_context(NonRecordingSpan(SpanContext(
                trace_id=int(trace_id, 16),
                span_id=int(span_id, 16) if span_id else 0,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED if sampled else 0),
            )))
        else:
            context = trace_api.set_span_in_context(INVALID_SPAN)
        record = LogRecord(
            timestamp=timestamp_ns,
            observed_timestamp=observed_ns,
            context=context,
            severity_text="INFO",
            severity_number=SeverityNumber.INFO,
            body=None,
            attributes=dict(attrs),
            event_name=event_name,
        )
        out.append(ReadableLogRecord(log_record=record, resource=resource,
                                     instrumentation_scope=scope))
    return out
