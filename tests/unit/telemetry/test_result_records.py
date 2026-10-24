"""Result log records (FR-38.16; FTDD §6.5; FTASKS 8.7, 8.8), through the real record path.

* AT08: the receiver gets one `mi.probe.result` record per recorded result — the stored object
  flattened under `mi.probe.*`, with `source_event_id`, the trace and span ids on the record and
  the timestamp equal to `event_time`;
* a first-attempt failure is retried with the SAME `result_id` (the same bytes);
* Batch results are exported (origin `batch`) and `TELEMETRY_RESULT_LOG_ORIGINS` filters them;
* a failed write is exported as `not_persisted`, with no `source_event_id`;
* the log SDK (a private module in 1.45.1) constructs and encodes a record at the pinned version;
* no record is ever a `gen_ai.evaluation.result` event.

Control M16: unregister `telemetry.emit_results` from `ProbeEventService._publish`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from millm import telemetry
from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush


def _ns(rfc3339: str) -> int:
    moment = datetime.strptime(rfc3339, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1_000_000) * 1000


def _results(stub) -> list[dict]:
    return [r for r in stub.logs if r["event_name"] == "mi.probe.result"]


async def test_at08_the_receiver_gets_every_recorded_result(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        r = h.post_chat(cap.BODY, {})
        assert r.status_code == 200
        rows = await h.event_rows(r.json()["id"])
        flush()
    records = _results(stub)
    assert len(records) == len(rows) == 3
    by_id = {rec["attributes"]["mi.probe.result_id"]: rec for rec in records}
    for row in rows:
        rec = by_id[row["result_id"]]
        attrs = rec["attributes"]
        assert attrs["mi.probe.source_event_id"] == row["id"]
        assert attrs["mi.probe.persistence"] == "persisted"
        assert attrs["mi.probe.probe_id"] == row["probe_id"]
        assert attrs["mi.probe.window"] == row["window"]
        assert attrs["mi.probe.origin"] == "live"
        assert attrs["mi.probe.behaviour.key"] in (cap.ALPHA_KEY, cap.BETA_KEY)
        assert rec["trace_id"] == row["trace_id"] and rec["span_id"] == row["span_id"]
        assert rec["time_unix_nano"] == _ns(row["result"]["event_time"])
        assert rec["scope"] == "millm" and rec["body"] is None
        # A record never CLAIMS a sampled trace (the result cannot know it was sampled).
        assert rec["flags"] & 0x01 == 0
        for key, value in row["result"].items():
            if isinstance(value, (str, int, float, bool)) and not isinstance(value, dict):
                assert attrs.get(f"mi.probe.{key}") == value, key
    assert all(rec["event_name"] != "gen_ai.evaluation.result" for rec in stub.logs)


async def test_a_first_attempt_failure_is_retried_with_the_same_result_id(stub, telemetry_on):
    stub.mode = "fail_first"
    telemetry_on(TELEMETRY_RETRY_HORIZON_S=10.0)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        # The import's lifecycle record goes first (and takes the first failure); then the
        # receiver fails the NEXT logs request once more, which is the results'.
        assert stub.wait_for(lambda s: s.logs, timeout=15)
        with stub.lock:
            stub.raw.clear()
            stub.requests_by_path.clear()
        r = h.post_chat(cap.BODY, {})
        assert r.status_code == 200
        ids = {e["result_id"] for e in r.json()["millm_probe_verdicts"]}
        assert stub.wait_for(lambda s: _results(s), timeout=15), "never delivered"
    attempts = [body for path, body, _h in stub.raw if path == "/v1/logs"]
    assert len(attempts) == 2 and attempts[0] == attempts[1], "the retry resent the same bytes"
    assert all(rid.encode() in attempts[0] for rid in ids)
    assert {rec["attributes"]["mi.probe.result_id"] for rec in _results(stub)} == ids
    from millm.telemetry import health

    logs = health.SIGNALS["logs"]
    assert logs.failed == 0 and logs.exported == 2  # the import record, then the result


async def test_batch_results_are_exported_and_the_origin_filter_applies(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        await h.batch_line(cap.BODY)
        assert h.post_chat(cap.BODY, {}).status_code == 200
        flush()
    origins = sorted(rec["attributes"]["mi.probe.origin"] for rec in _results(stub))
    assert origins == ["batch", "live"]
    batch = next(rec for rec in _results(stub) if rec["attributes"]["mi.probe.origin"] == "batch")
    assert batch["attributes"]["mi.probe.batch_line"] == 1
    assert "mi.probe.session_id" not in batch["attributes"]

    stub.logs.clear()
    telemetry_on(TELEMETRY_RESULT_LOG_ORIGINS="live")
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        await h.batch_line(cap.BODY)
        assert h.post_chat(cap.BODY, {}).status_code == 200
        flush()
    assert [rec["attributes"]["mi.probe.origin"] for rec in _results(stub)] == ["live"]


async def test_a_failed_write_is_exported_not_persisted(stub, telemetry_on, monkeypatch):
    telemetry_on()
    from millm.db.repositories.probe_repository import ProbeEventRepository

    async def refuse(self, rows):
        raise RuntimeError("SENTINEL-DB-MESSAGE")

    monkeypatch.setattr(ProbeEventRepository, "create_many", refuse)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        r = h.post_chat(cap.BODY, {})
        assert r.status_code == 200
        flush()
    records = _results(stub)
    assert len(records) == 1
    attrs = records[0]["attributes"]
    assert attrs["mi.probe.persistence"] == "not_persisted"
    assert "mi.probe.source_event_id" not in attrs
    assert attrs["mi.probe.result_id"] == r.json()["millm_probe_verdicts"][0]["result_id"]
    assert b"SENTINEL-DB-MESSAGE" not in stub.all_bytes()


def test_the_log_sdk_constructs_and_encodes_a_record_at_the_pinned_version():
    """U10: `opentelemetry.sdk._logs` is private in 1.45.1; this is the canary for its API."""
    from importlib.metadata import version

    from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.sdk.resources import Resource

    from millm.telemetry.logs import build_records

    assert version("opentelemetry-sdk") == "1.45.1"
    records = build_records(
        [("mi.probe.result", {"mi.probe.result_id": "x", "mi.probe.score": 0.5}, "1" * 32,
          "2" * 16, True, 1_700_000_000_000_000_000, 1_700_000_000_000_000_001)],
        Resource({"service.name": "millm"}), "0.5.0", telemetry.SEMCONV_SCHEMA_URL,
    )
    wire = ExportLogsServiceRequest.FromString(encode_logs(records).SerializeToString())
    rec = wire.resource_logs[0].scope_logs[0].log_records[0]
    assert rec.event_name == "mi.probe.result"
    assert rec.trace_id.hex() == "1" * 32 and rec.span_id.hex() == "2" * 16
    assert rec.time_unix_nano == 1_700_000_000_000_000_000
    assert {kv.key for kv in rec.attributes} == {"mi.probe.result_id", "mi.probe.score"}
    assert rec.body.WhichOneof("value") is None, "no body: everything is an attribute"


async def test_a_stateless_result_leaves_only_through_the_gate(stub, telemetry_on):
    """`emit_results` itself refuses a stateless result unless its caller passed the FR-38.17
    gate: a second path that one day records stateless results cannot export them by accident."""
    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        data = h.client.post("/api/probes/score", json={
            "probe_ids": [rows[0].id],
            "inputs": [{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3}]}).json()["data"]
    stateless = [{k: v for k, v in entry.items()} for item in data["results"]
                 for entry in item["verdicts"]]
    assert stateless and all(e["origin"] == "stateless" for e in stateless)
    assert telemetry.emit_results(stateless) == 0
    assert telemetry.emit_results(stateless, stateless_permitted=True) == len(stateless)
    flush()
    assert len([r for r in stub.logs if r["event_name"] == "mi.probe.result"]) == len(stateless)
