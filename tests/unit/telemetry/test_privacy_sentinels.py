"""Privacy, AT10 (FR-38.21; FTDD §8; FTASKS 8.11) — THE privacy-critical test of the export path.

Every signal on, transcripts on, stateless export permitted. Unique sentinels are planted where
user text and secrets live — the system prompt, the user message, the reply, request headers
(including `Authorization`), an exception message raised inside a probe and inside a request, a
text input to the stateless score route, the acknowledgement reason — plus the pilot's
identifier-shaped value (a PUBLISHED test value only, D28) in a prompt. Then:

* the RAW BYTES of every request the receiver got contain no sentinel, and not the identifier
  whole or without its delimiters; no decoded string value (ids excluded) contains its last four
  digits;
* nothing unlisted was even ATTEMPTED (`attributes_dropped` stays 0): the code never builds a
  content attribute that the allowlist then has to catch;
* an `http://` endpoint is refused without `TELEMETRY_ALLOW_PLAINTEXT`; an `https://` receiver
  whose certificate is not trusted is refused (and accepted once its CA is configured).

Controls (M21): add a prompt attribute to the inference span (`attributes_dropped` turns red);
remove the allowlist AND add the prompt attribute (the raw-bytes check turns red).
"""

from __future__ import annotations

import datetime as dt
import re
import ssl

import pytest

from millm import telemetry
from millm.core.config import settings
from millm.telemetry import allowlist, health
from tests.support import capture_pre038_goldens as cap
from tests.support.otlp_stub import OtlpStub
from tests.unit.telemetry.conftest import flush, telemetry_settings

#: A published test value (the long-retired specimen number printed on sample cards), never a
#: real person's. Checked whole, without delimiters, and by its last four digits.
PILOT_VALUE = "078-05-1120"
SENTINELS = {
    "system": "SENTINEL-SYSTEM-PROMPT",
    "user": "SENTINEL-USER-MESSAGE",
    "reply": "SENTINEL-REPLY-TEXT",
    "header": "SENTINEL-CUSTOM-HEADER",
    "auth": "SENTINEL-AUTH-BEARER",
    "probe_exc": "SENTINEL-PROBE-EXCEPTION",
    "request_exc": "SENTINEL-REQUEST-EXCEPTION",
    "score_text": "SENTINEL-SCORE-TEXT",
    "stream_user": "SENTINEL-STREAM-USER",
    "completion": "SENTINEL-COMPLETION-PROMPT",
    "reason": "SENTINEL-ARM-REASON",
    "client_session": "SENTINEL-CLIENT-SESSION",
}
ID_LIKE = re.compile(r"^([0-9a-f]{16}|[0-9a-f]{32}|[0-9a-f]{64}|[0-9a-f-]{36}|"
                     r"[0-9T:.\-]+Z|pr_[0-9a-f]{12}|chatcmpl-[0-9a-f]+|cmpl-[0-9a-f]+)$")


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": f"{system} w1"},
            {"role": "user", "content": f"{user} w1 w2 {PILOT_VALUE}"}]


def _strings(stub) -> list[str]:
    out = []
    for item in stub.spans + stub.logs + stub.metric_points:
        out.extend(v for v in item["attributes"].values() if isinstance(v, str))
        out.extend(str(item.get(k)) for k in ("name", "event_name", "status_message")
                   if item.get(k))
    for resource in stub.resources:
        out.extend(v for v in resource.values() if isinstance(v, str))
    return out


async def test_at10_no_user_text_or_secret_leaves(stub, telemetry_on, monkeypatch):
    monkeypatch.setattr(settings, "PROBE_TRANSCRIPTS", True)
    telemetry_on(TELEMETRY_STATELESS_EXPORT_PERMITTED=True)
    real_encoder = cap._encoder

    def encoder(kind):
        if kind == "raise":
            def broken(_x):
                raise RuntimeError(f"{SENTINELS['probe_exc']} {PILOT_VALUE}")
            return broken
        return real_encoder(kind)

    monkeypatch.setattr(cap, "_encoder", encoder)
    failing = cap.ProbeSpec("probe-gamma", layer=1, threshold=0.5, weight=0.1,
                            behaviour="b-gamma", encoder="raise")
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA, cap.BETA, failing])
        real_record = h.svc._probe_record

        async def record(*args, **kwargs):
            kwargs["reply_text"] = f"{SENTINELS['reply']} {PILOT_VALUE} " + (
                kwargs.get("reply_text") or "")
            return await real_record(*args, **kwargs)

        monkeypatch.setattr(h.svc, "_probe_record", record)
        headers = {"X-Custom-Note": SENTINELS["header"],
                   "Authorization": f"Bearer {SENTINELS['auth']}",
                   "X-miLLM-Session": SENTINELS["client_session"]}
        chat = h.post_chat({**cap.BODY, "messages": _messages(SENTINELS["system"],
                                                              SENTINELS["user"])}, headers)
        assert chat.status_code == 200, chat.text
        assert len(chat.json()["millm_probe_verdicts"]) == 4  # alpha 1, beta 2, gamma 1
        text = h.stream_chat({**cap.BODY, "messages": _messages(SENTINELS["system"],
                                                                SENTINELS["stream_user"])},
                             headers)
        assert "millm_probe_verdicts" in text
        completion = h.client.post("/v1/completions", json={
            "model": "tiny", "prompt": f"{SENTINELS['completion']} w1 {PILOT_VALUE}",
            "max_tokens": 2, "temperature": 0.0}, headers=headers)
        assert completion.status_code == 200, completion.text
        score = h.client.post("/api/probes/score", json={
            "probe_ids": [rows[0].id], "export_results": True,
            "inputs": [{"text": f"{SENTINELS['score_text']} w1 {PILOT_VALUE}"}]})
        assert score.status_code == 200, score.text
        assert score.json()["data"]["export"]["records_queued"] >= 1
        arm = h.client.post(f"/api/probes/{rows[1].id}/arm",
                            json={"acknowledge_below_rung2": True,
                                  "reason": f"{SENTINELS['reason']} {PILOT_VALUE}"})
        assert arm.status_code < 500, arm.text

        def boom(*a, **k):
            raise RuntimeError(f"{SENTINELS['request_exc']} {PILOT_VALUE}")

        monkeypatch.setattr(h.svc, "_format_chat_messages", boom)
        from fastapi.testclient import TestClient

        h._fresh_socket_throttle()
        failed = TestClient(h.client.app, raise_server_exceptions=False).post(
            "/v1/chat/completions", json=cap.BODY, headers=headers)
        assert failed.status_code >= 500
        flush()

    # Something of every signal was exported, so the scan below is not vacuous.
    assert stub.spans and stub.logs and stub.metric_points
    assert any(r["attributes"].get("mi.probe.reason") == "encoder_failed" for r in stub.logs)
    raw = stub.all_bytes()
    for name, sentinel in SENTINELS.items():
        assert sentinel.encode() not in raw, f"{name} sentinel left the deployment"
    assert PILOT_VALUE.encode() not in raw
    assert PILOT_VALUE.replace("-", "").encode() not in raw
    last_four = PILOT_VALUE[-4:]
    leaking = [s for s in _strings(stub) if last_four in s and not ID_LIKE.match(s)]
    assert leaking == [], leaking
    # ⚠ Nothing unlisted was attempted: the code never offered a content attribute.
    assert allowlist.attributes_dropped() == 0
    snapshot = str(health.snapshot())
    assert all(s not in snapshot for s in SENTINELS.values())


async def test_a_plaintext_endpoint_is_refused_without_the_trusted_local_choice(stub):
    telemetry.configure(telemetry_settings(stub.endpoint, TELEMETRY_ALLOW_PLAINTEXT=False))
    assert telemetry._PIPELINE is None
    telemetry.emit_lifecycle(None, "import")
    assert health.SIGNALS["logs"].view(60, 300)["state"] == "unavailable"
    assert health.SIGNALS["logs"].view(60, 300)["last_error"]["type"] == "PlaintextNotAllowed"
    assert stub.raw == []


def _self_signed(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    import ipaddress

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "otlp-stub")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "stub.pem", tmp_path / "stub.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context, cert_path


@pytest.mark.parametrize("trusted", [False, True])
async def test_an_untrusted_certificate_is_refused(tmp_path, trusted):
    context, cert_path = _self_signed(tmp_path)
    tls_stub = OtlpStub(tls=context).start()
    try:
        extra = {"TELEMETRY_TLS_CA_FILE": str(cert_path)} if trusted else {}
        telemetry.configure(telemetry_settings(tls_stub.endpoint, TELEMETRY_ALLOW_PLAINTEXT=False,
                                               TELEMETRY_RETRY_HORIZON_S=0.0, **extra))
        assert telemetry._PIPELINE is not None
        assert telemetry._PIPELINE.cfg.transport == "tls"
        telemetry.emit_lifecycle(None, "import")
        flush()
        view = health.SIGNALS["logs"].view(60, 300)
        if trusted:
            assert view["exported"] == 1 and tls_stub.logs, view
        else:
            assert view["exported"] == 0 and view["failed"] == 1, view
            assert view["last_error"]["type"] == "SSLError"
            assert tls_stub.logs == []
    finally:
        tls_stub.stop()


# ── the backstops behind the allowlist ───────────────────────────────────────────


async def test_a_span_never_carries_events_a_status_message_or_unlisted_attributes(
        stub, telemetry_on, monkeypatch):
    """`_QueueingProcessor.on_end` rebuilds every span: no events (an exception event carries its
    message), no links, a status WITHOUT description, attributes through the allowlist. Planted
    directly on an SDK span, as a future caller or library might."""
    from opentelemetry.trace.status import Status, StatusCode

    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "leak=SENTINEL-RESOURCE")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "SENTINEL-SERVICE")
    pipeline = telemetry_on()
    ref = pipeline.start_span("chat", {"gen_ai.operation.name": "chat"}, None)
    span = ref.sdk_span
    span.add_event("exception", {"exception.message": "SENTINEL-EVENT"})
    span.set_attribute("gen_ai.input.messages", "SENTINEL-ATTRIBUTE")
    span.set_status(Status(StatusCode.ERROR, "SENTINEL-STATUS"))
    span.end()
    flush()
    got = next(s for s in stub.spans if s["name"] == "chat")
    assert got["events"] == 0 and got["status_message"] == "" and got["status_code"] == 2
    assert "gen_ai.input.messages" not in got["attributes"]
    raw = stub.all_bytes()
    for sentinel in (b"SENTINEL-EVENT", b"SENTINEL-ATTRIBUTE", b"SENTINEL-STATUS",
                     b"SENTINEL-RESOURCE", b"SENTINEL-SERVICE"):
        assert sentinel not in raw, sentinel
    assert stub.resources and all(set(r) <= set(allowlist.RESOURCE) for r in stub.resources)


async def test_no_exemplar_even_inside_a_sampled_span_context(stub, telemetry_on):
    """The meter provider's exemplar filter is ALWAYS OFF: a measurement recorded while a sampled
    span is current would otherwise attach that trace and span id to the metric point."""
    from opentelemetry import context as otel_context
    from opentelemetry import trace as trace_api

    pipeline = telemetry_on(TELEMETRY_LOGS_ENABLED=False)
    ref = pipeline.start_span("chat", {}, None)
    token = otel_context.attach(trace_api.set_span_in_context(ref.sdk_span))
    try:
        pipeline._histograms["millm.probe.score_margin"].record(
            0.25, attributes={"probe_id": "pr_0123456789ab", "window": "all"})
    finally:
        otel_context.detach(token)
    pipeline.end_span(ref)
    flush()
    margins = [p for p in stub.metric_points if p["name"] == "millm.probe.score_margin"]
    assert margins and all(p["exemplars"] == 0 for p in margins)


async def test_the_environment_proxy_and_ca_settings_are_not_used(stub, telemetry_on,
                                                                  monkeypatch):
    """`trust_env = False`: a proxy, CA bundle or netrc in the environment cannot redirect the
    export (only the operator's TELEMETRY_* settings decide where data goes)."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    telemetry_on()
    telemetry.emit_lifecycle(None, "import")
    flush()
    assert stub.logs, "the export went to the configured receiver, not the environment's proxy"


async def test_a_record_or_span_outside_the_allowlist_never_leaves(stub, telemetry_on):
    """Backstops for a FUTURE caller: an unlisted key handed to `emit_results` (say a result that
    one day carries `context_text`) and a span with an unlisted name are dropped before the queue
    — and counted."""
    pipeline = telemetry_on()
    result = {"schema": "mi.probe.result/v1", "result_id": "0192f2a0-0000-7000-8000-000000000001",
              "origin": "live", "probe_id": "pr_0123456789ab", "window": "all",
              "outcome": "completed", "event_time": "2026-10-23T12:00:00.000001Z",
              "context_text": "SENTINEL-CONTEXT-TEXT", "reason": "free text SENTINEL-REASON"}
    assert telemetry.emit_results([result]) == 1
    ref = pipeline.start_span("SENTINEL-SPAN-NAME", {}, None)
    pipeline.end_span(ref)
    flush()
    raw = stub.all_bytes()
    assert b"SENTINEL" not in raw
    assert [r["attributes"].get("mi.probe.result_id") for r in stub.logs] == [result["result_id"]]
    assert not [s for s in stub.spans if "SENTINEL" in s["name"]]
    assert allowlist.attributes_dropped() == 3  # context_text, reason, the span
