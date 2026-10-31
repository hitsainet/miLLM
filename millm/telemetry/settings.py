"""The effective telemetry configuration (FTDD §5.8, §5.7): validated, and a NON-SECRET view of it.

⚠ NO OPENTELEMETRY IMPORT. This module is read on every startup and by the health route, with
every switch off; importing the SDK here would break "off means free" (D38.17).

⚠ NEVER A SECRET IN THE VIEW. `public_view()` is what `GET /api/health/detailed` shows. It reports
`headers_set: bool`, never the headers; the endpoint with userinfo, query and fragment stripped
(a `https://user:token@host/?key=…` endpoint is a credential in a URL); certificate PATHS only as
set / not set. `TelemetrySettings` itself keeps the headers and the endpoint for the export
thread, and its `repr` hides both (and the client key path) so a log line of the object cannot
leak them either.
"""

from __future__ import annotations

import secrets as _secrets
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROTOCOL = "http/protobuf"
SIGNALS = ("traces", "metrics", "logs")
#: The origins a result log record may come from (`TELEMETRY_RESULT_LOG_ORIGINS`). `stateless` is
#: governed by `TELEMETRY_STATELESS_EXPORT_PERMITTED` and the request's opt-in, never this list.
LOG_ORIGINS = ("live", "batch")
#: `service.instance.id` when `TELEMETRY_SERVICE_INSTANCE_ID` is unset: random, ONCE PER PROCESS.
#: ENH-006 D75 made the configuration changeable while miLLM runs, so `load()` runs again on every
#: save; a fresh id per `load()` would make one process look like a new instance after each change.
PROCESS_INSTANCE_ID = _secrets.token_hex(8)


def strip_endpoint(endpoint: str | None) -> str | None:
    """The endpoint as shown: scheme, host, port and path; no userinfo, no query, no fragment."""
    if not endpoint:
        return None
    try:
        parts = urlsplit(endpoint.strip())
        host = parts.hostname or ""
        if ":" in host:  # an IPv6 literal
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        return "<invalid>"


def check_written_endpoint(text: str) -> str:
    """ENH-006 D75: the rule for an endpoint SAVED on the Settings page (or by an approved agent).

    An `http`/`https` URL with a host. ⚠ A userinfo part, a query string or a fragment is REFUSED,
    not stripped: credentials belong in `TELEMETRY_HEADERS` (environment only, secret), and a
    saved value is stored in the database and the change history, where a token in a URL would
    sit in clear text. Whether plaintext is allowed is NOT checked here: that is
    `TELEMETRY_ALLOW_PLAINTEXT`'s rule, applied when the pipeline is built, and shown in health
    (`PlaintextNotAllowed`), so the order of two saves never matters. Raises `ValueError` with the
    operator-facing reason; never echoes the value.
    """
    if any(c.isspace() for c in text) or any(ord(c) < 0x20 or ord(c) == 0x7F for c in text):
        raise ValueError("an endpoint may not contain spaces or control characters")
    try:
        parts = urlsplit(text)
        _ = parts.port  # raises on a malformed port
    except ValueError:
        raise ValueError("the endpoint is not a valid URL") from None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError("the endpoint must be an http:// or https:// URL with a host")
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        raise ValueError("the endpoint may not carry a user name or password; set receiver "
                         "credentials in TELEMETRY_HEADERS in the environment")
    if parts.query or "?" in text:
        raise ValueError("the endpoint may not carry a query string; set receiver credentials "
                         "in TELEMETRY_HEADERS in the environment")
    if parts.fragment or "#" in text:
        raise ValueError("the endpoint may not carry a fragment (#...)")
    return text


def check_written_origins(text: str) -> str:
    """ENH-006 D75: `TELEMETRY_RESULT_LOG_ORIGINS` as saved: a comma list of `live` and `batch`
    (either, both, or empty for neither), stored in the canonical order."""
    wanted = [o.strip().lower() for o in text.split(",") if o.strip()]
    unknown = sorted(set(wanted) - set(LOG_ORIGINS))
    if unknown:
        raise ValueError("the origins may be live and batch only (stateless is governed by "
                         "TELEMETRY_STATELESS_EXPORT_PERMITTED)")
    return ",".join(o for o in LOG_ORIGINS if o in wanted)


def check_written_label(text: str) -> str:
    """A resource-attribute value saved on the Settings page: printable ASCII, no spaces."""
    if not text:
        raise ValueError("enter a value")
    if not all(0x21 <= ord(c) <= 0x7E for c in text):
        raise ValueError("use printable characters without spaces")
    return text


def parse_headers(text: str | None) -> dict[str, str]:
    """`k=v,k=v` → dict. Blank entries are ignored; a malformed entry refuses the whole value
    (`InvalidHeaders`), because sending half of an operator's credentials is worse than none."""
    out: dict[str, str] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("InvalidHeaders")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if not key or any(c in key for c in " \t\r\n:") or any(c in value for c in "\r\n"):
            raise ValueError("InvalidHeaders")
        out[key] = value.strip()
    return out


@dataclass(frozen=True)
class TelemetrySettings:
    """Everything the pipeline needs, resolved once at `configure()`."""

    switches: dict[str, bool]
    #: The configuration refusal (an error CLASS name, never text) that leaves every enabled
    #: signal `unavailable`, or None.
    refusal: str | None
    #: `repr=False` (review R1 of S2): the URL can carry a credential (module doc).
    endpoint: str | None = field(repr=False)
    transport: str | None
    headers: dict[str, str] = field(repr=False)
    ca_file: str | None = None
    client_cert_file: str | None = None
    client_key_file: str | None = field(default=None, repr=False)
    service_name: str = "millm"
    service_instance_id: str = ""
    deployment_environment: str | None = None
    queue_size: int = 8192
    batch_size: int = 512
    export_timeout_s: float = 10.0
    retry_horizon_s: float = 60.0
    drain_deadline_s: float = 5.0
    metric_interval_s: float = 15.0
    sampler_ratio: float = 1.0
    series_cap: int = 2000
    stateless_export_permitted: bool = False
    result_log_origins: tuple[str, ...] = LOG_ORIGINS
    degraded_window_s: float = 60.0
    unavailable_after_s: float = 300.0
    sdk_disabled: bool = False

    @property
    def any_on(self) -> bool:
        return any(self.switches.values())

    def enabled(self, signal: str) -> bool:
        """The switch is on AND nothing refused the configuration: this signal really exports."""
        return bool(self.switches.get(signal)) and self.refusal is None

    def public_view(self, sdk_version: str | None,
                    semconv_schema_url: str | None) -> dict[str, Any]:
        """The `config` block of the health route. ⚠ Non-secret by construction (module doc)."""
        return {
            "endpoint": strip_endpoint(self.endpoint),
            "protocol": PROTOCOL,
            "transport": self.transport,
            "headers_set": bool(self.headers),
            "tls_ca_file_set": bool(self.ca_file),
            "tls_client_cert_set": bool(self.client_cert_file),
            "tls_client_key_set": bool(self.client_key_file),
            "service_name": self.service_name,
            "service_instance_id": self.service_instance_id,
            "deployment_environment": self.deployment_environment,
            "sampler": {"kind": "parentbased_traceidratio", "ratio": self.sampler_ratio},
            "series_cap": self.series_cap,
            "queue_size": self.queue_size,
            "batch_size": self.batch_size,
            "export_timeout_s": self.export_timeout_s,
            "retry_horizon_s": self.retry_horizon_s,
            "drain_deadline_s": self.drain_deadline_s,
            "metric_interval_s": self.metric_interval_s,
            "stateless_export_permitted": self.stateless_export_permitted,
            "result_log_filter": {"origins": list(self.result_log_origins)},
            "sdk_disabled": self.sdk_disabled,
            "sdk_version": sdk_version,
            "semconv_schema_url": semconv_schema_url,
        }


def _origins(text: str) -> tuple[str, ...]:
    wanted = [o.strip().lower() for o in (text or "").split(",") if o.strip()]
    return tuple(o for o in LOG_ORIGINS if o in wanted)


def load(source: Any) -> TelemetrySettings:
    """Resolve `Settings` (the process singleton or a test copy) into the effective view.

    Never raises: a bad value becomes a refusal CLASS that leaves the enabled signals
    `unavailable` and is shown in health. Telemetry must never stop the server starting.
    """
    sdk_disabled = bool(source.OTEL_SDK_DISABLED)
    switches = {
        "traces": bool(source.TELEMETRY_TRACES_ENABLED) and not sdk_disabled,
        "metrics": bool(source.TELEMETRY_METRICS_ENABLED) and not sdk_disabled,
        "logs": bool(source.TELEMETRY_LOGS_ENABLED) and not sdk_disabled,
    }
    endpoint = (source.TELEMETRY_ENDPOINT or "").strip() or None
    refusal: str | None = None
    transport: str | None = None
    headers: dict[str, str] = {}
    if any(switches.values()):
        if (source.TELEMETRY_PROTOCOL or "").strip() != PROTOCOL:
            refusal = "UnsupportedProtocol"
        elif endpoint is None:
            refusal = "NoEndpoint"
        else:
            try:
                parts = urlsplit(endpoint)
                scheme, host = parts.scheme.lower(), parts.hostname
                _ = parts.port  # raises on a malformed port
            except ValueError:
                scheme, host = "", None
            if scheme not in ("http", "https") or not host:
                refusal = "InvalidEndpoint"
            elif scheme == "http" and not source.TELEMETRY_ALLOW_PLAINTEXT:
                # D26 OQ-16: plaintext only as a recorded trusted-local choice.
                refusal = "PlaintextNotAllowed"
            else:
                transport = "tls" if scheme == "https" else "plaintext_trusted_local"
        if refusal is None:
            try:
                headers = parse_headers(source.TELEMETRY_HEADERS)
            except ValueError:
                refusal = "InvalidHeaders"
    elif endpoint is not None:
        try:
            transport = "tls" if urlsplit(endpoint).scheme.lower() == "https" else (
                "plaintext_trusted_local" if source.TELEMETRY_ALLOW_PLAINTEXT else None)
        except ValueError:
            transport = None
    return TelemetrySettings(
        switches=switches,
        refusal=refusal,
        endpoint=endpoint,
        transport=transport,
        headers=headers,
        ca_file=source.TELEMETRY_TLS_CA_FILE or None,
        client_cert_file=source.TELEMETRY_TLS_CLIENT_CERT_FILE or None,
        client_key_file=source.TELEMETRY_TLS_CLIENT_KEY_FILE or None,
        service_name=source.TELEMETRY_SERVICE_NAME or "millm",
        service_instance_id=source.TELEMETRY_SERVICE_INSTANCE_ID or PROCESS_INSTANCE_ID,
        deployment_environment=source.TELEMETRY_DEPLOYMENT_ENVIRONMENT or None,
        queue_size=int(source.TELEMETRY_QUEUE_SIZE),
        batch_size=int(source.TELEMETRY_BATCH_SIZE),
        export_timeout_s=float(source.TELEMETRY_EXPORT_TIMEOUT_S),
        retry_horizon_s=float(source.TELEMETRY_RETRY_HORIZON_S),
        drain_deadline_s=float(source.TELEMETRY_DRAIN_DEADLINE_S),
        metric_interval_s=float(source.TELEMETRY_METRIC_INTERVAL_S),
        sampler_ratio=float(source.TELEMETRY_TRACES_SAMPLER_RATIO),
        series_cap=int(source.TELEMETRY_SERIES_CAP),
        stateless_export_permitted=bool(source.TELEMETRY_STATELESS_EXPORT_PERMITTED),
        result_log_origins=_origins(source.TELEMETRY_RESULT_LOG_ORIGINS),
        degraded_window_s=float(source.TELEMETRY_HEALTH_DEGRADED_WINDOW_S),
        unavailable_after_s=float(source.TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S),
        sdk_disabled=sdk_disabled,
    )
