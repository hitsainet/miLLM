"""Runtime overrides of environment settings (Feature 35; suite enhancement ENH-003 §3.2).

THE ENVIRONMENT STAYS THE SOURCE OF TRUTH. An override is a row in ``env_overrides``; while it
exists its value outranks the environment variable, and deleting it ("Use the environment value")
returns to the environment. Precedence: override > environment variable > code default (D4).

ONLY ``RUNTIME_KEYS`` CAN BE WRITTEN — 30 keys, the curated list of decision D2. Everything else on
``Settings`` is classified too (every field is in exactly one group, which a test holds), so the
Settings page can say when a read-only value would take effect:

* ``RUNTIME_KEYS``     — "Changeable while miLLM runs", each with the moment a change takes effect.
* ``MODEL_LOAD_KEYS``  — read at the next model load or SAE attach; read-only in v1 (D9).
* ``RESTART_KEYS``     — captured once at startup; a restart of ``millm-backend`` is needed.
* everything else      — "From the environment", read-only. ``SECRET_KEYS`` are shown only as set
  or not set, never their value or anything derived from it.

``NEVER_WRITABLE`` names the keys whose writability would be a defect, tested by name: moving one
into ``RUNTIME_KEYS`` fails the route test that ``PUT``s each of them.

ONE PROCESS, ONE SINGLETON. miLLM runs one uvicorn worker in one replica, and every reader reads the
module singleton ``millm.core.config.settings`` when it uses a value. An override is therefore
validated as a whole candidate ``Settings`` (so every field and model validator runs — assignment
alone runs none, ``validate_assignment`` is off) and then assigned onto the singleton under a lock.
Writes are serialised across validate → commit → apply, and nothing is applied before the commit.
The stored overrides are loaded by ``load_runtime_overrides_on_startup``, which ``main.lifespan``
calls FIRST — before logging is configured, before the inference service builds its queue, before
the batch API starts and before ``AUTO_LOAD_MODEL`` — and which ignores (and logs) any row whose key
is not writable or whose value no longer validates. If miLLM ever runs more than one process, an
in-memory apply reaches only one of them; this module then needs a per-request version check.

TELEMETRY (ENH-006 D75–D78, 2026-10-28). Every telemetry setting but ``TELEMETRY_ENV_ONLY`` (the
receiver's headers and client key, the CA and client-certificate paths, ``OTEL_SDK_DISABLED``) is a
runtime key. A save of any of them rebuilds the export pipeline ONCE (``_reconfigure_telemetry``,
deduplicated per save by ``_run_hooks``, run in a worker thread because the old pipeline drains
first). One save may name several keys (``save_changes``; ``PUT /api/environment``).

WHO WRITES. The operator's write applies at once, recorded as ``operator``. An AI agent's write of
a telemetry key (``X-miLLM-Agent``) waits in miLLM's approval queue (action ``telemetry_write``,
D76) and, once approved, is recorded as ``agent:<name>`` with its ``approval_id``. Other keys keep
Feature 35's posture (D5): no MCP tool writes them.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Optional

import structlog
from pydantic import ValidationError

from millm.core.config import Settings, settings
from millm.core.errors import MiLLMError

logger = structlog.get_logger()

#: Who made a change: `operator`, or `agent:<name>` for an approved agent request (D76).
ACTOR_OPERATOR = "operator"

#: Longest reason accepted with a change (the column's width).
REASON_MAX_CHARS = 500

GROUP_RUNTIME = "runtime"
GROUP_MODEL_LOAD = "model_load"
GROUP_RESTART = "restart"
GROUP_ENVIRONMENT = "environment"

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# =============================================================================
# Errors
# =============================================================================


class SettingNotFoundError(MiLLMError):
    """No setting has this name."""

    code = "SETTING_NOT_FOUND"
    status_code = 404


class SettingNotRuntimeError(MiLLMError):
    """The setting exists but cannot be changed while miLLM runs."""

    code = "SETTING_NOT_RUNTIME"
    status_code = 422


class SettingInvalidError(MiLLMError):
    """The value is refused by the setting's own rule (type, range, or a rule across settings)."""

    code = "SETTING_INVALID"
    status_code = 422


class SettingNotOverriddenError(MiLLMError):
    """A reset was asked for a setting that has no override."""

    code = "SETTING_NOT_OVERRIDDEN"
    status_code = 404


# =============================================================================
# The registry
# =============================================================================


@dataclass(frozen=True)
class RuntimeKey:
    """One writable setting: what the page shows, when a change takes effect, what it accepts."""

    key: str
    label: str
    description: str
    #: When a change takes effect, in the page's words. Pinned per key by a test.
    effect: str
    #: `int`, `float`, `choice` or `bool` (039 S2: a switch; a JSON boolean only).
    kind: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: tuple[str, ...] = ()
    #: Normalises a `choice` value before it is checked (case).
    normalise: Optional[Callable[[str], str]] = None
    #: `text` only (D75): the value's own rule. Returns the value to store, or raises
    #: `ValueError` with the operator-facing reason. Runs on the stripped text.
    check: Optional[Callable[[str], str]] = None
    #: `text` only: whether an empty value (or JSON null) means "not set" (`None`).
    nullable: bool = False
    #: `text` only: the longest value accepted.
    max_length: int = 256


def _rk(key: str, label: str, description: str, effect: str, kind: str, **kw: Any) -> RuntimeKey:
    return RuntimeKey(key=key, label=label, description=description, effect=effect, kind=kind, **kw)


_RUNTIME_SPECS: tuple[RuntimeKey, ...] = (
    _rk("LOG_LEVEL", "Log level",
        "How much the server writes to its log. DEBUG is the most detailed; it never adds "
        "request text or error details to responses, and chat text is not logged at any level.",
        "now", "choice", choices=LOG_LEVELS, normalise=str.upper),
    _rk("MAX_PENDING_REQUESTS", "Queue length",
        "How many requests may wait for the model at once. One more is refused with 503 "
        "QUEUE_FULL and a Retry-After hint. Batch API work never counts against it.",
        "next queue admission", "int", minimum=1, maximum=1000),
    _rk("CLUSTER_HUB_CACHE_TTL_S", "Cluster Hub cache (seconds)",
        "How long a Hugging Face listing of cluster definitions is reused before it is fetched "
        "again. 0 fetches every time.",
        "next Hub lookup", "int", minimum=0, maximum=86400),
    _rk("PROBE_HUB_CACHE_TTL_S", "Probe Hub cache (seconds)",
        "How long a Hugging Face listing of probe definitions is reused before it is fetched "
        "again. 0 fetches every time.",
        "next Hub lookup", "int", minimum=0, maximum=86400),
    _rk("BATCH_RETENTION_INTERVAL_S", "Batch file clean-up interval (seconds)",
        "How often expired Batch API files are deleted.",
        "within 30 seconds (next clean-up check)", "float", minimum=60, maximum=604800),
    _rk("EMBEDDINGS_MAX_INPUTS", "Embedding inputs per request",
        "The most strings one /v1/embeddings request may send. A request holds the model while "
        "it runs, so this bounds how long a chat waits behind it.",
        "next request", "int", minimum=1, maximum=2048),
    _rk("TRANSFORMERS_IDLE_CACHE_RELEASE_S", "Idle memory release (seconds)",
        "How long the graphics cards stay idle before unused cached memory is returned to them, "
        "so other programs on the node can use it. 0 releases at once; a negative value never "
        "releases.",
        "when the next request ends", "float", minimum=-1, maximum=86400),
    _rk("GRACEFUL_UNLOAD_TIMEOUT", "Unload wait (seconds)",
        "How long an unload waits for running requests to finish.",
        "next unload", "float", minimum=1, maximum=600),
    _rk("LEASE_DEFAULT_TTL_SECONDS", "Model lease default (seconds)",
        "How long a model lease lasts when the request names no duration. Must not exceed the "
        "maximum.",
        "next lease grant", "int", minimum=1, maximum=604800),
    _rk("LEASE_MAX_TTL_SECONDS", "Model lease maximum (seconds)",
        "The longest model lease a client may ask for. Must stay at or above the default and "
        "the Batch API's own lease.",
        "next lease grant", "int", minimum=1, maximum=604800),
    _rk("BATCH_MAX_ROWS", "Batch file rows",
        "The most lines a Batch API input file may have. A larger file is refused whole.",
        "next upload", "int", minimum=1, maximum=1_000_000),
    _rk("BATCH_MAX_FILE_BYTES", "Batch file size (bytes)",
        "The largest Batch API input file accepted.",
        "next upload", "int", minimum=1_048_576, maximum=2_147_483_648),
    _rk("BATCH_FILE_RETENTION_DAYS", "Batch file retention (days)",
        "How long an uploaded Batch API input file is kept.",
        "next upload", "int", minimum=1, maximum=365),
    _rk("BATCH_CHUNK_ROWS", "Batch rows per turn",
        "How many Batch API rows run before an interactive request may take the model. Smaller "
        "keeps chats responsive during a batch; larger finishes batches sooner.",
        "next chunk", "int", minimum=1, maximum=256),
    _rk("PROBE_MAX_ARMED", "Armed probes (backstop)",
        "The most probes that may be armed at once, whatever behaviours they belong to: a "
        "backstop behind the slowdown budget, which decides how many fit. The probe scoring route "
        "also scores at most this many probes per call.",
        "next arm", "int", minimum=1, maximum=64),
    _rk("PROBE_SLOWDOWN_BUDGET_PCT", "Probe slowdown budget (%)",
        "Probes may slow each token by at most this share of the model's own time: its measured "
        "time per generated token, and per 1,000 prompt tokens. As many probes arm as fit; there "
        "is no count of behaviours. A model that has not been measured uses the two fallback "
        "budgets below until its probe capacity is measured.",
        "next arm", "float", minimum=0.1, maximum=100.0),
    _rk("PROBE_VERDICT_HEADER_MAX_BYTES", "Probe verdict header size (bytes)",
        "The longest X-miLLM-Probe-Verdicts header a response carries. Over it, whole behaviours "
        "are dropped from the header and a final member says how many results there were; the "
        "response body, the stream and the events list always carry every result.",
        "next request", "int", minimum=256, maximum=65536),
    _rk("PROBE_MAX_IMPORTED", "Imported probes",
        "The most probes that may be imported. One more is refused.",
        "next import", "int", minimum=1, maximum=1_000_000),
    _rk("PROBE_MAX_PER_BEHAVIOUR", "Probes per behaviour",
        "The most probes one behaviour's set may hold when it is armed or scored.",
        "next arm", "int", minimum=1, maximum=256),
    _rk("PROBE_LIST_PAGE_SIZE", "Probe list page size",
        "How many probes or behaviours one page of a list returns when the caller names no limit.",
        "next request", "int", minimum=10, maximum=1000),
    _rk("RULE_SIGNALS_ENABLED", "Rule signals",
        "Whether attached rule signals run beside the probes. Off, every attached signal reports "
        "rule_signals_disabled and no rule runs; attachments are kept.",
        "next request", "bool"),
    _rk("RULE_TIME_BUDGET_MS", "Rule time budget (ms)",
        "The longest the rule signals of one request may run. A signal still running at the "
        "budget reports rule_timeout; the response is not held longer and the probes are "
        "unchanged.",
        "next request", "float", minimum=1.0, maximum=1000.0),
    _rk("APPROVAL_TTL_HOURS", "Approval expiry (hours)",
        "How long an AI agent's request waits on the Approvals page for your decision before it "
        "expires and can never run.",
        "next agent request", "int", minimum=1, maximum=720),
    _rk("RULE_SETS_MAX_IMPORTED", "Imported rule sets",
        "The most rule sets that may be imported. One more is refused.",
        "next import", "int", minimum=1, maximum=100_000),
    _rk("RULE_SET_MAX_SIGNALS", "Signals per rule set",
        "The most signals and named expressions one rule set may declare.",
        "next import", "int", minimum=1, maximum=1024),
    _rk("RULE_SET_MAX_BYTES", "Rule set size (bytes)",
        "The largest rule-set document accepted, checked before it is parsed.",
        "next import", "int", minimum=1024, maximum=16_777_216),
    _rk("RULE_CODE_CALL_WALL_MS", "Operator code wall time (ms)",
        "The wall-time backstop of one operator-code call in the rule runner. Code rules are "
        "scored after the response, so this never delays a chat; a call still running reports "
        "rule_timeout.",
        "next code-rule job", "int", minimum=1, maximum=60_000),
    _rk("RULE_CODE_CALL_MEMORY_MIB", "Operator code memory (MiB)",
        "The memory one operator-code call may use. Never below the runner contract's floor "
        "(64 MiB): below it, code rules are not run.",
        "next code-rule job", "int", minimum=64, maximum=512),
    _rk("RULE_CODE_CALL_OUTPUT_BYTES", "Operator code reply size (bytes)",
        "The longest reply one operator-code call may write. Never below the runner contract's "
        "floor (4,096 bytes).",
        "next code-rule job", "int", minimum=4096, maximum=65_536),
    _rk("RULE_CODE_CALLS_PER_REQUEST", "Operator code calls per request",
        "How many operator-code calls (signal x window) one request's after-response scoring may "
        "make. Signals past it stay pending (rule_worker_unavailable).",
        "next code-rule job", "int", minimum=1, maximum=256),
    _rk("RULE_CODE_MODULES_MAX_BYTES", "Operator code per import (bytes)",
        "The most bytes of operator-code documents one rule-set import may carry.",
        "next import", "int", minimum=1024, maximum=67_108_864),
    _rk("PROBE_MAX_AGE_DAYS", "Probe event retention (days)",
        "How long probe verdicts are kept.",
        "next recorded event", "int", minimum=1, maximum=3650),
    _rk("PROBE_MAX_OVERHEAD_MS_PER_PASS", "Fallback probe budget per generated token (ms)",
        "Used only for a model whose time per generated token has not been measured: the most the "
        "armed probes together may add to each forward pass while generating. A measured model "
        "uses the slowdown budget instead.",
        "next arm", "float", minimum=0.01, maximum=10),
    _rk("PROBE_MAX_PREFILL_MS_PER_1K", "Fallback probe budget per 1,000 prompt tokens (ms)",
        "Used only for a model whose prompt time has not been measured: the most the armed probes "
        "together may add per 1,000 prompt tokens read. A measured model uses the slowdown budget "
        "instead.",
        "next arm", "float", minimum=0.01, maximum=10),
    _rk("PROBE_BUDGET_HEADROOM", "Probe budget headroom",
        "The share of each probe budget arming may fill, leaving room for measurement noise.",
        "next arm", "float", minimum=0.5, maximum=1.0),
    _rk("PROBE_COST_SAMPLES", "Probe cost timings",
        "How many timings of each shape are taken when a probe is armed; the median is used.",
        "next arm", "int", minimum=5, maximum=200),
    _rk("PROBE_CAPACITY_DECODE_STEPS", "Capacity measurement steps",
        "How many one-token decode steps a model's probe-capacity measurement times; the median "
        "is its decode step time.",
        "next capacity measurement", "int", minimum=4, maximum=512),
    _rk("PROBE_TRANSCRIPTS", "Record session text",
        "Off by default. When on, each monitored chat turn stores the newest user message and the "
        "reply (not the reasoning), for the transcript retention below, so a judge can re-check "
        "it. This is personal data: it is never sent on the live feed and can be deleted per "
        "session.",
        "next request", "choice", choices=("off", "on"), normalise=str.lower),
    _rk("PROBE_TRANSCRIPT_MAX_AGE_DAYS", "Session text retention (days)",
        "How long recorded session text is kept.",
        "next retention pass", "int", minimum=1, maximum=365),
    _rk("PROBE_TRANSCRIPT_MAX_TURNS", "Session text kept (turns)",
        "The most recorded turns kept across all sessions; older ones are removed first.",
        "next retention pass", "int", minimum=100, maximum=1_000_000),
    _rk("PROBE_SESSION_IDLE_MINUTES", "Session idle end (minutes)",
        "A conversation idle this long ends; a later continuation opens a new, linked session.",
        "next request", "int", minimum=1, maximum=1440),
    _rk("PROBE_RETENTION_INTERVAL_S", "Probe clean-up interval (seconds)",
        "How often old probe verdicts, sessions and session text are removed.",
        "next retention pass", "int", minimum=30, maximum=86400),
    _rk("PROBE_SESSION_MAX_AGE_DAYS", "Session retention (days)",
        "How long a session not seen again is kept. Its verdicts follow their own retention.",
        "next retention pass", "int", minimum=1, maximum=3650),
    _rk("PROBE_MAX_EVENTS_PER_PROBE", "Probe events kept per probe",
        "The most live-traffic verdicts kept for one probe; older ones are removed first.",
        "next recorded event", "int", minimum=1, maximum=1_000_000),
    _rk("SENSING_MAX_AGE_DAYS", "Cluster sensing retention (days)",
        "How long cluster co-activation events are kept.",
        "next recorded event", "int", minimum=1, maximum=3650),
    _rk("CIRCUIT_SENSING_MAX_AGE_DAYS", "Circuit sensing retention (days)",
        "How long circuit edge events are kept.",
        "next recorded event", "int", minimum=1, maximum=3650),
    _rk("SENSING_EPSILON", "Cluster sensing threshold",
        "A feature counts as firing above this fraction of its largest recorded activation. "
        "Clusters already armed keep the value they were armed with.",
        "next arm", "float", minimum=0, maximum=1),
    _rk("CIRCUIT_SENSING_EPSILON", "Circuit sensing threshold",
        "The same threshold for circuit edge sensing. Circuits already armed keep theirs.",
        "next arm", "float", minimum=0, maximum=1),
    _rk("CIRCUIT_SENSING_MAX_TOKEN_LAG", "Circuit sensing window (tokens)",
        "How many tokens may pass between an upstream feature firing and its downstream partner "
        "for the pair to count as one edge. Circuits already armed keep theirs.",
        "next arm", "int", minimum=0, maximum=256),
    _rk("CLUSTER_INTENSITY_MIN", "Cluster strength minimum",
        "The lowest strength a cluster's dial allows when its definition gives no range. Must "
        "be below the maximum.",
        "next request", "float", minimum=0, maximum=10),
    _rk("CLUSTER_INTENSITY_MAX", "Cluster strength maximum",
        "The highest strength a cluster's dial allows when its definition gives no range.",
        "next request", "float", minimum=0, maximum=10),
    _rk("CIRCUIT_INTENSITY_MIN", "Circuit strength minimum",
        "The lowest strength the circuit dial allows. Must be below the maximum.",
        "next request", "float", minimum=0, maximum=10),
    _rk("CIRCUIT_INTENSITY_MAX", "Circuit strength maximum",
        "The highest strength the circuit dial allows.",
        "next request", "float", minimum=0, maximum=10),
    _rk("MULTISAE_VRAM_ENVELOPE_MB", "SAE set memory warning (MB)",
        "Attaching sparse autoencoders (SAEs) that need more graphics memory than this logs a "
        "warning. Advisory only: the real limit is the card's free memory.",
        "next attach or status read", "int", minimum=1, maximum=1_048_576),
    _rk("SCORING_FLOAT32_DEVICE", "Float32 scoring placement",
        "Where a float32 scoring request builds its full-precision copy: auto (the model's card "
        "when it fits, else the processor), cpu, or cuda (the card the model's inputs are on). "
        "A card cannot be named by number here.",
        "next float32 scoring request", "choice", choices=("auto", "cpu", "cuda"),
        normalise=str.lower),
    _rk("SCORING_FLOAT32_HEADROOM_MB", "Float32 scoring headroom (MB)",
        "Memory left free beyond the float32 copy itself before a scoring request is accepted.",
        "next float32 scoring request", "int", minimum=0, maximum=65536),
    _rk("PROBE_SCORE_MAX_INPUTS", "Probe scoring inputs per call",
        "The most inputs one /api/probes/score call may send. At least 64, because miStudio's "
        "tool sends up to 64.",
        "next request", "int", minimum=64, maximum=1024),
)

# =============================================================================
# Telemetry (ENH-006 D75–D78, 2026-10-28; reverses 038's OT-D5 for everything but secrets/files)
# =============================================================================

#: The telemetry settings that stay ENVIRONMENT-ONLY (D75): the receiver's credentials and the
#: client key's path (secrets), the CA and client-certificate paths (a file must be mounted into
#: the pod before it can be named), and the standard SDK kill switch. The only hand-kept telemetry
#: list: every other telemetry setting is writable, derived from `Settings` by prefix below, and a
#: test requires each derived key to have a registered `RuntimeKey`.
TELEMETRY_ENV_ONLY: frozenset[str] = frozenset({
    "TELEMETRY_HEADERS", "TELEMETRY_TLS_CLIENT_KEY_FILE",
    "TELEMETRY_TLS_CA_FILE", "TELEMETRY_TLS_CLIENT_CERT_FILE",
    "OTEL_SDK_DISABLED",
})

#: When a telemetry change takes effect: the save drains the running export pipeline (its own
#: `TELEMETRY_DRAIN_DEADLINE_S`) and builds it again from the settings in force.
TELEMETRY_EFFECT = "next export; drains and restarts the export pipeline"


def _tk(key: str, label: str, description: str, kind: str, **kw: Any) -> RuntimeKey:
    return _rk(key, label, description, TELEMETRY_EFFECT, kind, **kw)


def _telemetry_specs() -> tuple[RuntimeKey, ...]:
    from millm.telemetry import settings as tel

    return (
        _tk("TELEMETRY_TRACES_ENABLED", "Export traces",
            "Send one trace per request (and a child span per probe evaluation) to the receiver. "
            "Off by default. Nothing leaves until a switch is on AND an endpoint is set.",
            "bool"),
        _tk("TELEMETRY_METRICS_ENABLED", "Export metrics",
            "Send the probe coverage counters and histograms to the receiver. Off by default.",
            "bool"),
        _tk("TELEMETRY_LOGS_ENABLED", "Export result logs",
            "Send one log record per recorded probe result and per probe lifecycle step. Off by "
            "default. Never user text.",
            "bool"),
        _tk("TELEMETRY_ENDPOINT", "Receiver endpoint",
            "Base URL of the OpenTelemetry receiver (OTLP over HTTP); /v1/traces, /v1/metrics and "
            "/v1/logs are appended. A user name, password or query string is refused here: "
            "receiver credentials belong in TELEMETRY_HEADERS, in the environment. Empty: no "
            "receiver (an enabled signal then shows unavailable).",
            "text", check=tel.check_written_endpoint, nullable=True, max_length=2048),
        _tk("TELEMETRY_PROTOCOL", "Export protocol",
            "The OTLP protocol. http/protobuf is the only one miLLM sends.",
            "choice", choices=(tel.PROTOCOL,), normalise=str.lower),
        _tk("TELEMETRY_ALLOW_PLAINTEXT", "Allow plaintext (http://)",
            "Send to an http:// endpoint without TLS. Only for a receiver on a trusted local "
            "network; with this off an http:// endpoint is refused and the signals show "
            "unavailable (PlaintextNotAllowed).",
            "bool"),
        _tk("TELEMETRY_SERVICE_NAME", "Service name",
            "The service.name every exported record carries.",
            "text", check=tel.check_written_label, max_length=128),
        _tk("TELEMETRY_SERVICE_INSTANCE_ID", "Service instance id",
            "The service.instance.id every record carries. Empty: a random id for this process.",
            "text", check=tel.check_written_label, nullable=True, max_length=128),
        _tk("TELEMETRY_DEPLOYMENT_ENVIRONMENT", "Deployment environment",
            "The deployment.environment.name every record carries (for example production). "
            "Empty: not sent.",
            "text", check=tel.check_written_label, nullable=True, max_length=128),
        _tk("TELEMETRY_QUEUE_SIZE", "Export queue (records per signal)",
            "How many records may wait for export per signal. When it is full, new records are "
            "dropped and counted; a request never waits for export.",
            "int", minimum=1, maximum=1_000_000),
        _tk("TELEMETRY_BATCH_SIZE", "Export batch size",
            "The most records sent in one request to the receiver.",
            "int", minimum=1, maximum=100_000),
        _tk("TELEMETRY_EXPORT_TIMEOUT_S", "Export timeout (seconds)",
            "How long one send to the receiver may take. A save waits for a send in flight, up to "
            "this, while the pipeline restarts.",
            "float", minimum=0.1, maximum=120),
        _tk("TELEMETRY_RETRY_HORIZON_S", "Retry horizon (seconds)",
            "How long a failed batch is retried before its records are counted failed.",
            "float", minimum=0, maximum=86_400),
        _tk("TELEMETRY_DRAIN_DEADLINE_S", "Drain deadline (seconds)",
            "How long waiting records are sent for when the pipeline stops (a save here, or "
            "shutdown) before the rest are counted dropped.",
            "float", minimum=0, maximum=25),
        _tk("TELEMETRY_METRIC_INTERVAL_S", "Metric interval (seconds)",
            "How often metrics are sent.",
            "float", minimum=1, maximum=3600),
        _tk("TELEMETRY_TRACES_SAMPLER_RATIO", "Trace sampling ratio",
            "The share of new traces sent (1.0 = all). A request that arrives inside a sampled "
            "trace follows its parent's decision.",
            "float", minimum=0.0, maximum=1.0),
        _tk("TELEMETRY_SERIES_CAP", "Series per metric",
            "The most distinct series one metric may have; beyond it, series fold into one "
            "overflow series.",
            "int", minimum=1, maximum=100_000),
        _tk("TELEMETRY_STATELESS_EXPORT_PERMITTED", "Allow stateless result export",
            "Allow /api/probes/score results to be exported when the request also asks for it "
            "(export_results). Off: they never leave.",
            "bool"),
        _tk("TELEMETRY_RESULT_LOG_ORIGINS", "Result log origins",
            "Which recorded results become log records: live, batch, both (live,batch), or empty "
            "for neither.",
            "text", check=tel.check_written_origins, max_length=64),
        _tk("TELEMETRY_HEALTH_DEGRADED_WINDOW_S", "Degraded window (seconds)",
            "A failure or drop within this long shows the signal as degraded in health.",
            "float", minimum=1, maximum=86_400),
        _tk("TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S", "Unavailable after (seconds)",
            "Records waiting this long with no successful send show the signal as unavailable.",
            "float", minimum=1, maximum=86_400),
    )


_TELEMETRY_SPECS: tuple[RuntimeKey, ...] = _telemetry_specs()
_RUNTIME_SPECS = _RUNTIME_SPECS + _TELEMETRY_SPECS

RUNTIME_SPECS: dict[str, RuntimeKey] = {spec.key: spec for spec in _RUNTIME_SPECS}
RUNTIME_KEYS: frozenset[str] = frozenset(RUNTIME_SPECS)

#: Read at the next model load (or SAE attach): read-only in v1 (D9). Key -> plain description.
MODEL_LOAD_KEYS: dict[str, str] = {
    "TORCH_COMPILE": "Compile the model for speed when it loads (unset: automatic).",
    "GGUF_ENABLE_EMBEDDINGS": "Load GGUF models able to answer /v1/embeddings as well as chat.",
    "GGUF_CONTEXT_LENGTH": "The most context, in tokens, a GGUF model is loaded with.",
    "GGUF_TENSOR_SPLIT": "How a GGUF model that needs several cards divides its layers.",
    "GGUF_KV_CACHE_TYPE": "The precision of a GGUF model's key-value cache.",
    "GGUF_FLASH_ATTENTION": "Use flash attention for GGUF models (needed by a quantized cache).",
    "TRANSFORMERS_MIN_CONTEXT": "The context, in tokens, every card of a load must have room for.",
    "TRANSFORMERS_CUDA_CONTEXT_MB": "Memory each card keeps for its CUDA context.",
    "MULTISAE_ATTACH_DTYPE": "The precision attached SAE steering weights use.",
    "STRUCTURED_OUTPUT_GRAMMAR_CACHE": "Compiled JSON-schema grammars kept per loaded model.",
}

#: Captured once at startup: changing one needs a restart of `millm-backend`. Key -> description.
RESTART_KEYS: dict[str, str] = {
    "KV_CACHE_MODE": "The key-value cache mode the inference service is built with.",
    "SPECULATIVE_MODEL": "The draft model for speculative decoding (unset: off).",
    "SPECULATIVE_NUM_TOKENS": "Tokens the draft model proposes per step.",
    "ENABLE_CONTINUOUS_BATCHING": "Serve several requests in one batch (off in production).",
    "CBM_MAX_QUEUE_SIZE": "Continuous batching's own queue length.",
    "CBM_DEFAULT_TEMPERATURE": "The one temperature continuous batching batches.",
    "CBM_DEFAULT_TOP_P": "The one top-p continuous batching batches.",
    "CBM_DEFAULT_MAX_TOKENS": "Continuous batching's default token limit.",
    "CBM_FORCE_SERIAL_MONITORING": "Route monitored requests around continuous batching.",
    "AUTO_LOAD_MODEL": "A model to load when the server starts.",
    "LOG_FORMAT": "json or console log lines.",
    "QUEUE_DURATION_WINDOW": "Request durations kept for the queue's wait estimate.",
    "LEASE_ENDED_MEMORY": "Ended leases remembered so a late renew is answered clearly.",
    "RULE_RUNNER_SOCKET": "The rule runner's Unix socket (shared with the rule-runner sidecar).",
}

#: What a restart costs, shown with the restart group (k8s `strategy: Recreate`, one replica).
RESTART_EFFECT = "restart millm-backend (a restart ends in-flight requests and batches)"
MODEL_LOAD_EFFECT = "next model load"
ENVIRONMENT_EFFECT = "set in the environment; restart millm-backend"

#: Shown only as set or not set.
SECRET_KEYS: frozenset[str] = frozenset(
    {"DATABASE_URL", "HF_TOKEN", "SETTINGS_ENCRYPTION_KEY", "WEIGHTS_TOKENS",
     # 038 S2: the receiver's credentials and the client key's path (FTDD §5.8).
     "TELEMETRY_HEADERS", "TELEMETRY_TLS_CLIENT_KEY_FILE"}
)

#: Every telemetry setting, derived from `Settings` by prefix so a new TELEMETRY_* field joins
#: without a hand-kept list. 038 S2 made all of them environment-only (OT-D5); ENH-006 D75
#: (2026-10-28) reversed that for all but `TELEMETRY_ENV_ONLY`.
TELEMETRY_KEYS: frozenset[str] = frozenset(
    {k for k in Settings.model_fields if k.startswith("TELEMETRY_")} | {"OTEL_SDK_DISABLED"}
)
#: D75: the telemetry settings changeable while miLLM runs — every telemetry setting but the
#: environment-only ones. Derived, never listed.
TELEMETRY_RUNTIME_KEYS: frozenset[str] = TELEMETRY_KEYS - TELEMETRY_ENV_ONLY


def is_telemetry_key(key: Any) -> bool:
    """Whether a key NAMES a telemetry setting, read generously (case, spaces) because the
    approval gate uses it to decide whether an agent's write waits for the operator: a near-miss
    spelling must gate (and then be refused by the route's own check), never slip past."""
    text = str(key or "").strip().upper()
    return text.startswith(("TELEMETRY_", "OTEL_"))


#: Keys whose writability would be a defect: secrets, paths, the network surface, correctness
#: guarantees and cross-repo pins. A test `PUT`s each and requires 422.
NEVER_WRITABLE: frozenset[str] = SECRET_KEYS | frozenset({
    "MODEL_CACHE_DIR", "SAE_CACHE_DIR", "BATCH_FILES_DIR", "WEIGHTS_CHECKPOINT_DIR",
    "WEIGHTS_CHECKPOINT_READ_ROOTS",
    "HOST", "PORT", "CORS_ORIGINS", "DEBUG",
    "MAX_CONCURRENT_REQUESTS", "SENSING_FORCE_SERIAL", "CIRCUIT_SENSING_FORCE_SERIAL",
    "PROBE_FORCE_SERIAL", "PROBE_PARITY_TOLERANCE", "PROBE_PARITY_SCORE_TOLERANCE",
    "PROBE_PARITY_MATCHED_DTYPE_FLOOR", "PROBE_PARITY_MATCHED_RELATIVE_FLOOR",
    "PROBE_PARITY_MIN_ROBUST_FRACTION", "BATCH_PACK_DEFAULT", "TORCH_COMPILE_MODE",
    "CLUSTER_HUB_TAG", "PROBE_HUB_TAG", "WEIGHTS_COMMIT_WATCHDOG_MS",
    "LEASE_HOLDER_MAX_CHARS", "LEASE_REASON_MAX_CHARS",
}) | TELEMETRY_ENV_ONLY

#: 038 S2 (review): values shown only REDACTED. A receiver URL can carry a credential in its
#: userinfo or query (`https://user:KEY@host/?api_key=K`); the listing (and miStudio's
#: `millm_get_environment` tool, which carries it off-box) shows it as health does — scheme,
#: host, port and path only.
REDACTED_KEYS: frozenset[str] = frozenset({"TELEMETRY_ENDPOINT"})


def _redacted(key: str, value: Any) -> Any:
    if key not in REDACTED_KEYS or not isinstance(value, str):
        return value
    from millm.telemetry.settings import strip_endpoint

    return strip_endpoint(value)


#: Pairs that must stay strictly ordered (minimum key, maximum key).
_ORDERED_PAIRS: tuple[tuple[str, str], ...] = (
    ("CLUSTER_INTENSITY_MIN", "CLUSTER_INTENSITY_MAX"),
    ("CIRCUIT_INTENSITY_MIN", "CIRCUIT_INTENSITY_MAX"),
)


def all_setting_names() -> list[str]:
    return list(Settings.model_fields)


def group_of(key: str) -> str:
    if key in RUNTIME_KEYS:
        return GROUP_RUNTIME
    if key in MODEL_LOAD_KEYS:
        return GROUP_MODEL_LOAD
    if key in RESTART_KEYS:
        return GROUP_RESTART
    return GROUP_ENVIRONMENT


def effect_of(key: str) -> str:
    group = group_of(key)
    if group == GROUP_RUNTIME:
        return RUNTIME_SPECS[key].effect
    if group == GROUP_MODEL_LOAD:
        return MODEL_LOAD_EFFECT
    if group == GROUP_RESTART:
        return RESTART_EFFECT
    return ENVIRONMENT_EFFECT


# =============================================================================
# In-memory state
# =============================================================================

#: Held while assigning onto the singleton. Readers do not take it: each reads one attribute,
#: and every write is a single key whose candidate was validated as a whole.
_apply_lock = threading.Lock()
#: The overrides in force: key -> value.
_overrides: dict[str, Any] = {}
#: Last change per overridden key: {updated_at, updated_by, reason}.
_meta: dict[str, dict[str, Any]] = {}
#: The singleton's value before the key was first overridden — the environment's (or the code
#: default's) value, which a reset restores.
_baseline: dict[str, Any] = {}

_write_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _write_lock() -> asyncio.Lock:
    """One write at a time across validate → commit → apply (per event loop)."""
    loop = asyncio.get_running_loop()
    lock = _write_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _write_locks[loop] = lock
    return lock


def _apply_log_level(value: Any) -> None:
    from millm.core.logging import apply_log_level

    apply_log_level(str(value))


#: One telemetry rebuild at a time: two saves must never build two pipelines side by side.
_telemetry_lock = threading.Lock()


def _reconfigure_telemetry(_value: Any) -> None:
    """ENH-006 D75: rebuild the export pipeline from the settings in force.

    Drains the running pipeline first, for ITS OWN `TELEMETRY_DRAIN_DEADLINE_S` (records already
    queued are sent, the rest counted dropped), then `configure()` reads the singleton — every
    override this save applied — and starts a pipeline only if a signal is on and the
    configuration is accepted. A refused configuration (no endpoint, plaintext not allowed, …)
    shows in health. Off stays free: with every switch off `configure()` imports no SDK module.
    Registered for every telemetry key and run ONCE per save however many of them changed
    (`_run_hooks` deduplicates). Never raises: a save that reached the database must not answer
    500 because the exporter could not be built.
    """
    from millm import telemetry

    with _telemetry_lock:
        try:
            telemetry.shutdown()
            telemetry.configure()
        except Exception as exc:  # noqa: BLE001 - configure() never raises; belt and braces
            logger.warning("telemetry_reconfigure_failed", error_type=type(exc).__name__)


#: Run after a key's value changes in memory, for readers that cached it (D-captured keys).
#: ENH-006 D75: every writable telemetry key rebuilds the export pipeline (the pipeline captures
#: its whole configuration when it is built).
APPLY_HOOKS: dict[str, Callable[[Any], None]] = {
    "LOG_LEVEL": _apply_log_level,
    **dict.fromkeys(sorted(TELEMETRY_RUNTIME_KEYS), _reconfigure_telemetry),
}

#: Hooks NOT run when the stored overrides are loaded at startup, because `main.lifespan` does
#: their work itself right after the load: it calls `telemetry.configure()` once, with every
#: override already on the singleton (and before that, logging is not even configured).
STARTUP_DEFERRED_HOOKS: frozenset[Callable[[Any], None]] = frozenset({_reconfigure_telemetry})


def _run_hooks(changed: list[str], skip: frozenset[Any] = frozenset()) -> int:
    """Each distinct hook of the changed keys, ONCE, in the order its first key changed — so a
    save of several telemetry keys rebuilds the pipeline once, after every value is in place.
    Returns how many hooks ran."""
    seen: list[Any] = []
    for key in changed:
        hook = APPLY_HOOKS.get(key)
        if hook is None or hook in skip or hook in seen:
            continue
        seen.append(hook)
        hook(getattr(settings, key))
    return len(seen)


def env_value(key: str) -> Any:
    """The value the environment (or the code default) gives `key`, override or not."""
    return _baseline[key] if key in _baseline else getattr(settings, key)


def overrides() -> dict[str, Any]:
    return dict(_overrides)


def _apply(updates: dict[str, Any], removals: frozenset[str] | set[str] = frozenset(),
           meta: Optional[dict[str, dict[str, Any]]] = None, *, run_hooks: bool = True,
           skip_hooks: frozenset[Any] = frozenset()) -> list[str]:
    """Assign validated values onto the singleton (and restore removed keys), then run each
    distinct hook once (`run_hooks=False`: the caller runs them, off the event loop). Returns
    the changed keys."""
    changed: list[str] = []
    with _apply_lock:
        for key, value in updates.items():
            if key not in _baseline:
                _baseline[key] = getattr(settings, key)
            setattr(settings, key, value)
            _overrides[key] = value
            if meta and key in meta:
                _meta[key] = meta[key]
            changed.append(key)
        for key in removals:
            if key in _baseline:
                setattr(settings, key, _baseline.pop(key))
            _overrides.pop(key, None)
            _meta.pop(key, None)
            changed.append(key)
    if run_hooks:
        _run_hooks(changed, skip_hooks)
    return changed


async def _apply_and_run_hooks(updates: dict[str, Any],
                               removals: frozenset[str] | set[str] = frozenset(),
                               meta: Optional[dict[str, dict[str, Any]]] = None) -> None:
    """`_apply` for a route: the values land at once; the hooks run in a worker thread, because
    a telemetry rebuild waits for the old pipeline to drain (seconds) and must not stall every
    other request on the event loop. Awaited, so the response reports the rebuilt state."""
    changed = _apply(updates, removals, meta, run_hooks=False)
    await asyncio.to_thread(_run_hooks, changed)


def _forget_for_tests() -> None:
    """Forget every override, its metadata and its baseline, WITHOUT touching the singleton
    (test isolation: the test fixture restores the values it recorded before the test)."""
    with _apply_lock:
        _overrides.clear()
        _meta.clear()
        _baseline.clear()


# =============================================================================
# Validation
# =============================================================================


def _current_values() -> dict[str, Any]:
    return {name: getattr(settings, name) for name in Settings.model_fields}


def _validation_message(exc: ValidationError) -> str:
    """The validators' own messages, never the input values (a secret field could be one)."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
        msg = str(err.get("msg", "invalid value"))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid value"


def require_runtime(key: str) -> RuntimeKey:
    if key not in Settings.model_fields:
        raise SettingNotFoundError(f"There is no setting named {key!r}.", details={"key": key})
    spec = RUNTIME_SPECS.get(key)
    if spec is None:
        raise SettingNotRuntimeError(
            f"{key} cannot be changed while miLLM runs ({effect_of(key)}).",
            details={"key": key, "group": group_of(key)},
        )
    return spec


def _check_value(spec: RuntimeKey, value: Any, base: dict[str, Any]) -> Any:
    """`value` for `spec.key`, validated against `base` (the other values in force). Raises
    SettingInvalidError with the rule that refused it."""
    key = spec.key
    if spec.kind == "choice":
        if not isinstance(value, str):
            raise SettingInvalidError(
                f"{key} must be one of {', '.join(spec.choices)}.", details={"key": key}
            )
        value = spec.normalise(value.strip()) if spec.normalise else value.strip()
        if value not in spec.choices:
            raise SettingInvalidError(
                f"{key} must be one of {', '.join(spec.choices)}.", details={"key": key}
            )
    elif spec.kind == "bool":
        if not isinstance(value, bool):
            raise SettingInvalidError(f"{key} must be true or false.", details={"key": key})
    elif spec.kind == "text":
        # ENH-006 D75. ⚠ A refusal names the rule, never the value (an endpoint can carry a
        # credential, which is exactly what is being refused).
        if value is None:
            if not spec.nullable:
                raise SettingInvalidError(f"{key} must be set.", details={"key": key})
        elif not isinstance(value, str):
            raise SettingInvalidError(f"{key} must be text.", details={"key": key})
        else:
            value = value.strip()
            if len(value) > spec.max_length:
                raise SettingInvalidError(
                    f"{key} may be at most {spec.max_length} characters.", details={"key": key})
            if not value and spec.nullable:
                value = None
            elif spec.check is not None:
                try:
                    value = spec.check(value)
                except ValueError as exc:
                    raise SettingInvalidError(f"{key}: {exc}.", details={"key": key}) from None
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingInvalidError(f"{key} must be a number.", details={"key": key})
        if spec.kind == "int":
            if isinstance(value, float):
                if not value.is_integer():
                    raise SettingInvalidError(
                        f"{key} must be a whole number.", details={"key": key}
                    )
                value = int(value)
    try:
        candidate = Settings.model_validate({**base, key: value})
    except ValidationError as exc:
        message = _validation_message(exc)
        if key in REDACTED_KEYS:
            message = f"{key} is not a valid value."  # pydantic may quote the input
        raise SettingInvalidError(message, details={"key": key}) from None
    coerced = getattr(candidate, key)
    if spec.kind not in ("choice", "bool", "text"):
        if spec.minimum is not None and coerced < spec.minimum:
            raise SettingInvalidError(
                f"{key} must be at least {_num(spec.minimum)}, got {_num(coerced)}.",
                details={"key": key},
            )
        if spec.maximum is not None and coerced > spec.maximum:
            raise SettingInvalidError(
                f"{key} must be at most {_num(spec.maximum)}, got {_num(coerced)}.",
                details={"key": key},
            )
    _check_pairs(candidate, key)
    return coerced


def _check_pairs(candidate: Settings, key: Optional[str] = None) -> None:
    for low, high in _ORDERED_PAIRS:
        if key is not None and key not in (low, high):
            continue
        lo, hi = getattr(candidate, low), getattr(candidate, high)
        if not lo < hi:
            raise SettingInvalidError(
                f"{low} must be below {high} ({_num(lo)} is not below {_num(hi)}).",
                details={"key": key or low},
            )


def _num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _check_restore(key: str, restored: Any, base: Optional[dict[str, Any]] = None) -> None:
    """The environment's value must still fit the other overrides. The registry's bounds do not
    apply: they limit what may be SAVED here, and the environment's value is authoritative."""
    try:
        candidate = Settings.model_validate({**(base or _current_values()), key: restored})
        _check_pairs(candidate, key)
    except (ValidationError, SettingInvalidError) as exc:
        detail = exc.message if isinstance(exc, SettingInvalidError) else _validation_message(exc)
        raise SettingInvalidError(
            f"The environment's value for {key} does not fit the other overrides ({detail}). "
            "Change or reset those first.",
            details={"key": key},
        ) from None


def validate_change(key: str, value: Any) -> Any:
    """The value `key` would take, or SettingNotFound/NotRuntime/Invalid. Changes nothing."""
    spec = require_runtime(key)
    return _check_value(spec, value, _current_values())


def _clean_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    text = reason.strip()
    if not text:
        return None
    if len(text) > REASON_MAX_CHARS:
        raise SettingInvalidError(
            f"A reason may be at most {REASON_MAX_CHARS} characters.", details={"field": "reason"}
        )
    return text


# =============================================================================
# Writes
# =============================================================================


async def set_override(
    session: Any, key: str, value: Any, *, reason: Optional[str] = None,
    actor: str = ACTOR_OPERATOR, approval_id: Optional[str] = None,
) -> dict[str, Any]:
    """Validate, store (row + history, one commit), then apply. Returns the listing row."""
    await save_changes(session, {key: value}, (), reason=reason, actor=actor,
                       approval_id=approval_id)
    return row_for(key)


async def reset_override(
    session: Any, key: str, *, reason: Optional[str] = None, actor: str = ACTOR_OPERATOR,
    approval_id: Optional[str] = None,
) -> dict[str, Any]:
    """Return `key` to the environment's value. Refused when that value no longer fits the other
    overrides (e.g. a lease default above the environment's maximum)."""
    await save_changes(session, {}, (key,), reason=reason, actor=actor, approval_id=approval_id)
    return row_for(key)


def plan_changes(values: dict[str, Any], resets: tuple[str, ...] | list[str] = (),
                 stored: frozenset[str] | set[str] = frozenset(),
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one save of several keys AS A WHOLE, changing nothing (ENH-006 D75/D76).

    Returns ``(updates, restored)``: the coerced value each set key will take, and the
    environment value each reset key returns to. Resets are applied to the candidate first, then
    each set in the order given, every value checked against the candidate built so far (so a
    pair saved together, a minimum and its maximum, is checked as the pair it will be), and the
    whole candidate once more at the end. Raises SettingNotFound / NotRuntime / Invalid /
    NotOverridden exactly as a single-key save does. ``stored``: reset keys that have a database
    row (a row ignored at startup is not in force, and a reset still removes it).
    """
    if not values and not resets:
        raise SettingInvalidError("Nothing to change: name at least one setting.",
                                  details={"field": "values"})
    both = sorted(set(values) & set(resets))
    if both:
        raise SettingInvalidError(
            f"{', '.join(both)} cannot be both set and reset in one save.",
            details={"keys": both})
    if len(set(resets)) != len(tuple(resets)):
        raise SettingInvalidError("A setting is named twice in the resets.",
                                  details={"field": "reset"})
    base = _current_values()
    restored: dict[str, Any] = {}
    for key in resets:
        require_runtime(key)
        if key not in _overrides and key not in stored:
            raise SettingNotOverriddenError(
                f"{key} has no override; it already uses the environment's value.",
                details={"key": key},
            )
        restored[key] = env_value(key)
        if key in _overrides:
            _check_restore(key, restored[key], base)
        base[key] = restored[key]
    updates: dict[str, Any] = {}
    for key, value in values.items():
        spec = require_runtime(key)
        updates[key] = _check_value(spec, value, base)
        base[key] = updates[key]
    if len(values) + len(resets) > 1:
        try:
            _check_pairs(Settings.model_validate(base))
        except ValidationError as exc:
            raise SettingInvalidError(_validation_message(exc), details={}) from None
    return updates, restored


async def save_changes(
    session: Any, values: dict[str, Any], resets: tuple[str, ...] | list[str] = (), *,
    reason: Optional[str] = None, actor: str = ACTOR_OPERATOR, approval_id: Optional[str] = None,
) -> list[str]:
    """One save of one or several keys: validate as a whole, store every row and history line in
    ONE commit, then apply — every value onto the singleton, then each hook ONCE (a save of five
    telemetry keys rebuilds the export pipeline once). Returns the keys changed, in order.

    ``actor`` / ``approval_id``: who made the change; an approved agent request records
    ``agent:<name>`` and its approval (ENH-006 D76). History values of a redacted key (the
    receiver endpoint) are stored redacted: the environment's endpoint may carry a credential.
    """
    from millm.db.models.env_override import EnvOverride, EnvOverrideChange

    reason = _clean_reason(reason)
    async with _write_lock():
        stored = {key for key in resets
                  if key in RUNTIME_SPECS and await session.get(EnvOverride, key) is not None}
        updates, restored = plan_changes(values, resets, stored)
        now = datetime.now(UTC)
        for key, coerced in updates.items():
            previous = getattr(settings, key)
            row = await session.get(EnvOverride, key)
            if row is None:
                session.add(EnvOverride(key=key, value=coerced, updated_at=now, updated_by=actor,
                                        reason=reason))
            else:
                row.value, row.updated_at, row.updated_by, row.reason = (
                    coerced, now, actor, reason)
            session.add(EnvOverrideChange(
                key=key, action="set", previous_value=_shown(key, previous),
                new_value=_shown(key, coerced), actor=actor, approval_id=approval_id,
                reason=reason, changed_at=now,
            ))
        for key, value in restored.items():
            previous = getattr(settings, key)
            row = await session.get(EnvOverride, key)
            if row is not None:
                await session.delete(row)
            session.add(EnvOverrideChange(
                key=key, action="reset", previous_value=_shown(key, previous),
                new_value=_shown(key, value), actor=actor, approval_id=approval_id,
                reason=reason, changed_at=now,
            ))
        await session.commit()
        meta = {key: {"updated_at": now, "updated_by": actor, "reason": reason}
                for key in updates}
        await _apply_and_run_hooks(updates, set(restored), meta)
    for key in updates:
        logger.info("runtime_setting_changed", key=key, action="set", actor=actor,
                    approval_id=approval_id, effect=RUNTIME_SPECS[key].effect)
    for key in restored:
        logger.info("runtime_setting_changed", key=key, action="reset", actor=actor,
                    approval_id=approval_id, effect=RUNTIME_SPECS[key].effect)
    return [*updates, *restored]


def _shown(key: str, value: Any) -> Any:
    """A value as the history stores and shows it."""
    return _redacted(key, _jsonable(value))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


# =============================================================================
# Startup
# =============================================================================


async def load_runtime_overrides_on_startup(session_factory: Any) -> dict[str, Any]:
    """Apply every stored override that is still writable and valid. Never raises.

    A row whose key is not in RUNTIME_KEYS, or whose value no longer validates (a bound changed,
    or another override it depends on was refused), is IGNORED and logged — not applied, and not
    deleted: the operator sees it in the log and on the page's history, and a reset removes it.
    Each row is checked against the environment plus the rows accepted before it, repeating until
    nothing more is accepted, so the order rows are read in never decides the outcome.
    """
    from sqlalchemy import select

    from millm.db.models.env_override import EnvOverride

    applied: list[str] = []
    ignored: list[dict[str, str]] = []
    try:
        async with session_factory() as session:
            rows = list((await session.execute(
                select(EnvOverride).order_by(EnvOverride.key)
            )).scalars().all())
    except Exception as exc:  # noqa: BLE001 - an unmigrated database must not stop serving
        logger.warning("runtime_overrides_load_failed", error_type=type(exc).__name__)
        return {"applied": applied, "ignored": ignored, "error": type(exc).__name__}

    base = _current_values()
    accepted: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}
    pending = []
    for row in rows:
        if row.key in RUNTIME_SPECS:
            pending.append(row)
        else:
            ignored.append({"key": row.key, "reason": "not changeable while miLLM runs"})
    # Repeat until a pass accepts nothing more: rows that depend on each other (a lease default
    # above the environment's maximum, saved after the maximum was raised) were each valid when
    # saved in some order, and must not be refused for being read back in another.
    progress = True
    while pending and progress:
        progress = False
        still: list[Any] = []
        for row in pending:
            try:
                value = _check_value(RUNTIME_SPECS[row.key], row.value, base)
            except SettingInvalidError:
                still.append(row)
                continue
            base[row.key] = value
            accepted[row.key] = value
            meta[row.key] = {"updated_at": row.updated_at, "updated_by": row.updated_by,
                             "reason": row.reason}
            applied.append(row.key)
            progress = True
        pending = still
    for row in pending:
        try:
            _check_value(RUNTIME_SPECS[row.key], row.value, base)
            reason = "invalid"  # pragma: no cover - the loop above would have accepted it
        except SettingInvalidError as exc:
            reason = exc.message
        ignored.append({"key": row.key, "reason": reason})
    # The telemetry rebuild is deferred: `main.lifespan` calls `telemetry.configure()` itself,
    # right after this load, with every accepted override already on the singleton (D77).
    _apply(accepted, meta=meta, skip_hooks=STARTUP_DEFERRED_HOOKS)
    for item in ignored:
        logger.warning("runtime_override_ignored", key=item["key"], reason=item["reason"])
    return {"applied": applied, "ignored": ignored, "error": None}


# =============================================================================
# Reads
# =============================================================================


def _spec_fields(key: str) -> dict[str, Any]:
    spec = RUNTIME_SPECS.get(key)
    if spec is not None:
        return {
            "label": spec.label, "description": spec.description, "type": spec.kind,
            "minimum": spec.minimum, "maximum": spec.maximum,
            "choices": list(spec.choices) or None,
        }
    description = (MODEL_LOAD_KEYS.get(key) or RESTART_KEYS.get(key)
                   or SECRET_DESCRIPTIONS.get(key))
    return {"label": key, "description": description, "type": None, "minimum": None,
            "maximum": None, "choices": None}


#: What the page says about each secret. Values are never shown.
SECRET_DESCRIPTIONS: dict[str, str] = {
    "DATABASE_URL": "The database address, including its password.",
    "HF_TOKEN": "The server's Hugging Face token. A token saved in the Hugging Face card on this "
                "page outranks it, and a token sent with a request outranks both.",
    "SETTINGS_ENCRYPTION_KEY": "The key that encrypts the Hugging Face token saved on this page. "
                               "Without it, saving a token is refused.",
    "WEIGHTS_TOKENS": "The bearer tokens (as digests) that may use the attachable-weights routes.",
}


def _secret_is_set(key: str) -> bool:
    """Whether a secret is configured, read the way its own reader reads it — never its value."""
    if key == "HF_TOKEN":
        from millm.core.hf_credentials import _env_token

        return _env_token() is not None
    if key == "SETTINGS_ENCRYPTION_KEY":
        from millm.core.secret_box import _key_material

        return _key_material() is not None
    value = getattr(settings, key)
    if value is None:
        return False
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return bool(str(value).strip())


def row_for(key: str) -> dict[str, Any]:
    """One listing row. A secret carries only whether it is set."""
    secret = key in SECRET_KEYS
    overridden = key in _overrides
    meta = _meta.get(key, {})
    updated_at = meta.get("updated_at")
    row: dict[str, Any] = {
        "key": key,
        "group": group_of(key),
        "effect": effect_of(key),
        "secret": secret,
        "set": _secret_is_set(key) if secret else None,
        "value": None if secret else _redacted(key, _jsonable(getattr(settings, key))),
        "env_value": None if secret else _redacted(key, _jsonable(env_value(key))),
        "source": "override" if overridden else "env",
        "runtime_writable": key in RUNTIME_KEYS,
        "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else None,
        "updated_by": meta.get("updated_by"),
        "reason": meta.get("reason"),
    }
    row.update(_spec_fields(key))
    return row


def listing() -> dict[str, Any]:
    names = all_setting_names()
    order = {g: i for i, g in enumerate(
        (GROUP_RUNTIME, GROUP_MODEL_LOAD, GROUP_RESTART, GROUP_ENVIRONMENT))}
    runtime_order = {spec.key: i for i, spec in enumerate(_RUNTIME_SPECS)}
    names.sort(key=lambda k: (order[group_of(k)], runtime_order.get(k, 0), k))
    rows = [row_for(k) for k in names]
    counts = {g: sum(1 for r in rows if r["group"] == g) for g in order}
    return {"settings": rows, "counts": counts, "runtime_keys": [s.key for s in _RUNTIME_SPECS]}


async def history(session: Any, limit: int = 50) -> list[dict[str, Any]]:
    """The newest `limit` changes, newest first."""
    from sqlalchemy import select

    from millm.db.models.env_override import EnvOverrideChange

    rows = (await session.execute(
        select(EnvOverrideChange)
        .order_by(EnvOverrideChange.changed_at.desc(), EnvOverrideChange.id.desc())
        .limit(limit)
    )).scalars().all()
    return [
        {
            "id": r.id, "key": r.key, "action": r.action,
            "previous_value": r.previous_value, "new_value": r.new_value,
            "actor": r.actor, "approval_id": r.approval_id, "reason": r.reason,
            "changed_at": r.changed_at.isoformat() if r.changed_at else None,
        }
        for r in rows
    ]


# =============================================================================
# The approval card (ENH-006 D76)
# =============================================================================


def _word(value: Any) -> str:
    if value is None or value == "":
        return "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


async def describe_save(session: Any, values: dict[str, Any], resets: list[str] | tuple[str, ...],
                        reason: Optional[str]) -> dict[str, Any]:
    """The facts an agent's save shows on the approval card: each key's value now and the value
    it will take, the endpoint REDACTED both sides. Runs the save's own validation first, so a
    request that could only be refused is refused now (and never stored). Changes nothing."""
    from millm.db.models.env_override import EnvOverride

    _clean_reason(reason)
    stored = {key for key in resets
              if key in RUNTIME_SPECS and await session.get(EnvOverride, key) is not None}
    updates, restored = plan_changes(values, resets, stored)
    facts: dict[str, Any] = {}
    for key, new in updates.items():
        facts[f"{RUNTIME_SPECS[key].label} ({key})"] = (
            f"now {_word(_shown(key, getattr(settings, key)))}; "
            f"after approval {_word(_shown(key, new))}")
    for key, env in restored.items():
        facts[f"{RUNTIME_SPECS[key].label} ({key})"] = (
            f"now {_word(_shown(key, getattr(settings, key)))}; after approval "
            f"{_word(_shown(key, env))} (the environment's value)")
    effects = sorted({RUNTIME_SPECS[k].effect for k in [*updates, *restored]})
    facts["Takes effect"] = "; ".join(effects)
    facts["Reason"] = (reason or "").strip() or "none given"
    return facts
