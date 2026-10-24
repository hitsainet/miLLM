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

No MCP tool writes here in v1 (D5): miLLM's tools live in miStudio's server. Every write is the
operator's, recorded as ``operator`` in the history.
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

#: Who made a change. v1 has no agent path (D5); the field exists so the history's shape matches
#: the other apps' (`agent:<name>` there).
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
        "backstop behind the behaviour limit and the cost budget. The probe scoring route also "
        "scores at most this many probes per call.",
        "next arm", "int", minimum=1, maximum=64),
    _rk("PROBE_MAX_ARMED_BEHAVIOURS", "Armed behaviours",
        "The most behaviours (probe sets) that may be armed at once on a model with no approved "
        "limit of its own. A probe with no behaviour counts as one. Every probe of every set "
        "still counts against the cost budget.",
        "next arm", "int", minimum=1, maximum=64),
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
    _rk("PROBE_RECOMMEND_TARGET_SETS", "Recommended sets per model",
        "How many probe sets of the reference shape each model's recommended limits aim to fit.",
        "next recommendation", "int", minimum=1, maximum=16),
    _rk("PROBE_RECOMMEND_MAX_DECODE_SHARE", "Recommended decode share",
        "The largest share of a model's own decode step time its recommended probe budget may "
        "take (0.10 = 10%).",
        "next recommendation", "float", minimum=0.01, maximum=1.0),
    _rk("PROBE_RECOMMEND_MAX_PREFILL_SHARE", "Recommended prefill share",
        "The largest share of a model's own prefill step time its recommended probe budget may "
        "take (0.05 = 5%).",
        "next recommendation", "float", minimum=0.01, maximum=1.0),
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
    _rk("RULE_SETS_MAX_IMPORTED", "Imported rule sets",
        "The most rule sets that may be imported. One more is refused.",
        "next import", "int", minimum=1, maximum=100_000),
    _rk("RULE_SET_MAX_SIGNALS", "Signals per rule set",
        "The most signals and named expressions one rule set may declare.",
        "next import", "int", minimum=1, maximum=1024),
    _rk("RULE_SET_MAX_BYTES", "Rule set size (bytes)",
        "The largest rule-set document accepted, checked before it is parsed.",
        "next import", "int", minimum=1024, maximum=16_777_216),
    _rk("PROBE_MAX_AGE_DAYS", "Probe event retention (days)",
        "How long probe verdicts are kept.",
        "next recorded event", "int", minimum=1, maximum=3650),
    _rk("PROBE_MAX_OVERHEAD_MS_PER_PASS", "Probe budget per generated token (ms)",
        "The most the armed probes together may add to each forward pass while generating. A probe "
        "that would take the projected total over this (times the headroom) is refused at arm. "
        "Lowering it disarms nothing; the Probe Monitors page then shows the set as over budget.",
        "next arm", "float", minimum=0.01, maximum=10),
    _rk("PROBE_MAX_PREFILL_MS_PER_1K", "Probe budget per 1,000 prompt tokens (ms)",
        "The most the armed probes together may add per 1,000 prompt tokens read.",
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

#: 038 S2 (FTDD §5.8, OT-D5): every telemetry setting is environment-only. No route may change
#: where data goes or whether it leaves, so each is NEVER writable; derived from `Settings` by
#: prefix so a new TELEMETRY_* field joins without a hand-kept list.
TELEMETRY_KEYS: frozenset[str] = frozenset(
    {k for k in Settings.model_fields if k.startswith("TELEMETRY_")} | {"OTEL_SDK_DISABLED"}
)

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
}) | TELEMETRY_KEYS

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


#: Run after a key's value changes in memory, for readers that cached it (D-captured keys).
APPLY_HOOKS: dict[str, Callable[[Any], None]] = {"LOG_LEVEL": _apply_log_level}


def env_value(key: str) -> Any:
    """The value the environment (or the code default) gives `key`, override or not."""
    return _baseline[key] if key in _baseline else getattr(settings, key)


def overrides() -> dict[str, Any]:
    return dict(_overrides)


def _apply(updates: dict[str, Any], removals: frozenset[str] | set[str] = frozenset(),
           meta: Optional[dict[str, dict[str, Any]]] = None) -> None:
    """Assign validated values onto the singleton (and restore removed keys), then run hooks."""
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
    for key in changed:
        hook = APPLY_HOOKS.get(key)
        if hook is not None:
            hook(getattr(settings, key))


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
        raise SettingInvalidError(_validation_message(exc), details={"key": key}) from None
    coerced = getattr(candidate, key)
    if spec.kind not in ("choice", "bool"):
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


def _check_restore(key: str, restored: Any) -> None:
    """The environment's value must still fit the other overrides. The registry's bounds do not
    apply: they limit what may be SAVED here, and the environment's value is authoritative."""
    try:
        candidate = Settings.model_validate({**_current_values(), key: restored})
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
    actor: str = ACTOR_OPERATOR,
) -> dict[str, Any]:
    """Validate, store (row + history, one commit), then apply. Returns the listing row."""
    from millm.db.models.env_override import EnvOverride, EnvOverrideChange

    reason = _clean_reason(reason)
    async with _write_lock():
        coerced = validate_change(key, value)
        previous = getattr(settings, key)
        now = datetime.now(UTC)
        row = await session.get(EnvOverride, key)
        if row is None:
            row = EnvOverride(key=key, value=coerced, updated_at=now, updated_by=actor,
                              reason=reason)
            session.add(row)
        else:
            row.value, row.updated_at, row.updated_by, row.reason = coerced, now, actor, reason
        session.add(EnvOverrideChange(
            key=key, action="set", previous_value=_jsonable(previous),
            new_value=_jsonable(coerced), actor=actor, reason=reason, changed_at=now,
        ))
        await session.commit()
        _apply({key: coerced}, meta={key: {"updated_at": now, "updated_by": actor,
                                            "reason": reason}})
    logger.info("runtime_setting_changed", key=key, action="set", actor=actor,
                effect=RUNTIME_SPECS[key].effect)
    return row_for(key)


async def reset_override(
    session: Any, key: str, *, reason: Optional[str] = None, actor: str = ACTOR_OPERATOR,
) -> dict[str, Any]:
    """Return `key` to the environment's value. Refused when that value no longer fits the other
    overrides (e.g. a lease default above the environment's maximum)."""
    from millm.db.models.env_override import EnvOverride, EnvOverrideChange

    spec = require_runtime(key)
    reason = _clean_reason(reason)
    async with _write_lock():
        row = await session.get(EnvOverride, key)
        if key not in _overrides and row is None:
            raise SettingNotOverriddenError(
                f"{key} has no override; it already uses the environment's value.",
                details={"key": key},
            )
        restored = env_value(key)
        if key in _overrides:
            _check_restore(key, restored)
        previous = getattr(settings, key)
        now = datetime.now(UTC)
        if row is not None:
            await session.delete(row)
        session.add(EnvOverrideChange(
            key=key, action="reset", previous_value=_jsonable(previous),
            new_value=_jsonable(restored), actor=actor, reason=reason, changed_at=now,
        ))
        await session.commit()
        _apply({}, {key})
    logger.info("runtime_setting_changed", key=key, action="reset", actor=actor,
                effect=spec.effect)
    return row_for(key)


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
    _apply(accepted, meta=meta)
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
