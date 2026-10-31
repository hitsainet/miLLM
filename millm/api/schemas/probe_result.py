"""`mi.probe.result/v1` — one probe result, the same object on every surface (FR-38.2, D38.22).

miLLM is the producer. The model here is the ONE definition; both schemas are generated from it
(`MILLM_WRITE_CONTRACTS=1`, `tests/unit/api/test_probe_result_contract.py`):

* **published, open** — `docs/schemas/probe-result-v1.json`. Objects admit additional properties
  and every listed value is `anyOf [the known values, an identifier pattern]`, so a consumer reads
  a value added later in v1 as "other" (D38.10). miStudio byte-pins this file.
* **strict, test-only** — `tests/fixtures/contracts/probe-result-v1.strict.json`. Closed enums,
  `additionalProperties: false`, the published reason list and the conditional rules of FTDD
  §5.1. Every object miLLM's tests emit validates against it.

⚠ NO FIELD CAN CARRY TEXT. There is no place for a prompt, a reply, decoded context, token ids or
positions, activations, weights, acknowledgement text, an exception message or a credential:
every string property has an enum, a pattern or a format, and the contract test walks the strict
schema to keep it that way. `context_text` stays on `GET /api/probes/events/{id}`, outside.

⚠ ONE OBJECT, TWO MOMENTS (D38.23). The FINALIZED object (header, terminal chunk, body field,
Batch line) has neither `persistence` nor `source_event_id`; the RECORDED object (stored row,
socket, events API, session cells, telemetry) is the same object plus exactly those two. `wire()`
is the only serializer, so the two cannot drift: it drops each of the two while it is unset.

Field names are the suite's (cross-document review 2026-10-23): `detector_digest`,
`detector_digest_source`, `identity_algorithm`, `decision_digest` and `decision_revision` are
spelled the same in miStudio 039's `identity` block, in `probe_events` / `probes`, on
`GET /api/probes/{id}` and in miLLM 039's set reports.
"""

from __future__ import annotations

import copy
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter

from millm.services.probe_reasons import REASON_PATTERN, REASONS

SCHEMA_ID = "mi.probe.result/v1"
DEFINITION_SCHEMA = "mistudio.probe-definition/v1"
IDENTITY_ALGORITHM = "mistudio.probe-identity/1"

#: An identifier: request, batch, session, calibration and rule-set ids.
ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
DIGEST_OR_UNKNOWN = r"^([0-9a-f]{64}|unknown)$"
HEX64 = r"^[0-9a-f]{64}$"
RFC3339_UTC = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z$"
#: A probe's own name. Names are not restricted at import (FTID X-04), so a name outside this
#: pattern is reported `null` and counted (OI-3, D38.59). Parentheses are admitted because the
#: importer's own de-duplication writes `name (2)` (FTID §16 X-14).
NAME_PATTERN = r"^[A-Za-z0-9._:+\-() ]{1,200}$"
#: The served model's name: a repo id, a custom name, a quantization tag (`repo:Q8`).
MODEL_NAME_PATTERN = r"^([A-Za-z0-9._:/+@\- ]{1,255})$"
REVISION_PATTERN = r"^([0-9a-f]{40}|[A-Za-z0-9._/\-]{1,100})$"
PRECISION_PATTERN = r"^[a-z0-9_]{1,32}$"
BEHAVIOUR_KEY_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"
SEMVER_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.\-]+)?$"
#: The open schema's pattern for a value added to a published list later in v1 (D38.10).
OPEN_VALUE_PATTERN = r"^[a-z][a-z0-9_.\-]{0,63}$"
#: The strict schema's probe id: the importer's `pr_` + 12 hex (`probe_service.py`).
STRICT_PROBE_ID_PATTERN = r"^pr_[0-9a-f]{12}$"

ORIGINS = ("live", "batch", "stateless")
ENGINES = ("transformers", "llamacpp", "cbm")
EXECUTION_MODES = ("serial", "batched", "continuous_batching", "speculative")
WINDOW_NAMES = ("all", "prompt", "response", "last_user")
STATUSES = ("final", "partial", "unavailable")
PERSISTENCE = ("persisted", "not_persisted", "unknown")
DIGEST_SOURCES = ("producer", "computed", "unknown")
PARITY = ("passed", "failed", "never_run", "unknown")
STEERING_KINDS = ("none", "profile", "circuit", "inline", "unknown")
HOOKS = ("resid_post", "unknown")
BASES = ("residual", "sae_features", "unknown")
RULE_NAMES = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max", "unknown")
#: What the score answers. Every v1 probe is a binary linear readout of its definition's positive
#: class; `unknown` when the descriptor is missing (FTID §16 X-14: the BRD left the field path to
#: the FTDD, which named an enum and no value).
TARGETS = ("positive_class", "unknown")
SCORE_SEMANTICS = ("raw_margin",)


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


class Producer(_Closed):
    name: Literal["millm"] = "millm"
    version: str = Field(pattern=SEMVER_PATTERN)


class BehaviourOut(_Closed):
    """The behaviour the probe's DEFINITION names (never a code constant, FR-38.1)."""

    key: str = Field(pattern=BEHAVIOUR_KEY_PATTERN)
    version: Optional[int] = Field(default=None, ge=1)


class AttachedRuleSet(_Closed):
    rule_set_id: str = Field(pattern=ID_PATTERN)
    rule_set_digest: str = Field(pattern=HEX64)


class SetBlock(_Closed):
    """Part A's behaviour-set slot (OQ-38.3 (a), D31). 038 always sends `set: null`."""

    armed_members: int = Field(ge=0)
    total_members: int = Field(ge=0)
    complete: bool
    missing_probe_ids: list[Annotated[str, Field(pattern=ID_PATTERN)]] = Field(default_factory=list)
    members_digest: str = Field(pattern=HEX64)
    attached_rule_sets: list[AttachedRuleSet] = Field(default_factory=list)


class SteeringRef(_Closed):
    """What steered the residual this result read, from the request's steering report.

    `unknown` when no report was published: a result never claims "unsteered" it cannot show.
    """

    kind: Literal["none", "profile", "circuit", "inline", "unknown"]
    id: Optional[str] = Field(default=None, pattern=ID_PATTERN)


class DecisionOut(_Closed):
    calibration_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    target_fpr: Optional[float] = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    provisional: bool


class _Common(_Closed):
    """Identity, correlation and outcome fields both kinds carry."""

    schema_: Literal["mi.probe.result/v1"] = Field(default=SCHEMA_ID, alias="schema")
    result_id: str = Field(pattern=UUID_PATTERN)
    event_time: str = Field(pattern=RFC3339_UTC)
    event_time_source: Literal["finalized", "inserted"] = "finalized"
    origin: Literal["live", "batch", "stateless"]
    producer: Producer
    migrated: bool = False
    # correlation
    request_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    trace_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    span_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    #: RECORDED copies only (`probe_events.id`); absent from a finalized object.
    source_event_id: Optional[int] = Field(default=None, ge=1)
    batch_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    batch_line: Optional[int] = Field(default=None, ge=1)
    session_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    turn_index: Optional[int] = Field(default=None, ge=0)
    # grouping
    behaviour: Optional[BehaviourOut] = None
    set_: Optional[SetBlock] = Field(default=None, alias="set")
    # outcome
    outcome: Literal["completed", "skipped", "failed", "interrupted"]
    reason: Optional[str] = Field(default=None, pattern=REASON_PATTERN)
    #: RECORDED copies only; absent from a finalized object.
    persistence: Optional[Literal["persisted", "not_persisted", "unknown"]] = None
    window: Literal["all", "prompt", "response", "last_user"]
    status: Literal["final", "partial", "unavailable"] = "final"


class ProbeResult(_Common):
    """`kind: "probe"` — what one armed probe concluded over one window of one request."""

    kind: Literal["probe"] = "probe"
    choice_index: Optional[int] = Field(default=None, ge=0)
    input_index: Optional[int] = Field(default=None, ge=0)
    # detector
    probe_id: str = Field(pattern=ID_PATTERN)
    probe_name: Optional[str] = Field(default=None, pattern=NAME_PATTERN)
    detector_digest: str = Field(default="unknown", pattern=DIGEST_OR_UNKNOWN)
    detector_digest_source: Literal["producer", "computed", "unknown"] = "unknown"
    identity_algorithm: Optional[Literal["mistudio.probe-identity/1"]] = None
    definition_schema: Literal["mistudio.probe-definition/v1"] = DEFINITION_SCHEMA
    target: Literal["positive_class", "unknown"] = "unknown"
    score_semantics: Literal["raw_margin"] = "raw_margin"
    layer: Optional[int] = Field(default=None, ge=0)
    hook: Literal["resid_post", "unknown"] = "unknown"
    basis: Literal["residual", "sae_features", "unknown"] = "unknown"
    rule: Literal["mean", "max", "last", "softmax", "attention", "rolling_mean_max",
                  "unknown"] = "unknown"
    # execution
    served_model: str = Field(default="unknown", pattern=MODEL_NAME_PATTERN)
    model_revision: str = Field(default="unknown", pattern=REVISION_PATTERN)
    template_sha256: str = Field(default="unknown", pattern=DIGEST_OR_UNKNOWN)
    precision: str = Field(default="unknown", pattern=PRECISION_PATTERN)
    engine: Literal["transformers", "llamacpp", "cbm"] = "transformers"
    execution_mode: Literal["serial", "batched", "continuous_batching", "speculative"] = "serial"
    steering_ref: SteeringRef = Field(default_factory=lambda: SteeringRef(kind="unknown"))
    # measurement
    definition_scope: Literal["all", "prompt", "response", "last_user", "unknown"] = "unknown"
    n_scored_tokens: int = Field(default=0, ge=0)
    scored: bool
    # score
    score: Optional[float] = Field(default=None, allow_inf_nan=False)
    probability: None = None
    calibration_ref: None = None
    # decision
    threshold: Optional[float] = Field(default=None, allow_inf_nan=False)
    verdict: Optional[bool] = None
    #: v1 legacy duplicate of `verdict` (D38.10), kept for the life of v1.
    fires: Optional[bool] = None
    comparator: Literal[">="] = ">="
    decision_digest: str = Field(default="unknown", pattern=DIGEST_OR_UNKNOWN)
    decision_revision: Optional[int] = Field(default=None, ge=1)
    threshold_revision: int = Field(default=1, ge=1)
    decision: DecisionOut
    # evidence
    rung: int = Field(ge=0, le=3)
    rung_language: str = Field(pattern=r"^[a-z0-9 ,()'.\-]{1,80}$")
    parity_status: Literal["passed", "failed", "never_run", "unknown"] = "unknown"
    model_load_id: str = Field(default="unknown", pattern=r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|unknown)$")
    # timing
    #: Milliseconds the probe's finish took for this request; `null` = not recorded (migrated).
    evaluation_ms: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    completed_at: str = Field(pattern=RFC3339_UTC)

    #: The probe's name as the runtime knows it, for the LEGACY `name` key only. Never
    #: serialized: the contract's `probe_name` is pattern-gated and may be `null` (OI-3), while
    #: the legacy surfaces have always shown the name verbatim.
    _legacy_name: Optional[str] = PrivateAttr(default=None)


class RuleResult(_Common):
    """`kind: "rule"` — Part A's rule signal (D38.2). Defined in v1 so miLLM 039 emits it without
    a schema change; 038 never produces one."""

    kind: Literal["rule"] = "rule"
    rule_set_id: str = Field(pattern=ID_PATTERN)
    rule_set_digest: str = Field(pattern=HEX64)
    rule_set_version: int = Field(ge=1)
    signal: str = Field(pattern=ID_PATTERN)
    form: str = Field(pattern=OPEN_VALUE_PATTERN)
    fires: Optional[bool] = None
    count: int = Field(default=0, ge=0)
    #: Rule results have no legacy name; declared so the shared surfaces (order, record) can copy
    #: it alike for both kinds (039 S2).
    _legacy_name: Optional[str] = PrivateAttr(default=None)


ProbeResultV1 = Annotated[Union[ProbeResult, RuleResult], Field(discriminator="kind")]
_ADAPTER: TypeAdapter[Any] = TypeAdapter(ProbeResultV1)

#: Fields absent from a FINALIZED object; present on a RECORDED one when set.
RECORDED_ONLY = ("persistence", "source_event_id")


def wire(result: BaseModel) -> dict[str, Any]:
    """THE serializer of a result: JSON-mode, contract names, the two recorded-only fields dropped
    while unset. Every surface uses this, so a finalized and a recorded copy differ in exactly
    those two keys (T-S1-03)."""
    data = result.model_dump(mode="json", by_alias=True)
    for key in RECORDED_ONLY:
        if data.get(key) is None:
            data.pop(key, None)
    return data


def parse(data: dict[str, Any]) -> Any:
    """A stored or received object back into the model (by `kind`)."""
    return _ADAPTER.validate_python(data)


# ── schema generation ────────────────────────────────────────────────────────────

#: The conditional rules (FTDD §5.1), enforced by the builder (`probe_result._enforce`) and
#: stated in the strict schema. Each is violated once in the contract test.
CONDITIONAL_RULES: list[dict[str, Any]] = [
    {"if": {"properties": {"outcome": {"const": "completed"}}, "required": ["outcome"]},
     "then": {"properties": {"scored": {"const": True}, "score": {"type": "number"},
                             "status": {"const": "final"}, "reason": {"type": "null"}}}},
    {"if": {"properties": {"outcome": {"enum": ["skipped", "failed"]}}, "required": ["outcome"]},
     "then": {"properties": {"scored": {"const": False}, "score": {"type": "null"},
                             "verdict": {"type": "null"}, "reason": {"type": "string"}},
              "required": ["reason"]}},
    {"if": {"properties": {"outcome": {"const": "interrupted"}}, "required": ["outcome"]},
     "then": {"properties": {"verdict": {"type": "null"},
                             "status": {"enum": ["partial", "unavailable"]},
                             "reason": {"type": "string"}},
              "required": ["reason"]}},
    {"if": {"properties": {"status": {"const": "partial"}}, "required": ["status"]},
     "then": {"properties": {"score": {"type": "number"},
                             "n_scored_tokens": {"minimum": 1}}}},
    {"if": {"properties": {"status": {"const": "unavailable"}}, "required": ["status"]},
     "then": {"properties": {"score": {"type": "null"}}}},
    {"if": {"properties": {"threshold": {"type": "null"}}, "required": ["threshold"]},
     "then": {"properties": {"verdict": {"type": "null"}}}},
]

_RUNG_PHRASES = (
    "trained", "detects on held-out data", "detects on unseen tasks",
    "detects on unseen tasks, compared with a judge",
)

_DESCRIPTIONS = {
    "reason": "A code from the published list (millm/services/probe_reasons.py); null when "
              "outcome is completed. Consumers read an unknown code as other.",
    "persistence": "Recorded copies only.",
    "source_event_id": "Recorded copies only: the probe_events.id this result is stored as.",
    "decision_revision": "miStudio's revision of the decision (the producer's); null when the "
                         "producer stated none.",
    "threshold_revision": "miLLM's own counter of the bar that judged this result.",
}


def _base_schema() -> dict[str, Any]:
    schema = _ADAPTER.json_schema(by_alias=True, mode="serialization")
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs", {})
    for name in ("ProbeResult", "RuleResult"):
        body = defs[name]
        props = body["properties"]
        # Every field is always sent (nulls explicit) except the two recorded-only ones.
        body["required"] = sorted(k for k in props if k not in RECORDED_ONLY)
        for key, text in _DESCRIPTIONS.items():
            if key in props:
                props[key]["description"] = text
    schema["$id"] = "https://hitsai.net/schemas/millm/probe-result-v1.json"
    schema["title"] = SCHEMA_ID
    schema["description"] = (
        "One probe result, the same object on every miLLM surface (header projection, terminal "
        "chunk, millm_probe_verdicts body field, stored event, probe:event socket, events API, "
        "session cells, score route, Batch lines, telemetry). Generated from "
        "millm/api/schemas/probe_result.py; do not hand-edit."
    )
    return schema


def _walk(node: Any, fn: Any) -> None:
    if isinstance(node, dict):
        fn(node)
        for value in node.values():
            _walk(value, fn)
    elif isinstance(node, list):
        for item in node:
            _walk(item, fn)


def strict_schema() -> dict[str, Any]:
    """Closed enums, closed objects, the published reason list and the conditional rules."""
    schema = _base_schema()
    defs = schema["$defs"]
    probe = defs["ProbeResult"]
    probe["allOf"] = copy.deepcopy(CONDITIONAL_RULES)
    probe["properties"]["probe_id"] = {**probe["properties"]["probe_id"],
                                       "pattern": STRICT_PROBE_ID_PATTERN}
    probe["properties"]["rung_language"] = {"enum": list(_RUNG_PHRASES), "type": "string"}
    for name in ("ProbeResult", "RuleResult"):
        props = defs[name]["properties"]
        props["reason"] = {
            "anyOf": [{"enum": sorted(REASONS), "type": "string"}, {"type": "null"}],
            "description": _DESCRIPTIONS["reason"],
        }
    return schema


def open_schema() -> dict[str, Any]:
    """Objects open; every listed value also admits an identifier (D38.10)."""
    schema = _base_schema()

    def loosen(node: dict[str, Any]) -> None:
        if node.get("additionalProperties") is False:
            node["additionalProperties"] = True
        props = node.get("properties")
        if not isinstance(props, dict):
            return
        for key, prop in list(props.items()):
            if key in ("schema", "kind", "comparator"):
                continue  # identity and the one fixed operator stay closed
            props[key] = _open_value(prop)

    _walk(schema, loosen)
    for name in ("ProbeResult", "RuleResult"):
        props = schema["$defs"][name]["properties"]
        props["reason"] = {
            "anyOf": [{"pattern": REASON_PATTERN, "type": "string"}, {"type": "null"}],
            "description": _DESCRIPTIONS["reason"] + " Known codes: "
            + ", ".join(sorted(REASONS)) + ".",
        }
    return schema


def _open_value(prop: dict[str, Any]) -> dict[str, Any]:
    """A closed enum/const string becomes `anyOf [it, an identifier]`; nullables keep null."""
    if "const" in prop and isinstance(prop["const"], str):
        return {"anyOf": [prop, {"type": "string", "pattern": OPEN_VALUE_PATTERN}],
                **({"default": prop["default"]} if "default" in prop else {})}
    if "enum" in prop and all(isinstance(v, str) for v in prop["enum"]):
        return {"anyOf": [prop, {"type": "string", "pattern": OPEN_VALUE_PATTERN}],
                **({"default": prop["default"]} if "default" in prop else {})}
    if "anyOf" in prop:
        return {**prop, "anyOf": [_open_value(p) if isinstance(p, dict) else p
                                  for p in prop["anyOf"]]}
    return prop
