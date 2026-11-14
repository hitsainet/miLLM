"""The behaviour verdict: ONE served answer for a behaviour's probe stack (ENH-006 G19).

Operator decisions G19-1 … G19-6 (2026-11-08). **G19-1 revises D9**: miServer serves an ADDITIVE
`kind: "behaviour"` result built from a miWorkbench-exported definition (`mistudio.behaviour-verdict/v1`,
vendored byte-identically at `docs/schemas/behaviour-verdict-v1.json`). The member results — one per
(probe, window) and one per rule expression and window — are unchanged, so an application that blends
for itself loses nothing, and a behaviour with no definition answers byte-identically to before.

**The combiner** (`combine`), per behaviour and window, is 040's union with miWorkbench's precedence,
pinned by the shared cases (`docs/schemas/behaviour-verdict-cases.json`, owned by miWorkbench):

1. **rule match** — any of the definition's match expressions (D53's explicit `match`, G19-4) true on
   the window's text AFTER the vetoes → `fires: true`, `decided_by: rule_match`; a match beats a veto;
2. **rule veto** — a veto removed a span and nothing matched → the members whose `veto_applies` is true
   are suppressed (G19-3);
3. **members** — a verdict kind counts members whose OWN served verdict fired (`>= k`; no bar of its
   own: each member was already judged at its own bar and length bands); a score kind standardises
   each member's score with the recorded mean and standard deviation and compares `>=` one bar;
4. **unknown rules** (timed out, failed, pending operator code, the rule set not attached) are no
   match and no veto: the members decide (040 FR-35);
5. **missing members** (not armed, not scored, no decision) → `fires: true` only when a rule or a
   present member proves it, else `fires: null` with `decided_by: members_missing` — never `false`
   (G19-5).

Rule semantics at serving are the ROW semantics (`CompiledRuleSet.row_verdict`: signals, vetoes,
expressions), never `serve` — the existing `kind: "rule"` results keep their pre-veto meaning.

⚠ NO TEXT, NO SPANS. A verdict carries ids, digests, numbers and codes. The window text is read by the
rule evaluator and dropped; the leak test searches the verdict, the stored row and the telemetry record
for the text and its base64 form.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import rfc8785

logger = logging.getLogger(__name__)

SCHEMA_ID = "mistudio.behaviour-verdict/v1"
CASES_FORMAT = "mistudio.behaviour-verdict-cases/1"
SCHEMAS = Path(__file__).resolve().parents[2] / "docs" / "schemas"
SCHEMA_PATH = SCHEMAS / "behaviour-verdict-v1.json"
CASES_PATH = SCHEMAS / "behaviour-verdict-cases.json"
#: A blended value agrees within this (summation order may differ from miStudio's NumPy);
#: everything else must agree exactly.
VALUE_TOLERANCE = 1e-9
VERDICT_KINDS = ("any_of", "all_of", "k_of_n")
SCORE_KINDS = ("stack", "max", "best_single")
DECIDED_BY = ("rule_match", "members", "vetoed", "members_missing", "not_evaluated")

#: Reason codes a verdict's rule side can carry (published in `probe_reasons.REASONS`).
RULE_SET_NOT_ATTACHED = "rule_set_not_attached"
MEMBER_DECISION_CHANGED = "member_decision_changed"


class DefinitionRefused(ValueError):
    """An attach refused: `code` is the error class's tag, `details` name where — never text."""

    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


# ── identity ─────────────────────────────────────────────────────────────────────


def jcs_bytes(value: Any) -> bytes:
    return rfc8785.dumps(value)


def members_digest(detector_digests: Iterable[str]) -> str:
    """sha256 of the JCS of the SORTED detector digests — computed alike in miStudio."""
    return hashlib.sha256(jcs_bytes(sorted(str(d) for d in detector_digests))).hexdigest()


def definition_sha256(definition: Mapping[str, Any]) -> str:
    """The definition's identity: sha256 of its JCS bytes as received (miStudio exports JCS)."""
    return hashlib.sha256(jcs_bytes(definition)).hexdigest()


# ── the combiner (the shared cases pin it) ───────────────────────────────────────


def combine(definition: Mapping[str, Any], members_in: Mapping[str, Optional[Mapping[str, Any]]],
            rules_in: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """One behaviour verdict from one window's member inputs and rule inputs.

    `members_in`: detector digest → `{"fires": bool | None, "score": float | None}`, or None / absent
    (not armed, nothing produced). `rules_in`: `{"match", "veto", "unknown"}` or None (unknown)."""
    candidate = definition["candidate"]
    kind = candidate["kind"]
    with_rules = bool(candidate.get("rules"))
    members = list(definition["members"])
    digests = [str(m["detector_digest"]) for m in members]
    scoped = {str(m["detector_digest"]): bool(m.get("veto_applies", True)) for m in members}
    every_scoped = all(scoped.values())

    unknown = with_rules and (rules_in is None or bool(rules_in.get("unknown")))
    match = bool(with_rules and not unknown and rules_in.get("match"))
    # ⚠ A match beats a veto (040 FR-27): a veto holds only when nothing matched.
    veto = bool(with_rules and not unknown and rules_in.get("veto") and not match)

    needed = "fires" if kind in VERDICT_KINDS else "score"
    present = {d: members_in.get(d) for d in digests}
    missing = [d for d in digests if present[d] is None or present[d].get(needed) is None]
    fired = [d for d in digests if present[d] is not None and present[d].get("fires") is True]
    vetoed = [d for d in fired if veto and scoped[d]]

    value: Optional[float]
    if kind in VERDICT_KINDS:
        # With every member in scope the veto suppresses the whole blend below; a partial scope
        # suppresses its members here and the union then carries no veto of its own.
        value = float(len([d for d in fired if every_scoped or d not in vetoed]))
        base = value >= float(candidate["k"])
    elif missing:
        value, base = None, False
    else:
        value = _score_value(kind, candidate.get("params") or {},
                             [float(present[d]["score"]) for d in digests])
        threshold = candidate.get("threshold")
        base = threshold is not None and value >= float(threshold)
    if with_rules:
        final = match or (base and not (veto and every_scoped))
    else:
        final = base

    if final:
        fires: Optional[bool] = True
        decided_by = "rule_match" if match else "members"
    elif missing:
        fires, decided_by = None, "members_missing"  # G19-5: never a false clean
    elif veto:
        fires, decided_by = False, "vetoed"
    else:
        fires, decided_by = False, "members"
    return {
        "fires": fires,
        "decided_by": decided_by,
        "members_fired": fired,
        "members_vetoed": vetoed,
        "members_missing": missing,
        "rules": ({"match": None if unknown else match, "veto": None if unknown else veto,
                   "unknown": bool(unknown)} if with_rules else None),
        "value": value,
    }


def _score_value(kind: str, params: Mapping[str, Any], scores: Sequence[float]) -> float:
    """float64, member order, left to right — the arithmetic the shared rule text states."""
    z = [(s - float(params["mean"][j])) / float(params["std"][j]) for j, s in enumerate(scores)]
    if kind == "stack":
        total = 0.0
        for j, zj in enumerate(z):
            total += zj * float(params["weights"][j])
        return total + float(params["intercept"])
    if kind == "max":
        return max(z)
    return z[int(params["member_index"])]


def agrees(got: Mapping[str, Any], expect: Mapping[str, Any]) -> bool:
    for key in ("fires", "decided_by", "members_fired", "members_vetoed", "members_missing", "rules"):
        if got.get(key) != expect.get(key):
            return False
    a, b = got.get("value"), expect.get("value")
    if (a is None) != (b is None):
        return False
    return a is None or abs(float(a) - float(b)) <= VALUE_TOLERANCE


# ── checking a definition (attach) ───────────────────────────────────────────────


@lru_cache(maxsize=1)
def _validator() -> Any:
    from jsonschema import Draft202012Validator

    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def check(definition: Any, *, key: str) -> dict[str, Any]:
    """Every check that needs no registry: the vendored schema, the path's key, the members digest,
    the candidate's shape, a score kind's full veto scope, and EVERY combiner vector recomputed here
    (`verdict_parity_failed` on the first disagreement — layer 3 of parity). Returns a report."""
    if not isinstance(definition, dict):
        raise DefinitionRefused("behaviour_verdict_invalid", "the definition is not a JSON object")
    errors = sorted(_validator().iter_errors(definition), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        raise DefinitionRefused(
            "behaviour_verdict_invalid", "the definition does not match "
            f"{SCHEMA_ID} at /{'/'.join(str(p) for p in first.absolute_path)}",
            {"pointer": "/" + "/".join(str(p) for p in first.absolute_path),
             "validator": first.validator, "errors": len(errors)})
    if definition["behaviour"]["key"] != key:
        raise DefinitionRefused("behaviour_verdict_key_mismatch",
                                f"the definition is for {definition['behaviour']['key']!r}, not {key!r}",
                                {"behaviour": key, "definition_behaviour": definition["behaviour"]["key"]})
    digests = [m["detector_digest"] for m in definition["members"]]
    if len(set(digests)) != len(digests):
        raise DefinitionRefused("behaviour_verdict_invalid", "a detector digest is named twice",
                                {"pointer": "/members"})
    stated = definition["members_digest"]
    computed = members_digest(digests)
    if stated != computed:
        raise DefinitionRefused("behaviour_verdict_invalid", "members_digest is not the digest of the "
                                "members' detector digests", {"stated": stated, "computed": computed})
    candidate = definition["candidate"]
    kind, n = candidate["kind"], len(digests)
    if kind in VERDICT_KINDS:
        k = candidate.get("k")
        want = {"any_of": 1, "all_of": n}.get(kind)
        if not isinstance(k, int) or k < 1 or k > n or (want is not None and k != want):
            raise DefinitionRefused("behaviour_verdict_invalid", f"{kind} needs k in 1..{n}",
                                    {"pointer": "/candidate/k", "k": k})
    else:
        params = candidate.get("params") or {}
        need = {"stack": ("mean", "std", "weights", "intercept"), "max": ("mean", "std"),
                "best_single": ("mean", "std", "member_index")}[kind]
        absent = [p for p in need if p not in params]
        if absent or candidate.get("threshold") is None:
            raise DefinitionRefused("behaviour_verdict_invalid", f"{kind} needs {list(need)} and a threshold",
                                    {"pointer": "/candidate/params", "absent": absent})
        for name in ("mean", "std", "weights"):
            if name in params and len(params[name]) != n:
                raise DefinitionRefused("behaviour_verdict_invalid", f"params.{name} has "
                                        f"{len(params[name])} entries for {n} members",
                                        {"pointer": f"/candidate/params/{name}"})
        if "member_index" in params and not 0 <= int(params["member_index"]) < n:
            raise DefinitionRefused("behaviour_verdict_invalid", "params.member_index is not a member",
                                    {"pointer": "/candidate/params/member_index"})
        if candidate.get("rules") and not all(m["veto_applies"] for m in definition["members"]):
            raise DefinitionRefused("behaviour_verdict_invalid", "a score kind needs every member in the "
                                    "veto's scope", {"pointer": "/members", "candidate": candidate["name"]})
    if bool(candidate.get("rules")) != (definition.get("rules") is not None):
        raise DefinitionRefused("behaviour_verdict_invalid", "a rules_or_* candidate names a rule set, and "
                                "only it does", {"pointer": "/rules"})
    if definition.get("rules") is not None and not definition["rules"]["match"]["any_of"]:
        raise DefinitionRefused("behaviour_verdict_invalid", "the rule side names no match expression",
                                {"pointer": "/rules/match/any_of"})
    failed = [v["id"] for v in definition["combiner_vectors"]
              if not agrees(combine(definition, v["members"], v["rules"]), v["expect"])]
    if failed:
        raise DefinitionRefused("verdict_parity_failed", f"{len(failed)} of the definition's combiner vectors "
                                "disagree with miServer's combiner", {"vectors": failed[:20],
                                                                    "failed": len(failed)})
    return {"combiner_vectors": len(definition["combiner_vectors"]),
            "end_to_end_vectors": len(definition.get("end_to_end_vectors") or []),
            "end_to_end_status": definition.get("end_to_end_status")}


# ── serving ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttachedVerdict:
    """One behaviour's attached definition, as the request path reads it (no DB read per request)."""

    key: str
    definition: Mapping[str, Any]
    sha256: str
    #: `{detector digest: miStudio's decision digest}` — a member armed under another decision makes
    #: the verdict provisional (`member_decision_changed`).
    decisions: Mapping[str, Optional[str]] = field(default_factory=dict)

    @property
    def windows(self) -> tuple[str, ...]:
        return tuple(self.definition.get("windows") or ("all",))

    @property
    def rule_set_digest(self) -> Optional[str]:
        rules = self.definition.get("rules")
        return rules.get("rule_set_digest") if rules else None

    @property
    def match_names(self) -> tuple[str, ...]:
        rules = self.definition.get("rules")
        return tuple(rules["match"]["any_of"]) if rules else ()


@dataclass(frozen=True)
class RuleRow:
    """The rule side of one (behaviour, window): the row semantics' verdict, or unknown with why."""

    match: Optional[bool] = None
    veto: Optional[bool] = None
    unknown: bool = True
    reason: Optional[str] = None
    #: True while operator code decides it (revised in place after the response).
    pending: bool = False


_registry: dict[str, AttachedVerdict] = {}
generation = 0


def current() -> dict[str, AttachedVerdict]:
    return _registry


def install(attached: Iterable[AttachedVerdict]) -> None:
    """Swap the registry (rebuild on attach, clear and at startup)."""
    global _registry, generation
    _registry = {a.key: a for a in attached}
    generation += 1


def forget_for_tests() -> None:
    install(())


def attached_from(key: str, definition: Mapping[str, Any]) -> AttachedVerdict:
    return AttachedVerdict(
        key=key, definition=definition, sha256=definition_sha256(definition),
        decisions={m["detector_digest"]: m.get("decision_digest") for m in definition["members"]})


def armed_keys(probes: Iterable[Any]) -> set[str]:
    out = set()
    for probe in probes or ():
        key = getattr(getattr(probe, "descriptor", None), "behaviour_key", None)
        if isinstance(key, str) and key:
            out.add(key)
    return out


def work_for(probes: Iterable[Any]) -> list[AttachedVerdict]:
    """The attached definitions whose behaviour has an armed member on this request."""
    keys = armed_keys(probes)
    return [a for k, a in sorted(current().items()) if k in keys]


def _member_index(probes: Iterable[Any], key: str) -> dict[str, Any]:
    """`{detector digest: armed probe}` for the armed probes of `key`."""
    out: dict[str, Any] = {}
    for probe in probes or ():
        desc = getattr(probe, "descriptor", None)
        if getattr(desc, "behaviour_key", None) == key and getattr(desc, "detector_digest", None):
            out.setdefault(desc.detector_digest, probe)
    return out


def _ref(digest: str, by_digest: Mapping[str, Any]) -> dict[str, Any]:
    probe = by_digest.get(digest)
    return {"detector_digest": digest, "probe_id": getattr(probe, "probe_id", None)}


def verdict_items(attached: AttachedVerdict, verdicts: Sequence[Any], probes: Sequence[Any], *,
                  rule_rows: Mapping[tuple[str, str], RuleRow], not_scored_reason: Optional[str],
                  interrupted: Optional[str]) -> list[dict[str, Any]]:
    """The behaviour result FIELDS (no shared fields, no `result_id`) for one definition, one per
    window, in the definition's window order. Never raises on a well-formed definition."""
    definition = attached.definition
    by_digest = _member_index(probes, attached.key)
    by_probe_window = {(v.probe_id, v.window): v for v in verdicts or ()}
    candidate = definition["candidate"]
    out = []
    for window in attached.windows:
        members_in: dict[str, Optional[dict[str, Any]]] = {}
        used: list[Any] = []
        provisional: list[str] = []
        for member in definition["members"]:
            d = member["detector_digest"]
            probe = by_digest.get(d)
            v = by_probe_window.get((probe.probe_id, window)) if probe is not None else None
            if v is None or not getattr(v, "scored", False):
                members_in[d] = None
                continue
            used.append(v)
            members_in[d] = {"fires": v.fires, "score": v.score}
            if getattr(v, "provisional", False) and "member_provisional" not in provisional:
                provisional.append("member_provisional")
            armed_decision = getattr(getattr(getattr(probe, "descriptor", None), "decision", None),
                                     "decision_digest", None)
            recorded = attached.decisions.get(d)
            if recorded and armed_decision != recorded and MEMBER_DECISION_CHANGED not in provisional:
                provisional.append(MEMBER_DECISION_CHANGED)
        row = rule_rows.get((attached.key, window)) if candidate.get("rules") else None
        rules_in = (None if row is None or row.unknown
                    else {"match": bool(row.match), "veto": bool(row.veto), "unknown": False})
        outcome, status, reason = "completed", "final", None
        if not_scored_reason:
            outcome, status, reason = "skipped", "final", not_scored_reason
        elif interrupted:
            outcome, reason = "interrupted", interrupted
            status = "partial" if used and len(used) == len(definition["members"]) else "unavailable"
        if outcome == "skipped" or status == "unavailable":
            got = {"fires": None, "decided_by": "not_evaluated", "members_fired": [],
                   "members_vetoed": [], "members_missing": [m["detector_digest"] for m in definition["members"]
                                                             if members_in.get(m["detector_digest"]) is None],
                   "rules": None, "value": None}
        else:
            got = combine(definition, members_in, rules_in)
        if row is not None and row.pending and status == "final":
            status = "partial"  # revised in place when operator code decides it (AC 7)
        rules_out = None
        if candidate.get("rules"):
            rules_out = {"match": (got["rules"] or {}).get("match"), "veto": (got["rules"] or {}).get("veto"),
                         "unknown": bool(rules_in is None),
                         "reason": None if rules_in is not None else (
                             row.reason if row is not None else RULE_SET_NOT_ATTACHED),
                         "rule_set_digest": attached.rule_set_digest}
        out.append({
            "window": window,
            "fires": got["fires"],
            "decided_by": got["decided_by"],
            "members_fired": [_ref(d, by_digest) for d in got["members_fired"]],
            "members_vetoed": [_ref(d, by_digest) for d in got["members_vetoed"]],
            "members_missing": [_ref(d, by_digest) for d in got["members_missing"]],
            "rules": rules_out,
            "candidate": {"name": candidate["name"], "k": candidate.get("k"),
                          "value": got["value"] if got["value"] is None or math.isfinite(got["value"]) else None,
                          "threshold": candidate.get("threshold")},
            "provisional": bool(provisional),
            "provisional_reasons": provisional,
            "outcome": outcome, "status": status, "reason": reason,
            "_inputs": (members_in, rules_in),
        })
    return out


def build_results(verdicts: Sequence[Any], context: Any, *, shared: Optional[dict[str, Any]] = None,
                  origin: Optional[str] = None) -> list[Any]:
    """`kind: "behaviour"` results for every attached definition whose behaviour is armed on this
    request (none attached → `[]`, so the response is byte-identical to before). Never raises in
    production; under `MISERVER_STRICT_RESULTS=1` a builder defect is raised."""
    from miserver.services.probe_result import behaviour_result, shared_fields, strict_results

    probes = list(getattr(context, "probes", None) or ())
    work = work_for(probes)
    if not work:
        return []
    out = []
    try:
        shared = shared if shared is not None else shared_fields(context, origin=origin)
        rows = getattr(context, "verdict_rules", None) or {}
        for attached in work:
            items = verdict_items(attached, verdicts, probes, rule_rows=rows,
                                  not_scored_reason=getattr(context, "not_scored_reason", None),
                                  interrupted=getattr(context, "interrupted", None))
            for item in items:
                result = behaviour_result(attached, item, shared)
                out.append(result)
                pending = getattr(context, "verdict_pending", None)
                if isinstance(pending, dict):
                    pending[(attached.key, item["window"])] = (result.result_id, item["_inputs"])
    except Exception as exc:  # noqa: BLE001 - an observer never fails the request
        if strict_results():
            raise
        logger.error("behaviour_verdict_build_failed error=%s", type(exc).__name__)
        return []
    from miserver import telemetry

    telemetry.observe(out, admitted_ids=set())
    return out
