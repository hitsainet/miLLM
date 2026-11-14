"""Behaviour verdict definitions and a vetoing rule set for miLLM's G19 tests (ENH-006 G19).

A definition is built the way miStudio's `behaviour_verdict.build_definition` writes one: members in a
given order with their detector and decision digests, the members digest over the sorted detector
digests, the candidate, the rule side, and combiner vectors whose expectations come from the combiner
on the definition's own fields (miStudio computes them with ITS combiner; the shared cases pin that the
two agree). Random behaviour keys (`behaviour_factory.random_key`), nothing named for a product.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Optional, Sequence

from miserver.services import behaviour_verdict as bv
from tests.support import rule_set_factory as rf

MASKED = "ZQ-9999"


def masked_document(key: str) -> dict[str, Any]:
    """`rule_set_factory.document` plus a MASKED form that vetoes the token, and an explicit positive
    (D53) naming `token` — so a masked token is a veto and an unmasked one a match."""
    doc = copy.deepcopy(rf.document(key))
    doc["signals"].append({"id": "msk", "kind": "pattern", "pattern": r"\bZQ-9999\b",
                           "ignore_case": False, "fold_fullwidth_ascii": False, "reads": "text"})
    doc["expressions"]["masked"] = {"signal": "msk"}
    doc["vetoes"] = [{"id": "mask", "when": "masked", "removes": ["tok"]}]
    doc["match"] = {"version": 1, "any_of": ["token"]}
    doc["outputs"].append({"column": "masked", "type": "bool", "value": {"fired": "masked"}})
    for case in doc["cases"]:
        case["expect"]["outputs"]["masked"] = False
    doc["cases"].append({"id": "masked-token", "input": {"text": f"see {MASKED} here"}, "note": "constructed",
                         "source": {"kind": "constructed"},
                         "expect": {"outputs": {"token": False, "tokens": 0, "cued": False, "number": False,
                                                "lexicon": False, "both": False, "masked": True},
                                    "vetoed": True}})
    return doc


def rule_set_digest(doc: dict[str, Any]) -> str:
    return hashlib.sha256(bv.jcs_bytes(doc)).hexdigest()


def armed_members(key: str) -> list[Any]:
    from miserver.services.probe_runtime import ProbeRuntimeState

    return sorted((p for p in ProbeRuntimeState().armed()
                   if getattr(p.descriptor, "behaviour_key", None) == key), key=lambda p: p.probe_id)


def definition(key: str, members: Sequence[Any], *, candidate: str = "rules_or_any_of",
               rule_doc: Optional[dict[str, Any]] = None, veto_scope: Optional[Sequence[int]] = None,
               windows: Sequence[str] = ("all",), params: Optional[dict[str, Any]] = None,
               threshold: Optional[float] = None, k: Optional[int] = None,
               digest: Optional[str] = None) -> dict[str, Any]:
    """A `mistudio.behaviour-verdict/v1` definition over `members` (armed probes or digests)."""
    rules = candidate.startswith("rules_or_")
    base = candidate[len("rules_or_"):] if rules else candidate
    kind = base.split(":")[0]
    digests = [getattr(getattr(m, "descriptor", None), "detector_digest", None) or m for m in members]
    decisions = [getattr(getattr(getattr(m, "descriptor", None), "decision", None), "decision_digest", None)
                 for m in members]
    scope = set(range(len(digests))) if veto_scope is None else set(veto_scope)
    n = len(digests)
    out: dict[str, Any] = {
        "schema": bv.SCHEMA_ID,
        "behaviour": {"key": key, "version": 1},
        "model": {"id": "m_test", "template_sha256": None},
        "members": [{"detector_digest": d, "decision_digest": decisions[j], "decision_revision": 1,
                     "probe_id": f"pm_{j:012d}", "name": None, "role": "primary" if j == 0 else "candidate",
                     "forms": [], "veto_applies": rules and j in scope} for j, d in enumerate(digests)],
        "members_digest": bv.members_digest(digests),
        "candidate": {"name": candidate, "kind": kind, "rules": rules,
                      "k": (k if k is not None else {"any_of": 1, "all_of": n}.get(kind)) if kind in bv.VERDICT_KINDS
                      else None,
                      "params": params or {}, "threshold": threshold},
        "rules": ({"rule_set_digest": digest or rule_set_digest(rule_doc), "engine_digest": None,
                   "match": {"source": "explicit", "any_of": list((rule_doc.get("match") or {}).get("any_of") or [])},
                   "veto_scope": "all" if len(scope) == n else "per_member"} if rules else None),
        "windows": list(windows),
        "target_fpr": 0.05,
        "calibration": {"report_id": "pbr_0123456789ab", "report_sha256": "f" * 64, "view_id": "pmd_cal",
                        "n_negatives": 1000, "realised_union_fpr": 0.03, "over_budget": False,
                        "acknowledged": None},
        "end_to_end_vectors": [],
        "end_to_end_status": "not_generated",
        "statement": "fixture",
    }
    out["combiner_vectors"] = vectors(out)
    return out


def vectors(definition: dict[str, Any]) -> list[dict[str, Any]]:
    digests = [m["detector_digest"] for m in definition["members"]]
    score = definition["candidate"]["kind"] not in bv.VERDICT_KINDS

    def members(firing: Sequence[int], missing: Sequence[int] = ()) -> dict[str, Any]:
        return {d: (None if j in missing else {"fires": j in firing, "score": (3.0 if j in firing else -3.0)
                                                if score else None}) for j, d in enumerate(digests)}

    clean = {"match": False, "veto": False, "unknown": False} if definition["candidate"]["rules"] else None
    shapes = [("none_fire", members([]), clean), ("all_fire", members(range(len(digests))), clean),
              ("first_missing", members([], [0]), clean)]
    shapes += [(f"only_{j}", members([j]), clean) for j in range(len(digests))]
    if clean is not None:
        shapes += [(f"veto_only_{j}", members([j]), {**clean, "veto": True}) for j in range(len(digests))]
        shapes += [("match", members([]), {**clean, "match": True}),
                   ("unknown", members([0]), {**clean, "unknown": True})]
    return [{"id": i, "members": m, "rules": r, "expect": bv.combine(definition, m, r)} for i, m, r in shapes]
