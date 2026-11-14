"""Attaching, reading and clearing a behaviour's verdict definition; recording its results
(ENH-006 G19; `PUT/GET/DELETE /api/probes/behaviours/{key}/verdict`).

Attach checks, in order, and refuses on the first failure — nothing is stored on a refusal:

1. the definition itself (`behaviour_verdict.check`): the vendored schema, the path's key, the members
   digest, the candidate's shape, a score kind's full veto scope, and EVERY combiner vector recomputed by
   miServer's combiner (`VERDICT_PARITY_FAILED`, layer 3 of parity);
2. the members (D14, all or nothing): the behaviour's ARMED probes, by detector digest, must be exactly
   the definition's members (`BEHAVIOUR_VERDICT_MEMBERS_MISMATCH`, naming both differences);
3. the rule side of a `rules_or_*` definition: its rule set (by digest) is attached to the behaviour
   (`BEHAVIOUR_VERDICT_RULE_SET_NOT_ATTACHED`), every match expression is one of the set's, and a set that
   STATES its positive (D53) states the same one (`BEHAVIOUR_VERDICT_INVALID`, `match_mismatch`).

A member armed under another decision than the definition records is accepted and reported
(`decision_changed`): its verdicts read `provisional` with `member_decision_changed` until a new
definition is exported. Layer 4 (end-to-end vectors) is reported as the definition states it; v1
definitions carry none (`end_to_end_status: not_generated`).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from sqlalchemy import delete, select

from miserver.core.errors import (
    BehaviourVerdictInvalidError,
    BehaviourVerdictMembersMismatchError,
    BehaviourVerdictNotFoundError,
    BehaviourVerdictParityError,
    BehaviourVerdictRuleSetError,
)
from miserver.db.models.rule_set import BehaviourVerdict, BehaviourVerdictHistory, BehaviourVerdictResult
from miserver.services import behaviour_verdict as bv

logger = logging.getLogger(__name__)


def _armed_members(key: str) -> dict[str, Any]:
    """`{detector digest: armed probe}` of the behaviour, from THE REGISTRY (never the rows)."""
    from miserver.services.probe_runtime import ProbeRuntimeState

    out: dict[str, Any] = {}
    for probe in ProbeRuntimeState().armed():
        desc = getattr(probe, "descriptor", None)
        if getattr(desc, "behaviour_key", None) == key:
            out[getattr(desc, "detector_digest", None) or f"unknown:{probe.probe_id}"] = probe
    return out


def _refuse(exc: bv.DefinitionRefused) -> Exception:
    if exc.code == "verdict_parity_failed":
        return BehaviourVerdictParityError(str(exc), details=exc.details)
    return BehaviourVerdictInvalidError(str(exc), details={"code": exc.code, **exc.details})


def check_members(definition: dict[str, Any], key: str) -> dict[str, Any]:
    """D14: the armed set IS the definition's member set (by detector digest)."""
    armed = _armed_members(key)
    wanted = [m["detector_digest"] for m in definition["members"]]
    extra = sorted(d for d in armed if d not in set(wanted))
    absent = sorted(d for d in wanted if d not in armed)
    if extra or absent:
        raise BehaviourVerdictMembersMismatchError(
            f"the armed probes of {key!r} are not the definition's members",
            details={"armed_not_in_definition": extra, "definition_not_armed": absent,
                     "armed_probe_ids": sorted(p.probe_id for p in armed.values())})
    changed = []
    for member in definition["members"]:
        probe = armed[member["detector_digest"]]
        decision = getattr(getattr(probe.descriptor, "decision", None), "decision_digest", None)
        if member.get("decision_digest") and decision != member["decision_digest"]:
            changed.append({"detector_digest": member["detector_digest"], "probe_id": probe.probe_id,
                            "definition": member["decision_digest"], "armed": decision})
    return {"armed_probe_ids": sorted(p.probe_id for p in armed.values()), "decision_changed": changed}


def check_rules(definition: dict[str, Any], key: str) -> Optional[dict[str, Any]]:
    """The rule side: the set attached to this behaviour by digest, the match expressions its own."""
    from miserver.rules import registry as rule_registry
    from miserver.rules.document import match_of

    rules = definition.get("rules")
    if rules is None:
        return None
    digest = rules["rule_set_digest"]
    sets = [a for a in rule_registry.current().for_behaviour(key) if a.digest == digest]
    if not sets:
        raise BehaviourVerdictRuleSetError(
            f"the definition's rule set {digest[:12]}… is not attached to {key!r}",
            details={"rule_set_digest": digest, "behaviour": key})
    attached = sets[0]
    if attached.compiled is None:
        raise BehaviourVerdictInvalidError("the definition's rule set does not run here (its vectors failed)",
                                           details={"code": "rule_set_vectors_failed",
                                                    "rule_set_id": attached.rule_set_id})
    names = list(rules["match"]["any_of"])
    unknown = [n for n in names if n not in attached.compiled.expressions]
    if unknown:
        raise BehaviourVerdictInvalidError("the definition's match names expressions the rule set does not "
                                           "declare", details={"code": "match_unknown", "expressions": unknown})
    stated = match_of(attached.compiled.document)
    if stated["source"] == "explicit" and sorted(stated["any_of"]) != sorted(names):
        raise BehaviourVerdictInvalidError(
            "the rule set states its positive (D53) and the definition names another",
            details={"code": "match_mismatch", "rule_set": stated["any_of"], "definition": names})
    return {"rule_set_id": attached.rule_set_id, "match_source": rules["match"]["source"],
            "match_any_of": names, "pending_code": sorted(set(names) & attached.compiled.pending_expressions)}


async def attach(session: Any, key: str, definition: Any, *, by: Optional[str],
                 reason: Optional[str]) -> dict[str, Any]:
    try:
        parity = bv.check(definition, key=key)
    except bv.DefinitionRefused as exc:
        raise _refuse(exc) from None
    members = check_members(definition, key)
    rules = check_rules(definition, key)
    sha = bv.definition_sha256(definition)
    report = {"parity": {"members": "armed_digests_match", "combiner_vectors": parity["combiner_vectors"],
                         "end_to_end": parity["end_to_end_status"]},
              "members": members, "rules": rules}
    report_id = (definition.get("calibration") or {}).get("report_id")
    row = await session.get(BehaviourVerdict, key)
    if row is None:
        row = BehaviourVerdict(behaviour_key=key)
        session.add(row)
    row.definition, row.definition_sha256, row.report_id = definition, sha, report_id
    row.attach_report, row.attached_by, row.reason = report, by, reason
    from datetime import datetime, timezone

    row.attached_at = datetime.now(timezone.utc)
    session.add(BehaviourVerdictHistory(behaviour_key=key, action="attach", definition_sha256=sha,
                                        report_id=report_id, by=by, reason=reason))
    await session.commit()
    await rebuild(session)
    return await view(session, key)


async def clear(session: Any, key: str, *, by: Optional[str], reason: Optional[str]) -> dict[str, Any]:
    row = await session.get(BehaviourVerdict, key)
    if row is None:
        raise BehaviourVerdictNotFoundError(f"no verdict definition is attached to {key!r}",
                                            details={"behaviour": key})
    session.add(BehaviourVerdictHistory(behaviour_key=key, action="clear",
                                        definition_sha256=row.definition_sha256, report_id=row.report_id,
                                        by=by, reason=reason))
    await session.delete(row)
    await session.commit()
    await rebuild(session)
    return await view(session, key)


async def view(session: Any, key: str) -> dict[str, Any]:
    row = await session.get(BehaviourVerdict, key)
    history = (await session.execute(
        select(BehaviourVerdictHistory).where(BehaviourVerdictHistory.behaviour_key == key)
        .order_by(BehaviourVerdictHistory.changed_at.desc(), BehaviourVerdictHistory.id.desc())
        .limit(20))).scalars().all()
    attached = bv.current().get(key)
    return {
        "behaviour": key,
        "attached": row is not None,
        "definition": row.definition if row is not None else None,
        "definition_sha256": row.definition_sha256 if row is not None else None,
        "serving": attached is not None and row is not None and attached.sha256 == row.definition_sha256,
        "attach_report": row.attach_report if row is not None else None,
        "attached_at": row.attached_at.isoformat() if row is not None and row.attached_at else None,
        "attached_by": row.attached_by if row is not None else None,
        "history": [{"changed_at": h.changed_at.isoformat() if h.changed_at else None, "action": h.action,
                     "definition_sha256": h.definition_sha256, "report_id": h.report_id, "by": h.by,
                     "reason": h.reason} for h in history],
    }


async def rebuild(session: Any) -> int:
    """The serving registry from the rows (attach, clear and startup). Keeps the last good registry
    on a read failure; a stored definition that no longer checks is left out, logged."""
    try:
        rows = (await session.execute(select(BehaviourVerdict))).scalars().all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("behaviour_verdict_rebuild_failed error=%s", type(exc).__name__)
        return len(bv.current())
    attached = []
    for row in rows:
        try:
            bv.check(row.definition, key=row.behaviour_key)
        except bv.DefinitionRefused as exc:
            logger.warning("behaviour_verdict_not_serving key=%s code=%s", row.behaviour_key, exc.code)
            continue
        attached.append(bv.attached_from(row.behaviour_key, row.definition))
    bv.install(attached)
    return len(attached)


async def rebuild_with_new_session() -> int:
    try:
        from miserver.db.base import async_session_factory

        async with async_session_factory() as session:
            return await rebuild(session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("behaviour_verdict_rebuild_failed error=%s", type(exc).__name__)
        return len(bv.current())


# ── recording (beside the probe events, same retention) ──────────────────────────


def columns(result: Any) -> dict[str, Any]:
    """The `behaviour_verdict_results` row of one recorded result: every column a projection of it."""
    from datetime import datetime, timezone

    from miserver.services.probe_result import stored_result

    data = stored_result(result)
    return {
        "result": data, "result_id": data["result_id"], "trace_id": data.get("trace_id"),
        "span_id": data.get("span_id"),
        "event_time": datetime.strptime(data["event_time"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc),
        "outcome": data["outcome"], "reason": data.get("reason"), "status": data.get("status", "final"),
        "request_id": data.get("request_id"), "session_id": data.get("session_id"),
        "origin": data.get("origin") or "live", "batch_id": data.get("batch_id"),
        "batch_line": data.get("batch_line"), "behaviour_key": (data.get("behaviour") or {}).get("key"),
        "window": data["window"], "fires": data.get("fires"), "decided_by": data["decided_by"],
        "definition_sha256": data["definition_sha256"],
    }


def rows_for(results: Iterable[Any]) -> list[BehaviourVerdictResult]:
    return [BehaviourVerdictResult(**columns(r)) for r in results
            if getattr(r, "kind", "probe") == "behaviour"]


async def prune(session: Any, *, max_age_days: Optional[float], live_cap: int, batch_cap: int,
                keys: Optional[Iterable[str]] = None) -> int:
    """The age rule table-wide, then each behaviour's cap per origin (the probe events' rule)."""
    from datetime import datetime, timedelta, timezone

    removed = 0
    if max_age_days is not None and max_age_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=float(max_age_days))
        result = await session.execute(delete(BehaviourVerdictResult).where(
            BehaviourVerdictResult.created_at < cutoff).execution_options(synchronize_session=False))
        removed += int(result.rowcount or 0)
    if keys is None:
        keys = [r[0] for r in (await session.execute(
            select(BehaviourVerdictResult.behaviour_key).distinct())).all()]
    for key in keys:
        for origin, cap in (("live", live_cap), ("batch", batch_cap)):
            if cap is None or cap <= 0:
                continue
            keep = (select(BehaviourVerdictResult.id)
                    .where(BehaviourVerdictResult.behaviour_key == key, BehaviourVerdictResult.origin == origin)
                    .order_by(BehaviourVerdictResult.created_at.desc(), BehaviourVerdictResult.id.desc())
                    .limit(cap))
            result = await session.execute(delete(BehaviourVerdictResult).where(
                BehaviourVerdictResult.behaviour_key == key, BehaviourVerdictResult.origin == origin,
                BehaviourVerdictResult.id.not_in(keep.scalar_subquery()),
            ).execution_options(synchronize_session=False))
            removed += int(result.rowcount or 0)
    return removed
