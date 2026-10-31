"""Behaviours as the operator and the agent see them (039 FR-39.4, FR-39.5, FR-39.15).

`list_behaviours` is the paginated per-key summary behind `GET /api/probes/behaviours`;
`armed_summary` is what `GET /api/probes/status` reports per behaviour. Both read a behaviour's
membership from the `probes` table (a set is a selector, never a stored artifact) and whether a
probe is armed from THE REGISTRY (`ProbeRuntimeState`), never from the rows.

⚠ CONSTANT QUERIES PER PAGE (FR-39.15): one query for the page's keys, one for those keys' rows —
whatever the catalogue holds. The scale test counts statements with a SQLAlchemy listener and a
per-row lookup turns it red.

"Armed before unload / restart" (FR-39.5, TD39-12) is DERIVED: a row whose `paused_reason` is one
of `INVOLUNTARY_REASONS` was armed when it stopped without the operator asking. Re-arming clears it
(`paused_reason = None`); an operator disarm replaces it (`operator`).
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func, select

from millm.api.pagination import decode_cursor, encode_cursor


def attachments_of(key: str) -> list[dict[str, Any]]:
    """The rule sets attached to `key`, from the compiled rule registry (039 S2; no DB read).

    `{rule_set_id, rule_set_digest, rule_set_version, vector_status}` per set, in attach order —
    the `set` block's `attached_rule_sets` reads the first two (`probe_result_order`)."""
    from millm.rules import registry

    return [
        {"rule_set_id": a.rule_set_id, "rule_set_digest": a.digest,
         "rule_set_version": a.version,
         "vector_status": "vectors_failed" if a.vectors_failed else "passed"}
        for a in registry.current().for_behaviour(key)
    ]


def _registry() -> dict[str, Any]:
    from millm.services.probe_runtime import ProbeRuntimeState

    return {p.probe_id: p for p in ProbeRuntimeState().armed()}


def _versions(rows: list[Any], armed: dict[str, Any]) -> list[dict[str, Any]]:
    by_version: dict[Optional[int], list[Any]] = {}
    for row in rows:
        by_version.setdefault(row.behaviour_version, []).append(row)
    out = []
    for version, members in sorted(by_version.items(),
                                   key=lambda kv: (kv[0] is None, kv[0] or 0)):
        ids = sorted(r.id for r in members)
        armed_ids = [pid for pid in ids if pid in armed]
        out.append({
            "version": version,
            "probe_ids": ids,
            "imported": len(ids),
            "armed": len(armed_ids),
            "missing_probe_ids": [pid for pid in ids if pid not in armed] if armed_ids else [],
        })
    return out


def summarise(key: str, rows: list[Any], armed: Optional[dict[str, Any]] = None,
              attachments: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """One behaviour's summary from its rows and the registry."""
    from millm.services.probe_arming import INVOLUNTARY_REASONS

    armed = _registry() if armed is None else armed
    versions = _versions(rows, armed)
    armed_ids = sorted(r.id for r in rows if r.id in armed)
    armed_versions = [v for v in versions if v["armed"]]
    partial = None
    if armed_versions:
        v = armed_versions[0]
        partial = {
            "version": v["version"],
            "armed_members": v["armed"],
            "total_members": v["imported"],
            "complete": v["armed"] == v["imported"],
            "missing_probe_ids": v["missing_probe_ids"],
            "text": (None if v["armed"] == v["imported"]
                     else f"partial: {v['armed']} of {v['imported']} armed"),
        }
    before = sorted(r.id for r in rows
                    if r.id not in armed and r.paused_reason in INVOLUNTARY_REASONS)
    reasons = sorted({r.paused_reason for r in rows
                      if r.id not in armed and r.paused_reason in INVOLUNTARY_REASONS})
    return {
        "key": key,
        "imported": len(rows),
        "armed": len(armed_ids),
        "armed_probe_ids": armed_ids,
        "versions": versions,
        "set": partial,
        "armed_before_unload": ({"probe_ids": before, "reasons": reasons} if before else None),
        "attached_rule_sets": attachments or [],
    }


async def list_behaviours(session: Any, *, cursor: Optional[str], limit: int,
                          key_prefix: Optional[str] = None) -> tuple[list[dict[str, Any]],
                                                                      Optional[str]]:
    """One page of behaviours, ordered by key, and the cursor of the next page (or None)."""
    from millm.db.models.probe import Probe

    after = decode_cursor("behaviours", cursor, arity=1)
    keys_stmt = (select(Probe.behaviour_key).where(Probe.behaviour_key.is_not(None))
                 .group_by(Probe.behaviour_key).order_by(Probe.behaviour_key))
    if key_prefix:
        keys_stmt = keys_stmt.where(Probe.behaviour_key.like(f"{key_prefix}%"))
    if after is not None:
        keys_stmt = keys_stmt.where(Probe.behaviour_key > str(after[0]))
    keys = list((await session.execute(keys_stmt.limit(limit + 1))).scalars().all())
    more = len(keys) > limit
    keys = keys[:limit]
    if not keys:
        return [], None
    rows = (await session.execute(
        select(Probe.id, Probe.behaviour_key, Probe.behaviour_version, Probe.paused_reason)
        .where(Probe.behaviour_key.in_(keys)).order_by(Probe.behaviour_key, Probe.id)
    )).all()
    grouped: dict[str, list[Any]] = {key: [] for key in keys}
    for row in rows:
        grouped[row.behaviour_key].append(row)
    armed = _registry()
    items = [summarise(key, grouped[key], armed, attachments_of(key)) for key in keys]
    return items, encode_cursor("behaviours", [keys[-1]]) if more else None


async def armed_summary(session: Any) -> dict[str, Any]:
    """Status's per-behaviour block: the armed behaviours in full (with their partial state) and
    the behaviours armed before an involuntary stop. Bounded by the armed set and the stopped set,
    never by the catalogue."""
    from millm.db.models.probe import Probe
    from millm.services.probe_arming import INVOLUNTARY_REASONS

    armed = _registry()
    keys = {k for k in (getattr(getattr(p, "descriptor", None), "behaviour_key", None)
                        for p in armed.values()) if isinstance(k, str) and k}
    stopped_keys = set((await session.execute(
        select(Probe.behaviour_key).where(Probe.paused_reason.in_(list(INVOLUNTARY_REASONS)),
                                          Probe.behaviour_key.is_not(None)).distinct()
    )).scalars().all())
    wanted = sorted(k for k in (keys | stopped_keys) if k)
    rows = (await session.execute(
        select(Probe.id, Probe.behaviour_key, Probe.behaviour_version, Probe.paused_reason)
        .where(Probe.behaviour_key.in_(wanted)).order_by(Probe.behaviour_key, Probe.id)
    )).all() if wanted else []
    grouped: dict[str, list[Any]] = {key: [] for key in wanted}
    for row in rows:
        grouped[row.behaviour_key].append(row)
    summaries = [summarise(key, grouped[key], armed) for key in wanted]
    return {
        "behaviours": [s for s in summaries if s["armed"]],
        "armed_before_unload": [
            {"key": s["key"], **s["armed_before_unload"]}
            for s in summaries if s["armed_before_unload"] and not s["armed"]
        ],
    }


async def catalogue_counts(session: Any) -> dict[str, int]:
    """The catalogue as COUNTS (FR-39.15): never a list that grows with imports."""
    from millm.db.models.probe import Probe

    imported = int((await session.execute(select(func.count()).select_from(Probe))).scalar_one())
    behaviours = int((await session.execute(
        select(func.count(func.distinct(Probe.behaviour_key)))
    )).scalar_one())
    return {"imported": imported, "behaviours": behaviours}
