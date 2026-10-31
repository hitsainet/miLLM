"""Rule sets, attachments, history and rule results (039 S2; FTID §2.1).

Every write commits; every list is a keyset page (039 FR-39.15). Rule results are written with the
request's probe events in one session (`ProbeEventService.record`), and pruned by
`(rule_set_id, signal, origin)` under the probe-event caps (TD39-17).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.rule_set import (
    BehaviourRuleSignalHistory,
    BehaviourRuleSignals,
    RuleSet,
    RuleSignalResult,
)


class RuleSetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── rule sets ────────────────────────────────────────────────────────────────────────

    async def get(self, rule_set_id: str) -> Optional[RuleSet]:
        return await self.session.get(RuleSet, rule_set_id)

    async def by_digest(self, digest: str) -> Optional[RuleSet]:
        result = await self.session.execute(select(RuleSet).where(RuleSet.digest == digest))
        return result.scalar_one_or_none()

    async def by_ids(self, ids: Iterable[str]) -> list[RuleSet]:
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return []
        result = await self.session.execute(select(RuleSet).where(RuleSet.id.in_(wanted)))
        return list(result.scalars().all())

    async def all(self) -> list[RuleSet]:
        result = await self.session.execute(select(RuleSet).order_by(RuleSet.id))
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(RuleSet))
        return int(result.scalar_one())

    async def page(self, *, behaviour: Optional[str] = None, after: Optional[list[Any]] = None,
                   limit: int = 200) -> list[RuleSet]:
        """`ORDER BY behaviour_key, id` (both NOT NULL); `after` = the last `[key, id]`."""
        stmt = select(RuleSet).order_by(RuleSet.behaviour_key, RuleSet.id)
        if behaviour is not None:
            stmt = stmt.where(RuleSet.behaviour_key == behaviour)
        if after is not None:
            key, last_id = str(after[0]), str(after[1])
            stmt = stmt.where(or_(RuleSet.behaviour_key > key,
                                  and_(RuleSet.behaviour_key == key, RuleSet.id > last_id)))
        result = await self.session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def create(self, row: RuleSet) -> RuleSet:
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        return row

    async def delete(self, row: RuleSet) -> None:
        await self.session.delete(row)
        await self.session.commit()

    async def mark(self, rule_set_id: str, *, status: str, report: Optional[dict[str, Any]],
                   checked_at: datetime) -> None:
        row = await self.session.get(RuleSet, rule_set_id)
        if row is None:
            return
        row.vector_status = status
        row.vector_report = report
        row.checked_at = checked_at
        await self.session.commit()

    # ── attachments ──────────────────────────────────────────────────────────────────────

    async def attachment(self, behaviour_key: str) -> Optional[BehaviourRuleSignals]:
        return await self.session.get(BehaviourRuleSignals, behaviour_key)

    async def attachments(self) -> list[BehaviourRuleSignals]:
        result = await self.session.execute(
            select(BehaviourRuleSignals).order_by(BehaviourRuleSignals.behaviour_key))
        return list(result.scalars().all())

    async def attachments_naming(self, rule_set_id: str) -> list[BehaviourRuleSignals]:
        # The id list is JSON; read every row (one per attached behaviour, a small table).
        return [a for a in await self.attachments() if rule_set_id in (a.rule_set_ids or [])]

    async def set_attachment(self, behaviour_key: str, rule_set_ids: list[str],
                             signals: Optional[dict[str, list[str]]], *, action: str,
                             by: Optional[str], reason: Optional[str]) -> Optional[
                                 BehaviourRuleSignals]:
        """Write (or remove, when `rule_set_ids` is empty) a behaviour's attachment, with one
        history row, in one commit."""
        row = await self.session.get(BehaviourRuleSignals, behaviour_key)
        if rule_set_ids:
            if row is None:
                row = BehaviourRuleSignals(behaviour_key=behaviour_key, rule_set_ids=rule_set_ids,
                                           signals=signals)
                self.session.add(row)
            else:
                row.rule_set_ids = list(rule_set_ids)
                row.signals = signals
        elif row is not None:
            await self.session.delete(row)
            row = None
        self.session.add(BehaviourRuleSignalHistory(
            behaviour_key=behaviour_key, action=action, rule_set_ids=list(rule_set_ids),
            signals=signals, by=by, reason=reason,
        ))
        await self.session.commit()
        return row

    async def history(self, behaviour_key: str, limit: int = 100) -> list[
            BehaviourRuleSignalHistory]:
        result = await self.session.execute(
            select(BehaviourRuleSignalHistory)
            .where(BehaviourRuleSignalHistory.behaviour_key == behaviour_key)
            .order_by(BehaviourRuleSignalHistory.id.desc()).limit(limit))
        return list(result.scalars().all())

    # ── results ──────────────────────────────────────────────────────────────────────────

    async def results_for_request(self, request_id: str) -> list[RuleSignalResult]:
        result = await self.session.execute(
            select(RuleSignalResult).where(RuleSignalResult.request_id == request_id)
            .order_by(RuleSignalResult.id))
        return list(result.scalars().all())

    async def results_for_session(self, session_id: str) -> list[RuleSignalResult]:
        result = await self.session.execute(
            select(RuleSignalResult).where(RuleSignalResult.session_id == session_id)
            .order_by(RuleSignalResult.turn_index, RuleSignalResult.id))
        return list(result.scalars().all())

    async def results_page(self, *, request_id: Optional[str] = None,
                           session_id: Optional[str] = None, since: Optional[datetime] = None,
                           until: Optional[datetime] = None, verdict: Optional[str] = None,
                           window: Optional[str] = None, origin: Optional[str] = None,
                           behaviour: Optional[str] = None, result_id: Optional[str] = None,
                           trace_id: Optional[str] = None, before: Optional[list[Any]] = None,
                           limit: int = 100) -> list[RuleSignalResult]:
        """Newest first: `ORDER BY created_at DESC, id DESC`; `before` = `[created_at iso, id]`.
        The events list's filters, read as a rule result has them (`fired` = fires)."""
        stmt = select(RuleSignalResult).order_by(RuleSignalResult.created_at.desc(),
                                                 RuleSignalResult.id.desc())
        filters = {
            RuleSignalResult.request_id: request_id, RuleSignalResult.session_id: session_id,
            RuleSignalResult.window: window, RuleSignalResult.origin: origin,
            RuleSignalResult.behaviour_key: behaviour, RuleSignalResult.result_id: result_id,
            RuleSignalResult.trace_id: trace_id,
        }
        for column, value in filters.items():
            if value is not None:
                stmt = stmt.where(column == value)
        if since is not None:
            stmt = stmt.where(RuleSignalResult.created_at >= since)
        if until is not None:
            stmt = stmt.where(RuleSignalResult.created_at <= until)
        if verdict == "fired":
            stmt = stmt.where(RuleSignalResult.fires.is_(True))
        elif verdict == "not_fired":
            stmt = stmt.where(RuleSignalResult.outcome == "completed",
                              RuleSignalResult.fires.is_(False))
        elif verdict == "not_scored":
            stmt = stmt.where(RuleSignalResult.outcome != "completed")
        if before is not None:
            moment = datetime.fromisoformat(str(before[0]))
            last_id = int(before[1])
            stmt = stmt.where(or_(RuleSignalResult.created_at < moment,
                                  and_(RuleSignalResult.created_at == moment,
                                       RuleSignalResult.id < last_id)))
        result = await self.session.execute(stmt.limit(limit))
        return list(result.scalars().all())


async def prune_rule_results(session: AsyncSession, *, max_age_days: Optional[float],
                             live_cap: int, batch_cap: int,
                             keys: Optional[Iterable[tuple[str, str]]] = None) -> int:
    """The age rule table-wide, then each `(rule_set_id, signal)`'s cap per origin (TD39-17).

    `keys` limits the cap pass to the signals a record just wrote (the per-record pass); None
    walks every signal (the background pass). Never deletes by rule set: deleting a set keeps its
    results. Returns the rows removed. The caller commits.
    """
    from datetime import timedelta, timezone

    removed = 0
    if max_age_days is not None and max_age_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=float(max_age_days))
        result = await session.execute(
            delete(RuleSignalResult).where(RuleSignalResult.created_at < cutoff)
            .execution_options(synchronize_session=False))
        removed += int(result.rowcount or 0)
    if keys is None:
        rows = await session.execute(
            select(RuleSignalResult.rule_set_id, RuleSignalResult.signal).distinct())
        keys = [(r[0], r[1]) for r in rows.all()]
    for rule_set_id, signal in keys:
        for origin, cap in (("live", live_cap), ("batch", batch_cap)):
            if cap is None or cap <= 0:
                continue
            keep = (
                select(RuleSignalResult.id)
                .where(RuleSignalResult.rule_set_id == rule_set_id,
                       RuleSignalResult.signal == signal, RuleSignalResult.origin == origin)
                .order_by(RuleSignalResult.created_at.desc(), RuleSignalResult.id.desc())
                .limit(cap)
            )
            result = await session.execute(
                delete(RuleSignalResult).where(
                    RuleSignalResult.rule_set_id == rule_set_id,
                    RuleSignalResult.signal == signal,
                    RuleSignalResult.origin == origin,
                    RuleSignalResult.id.not_in(keep.scalar_subquery()),
                ).execution_options(synchronize_session=False))
            removed += int(result.rowcount or 0)
    return removed
