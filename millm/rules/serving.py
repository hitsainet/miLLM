"""`_rules_finish`: attached rule signals beside the probes, on every generation path (039 FR-39.23,
FR-39.24; FTDD §6.7; FTID §7.4 – §7.5).

Awaited by `InferenceService._rules_finish` immediately before each of the ten `_probe_finish`
sites. It leaves `context.rule_items` (a list of `RuleItem`) for 038's builder, which mints one
`kind: "rule"` result per item beside the probe results; the probe verdicts are never touched
(no veto, no blend, D9).

Order inside (FTID §7.5):

1. `context` None → nothing. No armed behaviour with an attachment → nothing (the common case;
   returns before any thread hop, FTID §13). Attached signals run only while at least one probe of
   their behaviour is armed (BEHAVIOUR-001 D40) — the request's `context.probes` are those.
2. The context is NOT SCORED (judge traffic, continuous batching, llama.cpp, a batched request…) →
   one entry per attached signal and window with the context's own reason; nothing runs
   ("nothing dropped silently", FR-39.23).
3. `RULE_SIGNALS_ENABLED` off → `failed` / `rule_signals_disabled`, attachments kept (TD39-38).
4. Otherwise the evaluator runs in `asyncio.to_thread` against an injected-clock deadline of
   `RULE_TIME_BUDGET_MS`; the await is bounded by the budget plus `GRACE_S`. Signals unfinished at
   the deadline report `rule_timeout`. A late thread result is DISCARDED by the context's
   generation counter: the thread writes its items only if the generation it started under is
   still current, and a timed-out await bumps it.

Windows (F39-5): a signal runs on the windows the behaviour's armed probes report (the producer's
v1 signals declare none, which means all of them). A window this request has no text for reports
`skipped` with why (`no_user_turn`, …).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Optional

from millm.rules import registry as rule_registry
from millm.rules.evaluator import (
    ERROR_REASON,
    PENDING_REASON,
    TIMEOUT_REASON,
    VECTORS_FAILED_REASON,
    WINDOW_ORDER,
)
from millm.rules.windows import WindowTexts, window_texts

logger = logging.getLogger(__name__)

#: The fixed bound past the budget the request may wait for the evaluator thread to return.
GRACE_S = 0.005
DISABLED_REASON = "rule_signals_disabled"
_items_lock = threading.Lock()


@dataclass(frozen=True)
class RuleItem:
    """One rule signal on one window, ready for the builder (`probe_result._rule_one`)."""

    rule_set_id: str
    rule_set_digest: str
    rule_set_version: int
    signal: str
    form: str
    behaviour_key: str
    window: str
    fires: Optional[bool]
    count: int
    outcome: str
    reason: Optional[str]
    status: str = "final"
    #: Matched spans of the CALLER'S OWN text — only on the stateless routes, only on request.
    spans: Optional[tuple[tuple[int, int], ...]] = None


def armed_behaviours(probes: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    """`{behaviour key: windows its armed probes report}` (a probe's windows, else its scope)."""
    out: dict[str, set[str]] = {}
    for probe in probes or ():
        key = getattr(getattr(probe, "descriptor", None), "behaviour_key", None)
        if not isinstance(key, str) or not key:
            continue
        windows = tuple(getattr(probe, "windows", None) or ()) or (getattr(probe, "scope", "all"),)
        out.setdefault(key, set()).update(windows)
    return {k: _ordered(v) for k, v in sorted(out.items())}


def _ordered(windows: Iterable[str]) -> tuple[str, ...]:
    seen = set(windows)
    return tuple([w for w in WINDOW_ORDER if w in seen] + sorted(seen - set(WINDOW_ORDER)))


def _work(probes: Iterable[Any], registry: Any) -> list[tuple[str, tuple[str, ...], Any]]:
    return [(key, windows, attached)
            for key, windows in armed_behaviours(probes).items()
            for attached in registry.for_behaviour(key)]


def _item(attached: Any, key: str, name: str, window: str, *, fires: Optional[bool] = None,
          count: int = 0, outcome: str, reason: Optional[str], status: str = "final",
          form: Optional[str] = None, spans: Any = None) -> RuleItem:
    return RuleItem(attached.rule_set_id, attached.digest, attached.version, name,
                    form or attached.form_of(name), key, window, fires, count, outcome, reason,
                    status, spans)


def _uniform(work: list[tuple[str, tuple[str, ...], Any]], outcome: str,
             reason: str) -> list[RuleItem]:
    """Every attached signal on every window with one outcome and reason (nothing ran)."""
    items = []
    for key, windows, attached in work:
        for name in attached.names():
            for window in windows:
                items.append(_item(attached, key, name, window, outcome=outcome, reason=reason))
    return items


def evaluate_work(work: list[tuple[str, tuple[str, ...], Any]], texts: WindowTexts, *,
                  clock: Callable[[], float], deadline: Optional[float],
                  return_positions: bool = False) -> list[RuleItem]:
    """Run every attached set on its behaviour's windows, in the fixed order. Never raises."""
    from millm.services.probe_reasons import OUTCOME_OF

    items: list[RuleItem] = []
    for key, windows, attached in work:
        names = attached.names()
        if attached.compiled is None or attached.vectors_failed:
            items.extend(_item(attached, key, n, w, outcome="failed",
                               reason=VECTORS_FAILED_REASON) for n in names for w in windows)
            continue
        present = {w: texts.texts[w] for w in windows if w in texts.texts}
        try:
            outcomes = attached.compiled.serve(present, only=names, clock=clock,
                                               deadline=deadline,
                                               return_positions=return_positions)
        except Exception as exc:  # noqa: BLE001 - an observer never fails the request
            logger.warning("rule_evaluate_failed rule_set_id=%s error=%s", attached.rule_set_id,
                           type(exc).__name__)
            outcomes = []
            items.extend(_item(attached, key, n, w, outcome="failed", reason=ERROR_REASON)
                         for n in names for w in present)
        by = {(o.signal, o.window): o for o in outcomes}
        for name in names:
            for window in windows:
                if window not in present:
                    reason = texts.missing.get(window, "no_scored_tokens")
                    items.append(_item(attached, key, name, window,
                                       outcome=OUTCOME_OF.get(reason, "skipped"), reason=reason))
                    continue
                o = by.get((name, window))
                if o is None:
                    continue
                items.append(_item(attached, key, name, window, fires=o.fires, count=o.count,
                                   outcome=o.outcome, reason=o.reason, status=o.status,
                                   form=o.form, spans=o.spans))
    return items


async def rules_finish(context: Any, *, messages: Optional[Iterable[Any]] = None,
                       prompt: Any = None, reply_text: Optional[str] = None,
                       clock: Callable[[], float] = time.monotonic) -> None:
    """Fill `context.rule_items`. Never raises (an observer never fails the request)."""
    if context is None:
        return
    context.rule_items = []
    registry = rule_registry.current()
    if not registry:
        return
    work = _work(getattr(context, "probes", None) or (), registry)
    if not work:
        return
    from millm.core.config import settings
    from millm.services.probe_reasons import OUTCOME_OF

    reason = getattr(context, "not_scored_reason", None)
    if reason:
        context.rule_items = _uniform(work, OUTCOME_OF.get(reason, "skipped"), reason)
        return
    if not bool(settings.RULE_SIGNALS_ENABLED):
        context.rule_items = _uniform(work, "failed", DISABLED_REASON)
        return
    texts = window_texts(messages=messages, prompt=prompt, reply_text=reply_text)
    budget = max(0.0, float(settings.RULE_TIME_BUDGET_MS)) / 1000.0
    generation = getattr(context, "rule_generation", 0) + 1
    context.rule_generation = generation
    deadline = clock() + budget

    def run() -> None:
        items = evaluate_work(work, texts, clock=clock, deadline=deadline)
        with _items_lock:
            # ⚠ THE GENERATION COUNTER: a thread that outlived its await writes nothing.
            if getattr(context, "rule_generation", None) == generation:
                context.rule_items = items

    try:
        await asyncio.wait_for(asyncio.to_thread(run), timeout=budget + GRACE_S)
    except asyncio.TimeoutError:
        with _items_lock:
            context.rule_generation = generation + 1
            context.rule_items = _uniform(work, "failed", TIMEOUT_REASON)
        logger.warning("rule_budget_exceeded sets=%s", len(work))
    except Exception as exc:  # noqa: BLE001
        logger.warning("rules_finish_failed error=%s", type(exc).__name__)
        context.rule_items = _uniform(work, "failed", ERROR_REASON)


__all__ = ["PENDING_REASON", "RuleItem", "armed_behaviours", "evaluate_work", "rules_finish"]
