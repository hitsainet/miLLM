"""Every stored rule set re-proves itself at EVERY startup (039 FR-39.20, TD39-26; FTID §3.5).

An evaluator change (a new build) can make a stored rule set disagree with its own cases. Such a
set is marked `vectors_failed` with a `vector_report` naming the first failing case (index, id,
column) and this build, stops being evaluated, and reports `rule_set_vectors_failed` for every
signal — never evaluated under changed rules silently. A set that passes again is marked `passed`.

Called from `lifespan` (`main.py`) before the rule registry is built, so the registry never
compiles a set the self-check has not seen under this build. Measured (FTDD §9.3): the duration
is logged; above `BACKGROUND_AFTER_S` for the catalogue it would move to a background task.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

#: FTDD §9.3: past this, the check belongs in a background task (recorded, not enforced in CI).
BACKGROUND_AFTER_S = 30.0


@dataclass
class SelfCheckReport:
    checked: int = 0
    passed: int = 0
    failed: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def build_id() -> str:
    """The evaluator build a report names: miLLM's version (the image that ran the check)."""
    try:
        from millm import __version__

        return f"millm {__version__}"
    except Exception:  # noqa: BLE001
        return "millm unknown"


def check_document(document: Any) -> tuple[str, Any]:
    """`("passed", None)` or `("vectors_failed", report)` for one stored document. Never raises."""
    from millm.rules import RuleVectorsFailed, prepare

    try:
        prepare(document, run_own_cases=True)
        return "passed", None
    except RuleVectorsFailed as exc:
        return "vectors_failed", {**exc.details, "build": build_id()}
    except Exception as exc:  # noqa: BLE001 - a set this build cannot compile is not evaluated
        return "vectors_failed", {"case_index": None, "case_id": None,
                                  "where": getattr(exc, "code", type(exc).__name__),
                                  "signal": (getattr(exc, "details", None) or {}).get("signal"),
                                  "build": build_id()}


async def run(session_factory: Any) -> SelfCheckReport:
    """Check every stored set and write its status. Never raises."""
    from millm.db.repositories.rule_set_repository import RuleSetRepository

    report = SelfCheckReport()
    started = time.monotonic()
    try:
        async with session_factory() as session:
            repository = RuleSetRepository(session)
            rows = await repository.all()
            now = datetime.now(timezone.utc)
            for row in rows:
                status, details = check_document(row.document)
                report.checked += 1
                if status == "passed":
                    report.passed += 1
                else:
                    report.failed.append(row.id)
                    logger.warning("rule_set_vectors_failed rule_set_id=%s where=%s", row.id,
                                   (details or {}).get("where"))
                await repository.mark(row.id, status=status, report=details, checked_at=now)
    except Exception as exc:  # noqa: BLE001 - startup must not fail on bookkeeping
        logger.warning("rule_selfcheck_failed error=%s", type(exc).__name__)
    report.duration_s = time.monotonic() - started
    logger.info("rule_selfcheck checked=%s failed=%s duration_s=%.3f", report.checked,
                len(report.failed), report.duration_s)
    return report
