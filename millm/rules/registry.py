"""The compiled rule registry: what the request path reads (039 FR-39.22, FR-39.23; FTID §3.5).

`{behaviour_key: (AttachedSet, ...)}` for every behaviour with an attachment, built from the
database and swapped atomically: the request path reads `current()` with no lock and no database
read (FTID §13). It is rebuilt — whole, under a lock — on import, delete, attach, detach, arm,
disarm and at startup (`rebuild`), each a call a test counts.

A set whose stored `vector_status` is `vectors_failed` is present but never evaluated: every one of
its signals reports `rule_set_vectors_failed` (FR-39.20). A set that no longer compiles under this
build (an evaluator change) is treated the same way, and the startup self-check marks it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from millm.rules.evaluator import CompiledRuleSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttachedSet:
    rule_set_id: str
    digest: str
    version: int
    behaviour_key: str
    #: None when the set did not compile or its vectors failed: it reports, never evaluates.
    compiled: Optional[CompiledRuleSet]
    vectors_failed: bool
    #: The expressions this behaviour narrowed the set to; None = every one.
    only: Optional[tuple[str, ...]]
    #: `{expression name: form}` read from the document (present even when it did not compile).
    forms: tuple[tuple[str, str], ...] = ()
    #: 039 S3: `(code_sha256, canonical text)` of the code documents stored beside the set. Read
    #: only by the after-response job (`code_followup`), which sends them to the runner.
    code_documents: tuple[tuple[str, str], ...] = ()

    def names(self) -> list[str]:
        """The expressions this attachment reports, in their fixed (sorted) order."""
        names = sorted(name for name, _form in self.forms)
        if self.only is not None:
            wanted = set(self.only)
            names = [n for n in names if n in wanted]
        return names

    def form_of(self, name: str) -> str:
        return dict(self.forms).get(name, "unknown")


@dataclass(frozen=True)
class Registry:
    by_behaviour: dict[str, tuple[AttachedSet, ...]]

    def for_behaviour(self, key: str) -> tuple[AttachedSet, ...]:
        return self.by_behaviour.get(key, ())

    def __bool__(self) -> bool:
        return bool(self.by_behaviour)


_EMPTY = Registry({})
_current: Registry = _EMPTY
_lock: Optional[asyncio.Lock] = None
_lock_loop: Any = None
#: Bumped on every rebuild (tests read it to count rebuilds; nothing else may).
generation = 0


def current() -> Registry:
    return _current


def _rebuild_lock() -> asyncio.Lock:
    global _lock, _lock_loop
    loop = asyncio.get_running_loop()
    if _lock is None or _lock_loop is not loop:
        _lock, _lock_loop = asyncio.Lock(), loop
    return _lock


def document_forms(document: Any) -> tuple[tuple[str, str], ...]:
    """`(name, form)` per expression, from the document alone. Never raises."""
    try:
        from millm.rules.document import expression_form

        kinds = {s["id"]: s["kind"] for s in document.get("signals") or []}
        out = []
        for name, expr in (document.get("expressions") or {}).items():
            form = expression_form(expr)
            out.append((name, str(kinds.get(expr.get("signal"), "unknown"))
                        if form == "signal" else form))
        return tuple(sorted(out))
    except Exception:  # noqa: BLE001
        return ()


def compile_row(row: Any) -> tuple[Optional[CompiledRuleSet], bool]:
    """`(compiled, vectors_failed)` for one stored rule set. Never raises."""
    if row.vector_status == "vectors_failed":
        return None, True
    try:
        return CompiledRuleSet(row.document), False
    except Exception as exc:  # noqa: BLE001 - a set that no longer compiles reports, never runs
        logger.warning("rule_set_compile_failed rule_set_id=%s error=%s", row.id,
                       type(exc).__name__)
        return None, True


async def rebuild(session: Any, *, reason: str = "") -> Registry:
    """Rebuild from the database and swap. Keeps the previous registry on a read failure."""
    global _current, generation
    from millm.db.repositories.rule_set_repository import RuleSetRepository

    async with _rebuild_lock():
        try:
            repository = RuleSetRepository(session)
            attachments = await repository.attachments()
            ids = [i for a in attachments for i in (a.rule_set_ids or [])]
            rows = {r.id: r for r in await repository.by_ids(ids)}
        except Exception as exc:  # noqa: BLE001 - serving keeps the last good registry
            logger.warning("rule_registry_rebuild_failed reason=%s error=%s", reason,
                           type(exc).__name__)
            return _current
        compiled_cache: dict[str, tuple[Optional[CompiledRuleSet], bool]] = {}
        built: dict[str, tuple[AttachedSet, ...]] = {}
        for attachment in attachments:
            sets = []
            for rule_set_id in attachment.rule_set_ids or []:
                row = rows.get(rule_set_id)
                if row is None:
                    continue
                if rule_set_id not in compiled_cache:
                    compiled_cache[rule_set_id] = compile_row(row)
                compiled, failed = compiled_cache[rule_set_id]
                narrowed = (attachment.signals or {}).get(rule_set_id)
                sets.append(AttachedSet(
                    rule_set_id=row.id, digest=row.digest, version=int(row.version or 1),
                    behaviour_key=attachment.behaviour_key, compiled=compiled,
                    vectors_failed=failed,
                    only=tuple(narrowed) if narrowed is not None else None,
                    forms=document_forms(row.document),
                    code_documents=tuple(sorted((row.code_modules or {}).items())),
                ))
            if sets:
                built[attachment.behaviour_key] = tuple(sets)
        _current = Registry(built)
        generation += 1
        return _current


async def rebuild_with_new_session(reason: str = "") -> Registry:
    """For callers without a session of their own (startup, arm/disarm). Never raises."""
    try:
        from millm.db.base import async_session_factory

        async with async_session_factory() as session:
            return await rebuild(session, reason=reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rule_registry_rebuild_failed reason=%s error=%s", reason,
                       type(exc).__name__)
        return _current


def forget_for_tests() -> None:
    global _current
    _current = _EMPTY
