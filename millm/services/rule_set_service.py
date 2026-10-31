"""Rule sets as data: validate, import, list, get, delete, evaluate; behaviour attachments
(039 FR-39.17, FR-39.21, FR-39.22, FR-39.25; FTDD §5.2).

Every path to an accepted rule set is `millm.rules.prepare` (size → JSON → forms → schema and
references → compile → own cases), so validate and import cannot disagree: validate is import
without the write. Refusals are typed `MiLLMError`s whose `details` name where — a pointer, a
signal, a case index — never the document's text.

Rule sets are never replaced in place: a new version is a new import with a new id and digest,
and an identical document (same identity digest) returns the existing id.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Optional

from millm.core.config import settings
from millm.core.errors import (
    CatalogueLimitError,
    RuleFormUnavailableError,
    RulePatternUnsupportedError,
    RuleSetBehaviourMismatchError,
    RuleSetDigestMismatchError,
    RuleSetInUseError,
    RuleSetInvalidError,
    RuleSetNotFoundError,
    RuleVectorsFailedError,
)
from millm.db.models.rule_set import RuleSet
from millm.db.repositories.rule_set_repository import RuleSetRepository
from millm.rules import Prepared, RuleSetInvalid, RuleSetNotRunnable, RuleVectorsFailed, prepare
from millm.rules import registry as rule_registry
from millm.rules.document import NOT_USED_AT_SERVING

logger = logging.getLogger(__name__)

_FORM_CODES = {"form_unavailable", "validator_unknown"}


def _refusal(exc: Exception) -> Exception:
    """The typed refusal for one `prepare` failure."""
    if isinstance(exc, RuleVectorsFailed):
        return RuleVectorsFailedError(
            f"the rule set disagrees with its own case {exc.details.get('case_index')} "
            f"({exc.details.get('case_id')}) at {exc.details.get('where')}",
            details=dict(exc.details))
    code = getattr(exc, "code", "rule_set_invalid")
    details = {"rule": code, **dict(getattr(exc, "details", {}) or {})}
    if code == "pattern_refused":
        return RulePatternUnsupportedError(
            f"signal {details.get('signal')!r}: the pattern uses {details.get('rule')!r}, which "
            "the linear-time dialect does not run", details=details)
    if code in _FORM_CODES:
        return RuleFormUnavailableError(str(exc), details=details)
    return RuleSetInvalidError(str(exc), details=details)


def body_limit() -> int:
    """The most bytes an import or validate body may have, checked BEFORE it is parsed: the
    rule-set document's own limit plus the code documents an envelope may carry (S3)."""
    return int(settings.RULE_SET_MAX_BYTES) + int(settings.RULE_CODE_MODULES_MAX_BYTES)


def _document_and_modules(raw: bytes) -> tuple[Any, Optional[dict[str, Any]]]:
    """A bare rule-set document, or the S3 envelope `{rule_set, code_modules}` (OQ-39.7 (a)).
    The rule-set document keeps its own size limit either way."""
    import json

    from millm.rules.code_modules import split_envelope
    from millm.rules.document import parse_bytes

    doc_limit = int(settings.RULE_SET_MAX_BYTES)
    parsed = parse_bytes(raw, body_limit())
    document, modules = split_envelope(parsed)
    size = len(raw) if modules is None else len(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if size > doc_limit:
        raise RuleSetInvalid("rule_set_too_large",
                             f"the document is {size} bytes; the limit is {doc_limit}",
                             {"bytes": size, "limit": doc_limit, "setting": "RULE_SET_MAX_BYTES"})
    return document, modules


def check(raw: bytes, *, declared_digest: Optional[str] = None,
          origin: Optional[str] = None) -> Prepared:
    """Every check import runs, writing nothing. Raises the typed refusal."""
    import dataclasses

    from millm.rules import code_modules

    try:
        document, modules = _document_and_modules(raw)
        prepared = prepare(document, declared_origin=origin)
        verified = code_modules.verify(prepared.document, modules,
                                       max_bytes=int(settings.RULE_CODE_MODULES_MAX_BYTES))
    except (RuleSetInvalid, RuleSetNotRunnable, RuleVectorsFailed) as exc:
        raise _refusal(exc) from None
    prepared = dataclasses.replace(prepared, code_modules=verified)
    names = len(prepared.document["signals"]) + len(prepared.document["expressions"])
    limit = int(settings.RULE_SET_MAX_SIGNALS)
    if names > limit:
        raise CatalogueLimitError(
            f"the rule set declares {names} signals and expressions; at most {limit} may",
            details={"limit": "RULE_SET_MAX_SIGNALS", "value": limit, "source": "global",
                     "declared": names},
            next_tool="millm_validate_rule_set")
    if declared_digest is not None and declared_digest != prepared.digest:
        raise RuleSetDigestMismatchError(
            "the declared digest is not this document's identity (RFC 8785 + SHA-256)",
            details={"declared": declared_digest, "recomputed": prepared.digest})
    return prepared


def summary(row: RuleSet, attachments: Optional[list[str]] = None) -> dict[str, Any]:
    document = row.document or {}
    return {
        "id": row.id,
        "name": row.name,
        "version": row.version,
        "behaviour_key": row.behaviour_key,
        "origin": row.origin,
        "digest": row.digest,
        "vector_status": row.vector_status,
        "signals": len(document.get("signals") or []),
        "expressions": sorted((document.get("expressions") or {}).keys()),
        "imported_at": row.imported_at.isoformat() if row.imported_at else None,
        "checked_at": row.checked_at.isoformat() if row.checked_at else None,
        # 039 S3: the code documents held (hashes only, never the source) and those missing.
        "code_modules": sorted(row.code_modules or {}),
        "code_modules_missing": _missing_modules(row),
        **({"attached_to": attachments} if attachments is not None else {}),
    }


def _missing_modules(row: RuleSet) -> list[str]:
    from millm.rules.code_modules import missing

    try:
        return missing(row.document or {}, row.code_modules)
    except Exception:  # noqa: BLE001 - a summary never fails on a stored document
        return []


def detail(row: RuleSet, attachments: list[str]) -> dict[str, Any]:
    """`GET /api/rule-sets/{id}`: the document, its provenance and vector status, where it is
    attached, and which of its fields serving does not use (OQ-39.2 (a))."""
    from millm.rules.registry import document_forms

    document = row.document or {}
    return {
        **summary(row, attachments),
        "description": document.get("description"),
        "forms": dict(document_forms(document)),
        "provenance": row.provenance,
        "vector_report": row.vector_report,
        "not_used_at_serving": {k: document[k] for k in NOT_USED_AT_SERVING if k in document},
        "document": document,
    }


async def import_rule_set(session: Any, raw: bytes, *, version: int = 1,
                          declared_digest: Optional[str] = None,
                          origin: Optional[str] = None) -> tuple[RuleSet, bool]:
    """`(row, created)`. An identical document returns the existing row, `created=False`."""
    prepared = check(raw, declared_digest=declared_digest, origin=origin)
    repository = RuleSetRepository(session)
    existing = await repository.by_digest(prepared.digest)
    if existing is not None:
        added = {k: v for k, v in prepared.code_modules.items()
                 if k not in (existing.code_modules or {})}
        if added:
            # 039 S3: the same rule set again, now WITH code documents it lacked (verified above
            # against this document's own signals): stored beside it, nothing else changes.
            existing.code_modules = {**(existing.code_modules or {}), **added}
            await session.commit()
            await rule_registry.rebuild(session, reason="import")
        return existing, False
    limit = int(settings.RULE_SETS_MAX_IMPORTED)
    if await repository.count() >= limit:
        raise CatalogueLimitError(
            f"{limit} rule sets are imported; delete one first",
            details={"limit": "RULE_SETS_MAX_IMPORTED", "value": limit, "source": "global"},
            next_tool="millm_delete_rule_set")
    document = prepared.document
    row = RuleSet(
        id="rs_" + secrets.token_hex(6),
        name=document["name"],
        version=max(1, int(version)),
        behaviour_key=document["behaviour_key"],
        origin=prepared.origin,
        digest=prepared.digest,
        document=document,
        provenance=document.get("provenance"),
        vector_status="passed",
        code_modules=prepared.code_modules or None,
    )
    row = await repository.create(row)
    await rule_registry.rebuild(session, reason="import")
    return row, True


async def get_or_404(session: Any, rule_set_id: str) -> RuleSet:
    row = await RuleSetRepository(session).get(rule_set_id)
    if row is None:
        raise RuleSetNotFoundError(f"rule set {rule_set_id!r} is not imported",
                                   details={"rule_set_id": rule_set_id})
    return row


def _armed_behaviours() -> set[str]:
    from millm.services.probe_runtime import ProbeRuntimeState

    out = set()
    for probe in ProbeRuntimeState().armed():
        key = getattr(getattr(probe, "descriptor", None), "behaviour_key", None)
        if isinstance(key, str) and key:
            out.add(key)
    return out


async def delete_rule_set(session: Any, rule_set_id: str, *, by: Optional[str] = None,
                          reason: Optional[str] = None) -> dict[str, Any]:
    """Refused while attached to a behaviour with an armed probe; otherwise detached from every
    behaviour (one history row each) and deleted. Its results are kept (no foreign key)."""
    repository = RuleSetRepository(session)
    row = await get_or_404(session, rule_set_id)
    attached = await repository.attachments_naming(rule_set_id)
    busy = sorted(a.behaviour_key for a in attached if a.behaviour_key in _armed_behaviours())
    if busy:
        raise RuleSetInUseError(
            f"rule set {rule_set_id!r} is attached to {len(busy)} behaviour(s) with an armed "
            "probe; clear the attachment or disarm first", details={"behaviours": busy})
    for attachment in attached:
        remaining = [i for i in attachment.rule_set_ids if i != rule_set_id]
        signals = {k: v for k, v in (attachment.signals or {}).items() if k != rule_set_id} \
            or None
        await repository.set_attachment(attachment.behaviour_key, remaining, signals,
                                        action="rule_set_deleted", by=by,
                                        reason=reason or f"rule set {rule_set_id} deleted")
    await repository.delete(row)
    await rule_registry.rebuild(session, reason="delete")
    return {"id": rule_set_id, "deleted": True,
            "detached_from": sorted(a.behaviour_key for a in attached)}


async def set_attachment(session: Any, key: str, rule_set_ids: list[str],
                         signals: Optional[dict[str, list[str]]], *, by: Optional[str],
                         reason: Optional[str]) -> dict[str, Any]:
    """`PUT /api/probes/behaviours/{key}/rule-signals` (FR-39.22): every id exists and is of
    this behaviour; a narrowing names expressions the set declares. Version lag is allowed."""
    repository = RuleSetRepository(session)
    wanted = list(dict.fromkeys(rule_set_ids))
    rows = {r.id: r for r in await repository.by_ids(wanted)}
    missing = [i for i in wanted if i not in rows]
    if missing:
        raise RuleSetNotFoundError(f"{len(missing)} rule set id(s) are not imported",
                                   details={"rule_set_ids": missing})
    foreign = [{"rule_set_id": i, "behaviour_key": rows[i].behaviour_key}
               for i in wanted if rows[i].behaviour_key != key]
    if foreign:
        raise RuleSetBehaviourMismatchError(
            f"{len(foreign)} rule set(s) are keyed to another behaviour than {key!r}",
            details={"behaviour": key, "rule_sets": foreign})
    for rule_set_id, names in (signals or {}).items():
        if rule_set_id not in rows:
            raise RuleSetInvalidError("a narrowing names a rule set that is not attached",
                                      details={"rule_set_id": rule_set_id})
        declared = set((rows[rule_set_id].document or {}).get("expressions") or {})
        unknown = sorted(set(names) - declared)
        if unknown:
            raise RuleSetInvalidError("a narrowing names signals the rule set does not declare",
                                      details={"rule_set_id": rule_set_id, "signals": unknown})
    await repository.set_attachment(key, wanted, signals or None, action="attach", by=by,
                                    reason=reason)
    await rule_registry.rebuild(session, reason="attach")
    return await attachment_view(session, key)


async def clear_attachment(session: Any, key: str, *, by: Optional[str],
                           reason: Optional[str]) -> dict[str, Any]:
    repository = RuleSetRepository(session)
    await repository.set_attachment(key, [], None, action="clear", by=by, reason=reason)
    await rule_registry.rebuild(session, reason="detach")
    return await attachment_view(session, key)


async def attachment_view(session: Any, key: str) -> dict[str, Any]:
    repository = RuleSetRepository(session)
    row = await repository.attachment(key)
    ids = list(row.rule_set_ids) if row is not None else []
    sets = {r.id: r for r in await repository.by_ids(ids)}
    history = await repository.history(key, limit=20)
    return {
        "behaviour": key,
        "rule_set_ids": ids,
        "signals": row.signals if row is not None else None,
        "rule_sets": [summary(sets[i]) for i in ids if i in sets],
        "history": [
            {"changed_at": h.changed_at.isoformat() if h.changed_at else None,
             "action": h.action, "rule_set_ids": h.rule_set_ids, "signals": h.signals,
             "by": h.by, "reason": h.reason}
            for h in history
        ],
    }
