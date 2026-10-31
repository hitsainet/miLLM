"""The facts on each approval card (ENH-008; BUILD_SPEC §3.1 "Card facts").

One describer per gated action: ``async (values, session) -> {label: value}``. The labels are
the words the operator reads, in the order they are shown; the admin UI renders keys and values as
text, in this order. Each describer runs the route's own first refusal (an unknown model is the
route's 404 ``MODEL_NOT_FOUND``) BEFORE anything is stored, so an agent never files a request that
could only fail.

⚠ Never a lease id: the load and unload describers say only whether one was sent.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from millm.core.errors import ModelNotFoundError

NOTHING_LOADED = "nothing (no model is loaded)"


async def _model(session: AsyncSession, model_id: int) -> Any:
    from millm.db.repositories.model_repository import ModelRepository

    row = await ModelRepository(session).get_by_id(int(model_id))
    if row is None:
        raise ModelNotFoundError(f"Model with ID '{model_id}' not found",
                                 details={"model_id": model_id})
    return row


def _name(row: Any) -> str:
    return f"{row.name} (id {row.id})"


def _resident() -> Any | None:
    """The model being served now, from the loader's process state."""
    from millm.ml.model_loader import LoadedModelState

    return LoadedModelState().current


def _attached_saes() -> list[str]:
    from millm.services.sae_service import AttachedSAEState

    return [f"{e.sae_id} at layer {e.layer}" for e in AttachedSAEState().entries()]


def _steering_active() -> bool:
    from millm.services.sae_service import AttachedSAEState

    return any(e.sae.is_steering_enabled for e in AttachedSAEState().entries())


def _gpu_words(request: Any) -> str:
    gpu = None
    if isinstance(request, dict):
        gpu = request.get("gpu")
    elif request is not None:
        gpu = getattr(request, "gpu", None)
    if gpu is None or gpu == "auto":
        return "automatic: the card with the most free memory that fits"
    if gpu == "all":
        return "split across every visible card"
    return str(gpu)


async def model_load(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    row = await _model(session, values["model_id"])
    resident = _resident()
    if resident is None:
        evicts = NOTHING_LOADED
    elif resident.model_id == row.id:
        evicts = "nothing (this model is already loaded)"
    else:
        evicts = f"{resident.model_name} (id {resident.model_id})"
    return {
        "Model": _name(row),
        "Repository": row.repo_id or row.local_path or "—",
        "Precision": getattr(row.quantization, "value", row.quantization),
        "Unloads first": evicts,
        "Graphics card": _gpu_words(values.get("request")),
        "Lease": ("sent by the agent; kept hidden and checked again when the load runs"
                  if values.get("x_millm_lease") else "none sent"),
    }


async def model_unload(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    row = await _model(session, values["model_id"])
    resident = _resident()
    loaded = resident is not None and resident.model_id == row.id
    saes = _attached_saes() if loaded else []
    return {
        "Model": _name(row),
        "Loaded now": "yes" if loaded else "no (unloading it changes nothing)",
        "Attached SAEs it drops": ", ".join(saes) if saes else "none",
        "Active steering it drops": "yes" if loaded and _steering_active() else "none",
        "Lease": ("sent by the agent; kept hidden and checked again when the unload runs"
                  if values.get("x_millm_lease") else "none sent"),
    }


async def model_delete(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    row = await _model(session, values["model_id"])
    size = row.disk_size_mb
    return {
        "Model": _name(row),
        "Repository": row.repo_id or row.local_path or "—",
        "Size on disk": f"{size:,} MB" if size else "unknown",
        "Warning": "A deleted model cannot be recovered; it has to be downloaded again.",
    }


def _limits_words(limits: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("max_armed", "max_behaviours", "decode_budget", "prefill_budget"):
        value = limits.get(key)
        if isinstance(value, dict) and "value" in value:
            value = f"{value['value']} {value.get('unit', '')}".strip()
        out[key] = "global default" if value is None else value
    return out


async def probe_limits_write(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    from millm.services import probe_limits

    row = await _model(session, values["model_id"])
    body: Any = values.get("request")
    requested: dict[str, Any] = (body.model_dump() if hasattr(body, "model_dump")
                                 else dict(body or {}))
    current: dict[str, Any] = {}
    try:
        capacity = await probe_limits.capacity(session, row)
        current = dict(capacity.get("configured") or {})
    except Exception:  # noqa: BLE001 - the card says it could not read them; never blocks filing
        current = {}
    facts: dict[str, Any] = {
        "Model": _name(row),
        "Current limits": _limits_words(current) if current else "could not be read",
        "Requested limits": _limits_words(requested),
        "Source": requested.get("source") or "operator",
    }
    if requested.get("source") == "recommended-approved":
        facts["Recommendation digest"] = requested.get("recommendation_digest")
        facts["Note"] = ("The values written are the recommendation miLLM recomputes when this "
                         "runs; a changed recommendation is refused then.")
    facts["Reason"] = requested.get("reason") or "none given"
    return facts


async def capacity_measure(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    row = await _model(session, values["model_id"])
    resident = _resident()
    loaded = resident is not None and resident.model_id == row.id
    return {
        "Model": _name(row),
        "Loaded now": "yes" if loaded else "no (the measurement needs it loaded and will refuse)",
        "Graphics card": (str(getattr(resident, "device", "")) or "unknown") if loaded else "—",
        "What it does": "Runs the resident model on its graphics card to time each step and the "
                        "cost of each probe. Chat requests wait while it runs.",
    }


async def code_rule_import(values: dict[str, Any], _session: AsyncSession) -> dict[str, Any]:
    """The D4 code-rule card: every code document IN FULL, its hash, functions and runner
    contract, and the set's own case results. Runs every check import runs (it writes nothing),
    so an import that would be refused is refused now, with import's own error."""
    from millm.rules import runner_contract as rc
    from millm.services import rule_set_service

    raw = str(values.get("request_body") or "").encode("utf-8")
    prepared = rule_set_service.check(raw, declared_digest=values.get("digest"),
                                      origin=values.get("origin"))
    document = prepared.document
    modules = []
    for code_sha256, text in sorted((prepared.code_modules or {}).items()):
        try:
            parsed = rc.parse_code_document(code_sha256, text)
            functions = sorted(parsed.functions)
            source = parsed.source
        except Exception:  # noqa: BLE001 - already verified by check(); never hide the text
            functions, source = [], text
        modules.append({"code_sha256": code_sha256, "functions": functions,
                        "runner_contract": rc.contract_format(), "source": source})
    named = sorted({s.get("code_sha256") for s in document.get("signals") or []
                    if s.get("kind") == "code_rule" and s.get("code_sha256")})
    missing = [h for h in named if h not in (prepared.code_modules or {})]
    return {
        "Rule set": f"{document.get('name')} (behaviour {document.get('behaviour_key')})",
        "Digest": prepared.digest,
        "Version": values.get("version"),
        "Cases": f"{prepared.cases.passed} passed, {prepared.cases.unverified} not verifiable here",
        "Code rules named": ", ".join(named) if named else "none",
        "Code documents missing": (", ".join(missing) + " (those rules stay pending)"
                                   if missing else "none"),
        "Revocation": "miLLM asks for each import; there is no revoked state here.",
        "code_modules": modules,
    }


def names_operator_code(values: dict[str, Any]) -> bool:
    """True when a rule-set import names operator code anywhere. Port of miStudio's
    ``names_operator_code``, FAILING CLOSED: any ``code_modules`` key, or any object anywhere
    whose ``kind`` is ``code_rule``. A body that is not JSON raises, and the gate then gates."""
    from millm.rules.code_modules import names_operator_code as scan

    return scan(json.loads(str(values.get("request_body") or "")))


# ENH-006 D76: an agent's change to telemetry settings. Each describer runs the save's own
# validation (`runtime_settings.plan_changes`): a refused value — an endpoint with a user name or a
# query string, an environment-only key — answers the route's own 422 and is never stored.


def _setting_values(values: dict[str, Any]) -> tuple[dict[str, Any], list[str], Any]:
    """(values to set, keys to reset, reason) from any of the three environment write routes."""
    body: Any = values.get("body")
    data: dict[str, Any] = (body.model_dump() if hasattr(body, "model_dump")
                            else dict(body or {}))
    key = values.get("key")
    if key is not None and "value" in data:  # PUT /api/environment/{key}
        return {str(key): data.get("value")}, [], data.get("reason")
    if key is not None:  # DELETE /api/environment/{key}
        return {}, [str(key)], values.get("reason")
    return dict(data.get("values") or {}), list(data.get("reset") or []), data.get("reason")


async def telemetry_write(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    from millm.services import runtime_settings

    to_set, to_reset, reason = _setting_values(values)
    return await runtime_settings.describe_save(session, to_set, to_reset, reason)


def names_telemetry(values: dict[str, Any]) -> bool:
    """True when an environment write names a telemetry setting (the gate's predicate). Read
    generously (`runtime_settings.is_telemetry_key`); a body it cannot read gates (it raises)."""
    from millm.services.runtime_settings import is_telemetry_key

    to_set, to_reset, _ = _setting_values(values)
    return any(is_telemetry_key(k) for k in [*to_set, *to_reset])
