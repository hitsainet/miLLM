"""
Circuit management API endpoints (Feature 13).

Import `mistudio.circuit-definition/v1` documents, list/activate/deactivate
circuits, dial the global intensity, and re-export lossless definitions.

Two invariants this surface enforces:
  * **Evidence honesty** — every response carries the server-rendered
    ``rung_language``; a circuit below rung 2 can only be activated with an
    explicit ``acknowledge_unvalidated`` (refused as ``UNVALIDATED_CIRCUIT``).
  * **Never a wrong-basis serve** — activation reports ``serving_mode``
    (``full`` vs ``slice_fallback``) so a per-layer projection is never
    presented as the whole circuit.

Additive-only per ``docs/mcp-contract.md`` v1.1 (§4 ``millm_circuits``).
"""

import json
from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, Query, Request
from pydantic import ValidationError as PydanticValidationError

from millm.api.dependencies import CircuitServiceDep
from millm.api.schemas.circuit import (
    CIRCUIT_DEFINITION_KIND,
    MAX_CIRCUIT_IMPORT_BYTES,
    CircuitActivationResponse,
    CircuitDeactivationResponse,
    CircuitIntensityResponse,
    CircuitListResponse,
    CircuitSummary,
    SetCircuitIntensityRequest,
)
from millm.api.schemas.common import ApiResponse
from millm.core.errors import UnvalidatedCircuitError

from millm.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/circuits", tags=["circuits"])

CircuitId = Annotated[str, Path(description="Circuit ID")]

#: The v1 contract nests ~5 levels; anything far beyond that is a nesting bomb.
MAX_IMPORT_NESTING_DEPTH = 32


def _max_depth(value: Any) -> int:
    """Maximum nesting depth of a parsed JSON value.

    Iterative (explicit stack) so measuring the depth cannot itself blow the
    stack on a hostile payload.
    """
    max_seen = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_seen:
            max_seen = depth
        if depth > MAX_IMPORT_NESTING_DEPTH:
            return depth  # already over the limit; stop walking
        if isinstance(node, dict):
            for v in node.values():
                stack.append((v, depth + 1))
        elif isinstance(node, list):
            for v in node:
                stack.append((v, depth + 1))
    return max_seen


@router.get(
    "",
    response_model=ApiResponse[CircuitListResponse],
    summary="List imported circuits",
)
async def list_circuits(
    service: CircuitServiceDep,
    min_rung: int | None = Query(None, ge=0, le=3, description="Only rung >= this"),
    serveable: bool | None = Query(None, description="Only fully-serveable circuits"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ApiResponse[CircuitListResponse]:
    """List circuits with their evidence rung, layers and serveability."""
    rows = await service.list_circuits(
        min_rung=min_rung, serveable=serveable, limit=limit, offset=offset
    )
    total = await service.repository.count(min_rung=min_rung, serveable=serveable)
    active = next((r["id"] for r in rows if r["is_active"]), None)
    return ApiResponse.ok(
        CircuitListResponse(
            circuits=[CircuitSummary(**r) for r in rows],
            active_circuit_id=active,
            total=total,
        )
    )


@router.post(
    "/claims/release",
    response_model=ApiResponse[dict],
    summary="Release a stuck layer claim",
)
async def release_claims(
    service: CircuitServiceDep,
    circuit_id: str = Query(
        ...,
        description=(
            "The circuit whose claims to release. Only claims belonging to "
            "THIS circuit are touched."
        ),
    ),
) -> ApiResponse[dict]:
    """Manually release one circuit's layer claims (F19 R2-10).

    Every claim-leak path in this feature previously had exactly one remedy: a
    full process restart, which drops every loaded model and every attached SAE
    — a multi-minute GPU outage to clear one stale row.

    Scoped to a single circuit ON PURPOSE. A "release everything" button is a
    foot-gun in a feature whose whole point is that several circuits serve at
    once: it would silently strip live circuits of the protection they are
    relying on.

    This does NOT stop steering. It releases the CLAIM, so the layers can be
    taken again. If the circuit is still active and steering, deactivate it
    instead — releasing its claim while it serves leaves it steering layers it
    does not hold, which is the state R2-07 exists to report.
    """
    from millm.services.circuit_claim_registry import CircuitClaimRegistry

    session = getattr(service.repository, "session", None)
    if session is None:
        logger.error("circuit_claims_unavailable_for_release")
        return ApiResponse.fail(
            code="INTERNAL_ERROR",
            message="Claims are not readable, so they cannot be released",
        )

    circuit = None
    try:
        circuit = await service.repository.get(circuit_id)
    except Exception:  # pragma: no cover - the warning below is the point
        pass

    released = await CircuitClaimRegistry(session).release(circuit_id)
    await session.commit()

    warnings: list[str] = []
    # R3-02: an unknown circuit id returned success with `released_layers: []`
    # — indistinguishable from "the claim was already gone". An operator
    # recovering from a stuck claim, working from a name they may have
    # mistyped, could not tell whether they had fixed anything.
    if circuit is None:
        warnings.append(
            f"No circuit '{circuit_id}' exists. Nothing was released — check "
            "the id against GET /api/circuits/claims."
        )
    elif not released:
        warnings.append(
            "This circuit held no live claims, so nothing changed. If an "
            "activation is still being refused, the layers are held by a "
            "different circuit — check GET /api/circuits/claims."
        )
    if circuit is not None and getattr(circuit, "is_active", False):
        # Say it plainly rather than refusing: an operator clearing a claim on
        # a circuit that still reads active is usually recovering from exactly
        # the divergence this warns about.
        warnings.append(
            "This circuit still reads ACTIVE. Its claims are released, so it "
            "is steering layers it no longer holds and another circuit can "
            "take them. Deactivate it unless you are deliberately recovering "
            "from a stuck claim."
        )

    # R3-01: a successful release PROVES the claims subsystem is working, so
    # clear the degraded flag. Without this, `/health/detailed` stayed DEGRADED
    # for the life of the process even after the operator had used the
    # documented remedy — a health signal that cannot recover.
    try:
        from millm.api.routes.system.health import note_claims_healthy

        note_claims_healthy()
    except Exception:  # pragma: no cover - reporting must not fail the release
        pass

    logger.warning(
        "circuit_claims_manually_released",
        circuit_id=circuit_id,
        layers=released,
        still_active=bool(circuit is not None and getattr(circuit, "is_active", False)),
    )
    return ApiResponse.ok(
        {
            "circuit_id": circuit_id,
            "released_layers": released,
            "warnings": warnings,
        }
    )


@router.get(
    "/claims",
    response_model=ApiResponse[list[dict]],
    summary="Which circuit holds which layer",
)
async def list_claims(service: CircuitServiceDep) -> ApiResponse[list[dict]]:
    """Live layer claims: `[{layer, circuit_id, circuit_name, composed}]`.

    F19. The unit of contention is the LAYER, so "what is serving" is not
    answerable from the circuit list alone — two circuits can both be active
    while contending for nothing. This is the view that makes a refusal
    intelligible before it happens.
    """
    from millm.services.circuit_claim_registry import CircuitClaimRegistry

    session = getattr(service.repository, "session", None)
    if session is None:
        # F19 R1-10: an empty list here is rendered by ClaimsStrip as the
        # affirmative statement "No layers are currently claimed" — a
        # confident, wrong all-clear at exactly the moment the claims
        # subsystem is unavailable. There is no third state in the UI for
        # "could not determine", so this must at least be loud in the logs;
        # an operator who then attempts an activation gets a contention
        # refusal contradicting the strip they just read.
        logger.error(
            "circuit_claims_unavailable",
            detail=(
                "no session to read claims through — the claims view will "
                "report NOTHING CLAIMED, which is indistinguishable from the "
                "truth and may contradict the next activation's refusal"
            ),
        )
        return ApiResponse.ok([])

    # R1-20: `live_claims()` now populates `circuit_name` itself, so this no
    # longer reaches through the API boundary into the registry's private
    # `_names_for`.
    claims = await CircuitClaimRegistry(session).live_claims()
    return ApiResponse.ok(
        [
            {
                "layer": c.layer,
                "circuit_id": c.circuit_id,
                "circuit_name": c.circuit_name,
                "composed": c.composed,
                "steering_keys": list(c.steering_keys),
            }
            for c in sorted(claims, key=lambda c: (c.layer, c.circuit_id))
        ]
    )


async def _active_rows_with_steering(
    service: Any, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach each row's own `steering` verdict (F19 R3-07).

    `steering` is the SERVER's verdict on whether a circuit is genuinely
    influencing generation — clients must not derive it from `is_active`,
    which overclaims for a slice-fallback, unparseable or unattached circuit.
    """
    from millm.api.dependencies import get_inference_service
    from millm.services.inference_service import reset_steering_memo

    out: list[dict[str, Any]] = []
    for row in rows:
        summary = CircuitSummary(**row).model_dump()
        # F19 R3-19: ask the OWNER MAP, not `_steering_circuit()`.
        #
        # R3-06 made `_steering_circuit()` return None when SEVERAL circuits
        # serve — correct for the dial and the rung header, since no single
        # circuit describes the response. But this field answers a different
        # question, per row: "is THIS circuit influencing generation?" Reusing
        # the singular predicate made every row report `steering: false` in
        # exactly the state the feature exists to support — two circuits both
        # genuinely steering, and the endpoint saying neither is.
        try:
            from millm.services.sae_service import AttachedSAEState

            summary["steering"] = bool(
                AttachedSAEState().owner_keys(f"circuit:{row['id']}")
            )
        except Exception:
            # An observability nicety must never fail this read.
            summary["steering"] = False
        out.append(summary)
    return out


@router.get(
    "/active",
    response_model=ApiResponse[Any],
    summary="The currently serving circuits",
)
async def get_active_circuit(
    service: CircuitServiceDep,
    single: bool = Query(
        False,
        description=(
            "Pre-F19 shape: return the most recently activated circuit as a "
            "single object instead of a list. Under-reports when several "
            "circuits serve."
        ),
    ),
) -> ApiResponse[Any]:
    """The active circuit (with serving_mode), or null when none is serving.

    Carries ``steering``: the server's own verdict on whether this circuit is
    genuinely influencing generation. R3 found the OWUI filter deriving that
    locally from ``is_active``, which overclaims for a slice-fallback,
    unparseable, or unattached circuit — the same rung overclaim the server
    already suppresses on its own headers. Clients read this field instead.
    """
    # F19 R3-07: several circuits can serve at once, so this returns a LIST.
    # `?single=true` keeps the pre-F19 shape for callers that have not been
    # migrated — but a client using it while two circuits serve is told about
    # one of them, so the compatibility path says so in a header rather than
    # quietly under-reporting.
    rows = await service.list_active()
    if single:
        if not rows:
            return ApiResponse.ok(None)
        if len(rows) > 1:
            logger.info(
                "circuit_active_single_shape_under_reports",
                count=len(rows),
                detail=(
                    "?single=true was used while several circuits are serving "
                    "— only the most recently activated is reported"
                ),
            )
        row = rows[0]
    else:
        # The list form is the truthful one. Wrapped per-row below so each
        # carries its own `steering` verdict.
        return ApiResponse.ok(await _active_rows_with_steering(service, rows))

    summary = CircuitSummary(**row)
    try:
        from millm.api.dependencies import get_inference_service
        from millm.services.inference_service import reset_steering_memo

        reset_steering_memo()
        steering = await get_inference_service()._steering_circuit()
        summary.steering = steering is not None and str(
            getattr(steering, "id", "")
        ) == str(summary.id)
    except Exception:  # a status nicety must never fail the status call
        summary.steering = None
    return ApiResponse.ok(summary)


@router.post(
    "/import",
    response_model=ApiResponse[CircuitSummary],
    summary="Import a circuit definition",
)
async def import_circuit(
    request: Request,
    service: CircuitServiceDep,
    payload: dict[str, Any] = Body(..., description="circuit-definition/v1 JSON"),
    on_conflict: str = Query("rename", pattern="^(rename|fail)$"),
) -> ApiResponse[CircuitSummary]:
    """
    Import a `mistudio.circuit-definition/v1` document. Kind-keyed; strict
    validation against the frozen v1 contract; the RAW document is stored so
    re-export is lossless.
    """
    # Size gates — honest scope: FastAPI has ALREADY read+parsed the body by the
    # time handler code runs, so neither check is pre-parse (a true stream-level
    # limit belongs in middleware/ingress). These bound the DOWNSTREAM work and
    # give hostile payloads a structured refusal instead of a stored megarow.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_CIRCUIT_IMPORT_BYTES:
        return ApiResponse.fail(
            code="PAYLOAD_TOO_LARGE",
            message=f"Import exceeds {MAX_CIRCUIT_IMPORT_BYTES} bytes",
        )
    # Depth gate BEFORE json.dumps: a nesting bomb is cheap in bytes (3000
    # levels ≈ 21 KB, 2% of the cap) but expensive in stack — json.dumps and
    # pydantic both recurse, and `extra="allow"` means the garbage would be
    # accepted and persisted, then re-walked on every activate and export.
    depth = _max_depth(payload)
    if depth > MAX_IMPORT_NESTING_DEPTH:
        return ApiResponse.fail(
            code="VALIDATION_ERROR",
            message=(
                f"Import nests {depth} levels deep (max {MAX_IMPORT_NESTING_DEPTH}) "
                "— the v1 contract nests only a few levels"
            ),
        )

    encoded = json.dumps(payload, separators=(",", ":"))
    if len(encoded) > MAX_CIRCUIT_IMPORT_BYTES:
        return ApiResponse.fail(
            code="PAYLOAD_TOO_LARGE",
            message=f"Import exceeds {MAX_CIRCUIT_IMPORT_BYTES} bytes",
        )

    kind = payload.get("kind")
    if kind != CIRCUIT_DEFINITION_KIND:
        return ApiResponse.fail(
            code="UNKNOWN_KIND",
            message=(
                f"kind {kind!r} is not a supported circuit document "
                f"(expected {CIRCUIT_DEFINITION_KIND!r})"
            ),
        )

    try:
        circuit = await service.import_definition(
            payload, raw_bytes=len(encoded), on_conflict=on_conflict
        )
    except PydanticValidationError as e:
        return ApiResponse.fail(
            code="VALIDATION_ERROR",
            message=f"Payload does not match the v1 contract: {e.error_count()} error(s)",
            details={"errors": json.loads(e.json())[:10]},
        )
    return ApiResponse.ok(CircuitSummary(**service.summarize(circuit)))


@router.post(
    "/{circuit_id}/activate",
    response_model=ApiResponse[CircuitActivationResponse],
    summary="Activate (serve) a circuit",
)
async def activate_circuit(
    circuit_id: CircuitId,
    service: CircuitServiceDep,
    acknowledge_unvalidated: bool = Query(
        False,
        description=(
            "Required to activate a circuit whose evidence rung is below 2 "
            "(not causally validated)"
        ),
    ),
    allow_layer_overlap: bool = Query(
        False,
        description=(
            "Compose onto layers another active circuit already serves. "
            "Refused by default: composition is additive and unbounded in "
            "aggregate, and close-out testing measured TWO steered layers at "
            "individually-harmless strength destroying generation. While any "
            "layer is composed the X-miLLM-Circuit-Rung header is omitted, "
            "because no single circuit's evidence describes the response."
        ),
    ),
) -> ApiResponse[CircuitActivationResponse]:
    """
    Serve a circuit. Fully-bound SAE set ⇒ `serving_mode="full"` (multi-SAE);
    otherwise it degrades to `serving_mode="slice_fallback"` (a per-layer
    projection — NOT the whole circuit). A rung<2 circuit is refused with
    `UNVALIDATED_CIRCUIT` unless acknowledged.
    """
    try:
        result = await service.activate(
            circuit_id,
            acknowledge_unvalidated=acknowledge_unvalidated,
            allow_layer_overlap=allow_layer_overlap,
        )
    # F19: `CircuitLayerContentionError` deliberately has NO handler here.
    #
    # A route-level `except` was written first, then a mutation removing it
    # SURVIVED the whole suite. Probing showed why: the error class sets
    # `status_code = 200`, so `millm_error_handler` already produces a
    # byte-identical envelope — same code, same message, same details, and it
    # logs the refusal too. The handler was pure duplication that only looked
    # load-bearing.
    #
    # Removed rather than kept-and-tested: two paths producing the same
    # response is a place for them to drift, and the class-level contract is
    # the one that also covers every other route raising this error.
    except UnvalidatedCircuitError as e:
        # House style: handler-level refusal as 200 + success:false so the
        # client can surface the rung and re-send with the acknowledgement.
        return ApiResponse.fail(
            code=e.code, message=e.message, details=e.details
        )
    # applied_per_layer keys are ints from the service; JSON needs str keys.
    applied = result.get("applied_per_layer")
    if applied:
        result["applied_per_layer"] = {
            str(layer): {str(idx): val for idx, val in steering.items()}
            for layer, steering in applied.items()
        }
    return ApiResponse.ok(CircuitActivationResponse(**result))


@router.post(
    "/{circuit_id}/deactivate",
    response_model=ApiResponse[CircuitDeactivationResponse],
    summary="Stop serving a circuit",
)
async def deactivate_circuit(
    circuit_id: CircuitId, service: CircuitServiceDep
) -> ApiResponse[CircuitDeactivationResponse]:
    """Deactivate the circuit and clear its steering on every layer."""
    result = await service.deactivate(circuit_id)
    return ApiResponse.ok(CircuitDeactivationResponse(**result))


@router.put(
    "/active/intensity",
    response_model=ApiResponse[CircuitIntensityResponse],
    summary="Set the active circuit's global intensity (lambda)",
)
async def set_active_circuit_intensity(
    service: CircuitServiceDep,
    body: SetCircuitIntensityRequest,
) -> ApiResponse[CircuitIntensityResponse]:
    """One global λ scales every layer of the active circuit together."""
    active = await service.get_active()
    if active is None:
        return ApiResponse.fail(
            code="NO_ACTIVE_CIRCUIT",
            message="No circuit is currently serving",
        )
    try:
        result = await service.set_intensity(
            active["id"],
            body.intensity,
            acknowledge_unvalidated=body.acknowledge_unvalidated,
        )
    except UnvalidatedCircuitError as e:
        # Same house style as activate: a handler-level refusal in the envelope
        # so the client can surface the rung and re-send with the ack.
        return ApiResponse.fail(code=e.code, message=e.message, details=e.details)
    return ApiResponse.ok(CircuitIntensityResponse(**result))


@router.delete(
    "/{circuit_id}",
    response_model=ApiResponse[dict],
    summary="Delete an imported circuit",
)
async def delete_circuit(
    circuit_id: CircuitId, service: CircuitServiceDep
) -> ApiResponse[dict]:
    """Delete a circuit (deactivating it first when it is serving)."""
    return ApiResponse.ok(await service.delete(circuit_id))


@router.get(
    "/{circuit_id}/export",
    summary="Export the raw circuit definition",
)
async def export_circuit(
    circuit_id: CircuitId, service: CircuitServiceDep
) -> dict[str, Any]:
    """Returns the raw `mistudio.circuit-definition/v1` document (no envelope —
    the response IS the portable artifact). Deliberately NO response_model:
    a mirror would strip unknown additive fields from newer producers."""
    return await service.export_definition(circuit_id)
