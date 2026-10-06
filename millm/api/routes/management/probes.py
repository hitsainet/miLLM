"""`/api/probes` — import, arm, inspect (FR-24.1, 24.2, 24.8, 24.9).

Shaped like `routes/management/sensing.py`: an `APIRouter` with a prefix, the `ApiResponse`
envelope, and `Annotated` query parameters.

⚠ **THE EVENT DETAIL ROUTE IS THE ONLY PLACE CONTEXT TEXT IS SERVED.** The list route and the
socket both omit `context_text` / `context_token_ids` — they are the decoded window around a
firing position, which is user content. A reviewer asks for one event and gets its context; a
dashboard subscribing to everything does not get a feed of prompts.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Body, Path, Query
from pydantic import BaseModel, ConfigDict, Field

from millm.api.dependencies import (
    DbSession,
    InferenceServiceDep,
    ProbeArmingDep,
    ProbeEventRepo,
    ProbeEventServiceDep,
    ProbeHubServiceDep,
    ProbeRepo,
    ProbeServiceDep,
)
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.probe import Decision
from millm.api.schemas.probe_scoring import ProbeScoreRequest
from millm.services.probe_event_service import event_summary
from millm.core.errors import ProbeNotFoundError
from millm.core.probe_evidence import probe_rung_language, probe_rung_next_step
from millm.core.probe_labels import concept_of, label_mapping_of
from millm.core.config import settings
from millm.services.probe_arm_bridge import (
    build_parity_forward,
    build_probe_encoder,
    loaded_identity,
)
from millm.services.probe_arming import armed_probe_from_row

router = APIRouter(prefix="/api/probes", tags=["probes"])


class ProbeHubImportRequest(BaseModel):
    """One definition from a Hub repo. Shaped like `HubImportRequest` for clusters."""

    repo_id: str = Field(..., min_length=3, max_length=200)
    filename: str = Field(..., min_length=1, max_length=300)
    revision: str | None = None
    on_conflict: Literal["rename", "fail"] = "rename"


class ProbeArmRequest(BaseModel):
    """⚠ `acknowledge_below_rung2` is the OPERATOR's acknowledgement, stored separately from the
    one inside the definition. The person who exported a weak probe and the person arming it on
    live traffic are not necessarily the same, and only the second is choosing to monitor with it.
    """

    acknowledge_below_rung2: bool = False
    reason: str = Field("", max_length=500)
    #: WHICH WINDOWS THIS PROBE REPORTS — `all`, `prompt`, `response`, `last_user`. `None` means
    #: `all`, `prompt` and `response`, plus `last_user` only when the definition carries a bar for
    #: it (`probe_arming.resolve_windows`); an explicit `[]` means the probe's own scope alone, which is what it
    #: did before windows existed. `last_user` (2026-10-04) is the newest user message alone.
    #:
    #: ⚠ THIS IS NOT THE PROBE'S SCOPE AND DOES NOT CHANGE IT. `scope` is identity: what the
    #: probe was trained on and what its threshold was cut under. A probe whose contract scope is
    #: `prompt` or `response` is still refused at arm time on reproducibility grounds. This
    #: chooses which slices of a request the same weights are READ over, because the prompt says
    #: something about the user and the response says something about the model, and a mean over
    #: both answers neither.
    windows: Optional[list[str]] = None


class ProbeRecalibrationRequest(BaseModel):
    """A RE-CUT BAR, AND NOTHING ELSE.

    ⚠ `extra="forbid"` IS THE DOCTRINAL BOUNDARY, EXPRESSED AS A TYPE. `on_conflict=replace` is
    refused on import because overwriting a definition in place would change the DETECTOR
    underneath a running monitor. Moving a bar is not that — the event row already records the
    threshold each verdict was judged against — but the distinction only holds if this route is
    INCAPABLE of carrying a detector. A route that accepted `head`, `read`, `scope`, `basis` or
    `aggregation` and ignored them would be one review away from honouring them.

    `decision` is the contract's own `Decision` model, not a second schema: it is `extra="allow"`
    (inherited from `_Contract`), so a newer miStudio can add `decision.*` fields and they survive
    the round trip, while the envelope around it is closed. Additive inside the bar, closed
    outside it.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    #: WHICH PROBE THE PRODUCER RE-CUT, compared against `definition.provenance.probe_id`.
    mistudio_probe_id: str = Field(min_length=1, max_length=64)
    #: WHICH FIT. Compared when both sides carry one — the same probe id from a different fit is
    #: a different detector.
    mistudio_run_id: Optional[str] = Field(None, max_length=64)
    #: The producer's identity for this cut, recorded verbatim against the revision.
    calibration_id: Optional[str] = Field(None, max_length=64)
    #: Why the bar moved. The operator's question on seeing a changed number is "who, and why",
    #: and `threshold_history` is the only place that can answer.
    reason: str = Field("", max_length=500)


def _probe_summary(probe: Any) -> dict[str, Any]:
    from millm.services.probe_arming import (
        length_bands_from_definition,
        window_thresholds_from_definition,
    )

    return {
        "id": probe.id,
        "name": probe.name,
        "hf_id": probe.hf_id,
        "layer": probe.layer,
        "rule": probe.rule,
        # ⚠ THE RULE'S PARAMETERS, e.g. `{"window": 32}`. Two `rolling_mean_max` probes at one
        # layer differ ONLY here, and without it their tiles read identically (operator,
        # 2026-10-04). Read from the same field arming reads (`aggregation.params`).
        "rule_params": ((probe.definition or {}).get("aggregation") or {}).get("params") or {},
        # Each window's OWN bar (`decision.windows`), and how many length bands refine the bar
        # over the probe's own scope — the tile states which tokens a probe reads and at what bar.
        "window_thresholds": window_thresholds_from_definition(probe.definition),
        "length_band_count": len(length_bands_from_definition(probe.definition)),
        "scope": probe.scope,
        "basis": probe.basis,
        "streamable": probe.streamable,
        "threshold": probe.threshold,
        "target_fpr": probe.target_fpr,
        # ⚠ THE ROW'S REVISION, NOT THE REGISTRY'S. This is a row serialiser; consulting the
        # live registry here would put the reconciliation in two places, and `status()` already
        # owns that boundary and reports disagreement rather than hiding it. A reader comparing
        # an event's revision to this one learns "the bar has since moved", which is true of the
        # stored bar whether or not a refresh landed.
        "threshold_revision": int(getattr(probe, "threshold_revision", 1) or 1),
        "rung": probe.rung,
        # ⚠ Rendered server-side from the shared vocabulary, never derived by a client from the
        # number. miLLM and miStudio must describe the same probe with the same words.
        "rung_language": probe_rung_language(probe.rung),
        "next_step": probe_rung_next_step(probe.rung),
        # ⚠ WHAT IT WAS FITTED ON. The tile used to carry the read point and nothing else, so a
        # list of probes said where each one reads and never what it detects. `label_mapping` is
        # sent RAW — the client names both sides of the boundary, and formatting it here would
        # put a presentation decision on the wire where nothing could check it against the
        # corpus. `concept` is miStudio's own sentence, carried through rather than recomposed.
        "concept": concept_of(probe.definition),
        "label_mapping": label_mapping_of(probe.definition),
        # The precision the probe was fitted at, as the definition STATES it; None = the
        # definition predates the field (it was float16, but the document does not say so).
        "load_dtype": ((probe.definition or {}).get("model") or {}).get("load_dtype"),
        "armed": probe.armed,
        "paused_reason": probe.paused_reason,
        "parity": probe.parity,
        "created_at": probe.created_at,
    }


#: ⚠ Deliberately the SERVICE's serialiser, not a copy. The socket and this route feed the same
#: UI list, and when each built its own dict the socket's lacked `id` and `created_at`.
_event_summary = event_summary


# ── import ────────────────────────────────────────────────────────────────────


@router.post("/import", response_model=ApiResponse)
async def import_probe(
    payload: Annotated[dict[str, Any], Body(...)],
    service: ProbeServiceDep,
    on_conflict: Annotated[str, Query(pattern="^(rename|fail)$")] = "rename",
) -> ApiResponse:
    """Import a definition from a request body.

    `on_conflict` is `rename|fail`, matching circuits and clusters. There is deliberately no
    `replace`: overwriting a definition in place while its probe is ARMED would change the
    detector underneath a running monitor while every event kept the same `probe_id`.

    MOVING A BAR IS NOT REPLACING A DETECTOR.
    A probe definition carries two kinds of fact. The DETECTOR is everything that determines what
    number the probe produces: `head.weights`, `bias`, `norm_mean`, `norm_std`, `attention_query`,
    `read.layer`, `read.hook_point`, `scope`, `basis`, the `sae` block and its `feature_indices`,
    `aggregation.rule` and its `params`, `model`, and the `evidence` that says what the number is
    evidence of. The BAR is everything that determines only where that number is cut:
    `decision.threshold`, `target_fpr`, `realised_fpr`, `threshold_source`, `calibration`,
    `windows` and `length_bands`.

    `replace` was refused because it replaces the first kind, and the objection above stands exactly
    as written. It turns on a specific property of `probe_events`: the row records a `score` whose
    MEANING comes from the detector, and nothing on the row records which detector produced it.
    Change the weights and event #1's `score = 2.9` and event #900's `score = 2.9` are measurements
    of different quantities under one id, with nothing to tell them apart.

    A moved bar is not that, for one concrete reason: the event row ALREADY records the bar it was
    judged against, per verdict, at judgement time — including the length-band override — and nothing
    joins an event back to `probes.threshold`. After a re-cut, event #1 still says it was judged at
    2.8786 and event #900 says 2.4011; both are true and both remain comparable, because the score
    beneath each was produced by the same weights at the same layer under the same scope with the
    same rule. The score is the measurement; the bar is the line drawn across it.

    THE RULE: a probe's identity is everything that determines its SCORE; its bar is everything that
    only determines the CUT. The first may never change in place under a probe id. The second may,
    through `POST /api/probes/{probe_id}/recalibrate`, which is `extra="forbid"` and therefore
    structurally incapable of carrying a detector, refuses any cut it cannot match to
    `provenance.probe_id`, never stores the incoming object as the definition, and refuses a
    threshold with no budget and no named source. `on_conflict` remains `rename|fail`.
    """
    raw_bytes = len(json.dumps(payload).encode("utf-8"))
    probe = await service.import_definition(
        payload, raw_bytes=raw_bytes, on_conflict=on_conflict, origin="file"
    )
    return ApiResponse.ok(_probe_summary(probe))


# ── hub ───────────────────────────────────────────────────────────────────────
#
# Anonymous, read-only. The TTL cache and the circuit breaker are inherited from
# `ClusterHubService`, so a flapping Hub becomes `HUB_UNAVAILABLE` rather than a stalled request.


@router.get("/hub/search", response_model=ApiResponse)
async def hub_search(
    hub: ProbeHubServiceDep,
    q: Annotated[Optional[str], Query(description="Free-text query")] = None,
    base_model: Annotated[Optional[str], Query(description="Filter by base model id")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 30,
) -> ApiResponse:
    """Repos tagged `mistudio-probe-definition`."""
    return ApiResponse.ok(await hub.search(query=q, base_model=base_model, limit=limit))


@router.get("/hub/{repo_id:path}/definitions", response_model=ApiResponse)
async def hub_definitions(
    hub: ProbeHubServiceDep,
    repo_id: Annotated[str, Path(description="Hub repo id (org/name)")],
    revision: Annotated[Optional[str], Query()] = None,
) -> ApiResponse:
    """`manifest.jsonl` preferred, falling back to loose `*.probe.json` files."""
    return ApiResponse.ok(await hub.list_definitions(repo_id, revision=revision))


@router.post("/hub/import", response_model=ApiResponse)
async def hub_import(
    request: ProbeHubImportRequest,
    hub: ProbeHubServiceDep,
    service: ProbeServiceDep,
) -> ApiResponse:
    """Fetch one definition and import it. `origin` records that it came from the Hub."""
    _definition, raw_payload, _hub_ref = await hub.fetch_definition(
        request.repo_id, request.filename, revision=request.revision
    )
    probe = await service.import_definition(
        raw_payload,
        raw_bytes=len(json.dumps(raw_payload).encode("utf-8")),
        on_conflict=request.on_conflict,
        origin="hub",
    )
    return ApiResponse.ok(_probe_summary(probe))


# ── list / get / delete ───────────────────────────────────────────────────────


@router.get("", response_model=ApiResponse)
async def list_probes(
    repository: ProbeRepo,
    armed: Annotated[Optional[bool], Query()] = None,
) -> ApiResponse:
    probes = await repository.list(armed=armed)
    return ApiResponse.ok([_probe_summary(p) for p in probes])


@router.get("/status", response_model=ApiResponse)
async def probe_status(service: ProbeEventServiceDep) -> ApiResponse:
    """What every armed probe is doing, and why any of them is not scoring."""
    return ApiResponse.ok(await service.status())


@router.get("/events", response_model=ApiResponse)
async def list_events(
    events: ProbeEventRepo,
    probe_id: Annotated[Optional[str], Query()] = None,
    request_id: Annotated[Optional[str], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse:
    """Recent verdicts. **Without context text** — fetch one event for that."""
    rows = await events.list_events(
        probe_id=probe_id, request_id=request_id, limit=limit, offset=offset
    )
    return ApiResponse.ok([_event_summary(e) for e in rows])


@router.get("/events/{event_id}", response_model=ApiResponse)
async def get_event(event_id: int, events: ProbeEventRepo) -> ApiResponse:
    """One event, WITH its decoded context window.

    This is the only route that serves it, and it is a deliberate, per-event request rather than
    anything a dashboard receives by subscribing.
    """
    event = await events.get(event_id)
    if event is None:
        raise ProbeNotFoundError(f"No probe event {event_id}")
    detail = _event_summary(event)
    detail["context_text"] = event.context_text
    detail["context_token_ids"] = event.context_token_ids
    return ApiResponse.ok(detail)


@router.delete("/events", response_model=ApiResponse)
async def clear_events(
    events: ProbeEventRepo, probe_id: Annotated[Optional[str], Query()] = None
) -> ApiResponse:
    removed = await events.clear(probe_id)
    return ApiResponse.ok({"removed": removed})


@router.post("/score", response_model=ApiResponse)
async def score_probes(
    request: ProbeScoreRequest,
    session: DbSession,
    repository: ProbeRepo,
    inference: InferenceServiceDep,
) -> ApiResponse:
    """Score stored inputs with imported probes — armed or not — and persist NOTHING.

    Feature 27 (FR-27.4 – FR-27.7). No `probe_events` row, no runtime request context, no change
    to the armed set or to any stored parity report. Each input runs in its own admission slot,
    unsteered (`InferenceService.run_model_work`). The probe is built, scored and decided by the
    same code live serving uses, so offline equals live.

    ⚠ NOT a generation endpoint: it carries no `X-miLLM-Steering` header (X-09). Its forward
    always runs with every SAE suppressed (T-73).
    """
    from millm.services.probe_scoring import ProbeScoringService

    return ApiResponse.ok(
        await ProbeScoringService(repository, inference).score(request, session)
    )


@router.get("/{probe_id}", response_model=ApiResponse)
async def get_probe(probe_id: str, repository: ProbeRepo) -> ApiResponse:
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    detail = _probe_summary(probe)
    detail["definition"] = probe.definition
    return ApiResponse.ok(detail)


@router.delete("/{probe_id}", response_model=ApiResponse)
async def delete_probe(probe_id: str, service: ProbeServiceDep) -> ApiResponse:
    """Delete a probe and its events. Refused while armed."""
    deleted = await service.delete(probe_id)
    if not deleted:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    return ApiResponse.ok({"deleted": probe_id})


# ── arm / disarm ──────────────────────────────────────────────────────────────


@router.post("/{probe_id}/arm", response_model=ApiResponse)
async def arm_probe(
    probe_id: str,
    request: ProbeArmRequest,
    session: DbSession,
    repository: ProbeRepo,
    arming: ProbeArmingDep,
    inference: InferenceServiceDep,
) -> ApiResponse:
    """Run the four gates and arm the probe if they all pass.

    ⚠ **THE ONLY CALLER OF `ProbeArmingService.arm`.** Every gate — the limit, identity, the
    evidence rung, parity — is reachable only through this route, and `test_probe_reachable.py`
    goes red if its registration is removed.

    A refusal is an error in the envelope naming which gate refused and why; the gates run in the
    order they are cheapest to fail, so a wrong-model probe never costs a forward pass.
    """
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")

    identity, model, tokenizer = await loaded_identity(session)
    armed = await arming.arm(
        probe,
        model=model,
        loaded=identity,
        forward=build_parity_forward(model, probe.layer),
        # The parity forward runs inside one admission slot, unsteered (FR-27.6f-g, T-73).
        executor=inference.run_model_work,
        tokenizer=tokenizer,
        acknowledge_below_rung2=request.acknowledge_below_rung2,
        reason=request.reason,
        encoder=await build_probe_encoder(probe),
        windows=request.windows,
    )
    # ⚠ `armed` is an `ArmedProbe`, which has NO `basis` — the basis lives on the row. This read
    # `armed.basis` and raised AttributeError AFTER the probe was already armed and hooked, so
    # the operator saw INTERNAL_ERROR for an operation that had fully succeeded. A response
    # serialiser that can fail after the side effect is worse than one that fails before it.
    detail = _probe_summary(probe)
    detail["armed"] = True
    detail["parity"] = probe.parity
    return ApiResponse.ok(detail)


@router.post("/{probe_id}/parity", response_model=ApiResponse)
async def check_parity(
    probe_id: str,
    session: DbSession,
    repository: ProbeRepo,
    inference: InferenceServiceDep,
) -> ApiResponse:
    """Re-run parity without arming, and store the report.

    Useful after a model reload: the same definition against a differently-loaded model is the
    case parity exists to catch, and an operator should be able to ask before arming.
    """
    from millm.services.probe_parity import ProbeParityEngine

    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")

    # The identity is not CHECKED here (that is arming's gate), but its precision is reported
    # with the result, so a failed parity names its likely cause.
    identity, model, tokenizer = await loaded_identity(session)
    # `windows=[]` — the probe's own scope alone. Parity compares against miStudio's recorded
    # scores, which were computed under that scope; other windows would be work with no reader.
    armed = armed_probe_from_row(probe, encoder=await build_probe_encoder(probe), windows=[])
    tolerance = max(
        float((probe.definition.get("test_vectors") or {}).get("tolerance", 0.0) or 0.0),
        settings.PROBE_PARITY_TOLERANCE,
    )
    from millm.services.probe_parity import model_summary

    forward = build_parity_forward(model, probe.layer)
    # ⚠ INSIDE ONE ADMISSION SLOT, UNSTEERED (FR-27.6e, T-73). This ran the engine directly: no
    # slot, so a parity forward could run beside a generation on the same model, and no
    # suppression, so a profile steering an earlier layer moved the residual being compared
    # against miStudio's recorded scores.
    report = await inference.run_model_work(lambda: ProbeParityEngine(forward).run(
        armed, probe.definition, tolerance=tolerance, tokenizer=tokenizer,
        loaded_dtype=identity.dtype,
        loaded_quantization=identity.quantization,
        model=model_summary(identity),
    ))
    await repository.update(probe, parity=report.as_details())
    return ApiResponse.ok(report.as_details())


@router.post("/{probe_id}/recalibrate", response_model=ApiResponse)
async def recalibrate_probe(
    probe_id: str,
    request: ProbeRecalibrationRequest,
    repository: ProbeRepo,
) -> ApiResponse:
    """Move this probe's decision bar in place, without touching its detector.

    ⚠ **THIS IS NOT `on_conflict=replace` BY ANOTHER NAME, AND THE DIFFERENCE IS LOAD-BEARING.**
    `replace` is refused on import because overwriting a definition in place would change the
    detector underneath a running monitor while every event kept the same `probe_id` — the history
    would describe two detectors as one. That objection turns on `probe_events.score` having no
    record of which detector produced it. A bar is different: the event row ALREADY records the
    threshold each verdict was judged against, so after a re-cut both the old and the new events
    remain true and remain comparable, because the score beneath each came from the same weights
    at the same layer under the same scope with the same rule.

    The rule: a probe's identity is everything that determines its SCORE; its bar is everything
    that only determines the CUT. The first may never change in place under a probe id. The second
    may. `ProbeRecalibrationRequest` is `extra="forbid"`, so this route cannot carry the first.

    Why a route rather than disarm -> delete -> import: that path assigns a new `probe_id`, renames
    the row, leaves a monitoring gap, and **cascade-deletes every `probe_event`** — the entire
    verdict history of the monitor, to change one number. Parity is also unaffected: it compares
    per-token and combined scores and never reads `fires` or `threshold`, so a moved bar needs no
    re-verification of the weights.

    Refusals, each before any write: 404 unknown probe · 422 any field outside the bar
    (`extra="forbid"`) · 409 the cut names a different probe or run, or the stored definition
    records no provenance to compare · 409 a threshold with no budget and no source · 409 a
    per-window or per-length entry the runtime's own parser would discard.

    An ARMED probe's live runtime shape is refreshed in the same call, because a database write
    alone would change nothing that is served — `ArmedProbe` is resolved once at arm time and held
    in process memory. `registry_updated` says whether that happened, and `stale_armed` reports a
    row claiming armed with no live entry rather than quietly reconciling it.
    """
    from millm.core.errors import (
        ProbeRecalibrationMismatchError,
        ProbeThresholdUncalibratedError,
    )
    from millm.services.probe_recalibration import (
        ProbeRecalibrationRefused,
        ProbeRecalibrationService,
    )

    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")

    service = ProbeRecalibrationService(repository)
    try:
        outcome = await service.recalibrate(
            probe,
            # `by_alias=True` because the contract's wire names are what a definition carries, and
            # the stored definition must keep speaking the contract rather than python field names.
            decision=request.decision.model_dump(by_alias=True, exclude_none=False),
            mistudio_probe_id=request.mistudio_probe_id,
            mistudio_run_id=request.mistudio_run_id,
            calibration_id=request.calibration_id,
            reason=request.reason,
        )
    except ProbeRecalibrationRefused as refusal:
        # Mapped to the envelope's own error classes, so a refusal is distinguishable by CODE
        # rather than by reading prose. Two codes, not one: "this is a different probe" and "this
        # is not a calibrated bar" send an operator to different places.
        if refusal.code == "probe_threshold_uncalibrated":
            raise ProbeThresholdUncalibratedError(refusal.detail) from refusal
        raise ProbeRecalibrationMismatchError(refusal.detail) from refusal
    return ApiResponse.ok(outcome)


@router.post("/{probe_id}/disarm", response_model=ApiResponse)
async def disarm_probe(
    probe_id: str, repository: ProbeRepo, arming: ProbeArmingDep
) -> ApiResponse:
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    await arming.disarm(probe, reason="operator")
    return ApiResponse.ok(_probe_summary(probe))
