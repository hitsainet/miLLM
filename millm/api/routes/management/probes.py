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
from datetime import datetime
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
from millm.api.schemas.common import ApiResponse, PageResponse
from millm.api.schemas.probe import Decision
from millm.api.schemas.probe_scoring import ProbeScoreRequest
from millm.services.probe_event_service import event_summary
from millm.core.errors import ProbeNotFoundError
from millm.core.probe_evidence import probe_rung_language, probe_rung_next_step
from millm.core.probe_labels import (
    behaviour_of,
    concept_of,
    label_mapping_of,
    source_probe_id_of,
)
from millm.core.config import settings
from millm.services.probe_arm_bridge import (
    build_parity_forward,
    build_probe_encoder,
    loaded_identity,
)
from millm.services.probe_arming import armed_probe_from_row

router = APIRouter(prefix="/api/probes", tags=["probes"])

#: A behaviour key, as the definition contract and the result contract spell it.
BEHAVIOUR_KEY = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"


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


class BehaviourArmRequest(BaseModel):
    """`POST /api/probes/behaviours/{key}/arm` (039 FR-39.2, FR-39.6).

    `version` narrows to one behaviour version — and `null` means THE UNCATALOGUED VERSION, so its
    absence and an explicit `null` are different requests (`model_fields_set` tells them apart).
    `probe_ids` narrows to a subset of the behaviour's probes; one of another key is refused.
    `dry_run` runs every gate for every probe and arms nothing.
    """

    model_config = ConfigDict(extra="forbid")

    acknowledge_below_rung2: bool = False
    reason: str = Field("", max_length=500)
    windows: Optional[list[str]] = None
    version: Optional[int] = Field(None, ge=1)
    probe_ids: Optional[list[str]] = Field(None, min_length=1, max_length=256)
    dry_run: bool = False


class BehaviourDisarmRequest(BaseModel):
    """`POST /api/probes/behaviours/{key}/disarm` (FR-39.3). Narrowing as for arming.

    `reason` is the operator's note, logged as its length only; the row records `operator`, as a
    single-probe disarm does (it is VARCHAR(64), and a free sentence is not a reason code)."""

    model_config = ConfigDict(extra="forbid")

    version: Optional[int] = Field(None, ge=1)
    probe_ids: Optional[list[str]] = Field(None, min_length=1, max_length=256)
    reason: str = Field("", max_length=500)


def _version_arg(request: BaseModel) -> Any:
    """`UNSET` when the body did not name a version; else the value, `None` included."""
    from millm.services import probe_selection

    return request.version if "version" in request.model_fields_set else probe_selection.UNSET


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
    #: 038 R-039 (D38.28): miStudio's `mistudio.probe-identity/1` digest of THIS decision. Verified
    #: against the decision this call stores; a mismatch is `PROBE_IDENTITY_MISMATCH` (409) and
    #: nothing is written. miStudio's tool sends it only when this OpenAPI advertises it.
    decision_digest: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$")
    #: miStudio's revision of the decision; stored as `decision_revision` (NULL when not sent).
    decision_revision: Optional[int] = Field(None, ge=1)
    #: Why the bar moved. The operator's question on seeing a changed number is "who, and why",
    #: and `threshold_history` is the only place that can answer.
    reason: str = Field("", max_length=500)


#: What a probe with no stored result reads as. Zero, never absent: the list route always answers.
NO_RESULTS: dict[str, Any] = {"results_total": 0, "results_fired": 0, "last_result_at": None}


def _probe_summary(probe: Any, totals: dict[str, Any] | None = None) -> dict[str, Any]:
    """One probe as the UI and the MCP tools read it.

    `totals` — this probe's entry from `ProbeEventRepository.totals_by_probe` — is passed by the
    list and get routes, which add `results_total`, `results_fired` and `last_result_at`
    (operator 2026-10-29: the tile showed no count until opened). Routes that answer with one
    probe after a change (import, arm, disarm) omit them rather than send a guess.
    """
    from millm.services.probe_arming import (
        length_bands_from_definition,
        window_thresholds_from_definition,
    )

    summary = {
        "id": probe.id,
        "name": probe.name,
        "hf_id": probe.hf_id,
        "layer": probe.layer,
        # How many layers the model has, as the definition recorded it, so a tile can say
        # "layer 4 of 16" (operator 2026-10-29). The row already stores it for the arm-time check.
        "n_layers": probe.n_layers,
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
        # Feature 36 (D18, D23): the behaviour this probe monitors, as the definition states it.
        # `None` = the definition names none; the session timeline then groups by probe name.
        "behaviour": behaviour_of(probe.definition),
        # 036 FR-36.14 (coordinator addition): the PRODUCER's id for this probe
        # (`definition.provenance.probe_id`), so miStudio can match its catalogue and spot-check
        # probes to this one. `None` when the definition records none.
        "source_probe_id": source_probe_id_of(probe.definition),
        "label_mapping": label_mapping_of(probe.definition),
        # The precision the probe was fitted at, as the definition STATES it; None = the
        # definition predates the field (it was float16, but the document does not say so).
        "load_dtype": ((probe.definition or {}).get("model") or {}).get("load_dtype"),
        "armed": probe.armed,
        "paused_reason": probe.paused_reason,
        "parity": probe.parity,
        "created_at": probe.created_at,
    }
    if totals is not None:
        summary.update(totals)
    return summary


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


@router.get("", response_model=PageResponse)
async def list_probes(
    repository: ProbeRepo,
    events: ProbeEventRepo,
    armed: Annotated[Optional[bool], Query()] = None,
    behaviour: Annotated[Optional[str], Query(max_length=64)] = None,
    cursor: Annotated[Optional[str], Query(max_length=2048)] = None,
    limit: Annotated[Optional[int], Query(ge=1, le=1000)] = None,
) -> ApiResponse:
    """One page of imported probes, ordered by behaviour key (none last) then id, plus `next`.

    ⚠ 039 FR-39.15 — A DOCUMENTED CHANGE: with no paging parameter this returns the FIRST PAGE
    (`PROBE_LIST_PAGE_SIZE`, 200) and `meta.next`, not every probe; follow `next` until it is
    `null`. Each item's shape is unchanged. Keyset pages: every probe appears exactly once even
    while the catalogue changes between two reads.

    Each item also carries `results_total`, `results_fired` and `last_result_at` (2026-10-29),
    counted for the whole page in ONE grouped query.
    """
    from millm.api.pagination import decode_cursor, encode_cursor, page_meta

    size = int(limit or settings.PROBE_LIST_PAGE_SIZE)
    after = decode_cursor("probes", cursor, arity=2)
    probes = await repository.page(armed=armed, behaviour=behaviour, after=after, limit=size + 1)
    more = len(probes) > size
    probes = probes[:size]
    next_cursor = (encode_cursor("probes", [probes[-1].behaviour_key, probes[-1].id])
                   if more else None)
    totals = await events.totals_by_probe([p.id for p in probes])
    return PageResponse.page(
        [_probe_summary(p, totals.get(p.id, NO_RESULTS)) for p in probes],
        page_meta(next_cursor, size),
    )


# ── behaviours (039 S1) ───────────────────────────────────────────────────────
#
# ⚠ DECLARED ABOVE `GET /{probe_id}`: a `GET /api/probes/behaviours` declared after it would be
# routed as probe id `behaviours` and answer PROBE_NOT_FOUND (`test_behaviour_routes.py` asserts
# 200 through the live app).


@router.get("/behaviours", response_model=PageResponse)
async def list_behaviours(
    session: DbSession,
    key: Annotated[Optional[str], Query(max_length=64, description="Key prefix")] = None,
    cursor: Annotated[Optional[str], Query(max_length=2048)] = None,
    limit: Annotated[Optional[int], Query(ge=1, le=1000)] = None,
) -> ApiResponse:
    """Every behaviour key with its probes per version, how many are armed (from the registry),
    the partial state ("partial: n of m armed"), "armed before unload / restart" and attached
    rule sets — paginated by key (FR-39.4, FR-39.5, FR-39.15)."""
    from millm.api.pagination import page_meta
    from millm.services import probe_behaviours

    size = int(limit or settings.PROBE_LIST_PAGE_SIZE)
    items, next_cursor = await probe_behaviours.list_behaviours(
        session, cursor=cursor, limit=size, key_prefix=key
    )
    return PageResponse.page(items, page_meta(next_cursor, size))


class RuleSignalAttachRequest(BaseModel):
    """`PUT /api/probes/behaviours/{key}/rule-signals` (039 FR-39.22)."""

    model_config = ConfigDict(extra="forbid")

    rule_set_ids: list[str] = Field(..., min_length=1, max_length=64)
    #: `{rule_set_id: [signal, ...]}` narrows a set to named signals; absent = every signal.
    signals: Optional[dict[str, list[str]]] = None
    reason: Optional[str] = Field(default=None, max_length=500)


@router.get("/behaviours/{key}/rule-signals", response_model=ApiResponse)
async def get_behaviour_rule_signals(
    key: Annotated[str, Path(pattern=BEHAVIOUR_KEY)], session: DbSession,
) -> ApiResponse:
    """The rule sets attached to one behaviour, and the attachment's history (FR-39.22)."""
    from millm.services import rule_set_service

    return ApiResponse.ok(await rule_set_service.attachment_view(session, key))


@router.put("/behaviours/{key}/rule-signals", response_model=ApiResponse)
async def set_behaviour_rule_signals(
    key: Annotated[str, Path(pattern=BEHAVIOUR_KEY)],
    request: RuleSignalAttachRequest,
    session: DbSession,
) -> ApiResponse:
    """Attach rule sets to a behaviour, replacing its list. Each set must be keyed to this
    behaviour (`RULE_SET_BEHAVIOUR_MISMATCH`); a version lag is allowed and shown per result."""
    from millm.services import rule_set_service

    return ApiResponse.ok(await rule_set_service.set_attachment(
        session, key, request.rule_set_ids, request.signals, by="operator",
        reason=request.reason))


@router.delete("/behaviours/{key}/rule-signals", response_model=ApiResponse)
async def clear_behaviour_rule_signals(
    key: Annotated[str, Path(pattern=BEHAVIOUR_KEY)], session: DbSession,
    reason: Annotated[Optional[str], Query(max_length=500)] = None,
) -> ApiResponse:
    from millm.services import rule_set_service

    return ApiResponse.ok(await rule_set_service.clear_attachment(
        session, key, by="operator", reason=reason))


@router.post("/behaviours/{key}/arm", response_model=ApiResponse)
async def arm_behaviour(
    key: Annotated[str, Path(pattern=BEHAVIOUR_KEY)],
    request: BehaviourArmRequest,
    session: DbSession,
    repository: ProbeRepo,
    arming: ProbeArmingDep,
    inference: InferenceServiceDep,
) -> ApiResponse:
    """Arm one behaviour's probes ALL OR NOTHING (FR-39.2, D14), or dry-run it (FR-39.6).

    One selector (`probe_selection.select_behaviour`) picks the members for the loaded model; each
    runs today's per-probe gates (`check`), and the count and cost gates run once over the whole
    set (`commit`). Members already armed are untouched; only the remainder is armed (FR-39.4). A
    refusal names every probe's gate outcome and `error.next`.
    """
    from millm.services import probe_arm_bridge, probe_behaviours, probe_selection
    from millm.services.probe_limits import effective_for_model, resident_model_id

    try:
        identity, model, tokenizer = await probe_arm_bridge.loaded_identity(session)
        limits = await effective_for_model(session, resident_model_id())
        # ⚠ 12.5a R1: the selection's own refusals (a probe id outside the behaviour, the
        # catalogue limit, no member for the loaded model) are refused arms before `arm_set`
        # too, so they sit inside this `try` as well.
        selection = await probe_selection.select_behaviour(
            repository, key, version=_version_arg(request), probe_ids=request.probe_ids,
            loaded=identity,
        )
        if not selection.probes:
            raise ProbeNotFoundError(
                f"No imported probe of behaviour {key!r} matches the loaded model",
                details={"behaviour": key, "skipped_identity": list(selection.skipped_identity)},
            )
    except Exception as exc:
        # 038 FR-38.20 (12.5a F3): refused before any member is armed (no model, an engine
        # with no hooks, nothing selectable), so ONE record naming the behaviour and no probe. A
        # dry run arms nothing and records nothing, as `arm_set`'s own dry-run refusals do.
        if not request.dry_run:
            from types import SimpleNamespace

            from millm import telemetry

            telemetry.lifecycle_refused(SimpleNamespace(behaviour_key=key), "arm_refused", exc)
        raise
    outcome = await arming.arm_set(
        selection,
        limits=limits,
        model=model,
        dry_run=request.dry_run,
        loaded=identity,
        # Each probe's parity forward runs inside one admission slot, unsteered (FR-27.6f-g).
        executor=inference.run_model_work,
        tokenizer=tokenizer,
        acknowledge_below_rung2=request.acknowledge_below_rung2,
        reason=request.reason,
        windows=request.windows,
        forward_for=lambda probe: probe_arm_bridge.build_parity_forward(model, probe.layer),
        encoder_for=probe_arm_bridge.build_probe_encoder,
    )
    by_id = {p.id: p for p in selection.probes}
    summary = probe_behaviours.summarise(key, list(selection.probes))
    return ApiResponse.ok({
        **outcome,
        "armed": [_probe_summary(by_id[pid]) for pid in outcome.get("armed", [])],
        "set": summary["set"],
        # D83: no behaviour count; `max_behaviours` stays (null) for older readers.
        "limits": {"max_behaviours": None,
                   "source": limits.sources.get("max_behaviours"),
                   "budget": limits.budget(),
                   "budget_sources": {k: limits.sources.get(k) for k in ("decode", "prefill")},
                   "stale": limits.stale, "max_armed": limits.max_armed},
        "skipped_identity": list(selection.skipped_identity),
    })


@router.post("/behaviours/{key}/disarm", response_model=ApiResponse)
async def disarm_behaviour(
    key: Annotated[str, Path(pattern=BEHAVIOUR_KEY)],
    request: BehaviourDisarmRequest,
    repository: ProbeRepo,
    arming: ProbeArmingDep,
) -> ApiResponse:
    """Disarm every armed probe of the behaviour (FR-39.3). A layer hook another behaviour's armed
    probe still uses stays installed."""
    from millm.services import probe_behaviours, probe_selection

    selection = await probe_selection.select_behaviour(
        repository, key, version=_version_arg(request), probe_ids=request.probe_ids,
    )
    disarmed = await arming.disarm_set(selection, reason="operator")
    return ApiResponse.ok({
        "behaviour": {"key": key, "version": selection.version},
        "disarmed": disarmed,
        "summary": probe_behaviours.summarise(key, list(selection.probes)),
    })


@router.get("/status", response_model=ApiResponse)
async def probe_status(service: ProbeEventServiceDep) -> ApiResponse:
    """What every armed probe is doing, and why any of them is not scoring."""
    return ApiResponse.ok(await service.status())


@router.get("/events", response_model=PageResponse)
async def list_events(
    events: ProbeEventRepo,
    probe_id: Annotated[Optional[str], Query()] = None,
    request_id: Annotated[Optional[str], Query()] = None,
    session_id: Annotated[Optional[str], Query(max_length=24)] = None,
    since: Annotated[Optional[datetime], Query(description="ISO 8601")] = None,
    until: Annotated[Optional[datetime], Query(description="ISO 8601")] = None,
    verdict: Annotated[
        Optional[Literal["fired", "not_fired", "not_scored"]], Query()
    ] = None,
    window: Annotated[Optional[str], Query(max_length=16)] = None,
    origin: Annotated[Optional[str], Query(max_length=8)] = None,
    behaviour: Annotated[Optional[str], Query(max_length=64)] = None,
    score_min: Annotated[Optional[float], Query()] = None,
    score_max: Annotated[Optional[float], Query()] = None,
    result_id: Annotated[Optional[str], Query(max_length=36)] = None,
    trace_id: Annotated[Optional[str], Query(pattern=r"^[0-9a-f]{32}$")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, description="DEPRECATED (039): use cursor")] = 0,
    cursor: Annotated[Optional[str], Query(max_length=2048)] = None,
    kind: Annotated[Optional[Literal["probe", "rule"]], Query()] = None,
) -> ApiResponse:
    """Recent verdicts, newest first. **Without context text** — fetch one event for that.

    Feature 36 (FR-36.6): filters by session, time, verdict, window, origin, behaviour and score.
    Feature 38 (FR-38.11): and by `result_id` (exactly one event) or `trace_id` (that request's
    events) — the join back from a header, a body field or an exported record.
    039 (FR-39.15): a keyset `cursor` and `meta.next`; `offset` is kept for one contract version.

    ⚠ 039 (FTID Discrepancy 7): each item's `behaviour` is the STORED result's — what was armed
    (038 FR-38.3). The route used to join today's definition per probe on every page, an N+1 that
    could also rewrite what an old event said; it now reads nothing but the page.
    """
    from millm.api.pagination import decode_cursor, encode_cursor, page_meta

    # 039 S2 (FR-39.23, FTDD §5.2): probe events and rule results in ONE ordered union,
    # `created_at DESC, kind, id DESC`, one keyset `[created_at, kind, id]` over both tables.
    # Each table keeps its own two-value keyset: after a probe row at t, rule rows at t still
    # follow (`kind` sorts probe first); after a rule row at t, no probe row at t does.
    from millm.db.repositories.rule_set_repository import RuleSetRepository
    from millm.services.probe_event_service import rule_summary

    after = decode_cursor("events", cursor, arity=3)
    probe_after = rule_before = None
    if after is not None:
        moment, last_kind, last_id = str(after[0]), str(after[1]), int(after[2])
        probe_after = [moment, last_id] if last_kind == "probe" else [moment, 0]
        rule_before = [moment, last_id] if last_kind == "rule" else [moment, 2**62]
    want_rules = kind != "probe" and probe_id is None and score_min is None and score_max is None
    probe_rows = [] if kind == "rule" else await events.list_events(
        probe_id=probe_id, request_id=request_id, session_id=session_id, since=since,
        until=until, verdict=verdict, window=window, origin=origin, behaviour=behaviour,
        score_min=score_min, score_max=score_max, result_id=result_id, trace_id=trace_id,
        limit=limit + 1, offset=offset if after is None else 0, after=probe_after,
    )
    rule_rows = [] if not want_rules else await RuleSetRepository(events.session).results_page(
        request_id=request_id, session_id=session_id, since=since, until=until,
        verdict=verdict, window=window, origin=origin, behaviour=behaviour,
        result_id=result_id, trace_id=trace_id, before=rule_before, limit=limit + 1,
    )

    def _order(item: tuple[str, Any]) -> tuple[float, str, int]:
        kind_, row = item
        moment = row.created_at.timestamp() if row.created_at is not None else 0.0
        return (-moment, kind_, -int(row.id))

    merged = sorted([("probe", r) for r in probe_rows] + [("rule", r) for r in rule_rows],
                    key=_order)
    more = len(merged) > limit
    merged = merged[:limit]
    next_cursor = (encode_cursor("events", [merged[-1][1].created_at.isoformat(), merged[-1][0],
                                            merged[-1][1].id])
                   if more and merged else None)
    items = [_event_summary(r) if k == "probe" else rule_summary(r) for k, r in merged]
    return PageResponse.page(items, page_meta(next_cursor, limit))


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


class ProbeIdentityDetail(BaseModel):
    """The identity names `GET /api/probes/{id}` reports (038 FR-38.7), documented in the OpenAPI.

    ⚠ A CROSS-REPO CONTRACT: miStudio 039's `test_millm_probe_identity_names.py` reads these names
    from this app's served schema. The suite spells them the same everywhere (cross-document
    review 2026-10-23). Every other summary key travels as before (`extra="allow"`).
    """

    model_config = ConfigDict(extra="allow")

    detector_digest: Optional[str] = Field(None, description="64 hex; null = not digestible")
    detector_digest_source: Literal["producer", "computed", "unknown"] = "unknown"
    identity_algorithm: Optional[str] = Field(None, description="mistudio.probe-identity/1")
    decision_digest: Optional[str] = Field(None, description="64 hex, over the current decision")
    decision_revision: Optional[int] = Field(None, description="miStudio's revision, when stated")
    threshold_revision: int = Field(1, description="miLLM's own counter of the bar")


@router.get(
    "/{probe_id}", response_model=None,
    responses={200: {"model": ApiResponse[ProbeIdentityDetail],
                     "description": "The probe, its definition and its identity digests"}},
)
async def get_probe(probe_id: str, repository: ProbeRepo, events: ProbeEventRepo) -> ApiResponse:
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")
    totals = await events.totals_by_probe([probe.id])
    detail = _probe_summary(probe, totals.get(probe.id, NO_RESULTS))
    detail["definition"] = probe.definition
    # 038 FR-38.7: the identity, under the suite's names. Read from the row (the stored digests);
    # results carry the ARMED probe's copy (`ProbeDescriptor`).
    detail.update({
        "detector_digest": getattr(probe, "detector_digest", None),
        "detector_digest_source": getattr(probe, "detector_digest_source", None) or "unknown",
        "identity_algorithm": getattr(probe, "identity_algorithm", None),
        "decision_digest": getattr(probe, "decision_digest", None),
        "decision_revision": getattr(probe, "decision_revision", None),
        "threshold_revision": int(getattr(probe, "threshold_revision", 1) or 1),
    })
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
    """Run the gates and arm the probe if they all pass — a set of one (039: `check` + `commit`).

    ⚠ **THE ONLY CALLER OF `ProbeArmingService.arm`.** Every gate — identity, the evidence rung,
    scope, parity, cost, and the behaviour and probe limits — is reachable through this route (and,
    for a whole behaviour, `POST /behaviours/{key}/arm`), and
    `tests/unit/api/test_probe_route_surface.py` goes red if its registration is removed.

    A refusal is an error in the envelope naming which gate refused and why; the gates run in the
    order they are cheapest to fail, so a wrong-model probe never costs a forward pass.
    """
    probe = await repository.get(probe_id)
    if probe is None:
        raise ProbeNotFoundError(f"No probe {probe_id}")

    from millm import telemetry
    # Per-model limits (operator 2026-10-21): the RESIDENT model's count backstop and budgets.
    from millm.services.probe_limits import effective_for_model, resident_model_id

    try:
        identity, model, tokenizer = await loaded_identity(session)
        limits = await effective_for_model(session, resident_model_id())
        forward = build_parity_forward(model, probe.layer)
        encoder = await build_probe_encoder(probe)
    except Exception as exc:
        # ⚠ 038 FR-38.20 (12.5a F3): a refusal BEFORE `arming.arm` (no model, an engine with no
        # hooks — `PROBE_HOOK_UNSUPPORTED`) is a refused arm too; `arm` emits only its own gates'.
        telemetry.lifecycle_refused(probe, "arm_refused", exc)
        raise
    await arming.arm(  # the ArmedProbe is not used: the row carries what the response reads
        probe,
        model=model,
        loaded=identity,
        forward=forward,
        # The parity forward runs inside one admission slot, unsteered (FR-27.6f-g, T-73).
        executor=inference.run_model_work,
        tokenizer=tokenizer,
        acknowledge_below_rung2=request.acknowledge_below_rung2,
        reason=request.reason,
        encoder=encoder,
        windows=request.windows,
        limits=limits,
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

    from millm import telemetry
    from millm.services.probe_parity import model_summary

    try:
        # The identity is not CHECKED here (that is arming's gate), but its precision is reported
        # with the result, so a failed parity names its likely cause.
        identity, model, tokenizer = await loaded_identity(session)
        # `windows=[]` — the probe's own scope alone. Parity compares against miStudio's recorded
        # scores, which were computed under that scope; other windows would be work with no
        # reader.
        armed = armed_probe_from_row(probe, encoder=await build_probe_encoder(probe), windows=[])
        tolerance = max(
            float((probe.definition.get("test_vectors") or {}).get("tolerance", 0.0) or 0.0),
            settings.PROBE_PARITY_TOLERANCE,
        )
        forward = build_parity_forward(model, probe.layer)
        # ⚠ INSIDE ONE ADMISSION SLOT, UNSTEERED (FR-27.6e, T-73). This ran the engine directly:
        # no slot, so a parity forward could run beside a generation on the same model, and no
        # suppression, so a profile steering an earlier layer moved the residual being compared
        # against miStudio's recorded scores.
        report = await inference.run_model_work(lambda: ProbeParityEngine(forward).run(
            armed, probe.definition, tolerance=tolerance, tokenizer=tokenizer,
            loaded_dtype=identity.dtype,
            loaded_quantization=identity.quantization,
            model=model_summary(identity),
        ))
    except Exception as exc:
        # ⚠ 038 FR-38.20 (12.5a F3): a parity check refused before it ran (no model, an engine
        # with no hooks — `PROBE_HOOK_UNSUPPORTED`) or failed while running is a record too.
        telemetry.lifecycle_refused(probe, "parity", exc)
        raise
    await repository.update(probe, parity=report.as_details())
    # 038 FR-38.20: the stand-alone parity check is a lifecycle step too (arm-time parity emits
    # from the arming service).
    telemetry.emit_lifecycle(probe, "parity", outcome="ok" if report.passed else "refused",
                             code=None if report.passed else "PROBE_PARITY_FAILED")
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
    per-window or per-length entry the runtime's own parser would discard · 409
    `PROBE_IDENTITY_MISMATCH` a sent `decision_digest` that is not the stored decision's own.

    An ARMED probe's live runtime shape is refreshed in the same call, because a database write
    alone would change nothing that is served — `ArmedProbe` is resolved once at arm time and held
    in process memory. `registry_updated` says whether that happened, and `stale_armed` reports a
    row claiming armed with no live entry rather than quietly reconciling it.
    """
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
            decision_digest=request.decision_digest,
            decision_revision=request.decision_revision,
        )
    except ProbeRecalibrationRefused as refusal:
        # Mapped to the envelope's own error classes, so a refusal is distinguishable by CODE
        # rather than by reading prose — by `as_error`, the one mapping the lifecycle record
        # uses too (12.5a F5).
        raise refusal.as_error() from refusal
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
