# Technical Design Document: Circuit Edge Sensing

## miLLM Feature 15

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `015_FPRD|Circuit_Edge_Sensing.md` · `011_FTDD|Coactivation_Sensing.md` · `docs/mcp-contract.md` (v1.1)

---

## 1. Executive Summary

Edge sensing is a directional REFINEMENT of the shipped cluster-sensing path, not a second inference pass.
Feature 11's `_sense` already runs a member-only encode (≤20 cached encoder columns) at every position of
every pass and records per-position firings; edge sensing keeps that fire-detection substrate and adds an
ordered up→down matcher: an upstream member firing at position p that is FOLLOWED by its declared downstream
partner firing at a position in (p, p+L]. Because a circuit spans layers, the encode runs per referenced SAE
(member columns of that layer only); a small per-SAE ring of recent upstream firings lets the downstream
detection close edges across the lag window. Hits buffer on the `LoadedSAE`s, the serial request queue
provides request boundaries, and a post-generation flush decodes ±K `context_parts`, persists bounded edge
events (carrying the edge rung verbatim), and emits `circuit:sensing:event`.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Detection point | EXTEND `_sense` — reuse fire detection, add an up→down matcher | Fire predicate + member encode already shipped and tuned; only the pairing is new |
| Edge model | Direct declared edges (up feat/layer → down feat/layer), matched within lag L | Circuit meaning IS the edge; transitive chains deferred (FPRD §12) |
| Lag window | `down_pos ∈ (up_pos, up_pos + L]`, L=CIRCUIT_SENSING_LAG_TOKENS (8, max 64) | Directional + bounded; a ring of recent up-firings, not full history |
| Cross-layer | Per-SAE member encode; edge closes when both endpoint layers are attached | Members live on different layers/SAEs (Feature 12 multi-SAE) |
| Unsensable edges | Edge whose endpoint layer's SAE is not attached ⇒ skipped, reported in status | Never sense through the wrong decoder (EC-15.4/15.6) |
| Attribution | Endpoint event = token being READ at that position | Sampled token unknowable in-pass; inherited from Feature 11 |
| Alone/within | `ambient_fired_count` best-effort (full-width monitoring), else NULL | Honest v1; same rule as Feature 11 |
| Rung | Carried verbatim per edge; "causal" forbidden below rung 2 in every string | BR-005 evidence-integrity policy |
| Concurrency | Armed ⇒ forced serial; CBM never sensed | Batch rows can't be attributed to requests (Feature 11 precedent) |
| Persistence | `circuit_edge_sensing_events` + per-circuit cap + age pruning | Bounded by construction |

## 2. System Architecture

```
                  per forward pass (per referenced SAE)         per request (serial queue)
 ┌──────────────┐   hidden_states   ┌────────────────────────┐   begin ─────────────┐
 │ SAEHooker    │ ────────────────► │ LoadedSAE (layer Lk)   │                      │
 │ hook_fn      │  if edge-armed:   │  W_enc_m cache (≤20)   │   … N passes,        │
 └──────────────┘   _sense_edges    │  fire detect (F11)     │   up-firing ring +   │
                                    │  up→down matcher (F15) │   closed edges buffer │
                                    └────────────────────────┘   collect ◄──────────┘
                                                                     │ edge hits + full ids
                                                         ┌───────────▼────────────┐
                                                         │ CircuitSensingService   │
                                                         │  decode ±K context_parts │
                                                         │  persist (cap+prune)     │
                                                         │  WS 'circuit:sensing:event'│
                                                         └──────────────────────────┘
```

## 3. Detection Design (LoadedSAE additions, `millm/ml/sae_wrapper.py`)

Reuses `SensingConfig`/`SensedHit` fire detection; adds an edge config + an ordered matcher. The armed
state is a `CircuitSensingConfig` keyed per referenced SAE (each SAE senses its own member columns; the
matcher closes edges whose endpoints land on the two SAEs).

```python
# millm/ml/sae_wrapper.py — edge-sensing additions
@dataclass
class EdgeSpec:
    edge_key: str                          # '{up_idx}@{up_layer}->{down_idx}@{down_layer}'
    up_col: int                            # column into this SAE's member matrix (if up on this layer)
    down_col: int                          # column into partner SAE's member matrix (if down on this layer)
    up_layer: int; down_layer: int
    rung: int; rung_language: str

@dataclass
class CircuitSensingConfig:
    circuit_id: str
    member_indices: list[int]              # ≤ 20 per SAE (real feature idxs on THIS layer)
    thresholds: torch.Tensor               # (m,) — θ_i = max(θ_floor, ε·max_activation_i) (F11 reuse)
    threshold_mode: str                    # 'epsilon_max' | 'floor_only'
    edges: list[EdgeSpec]                  # edges with an endpoint on this SAE's layer
    lag_tokens: int                        # L
    max_events_per_request: int = 20

@dataclass
class SensedEdge:
    edge_key: str; phase: str
    up_feature_idx: int; up_layer: int; up_pos: int; up_peak_act: float
    down_feature_idx: int; down_layer: int; down_pos: int; down_peak_act: float
    token_lag: int
    rung: int; rung_language: str
```

`_sense_edges(hidden_states)` (armed-only, `torch.no_grad`, respects `_suppressed`): computes the member
`acts` and `fired` mask exactly as Feature 11's `_sense`, then for each fired member that is an edge's
UPSTREAM endpoint pushes `(pos, peak_act)` into a bounded per-edge up-firing ring; for each fired member
that is an edge's DOWNSTREAM endpoint, pops any ring entry with `0 < down_pos - up_pos ≤ L` and emits a
`SensedEdge` (peak activations taken over the fired span). Positions are absolute
(`_sensing_token_offset += seq`; prefill→decode after the first pass). Cross-layer edges close when the
downstream SAE's pass observes a down-firing whose upstream partner ring (on the upstream SAE) has an entry
in-window — the rings live on a shared per-request buffer the service owns, so both SAEs write the same
structure. Per-request cap → `truncated`.

> **AS-BUILT AMENDMENT (2026-07-20, R3).** The implementation reads the ring
> **non-destructively** — `match_down` returns the nearest antecedent without
> removing it — so one upstream fire can be the antecedent of several
> downstream events. That is deliberate and is the better evidence model: an
> upstream feature firing once and two downstream partners responding is two
> real observations, and popping would silently report only the first.
>
> The consequence the design anticipated (unbounded ring growth) is handled
> differently: the ring bounds per-edge fire retention by count, and prunes to
> the SLOWEST layer's position via `note_layer_progress`. Getting that wiring
> right took three attempts — R1 declared pruning "request-level" without a
> caller, R2 added service methods without a caller, and only R3's design
> (the ring tracking layer progress itself) was wireable, because it does not
> require any hook to know about its siblings.

Hook change (`millm/ml/sae_hooker.py`, beside the Feature 11 sensing branch): one sibling guard —
`if sae.is_edge_sensing_armed: with torch.no_grad(): sae._sense_edges(hidden_states)` — before
`apply_steering` so positions reflect the pre-steer read. `suppressed()` already covers embeddings passes.

Threshold source: circuit members' `max_activation` (per the definition, preserved verbatim);
missing values ⇒ `floor_only` per member exactly as Feature 11 (EC-15.6). An edge whose endpoint layer's
SAE is not attached is dropped from `edges` at arm time and reported as UNSENSABLE in status (EC-15.4).

## 4. Request Lifecycle (InferenceService)

```python
# inside the queue semaphore, both generation paths (extends the F11 wiring):
if circuit_sensing_service.should_sense(active_circuit):
    for sae in attached_saes_for(active_circuit):
        sae.begin_edge_sensing_request(request_id, shared_edge_buffer)
try:
    ... generate ...
finally:
    if circuit_sensing_service.is_armed:
        rid, edges, truncated = circuit_sensing_service.collect_edges()
        await self._notify_circuit_sensing(rid, edges, truncated, full_ids)  # async; beside _notify_sensing
```
Token ids for context reuse Feature 11 exactly: non-streaming `outputs[0]`; streaming the shipped
`IdCaptureStoppingCriteria` (zero-copy, survives early stop); prefill `inputs.input_ids`. Routing:
`_use_cbm_for_request` gains `or (settings.CIRCUIT_SENSING_FORCE_SERIAL and circuit_sensing_service.is_armed)`;
non-forced CBM requests skip `begin_edge_sensing_request` entirely (unsensed, EC-15.5).

## 5. Service, Persistence, API

```python
# millm/services/circuit_sensing_service.py
class CircuitSensingService:
    def arm_for_circuit(self, circuit, attached_saes: dict[int, LoadedSAE]) -> None
        # build per-SAE CircuitSensingConfig; drop edges whose endpoint SAE is unattached (record unsensable)
    def disarm(self, saes) -> None
    def should_sense(self, active_circuit) -> bool
    def collect_edges(self) -> tuple[str, list[SensedEdge], bool]
    async def record(self, request_id, edges, truncated, full_ids, tokenizer) -> list[CircuitEdgeSensingEvent]
        # decode ±K context_parts, build summaries (NO 'causal' < rung 2), persist, prune, emit WS
    async def get_events(self, circuit_id=None, limit=50, since=None) -> list[...]
    def status(self) -> dict   # armed circuit, per-edge thresholds+mode, lag, sensable/unsensable edges, overhead
```
Arm/disarm lifecycle: circuit activate (if the circuit's edge-sensing intent is set) ⇒ arm; deactivate /
SAE-set detach ⇒ disarm; enable/disable endpoints toggle the intent and live-arm/disarm when that circuit is
active — the exact shape of `management/sensing.py::_toggle`, over circuits.

Repository (`db/repositories/circuit_edge_sensing_repository.py`): `create_many`, `list_events`, `count`,
`clear`, `prune_aged` / cap-prune — pruned on flush and on read (throttled, same 600 s interval as Feature 11).

Summary format (EDGE-R4): `"{up_label} → {down_label}: L{up_layer}→L{down_layer}, lag {lag}, {rung_language},
peak {up_act:.1f}/{down_act:.1f} during {phase} @ {up_pos}→{down_pos}"` — `rung_language` verbatim; a guard
test asserts the substring "causal" never appears when `rung < 2`.

WS: `ProgressEmitter.emit_circuit_sensing_event` (sockets/progress.py, mirror of `emit_sensing_event`:551,
same thread-safe `run_coroutine_threadsafe`; event name `circuit:sensing:event`; payload excludes context text).

Routes per FPRD §5, matching `management/sensing.py` (ProfileId→CircuitId path param, `_PRUNE_ON_READ_INTERVAL_S`
throttle, envelope). DI + router registration follow the clusters/sensing pattern. `NO_ACTIVE_CIRCUIT`
(200+envelope) on sensing calls with no active circuit; `CIRCUIT_SENSING_EVENT_NOT_FOUND` (404) on detail.

## 6. Admin UI Design
`components/circuits/sensing/`: `EdgeSensingPanel` (status strip: armed circuit, overhead, threshold mode,
unsensable-edge notes; event list newest-first with WS live-prepend), `EdgeSensingEventDetail` (up→down member
table, lag, rung badge rendering `rung_language` verbatim, `context_parts` with the span highlighted).
EdgeSensingToggle wires to enable/disable. Socket subscription follows the shipped cluster-sensing socket
client pattern (`services/socket.ts`); a `circuit:sensing:event` handler prepends into the list.

## 7. Testing Strategy

### Unit
- `tests/unit/ml/test_circuit_edge_sensing.py`: fire predicate reuse, up→down matcher within lag L,
  EC-15.1 (lone upstream → no event), EC-15.2 (reversed order → no event), cross-layer edge closing,
  per-request cap + truncated, offset/phase accounting, unsensable-edge exclusion (EC-15.4/15.6),
  suppressed() no-op, arm/disarm idempotence, buffer hygiene (begin resets; no-begin ⇒ empty collect).
- `tests/unit/services/test_circuit_sensing_service.py`: config build from circuit edges + attached-SAE set,
  `context_parts` slicing (edges: event at pos 0, end of stream), summary builder, **no-"causal"-below-rung-2
  assertion**, ambient rules, retention prune math, lifecycle.
- `tests/unit/db/test_circuit_edge_sensing_repository.py`: create_many, prune cap + age, CASCADE on circuit delete.

### Integration
- `tests/integration/test_circuit_edge_sensing_workflow.py`: arm→generate (known up→down fixture)→events with
  correct lag/context/rung; reversed + lone fixtures produce no events; streaming early-stop context; enable/disable
  lifecycle incl. SAE-set detach; serial forcing assertion; CBM-unsensed path; overhead accumulator; WS emission;
  **latency-budget assertion** (armed overhead ≤ CIRCUIT_SENSING_MAX_OVERHEAD_MS; un-armed zero-delta smoke).

### E2E (post-deploy)
Circuits-page edge-sensing flow: toggle→chat traffic→live edge events→detail (rung badge verbatim).

## 8. Risks
- Hot-path regression: armed-only branch + µs member matmul + bounded up-firing ring + per-request cap;
  `sensing_overhead_ms` observable with warn threshold (EDGE-S2). Un-armed cost = one boolean. Within CBM
  budget (NFR-1.5).
- Cross-layer ring coordination: the up-firing ring is per-request and shared across the circuit's SAEs; a
  missed `begin` on an unsensed path must yield an empty collect, never stale cross-request edges — pinned by test.
- Rejected speculative drafts may sense positions whose tokens are discarded — accepted v1 inaccuracy,
  documented (attribution convention section), inherited from Feature 11.
- Context text is user content in the DB — retention caps + K=0 option are the controls; privacy note in docs.
- Rung integrity: the summary/UI must render `rung_language` verbatim; a guard test forbids "causal" below rung 2.
