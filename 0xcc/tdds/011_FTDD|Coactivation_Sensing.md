# Technical Design Document: Co-Activation Sensing

## miLLM Feature 11

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `011_FPRD|Coactivation_Sensing.md` · `008_FTDD|Cluster_Import.md`

---

## 1. Executive Summary

Sensing adds a second, tiny, armed-only observation path to the SAE hook — deliberately independent of
Feature Monitoring, whose capture path (a) positionally compacts feature columns and (b) only surfaces
the LAST forward pass to `on_activation`. A member-only encode (≤20 cached encoder columns) evaluates
the co-activation predicate at every position of every pass; hits buffer on the `LoadedSAE`, the serial
request queue provides request boundaries, and a post-generation flush decodes context windows,
persists bounded events, and emits WebSocket updates.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Detection point | New `_sense()` branch in the hook, member-only encode | Monitoring compacts + keeps last pass only; a ≤20-column matmul is µs |
| Request boundaries | `begin_sensing_request`/`collect_sensing_hits` inside the queue semaphore | The hook has no request concept; the serial queue does |
| Position tracking | `_sensing_token_offset += seq_len` per pass | Correct for prefill (N), decode (1), speculative verify (k+1) |
| Attribution | Event = token being READ at position p | Sampled token is unknowable in-pass; never over-claim |
| Alone/within | `ambient_fired_count` best-effort (full-width monitoring co-running), else NULL | Honest v1; exclusivity tier deferred |
| Context ids | outputs[0] (non-stream) / IdCaptureStoppingCriteria (stream) / prompt ids (prefill) | Zero-copy; stopping criteria run every step |
| Concurrency | Armed ⇒ forced serial; CBM never sensed | Batch rows can't be attributed to requests |
| Persistence | `sensing_events` + per-cluster cap + age pruning | Bounded by construction |

## 2. System Architecture

```
                       per forward pass                      per request (serial queue)
 ┌──────────────┐   hidden_states    ┌──────────────────┐   begin ─────────────┐
 │ SAEHooker    │ ─────────────────► │ LoadedSAE        │                      │
 │ hook_fn      │  if armed: _sense  │  W_enc_m cache   │   … N passes,        │
 └──────────────┘                    │  θ,min_k predicate│     hits buffer …    │
                                     │  debounce→SensedHit│                     │
                                     └──────────────────┘   collect ◄──────────┘
                                                                │ hits + full token ids
                                                     ┌──────────▼───────────┐
                                                     │ SensingService        │
                                                     │  decode ±K context    │
                                                     │  persist (cap+prune)  │
                                                     │  WS 'sensing:event'   │
                                                     └──────────────────────┘
```

## 3. Detection Design (LoadedSAE additions, `millm/ml/sae_wrapper.py`)

```python
@dataclass
class SensingConfig:
    profile_id: str
    member_indices: list[int]              # ≤ 20
    thresholds: torch.Tensor               # (m,) — θ_i = max(θ_floor, ε·max_activation_i)
    threshold_mode: str                    # 'epsilon_max' | 'floor_only' (EC-11.4 observability)
    min_k: int
    context_tokens: int
    max_events_per_request: int = 20

@dataclass
class SensedHit:
    pos_start: int; pos_end: int           # absolute positions (debounced span)
    phase: str                             # 'prefill' | 'decode'
    fired: list[tuple[int, float]]         # (REAL feature_idx, peak act)
    fired_count: int
    score: float                           # max(act_i / θ_i) over fired members

class LoadedSAE:
    def arm_sensing(self, config: SensingConfig) -> None:
        # cache W_enc_m = self.W_enc[:, config.member_indices].contiguous(), b_enc_m
    def disarm_sensing(self) -> None: ...
    @property
    def is_sensing_armed(self) -> bool: ...
    def begin_sensing_request(self, request_id: str) -> None:
        # reset buffer, _sensing_token_offset = 0, _sensing_phase = 'prefill', truncated = False
    def _sense(self, hidden_states: torch.Tensor) -> None:
        # x: (batch=1, seq, d_in) → acts = relu(x @ W_enc_m + b_enc_m)  (no_grad)
        # fired = acts > thresholds; counts = fired.sum(-1)
        # event_positions = counts >= min_k  → merge consecutive → SensedHits
        # offset += seq; after first pass phase = 'decode'; stop at cap → truncated
    def collect_sensing_hits(self) -> tuple[str, list[SensedHit], bool]:  # (request_id, hits, truncated)
```
Hook change (`millm/ml/sae_hooker.py:159-165` area): one branch beside monitoring —
`if sae.is_sensing_armed: with torch.no_grad(): sae._sense(hidden_states)`. The `suppressed()`
contextmanager (sae_wrapper.py:449) already covers embeddings passes — `_sense` respects `_suppressed`.

Threshold source: cluster_meta members' `max_activation` (Feature 8 preserved them verbatim);
missing values ⇒ `threshold_mode='floor_only'` recorded in status (EC-11.4).

## 4. Request Lifecycle (InferenceService)

```python
# inside the queue semaphore, both generation paths:
if sensing_service.should_sense(active_profile):        # armed cluster == active profile
    sae.begin_sensing_request(request_id)
try:
    ... generate ...
finally:
    if sae.is_sensing_armed:
        rid, hits, truncated = sae.collect_sensing_hits()
        await self._notify_sensing(rid, hits, truncated, full_ids)   # async; beside _notify_monitoring (:857/:1096)
```
Token ids for context: non-streaming — `outputs[0]` (:860 area) holds prompt+generated ids; streaming —
`IdCaptureStoppingCriteria` piggybacks the existing stopping-criteria pattern
(`_make_event_stopping_criteria`, inference_service.py:59): its `__call__(input_ids, ...)` stores a
reference each step (zero-copy), surviving early stops; prefill events slice `inputs.input_ids`.
Routing: `_use_cbm_for_request` gains `or (settings.SENSING_FORCE_SERIAL and
sensing_service.is_armed)`; non-forced CBM requests skip `begin_sensing_request` entirely (unsensed).

## 5. Service, Persistence, API

```python
# millm/services/sensing_service.py
class SensingService:
    def arm_for_profile(self, profile: Profile, sae: LoadedSAE) -> None   # builds SensingConfig
    def disarm(self, sae: LoadedSAE | None) -> None
    def should_sense(self, active_profile) -> bool
    async def record(self, request_id, hits, truncated, full_ids, tokenizer) -> list[SensingEvent]
        # decode ±K windows, build summaries, persist, prune, emit WS
    async def get_events(self, profile_id=None, limit=50, since=None) -> list[SensingEvent]
    def status(self) -> dict   # armed profile, thresholds+mode, overhead stats
```
Arm/disarm lifecycle: cluster activate (if `profiles.sensing_enabled`) ⇒ arm; deactivate / SAE detach ⇒
disarm; enable/disable endpoints toggle the column and live-arm/disarm when that cluster is active.

Repository (`db/repositories/sensing_repository.py`): `create_many`, `list`, `clear`,
`prune(profile_id, cap, max_age_days)` — prune on flush and on read.

Summary format (SEN-R4): `"<display_token>: {fired}/{m} members fired (peak F{idx} '{label}'
{score:.1f}×θ) during {phase} @ {start}–{end}"`.

WS: `ProgressEmitter.emit_sensing_event` (sockets/progress.py, mirror of `emit_activation_update` :493,
same thread-safe fire-and-forget; event name `sensing:event`; payload excludes context_text — UI fetches
detail via REST).

Routes per FPRD §5; DI + router registration follow the clusters.py pattern (Feature 8).

## 6. Admin UI Design
`components/clusters/sensing/`: `SensingPanel` (status strip: armed cluster, overhead, threshold mode;
event list newest-first with WS live-prepend), `SensingEventDetail` (member table, context text with the
event span highlighted). SensingToggle (stub from Feature 8) wires to enable/disable. Socket
subscription follows the existing monitoring socket client pattern (`services/socket.ts`).

## 7. Testing Strategy

### Unit
- `tests/unit/ml/test_sensing.py`: predicate math (thresholds, ε fallback→floor_only, min_k),
  debounce spans, cap+truncated, offset accounting for shapes (seq=N prefill, seq=1 decode, seq=k+1
  speculative), suppressed() no-op, arm/disarm idempotence.
- `tests/unit/services/test_sensing_service.py`: config build from cluster_meta, context-window slicing
  (edges: event at position 0, end of stream), summary builder, retention prune math.
- `tests/unit/db/test_sensing_repository.py`: create_many, prune cap + age, CASCADE delete with profile.

### Integration
- `tests/integration/test_sensing_workflow.py`: arm→generate (known co-firing fixture)→events with
  correct spans+context; streaming early-stop context correctness; enable/disable lifecycle incl. SAE
  detach; serial forcing assertion; CBM-unsensed path; overhead accumulator populated; WS emission.

### E2E (post-deploy)
Clusters-page sensing flow: toggle→chat traffic→live events→detail.

## 8. Risks
- Hot-path regression: armed-only branch + µs matmul + per-request cap; `sensing_overhead_ms` observable
  with warn threshold (SEN-S2). Un-armed cost = one boolean.
- Rejected speculative drafts may sense positions whose tokens are discarded — accepted v1 inaccuracy,
  documented in the manual (attribution convention section).
- Context text is user content in the DB — retention caps + K=0 option are the controls; privacy note
  in docs.
