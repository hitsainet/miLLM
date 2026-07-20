# Technical Design Document: Request-Scoped Sensing Context

## miLLM Feature 17

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `017_FPRD|Request_Scoped_Context.md` · `015_FTDD|Circuit_Edge_Sensing.md` · `BRD-MILLM-CIRCUITS-002.md` (BR-001) · `000_PADR|miLLM.md` (v1.3) · `docs/circuit-contention-model.md` (§4)

---

## 1. Executive Summary

This is a **structural refactor with no new capability**. Today, three pieces of per-request state are
distributed across N `LoadedSAE` objects: each advances its own `_edge_token_offset`, all write into one
`EdgeFireRing` handed to them at arm time, and each enforces its own copy of `max_events_per_request`.
The correctness of that arrangement rests on three rules that live in comments — advance the offset on
every return path, never prune from a hook, keep recording upstream fires when you stop appending
events — and each of Feature 15's three review rounds found a new way to break one of them.

The design replaces the arrangement with one `SensingRequestContext` per request that owns position,
rings and budget, and moves the machinery into `millm/ml/edge_sensing.py`. The mechanism that makes this
worth doing is not encapsulation for its own sake: it is that **the context advances position at the
pass boundary, above the sensing body**, so no early return inside that body can skip it. R1-03's
critical becomes unrepresentable rather than test-guarded. The same move gives ring pruning an owner
that legitimately knows about all layers, and gives the budget an owner that legitimately knows about
all circuits.

The one genuinely new design content is the **N-circuit shape**, mandated by
`docs/circuit-contention-model.md` §4 before Feature 19 lands: rings become one per
`(request, circuit)`, and the budget is attributed per circuit.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Position ownership | ONE counter on the context, advanced at the pass boundary | N per-SAE counters must agree on a coordinate no component owns — the root cause of 3 of 8 F15 criticals |
| Where position advances | In the hook-facing wrapper, ABOVE the sensing body's early returns | R1-03 and the live §15.6 defect are both "a return path that skipped a step"; put the step where no return can reach past it |
| Ring cardinality | **One per `(request, circuit)`** | `edge_key` is not unique across circuits; a shared ring would fabricate an observation of an edge that fired in neither circuit (PADR v1.3) |
| Ring pruning | Retained: the ring tracks layer progress itself and prunes to the slowest | The only one of three designs that proved wireable; R1 and R2 both declared a prune with no caller |
| Budget ownership | Per request on the context, attributed per circuit | R1 deferred-A / R2 deferred-A / R3-02: per-SAE caps give the cap cross-layer blast radius |
| Budget exhaustion | Suppresses only the downstream append; upstream recording continues | R2-03 and R3-02 are the same starvation bug reached through shed and cap respectively |
| Truncation | Per-layer (and per-circuit) attribution, `truncated_layers` in status | BR-006; today one saturated layer marks every row of the request |
| Module boundary | `millm/ml/edge_sensing.py`, no `LoadedSAE` dependency | R3's harness blind spot: a hand-written stub made both R1's and R2's criticals unrepresentable in 37 tests |
| Verification | Characterization-first, mutation-after | FR-17.6; this is the most defect-dense code in the arc |
| Dead code | R2's three prune methods deleted, not moved | Zero production callers, verified live; carrying them preserves the ambiguity |

## 2. System Architecture

```
                  per request (serial queue)                     per forward pass
 ┌────────────────────────────────────────────┐        ┌──────────────────────────────┐
 │ InferenceService                            │        │ SAEHooker.hook_fn            │
 │   ctx = SensingRequestContext(request_id,    │        │   if sae.is_edge_sensing_armed│
 │        circuits=[...], budget=N)             │        │     sae.sense_edges(hidden)  │
 │   for sae in attached: sae.bind_context(ctx) │        └──────────────┬───────────────┘
 └───────────────┬────────────────────────────┘                        │
                 │                                      ┌──────────────▼───────────────┐
                 │                                      │ LoadedSAE.sense_edges (THIN) │
                 │                                      │  pos = ctx.advance(layer,seq)│ ◄── ALWAYS
                 │                                      │  ... encode / fired mask ... │
                 │                                      │  matcher.match(ctx, pos, …)  │ ◄── may return early
                 │                                      └──────────────┬───────────────┘
                 │                                                     │
   ┌─────────────▼─────────────────────────────────────────────────────▼──────────────┐
   │ SensingRequestContext          (millm/ml/edge_sensing.py)                         │
   │   position: int            ← the ONE absolute coordinate                          │
   │   phase: 'prefill'|'decode'                                                       │
   │   rings: dict[circuit_id -> EdgeFireRing]   ← ONE PER (request, circuit)           │
   │   budget: EventBudget       ← per request, attributed per circuit                 │
   │   circuit_ids: frozenset    ← snapshotted at open; a re-arm cannot rewrite it      │
   │   closed: bool              ← a late write is dropped + logged, never leaks        │
   └───────────────────────────────────┬──────────────────────────────────────────────┘
                                       │ collect() once, at request end
                          ┌────────────▼─────────────┐
                          │ CircuitSensingService     │  (unchanged responsibilities:
                          │  decode context, persist, │   decode/persist/WS/prune-on-read)
                          │  WS emit                  │
                          └───────────────────────────┘
```

The critical structural property is the split in `LoadedSAE.sense_edges`: **the position advance is in
the thin wrapper, the fallible work is below it.** Today both live in the same function, with the
advance duplicated across three exits (`sae_wrapper.py:1072`, `:1093`, `:1123`) — which is precisely how
R1-03 happened and how the live `note_layer_progress` gap in FPRD §15.6 persists.

## 3. The Context (`millm/ml/edge_sensing.py`)

```python
# millm/ml/edge_sensing.py — the new module. No import of sae_wrapper.

@dataclass
class EventBudget:
    """Per-request event budget, attributed per circuit.

    R1 deferred-A: the cap was per-SAE but conceptually per-request, so an
    N-layer circuit could emit N x cap. R3-02: a layer that hit its cap
    returned from the whole pass, silently blinding uncapped siblings.
    Both are fixed by one owner that knows every circuit and every layer.
    """
    limit_per_circuit: int
    _spent: dict[str, int] = field(default_factory=dict)
    _shed_layers: dict[str, set[int]] = field(default_factory=dict)

    def try_spend(self, circuit_id: str, layer: int) -> bool:
        """True if an event may be appended. False NEVER stops upstream recording."""
        spent = self._spent.get(circuit_id, 0)
        if spent >= self.limit_per_circuit:
            self._shed_layers.setdefault(circuit_id, set()).add(layer)
            return False
        self._spent[circuit_id] = spent + 1
        return True

    def truncated_layers(self, circuit_id: str) -> list[int]: ...


class SensingRequestContext:
    """Owns absolute position, the per-circuit rings, and the event budget
    for exactly one request.

    Feature 15 distributed these across N LoadedSAEs. Three of that feature's
    eight criticals share one root cause: N per-SAE counters must agree on an
    absolute coordinate that no component owns, and the ring's lifetime was
    managed by whoever remembered to call it. R1 put pruning in the hook
    (which cannot know sibling state), R2 put it on the service (which is
    never on the per-pass path), and only R3's self-pruning ring worked --
    because it removed the requirement that any caller know global state.
    This class removes the requirement that any caller know it either.
    """

    def __init__(self, request_id: str, circuit_ids: Iterable[str],
                 max_lag: int, budget: EventBudget) -> None:
        self.request_id = request_id
        self.circuit_ids = frozenset(circuit_ids)      # snapshot: R2-04 / R3-04
        self.position = 0
        self.phase = "prefill"
        self.budget = budget
        # ONE RING PER (request, circuit). See PADR v1.3 and the contention
        # model section 4. edge_key is '{up}@{L}->{down}@{M}' and is NOT unique
        # across circuits, so a shared ring would let circuit A's upstream fire
        # match circuit B's downstream fire and record an observation of an
        # edge that fired in NEITHER -- a fabricated observation on an evidence
        # surface, categorically worse than a missed one.
        self._rings = {cid: EdgeFireRing(max_lag) for cid in self.circuit_ids}
        self._closed = False

    def advance(self, layer: int, seq: int) -> int:
        """Return the base position for this pass and advance the counter.

        Called ONCE per pass per layer from the thin wrapper, ABOVE any
        early return in the sensing body. This is the whole point: R1-03 was
        an early return that skipped the advance, and the live code still has
        two return paths that skip note_layer_progress for the same reason.
        """
        if self._closed:
            logger.warning("sensing_write_after_close: request=%s", self.request_id)
            return -1
        base = self.position
        self.position += seq
        if self.phase == "prefill":
            self.phase = "decode"
        for ring in self._rings.values():
            ring.note_layer_progress(layer, self.position)
        return base

    def ring(self, circuit_id: str) -> Optional[EdgeFireRing]: ...
    def close(self) -> None: ...
```

Note `advance()` calls `note_layer_progress` for every ring **unconditionally**, before any decision
about whether this pass will be sensed. That closes FPRD §15.6: a layer suppressed for an entire
request still reports progress, so `_progress` reaches the two entries the prune requires and the ring
prunes as designed.

`EdgeFireRing` moves across essentially unchanged — `record_up`, `match_down` (the R3 `bisect` path),
`note_layer_progress`, `prune_before`, `clear`, `_MAX_FIRES_PER_EDGE = 512`. Its docstrings must move
with it: they carry the reasons R1's and R2's designs failed, and a reader who loses them will
reintroduce one.

## 4. What Changes in `LoadedSAE` (`millm/ml/sae_wrapper.py`)

The 13 `_edge_*` instance fields (`_edge_batch_warned`, `_edge_began`, `_edge_done`,
`_edge_member_fires`, `_edge_overhead_ms`, `_edge_phase`, `_edge_request_id`, `_edge_ring`,
`_edge_saturation_warned`, `_edge_sensing`, `_edge_thresholds_cpu`, `_edge_token_offset`,
`_edge_truncated`) reduce to:

| Retained on the SAE | Why |
|---|---|
| `_edge_sensing` (the `CircuitSensingConfig`) | Genuinely per-SAE: this layer's member columns and thresholds |
| `_edge_ctx` (context reference) | The binding |
| `_W_enc_e` / `_b_enc_e` | Per-SAE weight cache, already SAE-owned |
| `_edge_batch_warned`, `_edge_saturation_warned` | Per-SAE log de-duplication |

Moved to the context: `_edge_token_offset` → `ctx.position`; `_edge_phase` → `ctx.phase`;
`_edge_request_id` → `ctx.request_id`; `_edge_ring` → `ctx.rings[circuit_id]`; `_edge_done` /
`_edge_truncated` → `ctx.budget`. Deleted outright: `_edge_thresholds_cpu` (dead since R1-14, recorded
again in R2-E and R3-G), and `_edge_member_fires` moves to the context if BR-007 consumes it, otherwise
stays.

`sense_edges` becomes the thin wrapper:

```python
def sense_edges(self, hidden_states: Tensor) -> None:
    """Hook entry point. Advance FIRST, then decide whether to sense."""
    ctx = self._edge_ctx
    if ctx is None:
        return
    seq = (hidden_states.shape[1] if hidden_states.dim() == 3
           else hidden_states.shape[0])
    base = ctx.advance(self._layer, seq)          # <- unconditional; cannot be skipped
    if base < 0 or self._suppressed or self._edge_sensing is None:
        return
    try:
        edge_sensing.sense_pass(ctx, self._edge_sensing, self,
                                hidden_states, base)
    except Exception:
        logger.exception("edge_sensing_pass_failed")   # never breaks generation
```

Every guard that previously had to remember to advance now simply returns. That is the design.

## 5. Service and Inference Wiring

`CircuitSensingService` keeps its responsibilities (arm/disarm, decode, persist, WS, retention) and
loses its ownership of the ring:

```python
# circuit_sensing_service.py
def begin_request(self, request_id, layer_saes) -> Optional[SensingRequestContext]:
    ctx = SensingRequestContext(
        request_id=request_id,
        circuit_ids=self._armed_circuit_ids(),        # a SET, not one id (F19-ready)
        max_lag=self._max_token_lag,
        budget=EventBudget(settings.CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST),
    )
    for sae in layer_saes.values():
        sae.bind_context(ctx)
    return ctx

def collect_edges(self, ctx) -> tuple[str, list[SensedEdge], dict[str, list[int]]]:
    """Drain once from the context. Returns per-circuit truncated_layers
    rather than a single request-wide boolean (BR-006)."""
```

**Deleted:** `prune_ring` (:526), `safe_prune_boundary` (:538), `prune_between_passes` (:550). Verified
live: zero production callers, referenced only by
`tests/unit/services/test_circuit_sensing_service.py:412/423/437`. These are R2's superseded pruning
design, which R3 replaced with `note_layer_progress` and never removed. Their tests go with them —
keeping a test for a deleted design is how `TestRingPruningIsWired` came to assert the opposite of the
truth.

`InferenceService` changes shape only slightly: `_circuit_sensing_begin` (:1493) returns the context
instead of a layer-SAE map, `_notify_circuit_sensing` (:1524) takes it, and `close_request` moves onto
the context. The three call sites at :1857/:2025/:2349 and the three notify sites at
:1938/:2301/:2415 are updated in step. The existing `SensingRequestContext` for Feature 11
(`inference_service.py:110`) is the naming precedent and stays where it is — F17's context is the same
idea one scope up, for circuits rather than clusters, and the two must not be conflated: if the name
collision is confusing at implementation time, F17's is `EdgeSensingRequestContext`. Resolve at
implementation, record the choice.

## 6. Admin UI Design

None. The only surface touched is `truncated_layers` on the edge-sensing status payload, rendered by
the Feature 15 status strip. If that strip currently renders a request-wide truncation flag, it renders
the layer list instead; no new component.

## 7. Testing Strategy

### Characterization (WRITTEN AND GREEN BEFORE ANY MOVE — FR-17.6, gate)
`tests/unit/ml/test_edge_sensing_characterization.py`, written against the CURRENT code and never
modified afterwards. It pins observable matcher behaviour, not implementation:
strict ordering (up before down); the lag window boundary at exactly L and L+1; same-position co-fire
does not match; newest antecedent wins; the read is non-destructive so one upstream fire can father
several events (the R3-F semantic the FTDD amendment blessed); `_MAX_FIRES_PER_EDGE` eviction drops
oldest; saturation shedding still records upstream fires; the cap still records upstream fires;
position advances on every return path including suppressed and batched; phase flips exactly once;
`prefill`/`decode` attribution. A diff to this file after the move is a behaviour change and must be
justified in the review record.

### Unit (new behaviour)
`tests/unit/ml/test_sensing_request_context.py`: advance/close/double-close/write-after-close;
`advance` reports progress for every ring including on suppressed passes (FPRD §15.6 regression pin);
budget attribution across two circuits; `try_spend` returning False never suppresses upstream
recording; `truncated_layers` names the shedding layer only.
`tests/unit/ml/test_edge_sensing_ring_isolation.py`: **the sharpest test in the feature** — two
circuits containing the SAME `edge_key`, circuit A fires upstream, circuit B fires downstream in
window, assert **zero** events. Negative-control it against a single shared ring, which must produce
the fabricated event.

### Integration
`tests/integration/test_circuit_edge_sensing_workflow.py` runs UNCHANGED — that is the behaviour-
preservation proof at the outside boundary. Added: two-circuit concurrent sensing (constructed
directly if Feature 19's flag is not yet live), and post-hang release.

### Mutation (AFTER the move — FR-17.6)
R3's practice, applied to `edge_sensing.py`: break a load-bearing line, run the suite, revert, record.
Minimum mutation set: the strict-before comparison in `match_down`; the window comparison; the
`bisect` insertion point; the unconditional `advance`; the `note_layer_progress` call inside it;
`try_spend`'s boundary; the ring lookup key (mutate to a constant — this must fail, and if it does not,
the isolation test is not pinning what it claims). Every survivor either gets a test or a recorded
reason.

### Benchmark
The three F15 latency shapes, re-measured after the move: saturated 4096-token pass, 200-edge circuit,
and the cross-layer ordering where the upstream layer records a full prefill before the downstream
layer matches ascending. Each of F15's three latency fixes was validated by a benchmark that measured
the path it had not changed; the benchmark set must cover all three.

## 8. Risks

- **A refactor that changes behaviour while the suite stays green.** This is the primary risk and the
  reason for characterization-first. Mitigation is procedural and non-negotiable: the characterization
  suite exists and is green before the first line moves.
- **The move silently reintroduces a fixed critical.** Eight criticals were fixed in this code; each
  fix is a specific line, several of them counter-intuitive (the non-destructive read, the
  `continue`-not-`return` on cap, the `bisect` start). Mitigation: every fix has a named test in the
  characterization suite, and the FTID lists them with file:line so a reviewer can check them off.
- **Docstrings lost in the move.** The comments in `EdgeFireRing.prune_before` and
  `note_layer_progress` are the institutional memory of why two designs failed. Losing them invites a
  fourth attempt. Mitigation: an explicit task, and a review Watch item.
- **The N-circuit shape is built but unexercised** until Feature 19 lands, so the isolation guarantee
  could rot. Mitigation: the ring-isolation test constructs two circuits directly and does not depend
  on Feature 19's flag.
- **Naming collision with Feature 11's `SensingRequestContext`.** Two classes, same name, adjacent
  concerns, one file apart. Mitigation: decide at implementation and record; a reviewer confusing them
  is a realistic path to a real defect.
- **Scope creep into Feature 18/19.** The context makes the single derivation and concurrent serving
  easier, which is exactly when it becomes tempting to start them. Out of scope is enumerated in FPRD
  §12 for this reason.
