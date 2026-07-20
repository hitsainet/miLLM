# Technical Implementation Document: Request-Scoped Sensing Context

## miLLM Feature 17

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `017_FPRD|Request_Scoped_Context.md` · `017_FTDD|Request_Scoped_Context.md` · `BRD-MILLM-CIRCUITS-002.md` (BR-001) · `docs/circuit-contention-model.md` (§4)

---

## 1. File Structure

```
millm/
├── ml/edge_sensing.py                   (NEW — EdgeSpec, CircuitSensingConfig, SensedEdge,
│                                          EdgeFireRing, EventBudget, SensingRequestContext,
│                                          sense_pass/match. NO import of sae_wrapper)
├── ml/sae_wrapper.py                    (MOD — remove 145 lines of edge machinery + 13 _edge_* fields;
│                                          keep a thin sense_edges wrapper + bind_context)
├── ml/sae_hooker.py                     (MOD — call site rename only, :181-183)
├── services/circuit_sensing_service.py  (MOD — owns arm/decode/persist/WS; DELETES prune_ring,
│                                          safe_prune_boundary, prune_between_passes)
├── services/inference_service.py        (MOD — begin returns a context; notify takes it; close on it)
├── api/schemas/circuit_sensing.py       (MOD — truncated_layers on the status schema)
tests/unit/ml/test_edge_sensing_characterization.py   (NEW — WRITTEN FIRST, never modified after)
tests/unit/ml/test_sensing_request_context.py         (NEW)
tests/unit/ml/test_edge_sensing_ring_isolation.py     (NEW — the cross-circuit collision test)
tests/unit/ml/test_edge_sensing.py                    (MOD — retarget imports; drop stub-based fixtures)
tests/unit/services/test_circuit_sensing_service.py   (MOD — delete the 3 prune tests at :412/:423/:437)
tests/integration/test_circuit_edge_sensing_workflow.py (UNCHANGED — the preservation proof)
```

## 2. Load-Bearing Implementation Points (verified against live code, 2026-07-20)

- **The machinery to move is 145 lines in a 1373-line file**, with **91 `_edge` references** and
  **13 `_edge_*` instance fields** on `LoadedSAE` (the F15 record said 11; recount at
  `sae_wrapper.py:380` and around `_reset_edge_buffer` :1012 — the two added since are
  `_edge_member_fires` and `_edge_thresholds_cpu`). Symbols and their live lines:
  `EdgeSpec` :51, `CircuitSensingConfig` :78, `SensedEdge` :98, `EdgeFireRing` :123
  (`_MAX_FIRES_PER_EDGE` :146, `record_up` :148, `match_down` :156, `prune_before` :189,
  `note_layer_progress` :211, `clear` :230), `arm_edge_sensing` :938, `_reset_edge_buffer` :1012,
  `begin_edge_sensing_request` :1028, `collect_sensed_edges` :1036, `_sense_edges` :1050,
  `_match_edges` :1137, `to_device` :1316.
- **`to_device` (:1316) must be checked, not assumed.** It moves cached tensors between devices; the
  edge caches (`_W_enc_e`/`_b_enc_e`) are among them. They stay on the SAE, so `to_device` should be
  untouched — but verify, because a cache left on the wrong device after the split is a silent
  wrong-answer path, not a crash.
- **`_sense_edges` advances the offset in THREE places** — :1072 (guard return), :1093 (batched-pass
  return) and :1123 (`finally`). That triplication is R1-03's residue and the thing the context
  removes. Do not port it; replace it with one `ctx.advance()` above the body (FTDD §4).
- **LIVE DEFECT to fix in the move (FPRD §15.6, EC-17.1):** both early returns (:1069-1075, :1080-1096)
  advance the offset and `return` **before** reaching `note_layer_progress`, which lives in the
  `finally` at :1128-1135. A layer suppressed for a whole request therefore never reports progress,
  `_progress` (:138) can stay below the `len(self._progress) < 2` guard at :222, and **the ring never
  prunes**. This is R3's own fix carrying R1's shape. `ctx.advance()` calling `note_layer_progress`
  unconditionally closes it.
- **DELETE, do not move: `prune_ring` (`circuit_sensing_service.py:526`), `safe_prune_boundary`
  (:538), `prune_between_passes` (:550).** Verified live: zero production callers. The only references
  are `tests/unit/services/test_circuit_sensing_service.py:412, :423, :437`. This is R2's superseded
  pruning design that R3 replaced with `note_layer_progress` and never removed — the codebase carries
  two pruning designs today, one live and one dead. Delete the tests with them.
- **`begin_edge_sensing_request` (:1028) documents a convention the context must absorb:** *"The CALLER
  clears the shared ring once for the whole circuit — clearing it here would wipe upstream fires
  recorded by a sibling SAE that began first."* That is exactly the class of rule this feature exists
  to eliminate. The context creates its rings in `__init__`; no participant clears anything.
- **`arm_edge_sensing` (:938) takes `(config, ring)`.** After the move it takes `config` only — the
  ring comes from the context at request time, not at arm time. This matters for Feature 19: rings are
  per `(request, circuit)`, and arm time does not know the request.
- **The `d_sae` bounds check at :947-957 must survive verbatim.** Its comment explains that an
  out-of-range `index_select` on CUDA is a device-side assert that poisons the process context. R2-07
  then tightened the column check to `-1 <= col < width` because `-1` is the legitimate "not my half"
  sentinel and `-2` was passing. Both checks move together.
- **`match_down` (:156) contains three generations of fix in one method** — R1's count-based bounding,
  R2's backward scan with `break`, R3's `bisect` insertion point. The `import bisect` is function-local
  (:180); hoist it to module scope in the new file, and re-run the benchmark, because a hoist is a
  behaviour-neutral change that a latency test should confirm rather than assume.
- **Hook call site is one line** — `sae_hooker.py:181-183`, `if sae.is_edge_sensing_armed: with
  torch.no_grad(): sae._sense_edges(hidden_states)`. It becomes `sae.sense_edges(...)` (public, thin).
  The `is_edge_sensing_armed` boolean check stays: un-armed cost must remain one boolean (EDGE-S3).
- **Inference call sites**: `_circuit_sensing_begin` :1493, `_notify_circuit_sensing` :1524, begin at
  :1857 / :2025 / :2349, notify at :1938 / :2301 / :2415, `close_request` at :2288. All six begin/notify
  sites must be updated together — F15's R3-04 was precisely a teardown that existed on one path.
- **Feature 11's `SensingRequestContext` already exists at `inference_service.py:110`** — a frozen
  begin-time snapshot dataclass `(sae, profile_id, config)`. F17's context is the same idea one scope
  up. Decide the name before writing code (`EdgeSensingRequestContext` is the safe choice) and record
  it; two same-named classes one file apart is a realistic path to a real defect.

## 3. Key Implementations

```python
# millm/ml/edge_sensing.py — the position/progress advance, the heart of the feature.
def advance(self, layer: int, seq: int) -> int:
    """Return this pass's base position and advance the shared counter.

    Called ONCE per pass per layer from LoadedSAE.sense_edges, ABOVE every
    guard. F15 advanced the offset at three separate exits of _sense_edges
    (sae_wrapper.py:1072, :1093, :1123) because each early return had to
    remember to; R1-03 was one that did not, and the surviving gap is that
    two of those three exits still return before note_layer_progress, so a
    layer suppressed for a whole request stalls pruning entirely.
    Advancing above the guards makes both unrepresentable.
    """
    if self._closed:
        # A hung generation thread waking after close (F15 R3-06). Dropping it
        # loudly is correct; letting it write would corrupt the NEXT request.
        logger.warning("sensing_write_after_close: request=%s layer=%s",
                       self.request_id, layer)
        return -1
    base = self.position
    self.position += seq
    if self.phase == "prefill":
        self.phase = "decode"
    # Unconditional, for EVERY ring: pruning must not depend on a layer
    # having had anything to sense this pass.
    for ring in self._rings.values():
        ring.note_layer_progress(layer, self.position)
    return base
```

```python
# millm/ml/edge_sensing.py — per-circuit ring lookup. The fabrication guard.
def ring(self, circuit_id: str) -> Optional[EdgeFireRing]:
    """The ring for ONE circuit. Never a shared ring.

    edge_key is '{up_idx}@{up_layer}->{down_idx}@{down_layer}' and is NOT
    unique across circuits: two circuits can legitimately declare the same
    edge. With one shared ring, circuit A's upstream fire would be a valid
    antecedent for circuit B's downstream fire, and the match would be
    recorded as an observation of an edge that fired in NEITHER circuit.
    A fabricated observation on an evidence surface is categorically worse
    than a missed one. Feature 19 makes this reachable by serving two
    circuits at once. See PADR v1.3 and circuit-contention-model.md section 4.
    """
    return self._rings.get(circuit_id)
```

```python
# millm/ml/edge_sensing.py — budget spend. Never stops upstream recording.
def try_spend(self, circuit_id: str, layer: int) -> bool:
    """May an event be appended for this circuit? False must CONTINUE, never RETURN.

    F15 hit this twice: R2-03 (a saturated layer's load-shed returned before
    recording upstream fires, starving quiet siblings) and R3-02 (the cap did
    the same thing through a different door -- _edge_done caused an early
    return from the whole pass). Upstream recording is a dict append and every
    sibling layer depends on it. The caller must `continue`, not `return`.
    """
    spent = self._spent.get(circuit_id, 0)
    if spent >= self.limit_per_circuit:
        self._shed_layers.setdefault(circuit_id, set()).add(layer)   # BR-006
        return False
    self._spent[circuit_id] = spent + 1
    return True
```

```python
# tests/unit/ml/test_edge_sensing_ring_isolation.py — the sharpest test in the feature.
def test_two_circuits_sharing_an_edge_key_do_not_cross_match():
    """circuit A's upstream must never be an antecedent for circuit B's downstream."""
    key = "7@10->42@13"                      # the SAME edge_key in both circuits
    ctx = EdgeSensingRequestContext("req-1", ["circ_a", "circ_b"], max_lag=8,
                                    budget=EventBudget(20))
    ctx.ring("circ_a").record_up(key, pos=3, act=9.0)     # only A fired upstream
    assert ctx.ring("circ_b").match_down(key, down_pos=5) is None   # B must not match
    assert ctx.ring("circ_a").match_down(key, down_pos=5) == (3, 9.0)
    # Negative control (documented, run once by hand): give both circuits ONE
    # shared ring and this assertion flips -- that is the fabricated observation.
```

## 4. Implementation Pitfalls

1. **Write the characterization suite first, and do not touch it afterwards.** FR-17.6 makes this a
   gate. If a characterization test needs editing after the move, the refactor changed behaviour —
   stop and record it as a finding rather than editing the test to match. This is the single most
   important instruction in the document.
2. **Advance above the guards, always.** Every `return` added to the sensing body in future must be
   safe by construction. If a reviewer can point at a return path and ask "does that skip the
   advance?", the refactor has failed its purpose.
3. **`try_spend` returning False means `continue`, not `return`.** Two of F15's criticals (R2-03,
   R3-02) are the same starvation bug through different doors. A third door is exactly the shape of
   defect this arc keeps producing.
4. **One ring per `(request, circuit)`.** Not per request, not per SAE, not per layer. The isolation
   test with its negative control is the pin.
5. **Move the docstrings.** `prune_before` (:189) and `note_layer_progress` (:211) carry the reasons
   R1's and R2's pruning designs failed. Losing them invites a fourth attempt — three have already
   been tried in this file.
6. **Delete R2's prune trio and their tests together.** Leaving the tests behind recreates
   `TestRingPruningIsWired`: a test named for a property it does not check, pinning a dead design.
7. **Do not regress the three latency fixes.** `bisect` start (R3-03), backward scan with `break`
   (R2-02), count-based bounding (R1-02). Benchmark all three shapes; each of F15's benchmarks
   measured a path its own fix had not changed.
8. **The non-destructive read is deliberate.** `match_down` does not remove the matched entry, so one
   upstream fire can father several downstream events. The FTDD amendment blessed this as the better
   evidence model. A "cleanup" that makes it consuming is a behaviour change.
9. **`_edge_thresholds_cpu` is dead — delete it, do not move it.** Recorded unresolved in R1-14, R2-E
   and R3-G. Moving dead code into a new module launders it into looking intentional.
10. **Keep `is_edge_sensing_armed` a plain boolean on the SAE.** Un-armed cost is one check (EDGE-S3);
    routing it through the context would put an attribute lookup on every hook call of every request.
11. **Update all six begin/notify call sites together** (:1857/:2025/:2349, :1938/:2301/:2415). F15's
    R3-04 was a teardown present on one path and absent on another.
12. **Resolve the `SensingRequestContext` name collision before writing code**, not during review.

## 5. Config Additions

**None.** No new config keys, no flags, no migration. If the implementation finds itself wanting a
flag to switch between the old and new arrangement, that is a signal the change is being landed in a
half-state — FPRD §11 rules that out explicitly, because two coexisting position counters are strictly
worse than one badly-owned one.
