# Technical Implementation Document: Circuit Edge Sensing

## miLLM Feature 15

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `015_FPRD|Circuit_Edge_Sensing.md` · `015_FTDD|Circuit_Edge_Sensing.md` · `docs/mcp-contract.md` (v1.1)

---

## 1. File Structure

```
millm/
├── ml/sae_wrapper.py                    (MOD — EdgeSpec/CircuitSensingConfig/SensedEdge, arm/disarm/
│                                          _sense_edges/begin_edge_sensing_request/collect; shared up-firing ring)
├── ml/sae_hooker.py                     (MOD — one edge-armed branch, sibling of the F11 sensing branch)
├── services/circuit_sensing_service.py  (NEW — extends the F11 sensing service pattern)
├── services/inference_service.py        (MOD — begin/collect + _notify_circuit_sensing across attached SAEs,
│                                          routing condition; reuse IdCaptureStoppingCriteria)
├── services/circuit_service.py          (MOD — arm/disarm on circuit activate/deactivate)  [Feature 13 owner]
├── services/sae_service.py              (MOD — disarm on SAE-set detach)
├── db/models/circuit_edge_sensing_event.py          (NEW — mirrors db/models/sensing_event.py)
├── db/repositories/circuit_edge_sensing_repository.py (NEW)
├── db/migrations/versions/012_add_circuit_edge_sensing.py (NEW)
├── api/schemas/circuit_sensing.py       (NEW)
├── api/routes/management/circuit_sensing.py (NEW)  + dependencies.py, routes/__init__.py (MOD)
├── sockets/progress.py                  (MOD — emit_circuit_sensing_event)
├── core/config.py                       (MOD — CIRCUIT_SENSING_* keys)
├── core/errors.py                       (MOD — NoActiveCircuitError, CircuitSensingEventNotFoundError)
admin-ui/src/components/circuits/sensing/{EdgeSensingPanel,EdgeSensingEventDetail}.tsx (NEW)
admin-ui/src/services/circuitSensing.ts, hooks/useCircuitSensing.ts (NEW)
tests/unit/{ml,services,db}/test_circuit_edge_sensing*.py (NEW)
tests/integration/test_circuit_edge_sensing_workflow.py (NEW)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **Extend `_sense`, do not fork it** — `sae_wrapper.py` already has the full fire-detection substrate:
  `arm_sensing`/`disarm_sensing` (:516/:545), `_W_enc_m` member cache (:527, `index_select` + `.contiguous()`),
  `_sensing_thresholds_cpu` (:533), `_reset_sensing_buffer` (:561), `begin_sensing_request` (:569),
  `_sensing_token_offset`/`_sensing_phase`/`_sensing_done`/`_sensing_truncated` (:167-170). `_sense_edges`
  reuses the acts/fired computation and adds the up→down ring matcher. Keep the two armed states separate
  (`is_sensing_armed` vs a new `is_edge_sensing_armed`) so cluster and circuit sensing never cross-fire.
- **Hook insertion** — put the edge branch beside the Feature 11 sensing branch in `sae_hooker.py::hook_fn`,
  a SIBLING (evaluate even when cluster sensing is off), BEFORE `apply_steering` so positions reflect the
  pre-steer residual read. `suppressed()` (sae_wrapper.py:509/`_suppressed` :145) — `_sense_edges` early-returns.
- **Do NOT reuse `_capture_activations`** — it positionally compacts columns and keeps only the last pass;
  fatal for sensing, same as the Feature 11 note. The member-only encode avoids both.
- **Request boundaries live in the serial queue** — the same semaphore-guarded blocks where Feature 11 calls
  `begin_sensing_request`/`collect_sensing_hits`. Circuit edges span SAEs, so `begin_edge_sensing_request` is
  called for EACH attached SAE the circuit references, all sharing ONE per-request up-firing ring owned by the
  service; `collect_edges` drains it once.
- **Flush placement** — beside `_notify_sensing`; `_notify_circuit_sensing` is async (DB) — await it.
- **Streaming/prefill token ids** — reuse the shipped `IdCaptureStoppingCriteria` and `outputs[0]`/`inputs.input_ids`
  exactly as Feature 11 (inference_service.py). Zero new capture machinery.
- **Routing** — `_use_cbm_for_request`: add `or (settings.CIRCUIT_SENSING_FORCE_SERIAL and
  circuit_sensing_service.is_armed)` beside the F11 condition. Non-forced CBM: skip begin ⇒ empty collect ⇒ unsensed.
- **WS emitter** — copy `emit_sensing_event` (sockets/progress.py:551-573) exactly; event name
  `circuit:sensing:event`; payload excludes context text (UI fetches detail via REST).
- **Routes shape** — copy `api/routes/management/sensing.py` (ProfileId→CircuitId, `_PRUNE_ON_READ_INTERVAL_S`
  read-throttle :29-32, envelope, `_toggle` persist+live-arm). `NO_ACTIVE_CIRCUIT` is 200+envelope (contract §5).
- **Migration numbering** — next free number is **011** (`ls millm/db/migrations/versions` confirms disk ends at
  `010_add_sensing_context_parts.py`; `008` is the sensing-events table, NOT circuits). Do NOT collide with the
  circuits table migration Feature 13 owns (`011_add_circuits_table.py`) — chain `down_revision` after it (this = `012`).
- **Model shape** — mirror `db/models/sensing_event.py`: `JSONVariant = JSON().with_variant(JSONB(),"postgresql")`,
  `context_parts` JSONB, `to_dict(include_context)` excluding context on WS payloads.

## 3. Key Implementations

```python
# sae_wrapper.py — _sense_edges core (reuses F11 acts/fired; adds up→down matcher)
def _sense_edges(self, hidden_states: torch.Tensor) -> None:
    if self._suppressed or self._edge_cfg is None or self._sensing_done:
        return
    x = hidden_states[0] if hidden_states.dim() == 3 else hidden_states     # (seq, d_in)
    acts = torch.relu(x.to(self._W_enc_m.dtype) @ self._W_enc_m + self._b_enc_m)  # (seq, m)
    fired = acts > self._edge_cfg.thresholds                                # (seq, m)
    for local_pos in fired.any(dim=-1).nonzero(as_tuple=True)[0].tolist():
        abs_pos = self._sensing_token_offset + local_pos
        for e in self._edge_cfg.edges:
            if e.up_col >= 0 and fired[local_pos, e.up_col]:
                self._edge_ring.push_up(e.edge_key, abs_pos,
                                        float(acts[local_pos, e.up_col]),
                                        e)                                   # bounded per-edge ring
            if e.down_col >= 0 and fired[local_pos, e.down_col]:
                up = self._edge_ring.pop_in_window(e.edge_key, abs_pos,
                                                   self._edge_cfg.lag_tokens)
                if up is not None:                                          # 0 < lag <= L, up before down
                    self._append_edge(up, abs_pos,
                                      float(acts[local_pos, e.down_col]), e,
                                      self._sensing_phase)                  # cap -> _sensing_done
    self._sensing_token_offset += x.shape[0]
    if self._sensing_phase == "prefill":
        self._sensing_phase = "decode"
```

```python
# circuit_sensing_service.py — arm: build per-SAE config, drop unsensable edges
def arm_for_circuit(self, circuit, attached: dict[int, LoadedSAE]) -> None:
    unsensable = []
    ring = EdgeRing()                              # ONE ring shared across the circuit's SAEs this arm
    for layer, sae in attached.items():
        members = [m for m in circuit.members if m["layer"] == layer]
        idxs = [int(m["feature_idx"]) for m in members]
        thetas = [self._theta(m) for m in members]      # F11 rule: max(floor, eps*max_activation) or inf
        edges = []
        for e in circuit.edges:
            up_here  = e["up_layer"]  == layer
            down_here = e["down_layer"] == layer
            if e["up_layer"] not in attached or e["down_layer"] not in attached:
                unsensable.append(e["edge_key"]); continue
            edges.append(EdgeSpec(
                edge_key=e["edge_key"],
                up_col=idxs.index(e["up_idx"]) if up_here and e["up_idx"] in idxs else -1,
                down_col=idxs.index(e["down_idx"]) if down_here and e["down_idx"] in idxs else -1,
                up_layer=e["up_layer"], down_layer=e["down_layer"],
                rung=e["rung"], rung_language=e["rung_language"]))
        cfg = CircuitSensingConfig(circuit.id, idxs, torch.tensor(thetas),
                                   self._mode(thetas), edges,
                                   settings.CIRCUIT_SENSING_LAG_TOKENS)
        sae.arm_edge_sensing(cfg, ring)
    self._armed_circuit_id = circuit.id
    self._unsensable_edges = unsensable            # surfaced in status()
```

```python
# circuit_sensing_service.py — summary builder (rung verbatim; NEVER 'causal' below rung 2)
def _summary(self, ev: SensedEdge, up_label: str, down_label: str) -> str:
    # ev.rung_language is server-rendered and already avoids 'causal' for rung<2;
    # the builder NEVER interpolates its own claim word — a guard test asserts
    # "causal" not in summary when ev.rung < 2.
    return (f"{up_label} -> {down_label}: L{ev.up_layer}->L{ev.down_layer}, "
            f"lag {ev.token_lag}, {ev.rung_language}, peak "
            f"{ev.up_peak_act:.1f}/{ev.down_peak_act:.1f} during {ev.phase} "
            f"@ {ev.up_pos}->{ev.down_pos}")[:300]
```

```python
# circuit_sensing_service.py — context_parts (off hot path; span covers up->down)
def _context(self, full_ids, ev: SensedEdge, k: int, tokenizer):
    if k == 0 or full_ids is None:
        return None, None
    ids1 = full_ids[0] if full_ids.dim() == 2 else full_ids
    lo = max(0, ev.up_pos - k); hi = min(ids1.shape[-1], ev.down_pos + 1 + k)
    seg = lambda a, b: tokenizer.decode(ids1[a:b].tolist(), skip_special_tokens=True)
    parts = {"before": seg(lo, ev.up_pos),
             "span":   seg(ev.up_pos, ev.down_pos + 1),
             "after":  seg(ev.down_pos + 1, hi)}
    return parts, ids1[lo:hi].tolist()
```

## 4. Implementation Pitfalls

1. **Ordering is strict** — only an up-firing FOLLOWED by a down-firing is an edge (EC-15.1/15.2). Never emit
   on a lone upstream, and never match a downstream firing to a LATER upstream. `pop_in_window` requires
   `0 < down_pos - up_pos ≤ L`.
2. **One ring per request, shared across SAEs** — a circuit's up and down endpoints may live on different SAEs;
   both `arm_edge_sensing` calls receive the SAME `EdgeRing`. `begin_edge_sensing_request` MUST reset it once
   (guard against per-SAE double reset); a missed begin on an unsensed path must yield empty collect.
3. **Buffer hygiene across passes** — decode positions span pass boundaries; the ring holds ABSOLUTE positions
   (`_sensing_token_offset`), so lag matching is correct across passes. Prune ring entries older than L to bound it.
4. **Unsensable edges are dropped at arm, not silently mis-decoded** — an edge whose endpoint SAE is unattached
   (slice-fallback, EC-15.4) or whose member has an infinite threshold (EC-15.6) is excluded from `edges` and
   listed in `status().unsensable_edges`. Never sense through the wrong decoder.
5. **Rung verbatim** — the summary and UI render `rung_language` exactly; never synthesize "causal" for rung<2.
   A guard test asserts the substring is absent below rung 2 (FPRD §9 crit 6).
6. **dtype/device** — cast `x` to `_W_enc_m.dtype` exactly as `encode()` (:298) does (hidden states may be fp16/bf16).
7. **WS payload excludes context text** (user content; size) — UI fetches detail via REST.
8. **CASCADE test** — deleting a circuit must remove its edge events; pin with a test.
9. **Armed state vs intent** — the circuit's edge-sensing intent flag is persistent; ARMED is runtime state
   (active circuit + intent + both endpoint SAEs attached). Status reports both distinctly (F11 pitfall 8).

## 5. Config Additions (millm/core/config.py)

```python
CIRCUIT_SENSING_LAG_TOKENS: int = 8            # L; up->down window, hard max 64
CIRCUIT_SENSING_EPSILON: float = 0.1           # reuse F11 fire predicate
CIRCUIT_SENSING_THETA_FLOOR: float = 0.0
CIRCUIT_SENSING_CONTEXT_TOKENS: int = 16       # ±K; hard max 64
CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST: int = 20
CIRCUIT_SENSING_MAX_EVENTS_PER_CIRCUIT: int = 1000
CIRCUIT_SENSING_MAX_AGE_DAYS: int = 7
CIRCUIT_SENSING_FORCE_SERIAL: bool = True
CIRCUIT_SENSING_MAX_OVERHEAD_MS: float = 5.0
```
