# Technical Implementation Document: Probe Monitor Runtime

**Specified in:** `~/app/enhance/specs/ENH-001-probe-monitors` (handoff 2026-09-25)
## miLLM Feature 24

**Version:** 1.0 · **Created:** 2026-09-25 · **Status:** Planned
**References:** 024_FPRD, 024_FTDD
**Load-bearing points verified against** miLLM `main` @ `d399df5` (2026-09-24) by read-only survey.
**Re-check every line number before editing.** `inference_service.py` is 5k lines and moves.

---

## 1. File Structure
**New files:**
```
millm/ml/probe_head.py                         head + online accumulators (pure torch)
millm/ml/probe_hooker.py                       prepended read-only hook
millm/services/probe_runtime.py                ProbeRuntimeState, LoadedProbe, ProbeRequestContext
millm/services/probe_parity.py                 ProbeParityEngine
millm/services/probe_service.py                import, hub, identity, arm/disarm, record, status
millm/api/schemas/probe.py                     contract mirror + request/response models
millm/api/routes/management/probes.py          /api/probes routes
millm/core/probe_evidence.py                   rung ladder mirror
millm/db/models/probe.py                       Probe, ProbeEvent
millm/db/repositories/probe_repository.py      ProbeRepository, ProbeEventRepository
millm/db/migrations/versions/016_add_probe_monitors.py
docs/schemas/probe-definition-v1.json          vendored byte-identical from miStudio
manual/docs/features/probe-monitors.md         (+ manual/sidebars.ts entry)
```
Tests: `tests/unit/{ml/test_probe_head.py, ml/test_probe_hooker.py, services/test_probe_runtime.py,
services/test_probe_parity.py, services/test_probe_service.py, services/test_probe_wiring.py,
api/test_probe_schema_sync.py, api/test_probe_routes.py, core/test_probe_evidence.py,
sockets/test_probe_event_privacy.py}`, `tests/integration/test_probe_workflow.py`,
`tests/performance/test_probe_overhead.py`, and `tests/fixtures/lfm2_probe_definition.json` (at
acceptance).
Admin UI: `admin-ui/src/{pages/ProbeMonitorsPage.tsx, components/probes/*, hooks/useProbes.ts,
types/probe.ts}` plus `__tests__`.

**Modified files:**
- `millm/services/inference_service.py`: begin, finish and record in each path; CBM routing;
  speculative pause
- `millm/api/routes/openai/chat.py`: the header (and in completions if it has a non-stream path)
- `millm/api/routes/openai/errors.py`: `ERROR_STATUS_MAP` rows if any probe error can surface on `/v1`
- `millm/api/routes/__init__.py`: router registration
- `millm/core/config.py`: a `PROBE_*` block
- `millm/core/errors.py`: the new errors
- `millm/db/models/__init__.py`
- `millm/sockets/progress.py`: `emit_probe_event`, and the duplicate-definition cleanup
- `millm/ml/sae_hooker.py`: lift the layer helpers
- `millm/services/model_service.py` (or the load/unload seam): `ProbeRuntimeState.disarm_all("model_changed")`
- `docs/mcp-contract.md`: v1.6
- `tests/unit/test_mcp_tool_paths_are_real.py` (or a probes sibling)
- `.env.example`
- The admin-ui rename sites (FTDD §6)

## 2. Load-Bearing Implementation Points
1. **Layer resolution:** `SAEHooker._get_layer` (`ml/sae_hooker.py` ~L241). It tries
   `model.model.layers` first, which is correct for LFM2 and Llama. `layer_device` is ~L95 and
   `get_layer_count` ~L298. Lift these to module functions and make `SAEHooker` delegate. The
   existing hooker tests must stay green, unchanged.
2. **Prepend:** `target_layer.register_forward_hook(fn, prepend=True)`. The SAE hooks are registered
   without prepend (~L83), so a prepended probe hook always runs first and sees the pre-steer
   residual, which is what `test_hook_senses_pre_steer_values` protects for sensing.
3. **Dynamo:** call `sae_service._reset_dynamo_for_hook_change()` (~L2469) on install and remove.
   Import the function; don't duplicate it.
4. **Sensing seams to mirror:**
   - `_sensing_begin` (~L2343) and its speculative skip (~L2363)
   - `_sensing_mark_history` (~L2503)
   - `_notify_sensing` (~L2523)
   - call sites: non-stream chat begin ~L3283 / notify ~L3366; stream begin ~L4007 / final chunk
     ~L4243 / `[DONE]` ~L4260 / notify ~L4344; completions ~L4399 / ~L4466
   - batched, CBM and llama.cpp skip paths (~L3036, L4656–4819, L3587/3639)
5. **CBM:** `_use_cbm_for_request` (~L849). Add `if settings.PROBE_FORCE_SERIAL and
   ProbeRuntimeState().has_armed(): return False`, next to the sensing clause (~L901).
6. **Header:** `api/routes/openai/chat.py`. Non-streaming sets post-generation headers after ~L239,
   where `X-miLLM-Circuit-Rung` is set. Read the verdicts from a new ContextVar
   `_PROBE_VERDICTS` (the `_STEERING_CIRCUIT_MEMO` pattern, inference_service ~L220–255), reset in
   `reset_steering_memo()` (~L154) or a sibling `reset_probe_memo()` called at the same spot.
7. **Stream chunk:** build it with the same `id/created/model` as the final chunk (~L4243) and yield
   it **between the final chunk and `[DONE]`**. `_probe_finish` must run before it, not in the
   `finally`.
8. **Record and emit:** copy `SensingService.record` (~L226) and `_emit_events` (~L396–421),
   including the `context_` stripping and throttle constants. The emitter method goes in
   `sockets/progress.py` next to `emit_sensing_event` (~L550).
9. **Identity sources:**
   - `LoadedModelState().current` (model_loader ~L271)
   - the DB `Model` row by `model_id` for `repo_id` (~L94), `revision` (~L127) and `cache_path`
   - `model.config.hidden_size`
   - `get_layer_count(model)`
   - `tokenizer.chat_template`
   - the snapshot SHA: parse `cache_path` for `snapshots/<40-hex>`; if absent, `REVISION_UNVERIFIED`
10. **Hub:** `cluster_hub_service.py` `_list_models_sync(tag, query, base_model, limit)`,
    `_list_repo_files_sync`, `_download_file_sync`, the `cluster_hub_circuit` breaker, and
    `_cache_get` / `_cache_put`. Parameterise by tag and suffix (`.probe.json`); `manifest.jsonl` is
    preferred.

## 3. Key Implementations
**Hook body:**
```python
def _make_layer_hook(layer: int, state: "ProbeRuntimeState"):
    def hook(module, inputs, output):
        ctx = state.current_request()            # None → not scoring this pass
        if ctx is None or ctx.not_scored_reason:
            return None
        h = output[0] if isinstance(output, tuple) else (output if torch.is_tensor(output) else output[0])
        if h.dim() != 3 or h.shape[0] != 1:      # batch>1 → not scored (sensing rule)
            ctx.mark_not_scored("batched_request"); return None
        t0 = time.perf_counter()
        probes = state.probes_at(layer)
        z = h[0].to(probes[0].dtype)             # [seq, d]
        S = torch.stack([p.head.token_scores(z) for p in probes])   # [P, seq]
        A = torch.stack([p.head.attn_logits(z) if p.head.query is not None else torch.zeros_like(S[0]) for p in probes])
        S_cpu, A_cpu = S.float().cpu(), A.float().cpu()              # ONE D2H per pass
        ctx.update(layer, S_cpu, A_cpu)          # scope mask + online accumulators + position offset
        ctx.overhead_ms += (time.perf_counter() - t0) * 1e3
        return None                              # never modify the residual
    return hook
```
- `current_request()` is a request-scoped slot set by `_probe_begin` and cleared by `_probe_record`.
  Serial execution is guaranteed by the queue (`MAX_CONCURRENT_REQUESTS=1`) and forced serial CBM.
- **The prompt-scope mask:** `_probe_begin(messages, prompt_ids)` renders prefixes with miLLM's
  `_render_chat_template` and tokenizes them, then marks system and user spans. Decode tokens count as
  `response`. If the prefix property fails, a `prompt`-scope probe is `not_scored` with reason
  `role_mask_unreliable`. That matches miStudio's fallback.

**Parity:**
```python
report = []
for i, v in enumerate(defn.test_vectors.vectors):
    ids = torch.tensor([v.token_ids], device=model_device)
    with temporary_hook(model, layer, capture) : model(input_ids=ids)
    s = head.token_scores(capture.z[0]).float().cpu()
    agg = head.aggregate(s, capture.z[0], scope_mask_from(v))
    report.append(max(abs(s - tensor(v.token_scores)).max(), abs(agg - v.score)))
drift = [render_ids(v.messages) != v.token_ids for v in vectors]
ok = max(report) <= tol
```
Parity runs inside a `RequestQueue` slot (acquire as a request does) under `torch.inference_mode()`.

## 4. Implementation Pitfalls
1. **Don't compute the verdict in `finally`.** In streaming, `[DONE]` is already sent by then.
2. **Don't register the hook without `prepend=True`.** A same-layer SAE would then steer what the
   probe reads, and it would still pass most tests. The prepend test must use an SAE that steers the
   same layer.
3. **One D2H per pass.** A `.item()` or `.tolist()` per probe per token repeats F11's R1 regression.
4. **A GGUF model has no tokenizer and no hooks.** Check `supports_hooks` before anything else in
   `arm`.
5. **Model swap:** hook handles belong to the old `nn.Module`. `disarm_all("model_changed")` must run
   *before* the old model is released, and the handles must be removed from the old module.
6. **Socket privacy:** strip every `context_*` key. The global review rule records that a leaked
   prompt once passed 135/135 tests, so the privacy test must assert **absence** on the emitted
   payload, with a mutation control.
7. **ContextVar reset:** reset the probe verdict memo at request start, or a previous request's
   verdict leaks into the next header.
8. **The `progress.py` duplicates:** remove the second definitions of `emit_monitoring_state_changed`
   (~L637) and `create_socket_io` (~L697) after confirming the live one is the last definition
   (Python keeps the last). Keep the one that is effectively used today, to preserve behaviour.
9. **The mirror is permissive (`extra="allow"`), the producer strict.** Never "fix" the vendored JSON.
   Fix the mirror.
10. **Layer indexing:** the contract's `layer` is a 0-based decoder block (`model.model.layers[L]`),
    the same as `_get_layer`. Add a test that `_get_layer(model, d.read.layer)` is the module
    miStudio's `layer_discovery` would pick on LFM2 (a structure fixture).

11. **SAE probes (FR-24.15):**
    - Find the downloaded SAE through the existing SAE repository by HF repo/path and revision.
    - Hash the weights file (SHA-256) once at arm time and cache the result by file mtime.
    - **Never** go through `AttachedSAEState` or `attach_sae`: the slice is private to the probe.
    - JumpReLU: apply `(pre > threshold[idx]) * pre`, and match the SAE's architecture exactly as
      `LoadedSAE.encode` does.
    - The pre-encode normalization comes from the definition's `normalization` (miStudio's
      training-normalization constants), not from the SAE's own config. The spike (OQ-4) checks the two
      agree.
    - A test attaches a *different* SAE on the same layer with steering on, and asserts the SAE probe's
      scores are unchanged (pre-steer read plus private encoder).

## 5. Config Additions (`core/config.py`, `PROBE_*` block; `.env.example`)
```
PROBE_FORCE_SERIAL=true
PROBE_MAX_ARMED=8
PROBE_PARITY_TOLERANCE=0.001
PROBE_MAX_OVERHEAD_MS=5.0
PROBE_MAX_EVENTS_PER_PROBE=5000
PROBE_MAX_AGE_DAYS=30
PROBE_EVENT_CONTEXT_TOKENS=24
PROBE_HUB_TAG=mistudio-probe-definition
PROBE_HUB_CACHE_TTL_S=300
PROBE_MAX_IMPORT_BYTES=2097152
```

## 6. Testing Notes (mutation controls to record)
- `test_probe_hooker.py`: remove `prepend=True` → the pre-steer test fails.
- `test_probe_wiring.py`: remove `_probe_begin` from any path → that path's test fails; remove the
  CBM clause → the serial-routing test fails.
- `test_probe_event_privacy.py`: stop stripping `context_text` → fails.
- `test_probe_routes.py`: unregister the router → fails (membership in the app's routes, **plus** a
  call with the payload).
- `test_probe_schema_sync.py`: edit one byte of the vendored file → the byte-identity test fails when
  miStudio is present, and the structural test fails for a missing field.
- `test_probe_evidence.py`: change one word of the rung language → the cross-repo identity fails.
- The MCP consistency test: add the `### millm_probes` heading without miStudio's registry → fails
  (proves co-release).

## 7. Decisions
FPRD R1–R8, FTDD U1–U3. Implementation defaults:

| # | Question | Default |
|---|---|---|
| V1 | How is the request slot scoped? | A single `current_request()` slot guarded by queue serialisation, with a test that concurrent begin raises |
| V2 | How is the header list ordered? | By probe name, for deterministic headers |
| V3 | Does parity use a temporary hook or the armed one? | A temporary hook, so arming isn't needed to test |
