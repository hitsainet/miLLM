# Technical Design: Probe Monitor Runtime

**Specified in:** `~/app/enhance/specs/ENH-001-probe-monitors` (handoff 2026-09-25)
## miLLM Feature 24

**Version:** 1.0 · **Created:** 2026-09-25 · **Status:** Planned
**References:** 024_FPRD · BRD-MILLM-PROBES-001 · vendored `docs/schemas/probe-definition-v1.json`
(miStudio 033) · F11 sensing, F15 edge sensing, F20 MCP circuit surface, F023 GGUF

---

## 1. Executive Summary

| Area | Decision | Rationale |
|---|---|---|
| Hook | One prepended, read-only forward hook per armed layer (`ProbeHooker`), independent of SAEs | Probes need no SAE; prepending reads pre-steer like sensing |
| Runtime state | `ProbeRuntimeState` singleton: layer → shared hook, list of `LoadedProbe` | Several probes on one layer cost one hook |
| Request lifecycle | `ProbeRequestContext` created alongside `SensingRequestContext`; `_probe_begin` / `_probe_finish` / `_probe_record` in every generation path | Reuses the proven sensing seams; finish runs before the final chunk |
| Parity | `ProbeParityEngine` runs test vectors via `token_ids` inside the request queue | Proves identical scoring on the live model; separates tokenizer drift |
| Identity | Strict compare, with `REVISION_UNVERIFIED` as a warning when only the requested revision is known | Wrong-model scoring is silent; refuse loudly |
| Persistence | Migration 016: `probes`, `probe_events` | Probes aren't profiles |
| Delivery | RFC 8941 header (non-stream); a `choices: []` chunk (stream) | OpenAI-compatible shapes |
| Evidence | `core/probe_evidence.py` mirrors miStudio's ladder verbatim | Same no-overclaiming rule as circuits |

## 2. System Architecture
```
import (file | hub | MCP) → mirror validate → probes row
arm → identity check → parity (queue slot) → ProbeRuntimeState.arm → ProbeHooker.install(prepend)
request:
  InferenceService.<path>
    _probe_begin(request_id, messages, prompt_ids) → ProbeRequestContext (scope masks, reset aggregates)
    forward passes → probe hook: z=out[0] → for probes@layer: s_t = ((z-μ)/σ)·w+b → masked online agg
                      (one .cpu() of the stacked scores per pass)
    _probe_finish(ctx) → verdicts (before final chunk / before response returns) → ContextVar
    stream: yield chunk{choices:[], millm_probe_verdicts} → [DONE]
    non-stream: chat.py sets X-miLLM-Probe-Verdicts from ContextVar
    finally: _probe_record(ctx) → ProbeService.record → probe_events + prune → emit probe:event (stripped)
```

## 3. Components

**`millm/ml/probe_head.py`:** `ProbeHead(weights, bias, mean, std, query|None, rule, params)` with
`token_scores(z)`. The rules and their online accumulators are `MeanAcc`, `MaxAcc`, `SoftmaxAcc`
(log-sum-exp), `AttentionAcc` (online softmax over `q·ẑ`), `RollingMeanMaxAcc` (a ring buffer) and
`LastAcc`. The math is the same as miStudio's `ml/probe_monitor_model.py`. **Parity tests pin it.**

**`millm/ml/probe_hooker.py`:** `ProbeHooker.install(model, layer, fn) -> RemovableHandle`. It reuses
`SAEHooker._get_layer` and `layer_device` (call them through an `SAEHooker()` instance or lift them to
module functions). It registers with `prepend=True`, and the hook returns `None` (output unchanged).

**`millm/services/probe_runtime.py`:**
- `ProbeRuntimeState` (singleton): `arm(probe_row, model)`, `disarm(id, reason)`,
  `disarm_all(reason)`, `armed()`, `paused_reason`.
- `LoadedProbe`: the head on the layer's device.
- `ProbeRequestContext`: `request_id`, per-probe accumulator, scope mask provider, position offset,
  phase, `overhead_ms`, `not_scored_reason`.

**`millm/services/probe_sae_slice.py`:** `SaeFeatureSlice.load(sae_record, feature_indices, normalization,
device)` reads the downloaded SAE file through the existing `sae_loader` / `sae_config` path, verifies
the weights SHA-256, and keeps only `W_enc[:, idx]`, `b_enc[idx]`, thresholds and normalization
constants. `encode(z) -> [seq, k]` mirrors `LoadedSAE`'s activation semantics restricted to `idx`.
`ProbeHead` takes an optional slice: the basis is `residual`, or `slice.encode(z)`.

**`millm/services/probe_parity.py`:** `ProbeParityEngine.run(model, tokenizer, probe) -> ParityReport`
- a forward pass over each vector's `token_ids` with a temporary hook
- compare `token_scores` and `score`
- re-render `messages` through `_render_chat_template` and compare token ids (drift)

**`millm/services/probe_service.py`:**
- import (size and kind gates, mirror validation, dedupe)
- the hub (a `ProbeHubService` over the cluster hub helpers, parameterised by tag and suffix)
- `check_identity`, `arm`, `disarm`, `parity`
- `record` (the `SensingService.record` pattern)
- `status`
- event queries

**`millm/api/schemas/probe.py`:** the mirror `ProbeDefinitionV1` (`extra="allow"`), request and
response models.

**`millm/api/routes/management/probes.py`:** routes registered in `api/routes/__init__.py`.

**`millm/core/probe_evidence.py`:** the ladder mirror.

**`millm/core/errors.py`:** the new errors listed in §5.

## 4. Data Design

`probes` also stores `basis` (`residual`|`sae_features`) and `sae_ref` JSON (the definition's `sae`
block), with an index on `sae_ref->>hf_repo` for the missing-SAE lookup.
Migration `016_add_probe_monitors.py` (`revision="016"`, `down_revision="015"`) uses
`sa.JSON().with_variant(postgresql.JSONB(), "postgresql")` for JSON columns (SQLite-compatible tests).

**`probes`:**
- `id` String(24)
- `name` String(120)
- `definition` JSON
- identity columns (`hf_id`, `revision`, `d_model`, `n_layers`, `template_sha256`)
- read columns (`layer`, `rule`, `streamable`, `scope`)
- decision (`threshold`, `target_fpr`), `rung`, `definition_acknowledgement` JSON
- `parity` JSON, `armed` Boolean, `arm_acknowledgement` JSON, `paused_reason` String(64)
- `provenance` JSON
- timestamps

A unique index on `name`.

**`probe_events`:**
- `id`, `probe_id` FK CASCADE, `request_id` String(64)
- `scored` Boolean, `not_scored_reason` String(64)
- `score`, `threshold`, `verdict` Boolean, `rung`, `top_positions` JSON, `n_scored_tokens`
- `context_text` Text, `context_token_ids` JSON, `summary` String(300)
- `created_at` (timezone-aware)
- indexes `(probe_id, created_at)` and `request_id`

**The mirror** keeps unknown fields (`extra="allow"`), so additive v1 fields round-trip in
`definition`. The typed columns are a projection used for queries only.

## 5. API Design
Routes (§7 of the FPRD) use the `ApiResponse` envelope. Error classes:

| Class | code | status | Notes |
|---|---|---|---|
| `ProbeModelMismatchError` | `PROBE_MODEL_MISMATCH` | 409 | `details.mismatches=[{field, expected, actual}]` |
| `ProbeParityFailedError` | `PROBE_PARITY_FAILED` | 409 | `details={max_abs_diff, vector_index, tolerance, tokenization_drift}` |
| `UnvalidatedProbeError` | `UNVALIDATED_PROBE` | 200 (envelope refusal, circuit house style) | `{rung, rung_language, next_step}` |
| `ProbeLimitError` | `PROBE_LIMIT` | 409 | max armed |
| `ProbeNotFoundError` | `PROBE_NOT_FOUND` | 404 | |
| (reuse) `EngineUnsupportedError` | `ENGINE_UNSUPPORTED` | 400 | GGUF arm |
| (reuse) `ValidationError` / `UNKNOWN_KIND` / `PAYLOAD_TOO_LARGE` | | 422/413 | circuit import gates |

**Verdict header:**
`X-miLLM-Probe-Verdicts: "high-stakes";score=2.31;threshold=1.07;verdict=?1;rung=3, "x";not-scored;reason="speculative_decoding"`

**Final stream chunk:**
`{"id":…,"object":"chat.completion.chunk","created":…,"model":…,"choices":[],"millm_probe_verdicts":[{name,score,threshold,verdict,rung,rung_language}]}`

## 6. Admin UI Design
- `src/pages/ProbeMonitorsPage.tsx` (route `/probe-monitors`, sidebar item "Probe Monitors", icon
  `ScanSearch`).
- Components in `src/components/probes/`: `ProbeImportDialog` (file / hub), `ProbeList`,
  `ProbeArmDialog` (ack below rung 2), `ParityReport`, `ProbeEventsFeed`, `ProbeStatusCard`,
  `ProbeRungChip`.
- `src/hooks/useProbes.ts`: react-query keys `['probes', …]` and a `socketClient.on('probe:event')`
  subscription following `useSensing`.
- `src/services/api.ts` `probesApi`; `src/services/socket.ts` `SocketEventHandlers['probe:event']`;
  `src/types/probe.ts`.
- **The rename:** `Sidebar.tsx:33` label "Probe" → "Feature Monitor". Also `DashboardPage.tsx:153`,
  `QuickActions.tsx:73,192`, the manual page `probe-monitoring.md` title, and
  `e2e/navigation.spec.ts`. The route stays `/monitoring`.

## 7. Testing Strategy
- **Pure:** head math and accumulators (streaming equals batch); header formatting; chunk shape;
  identity comparison; parity comparison.
- **The hook:** fake models (`tests/unit/ml/test_sae_hooker.py` style). Assert prepend order (probe
  sees pre-steer values when an SAE steers the same layer), output unchanged, one hook per layer for
  several probes, removal.
- **Lifecycle wiring:** for each path (non-stream chat, stream chat, completions), assert
  `_probe_begin` and `_probe_finish` are called with the payload. CBM routing is forced serial;
  speculative decoding gives `not_scored` with the reason; n>1 gives `not_scored`; llama.cpp arm is
  refused. This is the `test_circuit_sensing_wiring.py` style.
- **Delivery:** the header is present iff armed; the stream chunk comes before `[DONE]` and agrees
  with the non-stream verdict.
- **Recording:** events and prune; socket payload stripped of `context_*` (a privacy test with a
  mutation control).
- **Parity:** pass, fail and drift cases with a tiny model and synthetic vectors, plus the real
  miStudio fixture `tests/fixtures/lfm2_probe_definition.json` (added at acceptance).
- **Sync:** the vendored schema is byte-identical with miStudio (cross-repo); the mirror is
  structurally complete (the circuit structural pattern); the rung language equals miStudio's.
- **Performance:** overhead at 4k tokens with 2 probes (the `test_edge_sensing_baseline.py` style).
- **MCP:** `test_mcp_contract_consistency.py` (the category heading requires miStudio's registry) and
  a probes variant of `test_mcp_tool_paths_are_real.py`.
- **Frontend:** vitest for the page and hooks; the rename; Playwright navigation.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Streaming verdict computed after `[DONE]` | `_probe_finish` is called explicitly before the final chunk (L~4243), not in `finally` |
| Prompt-scope mask differs from miStudio's | The same prefix-rendering algorithm; parity vectors include `messages`; a drift report |
| Hook order with same-layer SAE steering | `prepend=True`, pinned by a test |
| torch.compile recompiles | `_reset_dynamo_for_hook_change()` on arm and disarm, as for SAEs |
| Clients choke on the `choices: []` chunk | Spike OQ-3; opt-in header fallback |
| The existing duplicate definitions in `sockets/progress.py` (`emit_monitoring_state_changed`, `create_socket_io` twice) confuse the new emitter | Clean them up in this feature (touched code), with a test that each is defined once |

## 9. Decisions
FPRD R1–R8. Technical defaults:

| # | Question | Default |
|---|---|---|
| U1 | Where do the layer helpers live? | Lift `_get_layer` / `layer_device` into module functions in `sae_hooker.py`; `SAEHooker` delegates (no behaviour change) |
| U2 | How is the hub implemented? | Parameterise the cluster hub helpers (tag, suffix) rather than copying them |
| U3 | Where is the accumulator math? | `ml/probe_head.py`, with no service imports |
