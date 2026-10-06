# BRD-04 · miLLM — Dataworks Support

**Owner:** Human in the Stream, LLC
**Status:** Draft v0.1 · 2026-10-06
**Depends on:** none new.
**Needed by:**
- BRD-03 (miDataworks — Dataset Curation, Labeling and Publishing): R-03.18, R-03.22, R-03.26, R-03.27, R-03.28, R-03.29, R-03.35, R-03.52, R-03.57, R-03.64. BRD-03 §7.1 maps each addition to the requirement that needs it.
- BRD-MIS-DATAWORKS-001 (miStudio, written in parallel). It owns the Model Context Protocol (MCP) tools for miLLM scoring, generation and probe scoring. Audit item 13 belongs to that document. This BRD provides the HTTP surface those tools call.

**Existing application, incremental.**

**Sources:**
- `~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, decision 7: all 14 audited gaps except authentication are in version 1, tracked in a separate miLLM BRD;
- `~/app/miDataworks/0xcc/docs/brd-source-research-2026-10-05.md` §1, the gap audit;
- the humor labelling project in `~/app/miDataworks` (`PLAN-humor-labeling.md`, `records/`).

Code references are to miLLM at `61bed07`, re-checked on 2026-10-06. Where a line moved since the audit, the line given here is the current one.

---

## 1. Why this exists

On 2026-10-04 and 2026-10-05, miLLM served as a classifier for a real labelling job. JEV-9B-decision ran through the scoring mode added to `/v1/completions` on 2026-10-04. The main run labelled 25,000 rows at 19–21 rows per second (`records/labelling_run.md`). Every row went through a client loop of single HTTP requests.

It worked. It also showed what miDataworks will hit when it turns that job into a product:

- **Scoring exists on one endpoint only.** `/v1/completions` takes `logprobs` and `allowed_token_ids` (`millm/api/schemas/openai.py:237-241`). `/v1/chat/completions` takes neither, so a chat-format classifier must render its own template.
- **There is no job API.** A prompt list on `/v1/completions` runs one prompt after another inside one request (`inference_service.py:4963-4973`). A 50,000-row run is 50,000 requests, with no server-side progress, cancel or resume.
- **Nothing protects a job from a model swap.** A request naming another model loads it on demand (`api/routes/openai/completions.py:110-113`, `chat.py:143-146`). The only guard is `locked`, a bare boolean with no holder and no expiry (`millm/db/models/model.py:152`).
- **GPU memory was invisible.** After the run, miLLM held 17,396 MiB of the RTX 3090 with no model loaded (`PLAN-humor-labeling.md:199`). Only `nvidia-smi` on the node showed it. Commit `61bed07` fixed the leak. Nothing in miLLM's REST API would have reported it.
- **Probe verdicts on stored text are impossible.** Scoring mode records no verdicts by design (`inference_service.py:4934-4936`). Parity scores only a definition's own test vectors. So a detector operator cannot ask "what would this probe say about these 10,000 rows".

**The cross-cutting hazard is the silent drop.** Every `/v1` request schema, and the chat message schema, sets `extra="ignore"` (`openai.py:39, 198, 249, 296`). A client sending `response_format`, `seed`, chat `logprobs` or `top_logprobs` gets a 200 and an answer that ignored them. Nothing in the response says so. A labelling job built on that would record "structured output, seed 7" against rows produced with neither. The code already knows this is a trap: the batched-chat field carries a comment warning that an older server accepts it and silently returns one choice (`openai.py:87-89`), and `X-miLLM-Batch` exists only to make that observable (`chat.py:264-272`).

The audit also found a latent defect. `_create_batched_chat_completion` (`inference_service.py:3393-3572`) never calls `_probe_begin`. Armed probes are silently skipped on batched chat. Probe Monitors' own rule (BRD-MILLM-PROBES-001 BR-006) is that a probe never goes silently quiet.

## 2. Goal in one sentence

Make miLLM a dependable backend for offline labelling, steered generation and detector work: every request field is honoured or refused, every long job survives and reports, and every answer says how it was produced.

## 3. Target environment

- One GPU node: an RTX 3090 (24 GB) and an RTX 3080 Ti (12 GB), both visible to the miLLM pod.
- One model resident at a time. A load unloads the previous model first (`model_service.py:782-786`).
- `MAX_CONCURRENT_REQUESTS` is 1 and `MAX_PENDING_REQUESTS` is 10 (`millm/core/config.py:255-256`). Steering, sensing and probes all depend on serial execution through the request queue.
- The continuous batching manager (CBM) is off in Kubernetes: `ENABLE_CONTINUOUS_BATCHING` is `"false"` (`k8s/base/backend.yaml:114-115`).
- No user sign-in. miLLM runs on the local network, as miStudio and miDataworks do.
- Callers: miDataworks (directly over HTTP), miStudio's MCP server (BRD-MIS-DATAWORKS-001), Open WebUI, and scripts.

## 4. Scope

**In scope:**
- the 13 additions in BRD-03 §7.1, less item 13 (MCP tools, owned by miStudio);
- the unknown-field rule for every `/v1` request;
- the batched-chat probe-skip defect and a test that guards every generation path.

**Out of scope:**
- **Authentication** (audit item 12). Decision 7 excluded it.
- **Co-residency** of several models, and graphics processing unit (GPU) scheduling shared with miStudio's workers. This BRD provides a VRAM (video memory) read per card and a model lease. Coordination beyond that is a later effort.
- MCP tools. BRD-MIS-DATAWORKS-001 owns them.

## 5. Requirements

Requirement numbers are stable. Other repositories cite them. A requirement may be amended in place but is never renumbered or reused.

### 5.1 Request validation

**R-04.1** Every `/v1` endpoint reports the request fields it did not use. This covers top-level fields and fields inside `messages`. The response carries `X-miLLM-Ignored-Fields` listing them, and the server logs a warning. A field is never dropped without a trace.

*Why:* `extra="ignore"` on every request schema (`openai.py:39, 198, 249, 296`) makes an unimplemented field look implemented.

**R-04.2** A request sending `X-miLLM-Strict: true` gets `400` instead of a warning when any field would be ignored. The error names every such field. miDataworks sends this header on every request.

**R-04.3** A named list of output-changing fields is never ignored, strict or not. Each is honoured or refused with `400`. The list includes at least `response_format`, `seed`, `logprobs`, `top_logprobs`, `allowed_token_ids`, `n`, `dimensions`, `tools`, `tool_choice`, `logit_bias` and `steering`. The list lives in one place, and a test asserts every entry on every endpoint.

**R-04.4** `n` on `/v1/completions` is implemented or refused. Implemented means `n` choices per prompt, indexed as OpenAI indexes them. Until then, `n > 1` returns `400`.

*Why:* `create_text_completion` loops over prompts and never reads `n` (`inference_service.py:4815-4900`). The only read of `n` on that schema is the scoring-mode check (`openai.py:269-270`).

**R-04.5** `dimensions` on `/v1/embeddings` is honoured or refused. It is honoured only for a model whose metadata declares support for truncated embeddings. Otherwise the request returns `400`.

*Why:* the field is accepted (`openai.py:293`) and never read by `create_embeddings` (`inference_service.py:5071-5148`).

### 5.2 Scoring on chat completions

**R-04.6** `/v1/chat/completions` accepts `logprobs` (boolean), `top_logprobs` (0–20) and `allowed_token_ids`. miLLM renders the chat template with the generation prompt, then runs the same next-token scoring path `/v1/completions` uses (`_score_text_completion` and `next_token_scores`, `inference_service.py:4930-5030`). The rendered prompt's special tokens are not added a second time at tokenisation.

**R-04.7** Chat scoring has the same limits as completion scoring: `max_tokens` 1, `n` 1, no streaming, and temperature 0 or at least the existing floor (`openai.py:259-281`). A request outside them gets `400`. A GGUF model (llama.cpp's file format) is refused before any auto-load, as completions already does (`completions.py:86-94`).

**R-04.8** Chat scoring is unsteered. Every attached sparse autoencoder (SAE) is suppressed through `_unsteered` (`inference_service.py:1176-1199`). Probes and sensing record nothing for it.

**R-04.9** The response uses OpenAI's chat logprobs shape: `choices[].logprobs.content[]` with `token`, `logprob`, `bytes` and `top_logprobs[]`. `return_tokens_as_token_ids` behaves as it does on completions.

**R-04.10** A scoring request may carry `extra_messages`. Each conversation is scored, and the response holds one choice per conversation, `index` in input order.

### 5.3 Structured output

**R-04.11** `response_format` on `/v1/chat/completions` supports `{"type": "json_object"}` and `{"type": "json_schema", "json_schema": {...}}` on the transformers engine. It uses constrained decoding, so the output parses as JSON and validates against the schema.

**R-04.12** Where structured output is not supported, the request returns `400` naming `response_format` and the reason. Cases include a GGUF model (unless R-04.11 is extended to llama.cpp), a schema feature outside the supported subset, and the CBM path. When the answer is decidable from the model row, the refusal comes before any auto-load.

**R-04.13** A constrained generation stopped by `max_tokens` reports `finish_reason: "length"`. miLLM never returns truncated JSON as a complete answer. The response carries `X-miLLM-Constrained` naming the format applied.

### 5.4 Reproducibility

**R-04.14** `seed` is accepted on chat and text completions and applied to sampling. The same seed, request, model and batch shape give identical output on the serial transformers path. The response echoes the applied seed in `X-miLLM-Seed` and carries a `system_fingerprint` naming model, revision, precision and engine.

**R-04.15** Where a seed cannot promise identical output, the response says so. Batched rows are deterministic per batch shape only (`inference_service.py:3409-3420`). On llama.cpp the seed is forwarded to the engine, or the request is refused.

### 5.5 Batch API

**R-04.16** miLLM provides an OpenAI-shaped batch API. `POST /v1/files` uploads a JSON Lines (JSONL) file with `purpose: "batch"`. `POST /v1/batches` starts a batch from `input_file_id`, `endpoint` and `completion_window`. Each line holds `custom_id`, `method`, `url` and `body`. Supported endpoints are `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and `/api/probes/score` (5.7).

**R-04.17** Every line is validated against its endpoint's schema before any row runs. Strict mode (R-04.2) applies to every line. Invalid lines go to the error file with their line number and reason. A file with no valid line is refused.

**R-04.18** Batches are persisted in PostgreSQL: status, request counts, input, output and error files. Status values follow OpenAI: `validating`, `in_progress`, `finalizing`, `completed`, `failed`, `cancelling`, `cancelled`, `expired`. A batch survives a pod restart and resumes from the first row without a recorded result. No recorded row runs twice.

*Why:* in this suite a rollout takes in-flight GPU work with it, and nothing requeues it. Two hour-long miStudio runs were lost that way.

**R-04.19** Batch rows reach the model only through `_admit()` (`inference_service.py:649-690`), the single admission path. A batch takes one slot per chunk and releases it between chunks, so interactive requests interleave and are not starved. A batch does not count against `MAX_PENDING_REQUESTS`.

**R-04.20** Scoring and embedding rows may be packed into one padded forward pass. Packing is on by default. A batch may set `pack: false` to get single-row semantics. The packed-versus-single difference is measured on the reference model and stated in the API reference.

*Why:* bfloat16 is not batch-invariant. miStudio measured batched against one-at-a-time probe scores differing by up to 0.177 (miStudio `0xcc/reviews/native_dtype_2026-10-03.md`).

**R-04.21** `GET /v1/batches/{id}` reports request counts. Progress is also emitted on a Socket.IO event. `POST /v1/batches/{id}/cancel` stops at the next row boundary and keeps completed rows in the output file. `GET /v1/batches` lists batches. `GET /v1/files/{id}/content` returns a file.

**R-04.22** A batch names one model. It takes the model lease (5.10) for its whole run, refuses to start if another holder has the lease, and never makes miLLM load or swap a model.

**R-04.23** Limits on rows per batch and bytes per file are configurable, with stated defaults. A file over a limit is refused at upload, not truncated.

### 5.6 Per-request SAE activations

**R-04.24** Chat and text completions accept `return_sae_activations: {sae_id?, features?, top_k, positions}`. `positions` is `last`, `prompt`, `completion`, `all` or an index range. The response body returns this request's activations only, keyed by position, under a `millm` extension object.

*Why:* today the only per-request read is a shared history buffer of 100–1000 entries (`api/schemas/monitoring.py:22-27`, `monitoring_service.py:122`), which keeps the last position only (`monitoring_service.py:286-289`). Under load a caller's entries are evicted by others, and a filter by `request_id` finds nothing.

**R-04.25** The request needs an attached SAE matching `sae_id`, or it is refused, naming the SAE. The response states whether the activations were read before or after steering. A request whose positions × `top_k` exceed a configured cap is refused with `400`.

**R-04.26** `return_sae_activations` works in scoring mode, so stored text can be tagged with features without generating anything.

### 5.7 Stateless probe scoring

**R-04.27** `POST /api/probes/score` takes `{probe_ids?, inputs, windows?}`. Each input is `token_ids`, `messages` or `text`; `token_ids` is authoritative when given. It returns, per input, probe and window: score, threshold, verdict, evidence rung and the provisional flag. It generates nothing, writes no probe event and changes no armed state.

**R-04.28** It works on any imported probe, armed or not. It reuses the parity forward (`build_parity_forward`, `probe_arm_bridge.py:119`) and the armed-probe construction parity uses (`api/routes/management/probes.py:405-415`). The decision applies the same window bars and length bands as live scoring. A probe whose identity check fails against the loaded model is refused, naming the mismatch, as arming refuses it (BRD-MILLM-PROBES-001 BR-001). A GGUF model is refused.

**R-04.29** Scoring takes a request slot through `_admit()`. Probes on the same layer share one forward pass per input.

*Why:* the existing parity route runs its forward pass with no request slot (`probes.py:385-418` contains no `_admit`), so it can overlap a generation. The new endpoint must not copy that. The parity route is brought under `_admit()` too.

**R-04.30** Offline scoring needs no global arming and forces no other traffic onto the serial path. This closes audit item 6. Today arming is global, and any armed probe pushes every request off continuous batching (`inference_service.py:1038-1046`).

### 5.8 Inline steering

**R-04.31** Chat and text completions accept `steering: {sae_id?, features: [{index, strength}]}`. It applies to this request only, inside the admission slot, and is restored afterwards. It follows the lifecycle `_apply_request_steering` and `_restore_request_profile` already use (`inference_service.py:1932, 2149`). No saved profile is created. The named SAE must be attached, or the request is refused.

**R-04.32** `steering` and `profile` are mutually exclusive. `steering: {"features": []}` means explicitly unsteered: every attached SAE is suppressed for the request.

**R-04.33** Every generation response states its steering state in `X-miLLM-Steering`: none, a profile with its intensity, inline steering with a feature count and hash, or a circuit. A non-streaming response sets it after generation, as `X-miLLM-Circuit-Rung` already is (`chat.py:287-288`). A streaming response carries it in a final Server-Sent Events (SSE) chunk. Batch output lines carry it in the body.

*Why:* today only `X-miLLM-Steering-Intensity` exists, and only when the request sent a dial (`chat.py:212-222`). A request steered by a globally active profile says nothing. BRD-03 R-03.35 must check each steered pair against what miLLM reports.

**R-04.34** `/v1/completions` gains `profile`, `steering_intensity` and `steering`. Today a text completion runs under whatever profile is active, with no way to opt out. `_has_steering_override` treats text completions as having no steering fields (`inference_service.py:945-956`).

### 5.9 Embeddings

**R-04.35** `/v1/embeddings` accepts `pooling`: `mean` (the default, as today), `last` or `cls`. It accepts `normalize` (default false, as today).

*Why:* the pool is a fixed mean over the last hidden layer (`inference_service.py:5123-5124`).

**R-04.36** An input longer than the model's limit returns `400` naming its index. It is never truncated silently. The number of inputs per request is capped, with a stated default.

*Why:* inputs are tokenised with `truncation=True` (`inference_service.py:5108-5110`). Where the tokenizer has a `model_max_length`, long input is cut without notice.

**R-04.37** The route's comments match its code. `embeddings.py:64-73` describes a GGUF guard the route does not contain. GGUF embeddings are in fact served (`_llamacpp_embeddings`, `inference_service.py:5150`).

### 5.10 Model lease

**R-04.38** `POST /api/models/{id}/lease` takes `{holder, ttl_seconds, reason}` and returns a lease ID. One lease exists at a time. The holder can renew and release it. `GET` reports holder, reason and expiry. A lease expires on its own at its time to live (TTL).

*Why:* `locked` has no owner and no expiry (`db/models/model.py:152`). A crashed holder leaves it set, and nobody can tell whose it is.

**R-04.39** While a model is leased, a load, unload or swap by anyone else is refused with `409 MODEL_LEASED`, naming holder and expiry. This covers the auto-load in every `/v1` route (`chat.py:143-146`, `completions.py:112-115`, `embeddings.py:81-84`) and `POST /api/models/{id}/load` and `/unload` (`management/models.py:153, 178`).

**R-04.40** The holder sends `X-miLLM-Lease: <id>`. Its own requests and model operations proceed.

**R-04.41** A request sending `X-miLLM-Load-Policy: refuse` fails with `409` instead of auto-loading a model that is not resident. The default stays auto-load, which Open WebUI relies on.

**R-04.42** Lease state appears in `/api/health/detailed` and on the Admin UI's model page.

### 5.11 Backpressure

**R-04.43** Every `503` carries `Retry-After` in seconds. This covers `QUEUE_FULL`, `MODEL_BUSY`, `MODEL_LOADING`, `MODEL_NOT_LOADED` and `INSUFFICIENT_MEMORY` (`api/routes/openai/errors.py:76-111`). The OpenAI error envelope and codes stay as they are.

*Why:* no `Retry-After` is set anywhere in `millm/`. BRD-03 R-03.29 must honour it.

**R-04.44** Queue depth is already in `/api/health/detailed` (`queue_pending`, `queue_max_pending`; `health.py:357-367`). This BRD adds the in-flight count, the batch backlog in rows and an estimated wait. These fields are documented as a stable contract.

### 5.12 GPU visibility

**R-04.45** A REST endpoint returns, per card: index, UUID, name, total, used and free memory in MiB, and miLLM's own allocated and reserved memory on that card.

*Why:* per-card figures exist only on the `system:metrics` Socket.IO stream (`sockets/progress.py:29-56, 107-121`) and in the loaded model's placement (`health.py:265-279`). Neither can show miLLM holding 17,396 MiB with no model loaded, which is what happened on 2026-10-05.

### 5.13 Probes on every generation path

**R-04.46** `_create_batched_chat_completion` opens a probe context. Until it can score each row, it marks the request not scored with reason `batched_request`, as the `n > 1` path does (`inference_service.py:3641-3648`). The verdict header and probe status then say why no verdict exists.

*Why:* the batched path never calls `_probe_begin` (`inference_service.py:3393-3572`). Armed probes are skipped with no trace.

**R-04.47** A test enumerates every generation entry point: serial, streaming and batched chat, text completion, the CBM chat, streaming and text paths, and the llama.cpp paths. It fails if any reaches generation with a probe armed and neither a probe context nor a recorded not-scored reason. The test discovers entry points from the service; a hand-kept list does not count.

*Why:* the CBM non-streaming chat and text paths (`inference_service.py:5230, 5418`) also never call `_probe_begin`. They are safe today only because `PROBE_FORCE_SERIAL` routes armed traffic away from CBM (`inference_service.py:1041-1046`). The setting's own comment says not to rely on that (`config.py:177-179`).

## 6. Acceptance criteria

Each wiring item is accepted only by a test that fails when its registration or call line is removed. The test asserts the payload and the call count, not only that a call happened.

1. **Unknown fields.** A chat request with `foo: 1` returns `X-miLLM-Ignored-Fields: foo`. The same request with `X-miLLM-Strict: true` returns `400` naming `foo`. Every R-04.3 field, sent where unimplemented, returns `400` with or without strict mode.
2. **`n` and `dimensions`.** `n: 2` on `/v1/completions` returns two choices per prompt or `400`, never one. `dimensions` on a model without declared support returns `400`.
3. **Chat scoring parity (hardware, JEV-9B-decision, bfloat16).** For 200 prompts, chat scoring of `messages` and completion scoring of the same template-rendered prompt get identical token IDs. Every reported logprob agrees within 1e-5 absolute. The chat path calls the shared scoring function, asserted by test.
4. **Chat scoring is unsteered.** With a profile active, chat scoring returns the same logprobs as with no SAE attached.
5. **Structured output.** 100 `json_schema` requests on the transformers engine all parse and validate. The same request on a GGUF model returns `400` before any model load.
6. **Seed.** The same sampled request with `seed: 7` run twice on the serial path gives byte-identical text, and the response echoes the seed.
7. **Batch (hardware).** A 10,000-row JEV-9B-decision scoring batch completes. Progress is visible throughout. Unpacked throughput is at least 19 rows per second, the client-loop rate in `records/labelling_run.md`. The packed rate and the measured packed-versus-single difference are recorded.
8. **Batch cancel and resume.** Cancel at about 3,000 rows keeps every completed row and runs none twice. A pod restart during a second batch resumes it with no recorded row repeated.
9. **Batch and chat interleave.** A chat request sent during a running batch is answered within one chunk's duration.
10. **SAE activations.** Two interleaved requests each get only their own activations. A request over the size cap returns `400`.
11. **Probe scoring (hardware).** On LFM2.5-1.2B, `/api/probes/score` reproduces a probe definition's test vectors within the parity tolerance, unarmed. Arming the probe and scoring the same inputs live gives the same scores. No probe event row is written.
12. **Inline steering.** Inline steering and a saved profile with the same features give identical output. `X-miLLM-Steering` is correct for none, profile, inline and circuit, streaming and not.
13. **Embeddings.** An over-limit input returns `400` naming its index. `pooling: last` with `normalize: true` returns unit vectors.
14. **Lease (hardware).** With a lease held by `midataworks`, a chat request naming another model gets `409 MODEL_LEASED`. So does `POST /api/models/{other}/load` without the lease ID. After the TTL passes without renewal, the same load succeeds.
15. **Backpressure.** Eleven concurrent requests produce a `503 queue_full` carrying `Retry-After`.
16. **GPU visibility (hardware).** After a model is unloaded, the per-card endpoint shows miLLM's reserved memory on each card, and it matches `nvidia-smi` within 256 MiB.
17. **Probe paths.** With a probe armed, a batched chat request records reason `batched_request`. Removing the R-04.46 call turns the R-04.47 test red.

## 7. Non-goals

- Authentication and per-caller identity on `/v1` (decision 7).
- More than one resident model, or a GPU scheduler shared with miStudio's workers.
- MCP tools (BRD-MIS-DATAWORKS-001).
- Raising `MAX_CONCURRENT_REQUESTS` above 1.
- Re-enabling continuous batching in production.
- Probe scoring of each row inside a batched chat request. R-04.46 records the gap honestly.
- Function calling (`tools`). R-04.3 refuses it; it is not implemented here.

## 8. Risks

| ID | Risk | Effect | Mitigation |
|---|---|---|---|
| RSK-01 | Strict refusal breaks clients that send harmless extra fields, such as Open WebUI | Chat UI stops working | Warning by default; strict only on request (R-04.1, R-04.2) |
| RSK-02 | Packed scoring differs from single-row scoring in bfloat16 | Labels move with batch composition | `pack: false`; measured difference published (R-04.20) |
| RSK-03 | A long batch starves interactive chat | Open WebUI stalls during labelling | Slot released between chunks (R-04.19); acceptance item 9 |
| RSK-04 | A stale lease blocks every model change | Operator cannot load a model | TTL with automatic expiry; holder and expiry shown (R-04.38, R-04.42) |
| RSK-05 | Constrained decoding is slow or unsupported on some tokenizers | Structured requests time out or fail | Refuse with `400` naming the reason (R-04.12) |
| RSK-06 | Batch persistence fills the data volume | Disk pressure on the GPU node | Row and byte limits (R-04.23); retention to be set |
| RSK-07 | Stateless probe scoring drifts from live scoring | Detector operators report scores miLLM would not record live | Same construction and decision code as arming; acceptance item 11 |
| RSK-08 | A new generation path skips probes, as batched chat did | A probe goes quiet unnoticed | Discovery-based test over every entry point (R-04.47) |
| RSK-09 | Thirteen additions ship as one release | Long time to first use | Order by BRD-03's phase needs: scoring, lease, backpressure and unknown fields first; batch API next |

## 9. Open questions

1. Should the `locked` flag be replaced by the lease, or kept for steering alongside it?
2. What packed-versus-single difference is acceptable on JEV-9B-decision before packing must default to off?
3. Structured output on GGUF: implement through llama.cpp grammars in version 1, or refuse?
4. Batch file retention: how long are input, output and error files kept on `/data`?
5. Should `return_sae_activations` default to pre-steering or post-steering activations?
6. Maximum lease TTL, and whether miStudio's GPU workers should take the lease too. That second part edges into co-residency, which is out of scope here.

**Status (Stage 3, 2026-10-06):** all six are closed by the checkpoint decisions of 2026-10-06 (`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`) and the register's defaults. The answers are recorded in PPRD v1.5, "BRD-04 Coverage". Question 3 is answered "refuse": structured output on GGUF is refused in v1. The question text above is kept unchanged, as requirement numbers are.
