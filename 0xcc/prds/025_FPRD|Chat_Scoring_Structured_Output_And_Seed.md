# Feature PRD: Chat Scoring, Structured Output, Seed and Request Validation

**Document ID:** 025_FPRD|Chat_Scoring_Structured_Output_And_Seed
**Version:** 1.0 (planned)
**Status:** Planned. Feature PRD written 2026-10-06; FTDD, FTID and FTASKS follow.
**Source:** BRD-04 (miLLM — Dataworks Support) §5.1–§5.4: R-04.1–R-04.4 and R-04.6–R-04.15.
R-04.5 (`dimensions`) moved to Feature 30; its *refusal* path is built here (FR-25.3).
**PPRD:** Feature 25 (FR-25.1 – FR-25.14), PPRD v1.5 · **PADR:** v1.5 §1 rows "Request validation
(v1.5)" and "Chat scoring (v1.5)"; v1.5 §10 "Dataworks Support (Features 25–30)" trade-offs
**Binding decisions:** checkpoint technical defaults of 2026-10-06
(`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, "Checkpoint decisions")
**Depends on:** none new. Reuses `_score_text_completion`, `next_token_scores` and `_unsteered`;
Feature 23 (GGUF refused before auto-load).
**Co-release:** miStudio BRD-MIS-DATAWORKS-001 tools `millm_score_chat` and `millm_generate`
(miStudio file `034_*`). Consumers: miDataworks BRD-03 R-03.22 (classifier protocols, chat
scoring) and R-03.28 (judge controls: recorded seed, structured output).

Code references are to miLLM at `7aa659c` (HEAD, 2026-10-06).

---

## 1. Feature Overview

**Name:** Chat Scoring, Structured Output, Seed and Request Validation.

**What it is:** four changes to the OpenAI-compatible `/v1` surface.

1. **Request validation.** Every `/v1` request reports the fields miLLM did not use. A strict mode
   refuses them. A named list of output-changing fields is always honoured or refused.
2. **Chat scoring.** `/v1/chat/completions` returns next-token log-probabilities. It renders the
   chat template, then calls the scorer `/v1/completions` already uses.
3. **Structured output.** `response_format` (`json_object`, `json_schema`) constrains generation on
   the transformers engine. Everywhere else it is refused with a reason.
4. **Reproducibility.** `seed` is applied and echoed. Every response carries a
   `system_fingerprint`. Where a seed cannot promise identical output, the response says so.

**Problem.** Every `/v1` request schema sets `extra="ignore"` (`millm/api/schemas/openai.py:39`,
`:198`, `:249`, `:296`). A client sending `response_format`, `seed` or chat `logprobs` gets a 200.
The answer silently ignored them, and nothing in the response says so. A labelling job built on
that would record "structured output, seed 7" against rows produced with neither (BRD-04 §1).

Scoring exists on `/v1/completions` only (`openai.py:237-247`). A chat-format classifier must
therefore render its own chat template. miDataworks' real client does exactly that today: it hand-
writes the JEV-9B prompt and calls `/v1/completions`
(`~/app/miDataworks/scripts/jev_client.py:27-29`, `:62-67`).

**Goals:**
- No request field is dropped without a trace.
- A field that changes the output is never ignored, on any endpoint or engine.
- Chat scoring and completion scoring of the same rendered prompt agree by construction.
- Structured output parses and validates, or the request is refused, or the response says it was
  cut short.
- A seeded request is reproducible on the serial path, and the response states the scope of that
  promise.

**Connection to the project.** This is the first feature of the Dataworks Support increment
(BRD-04 RSK-09). Its rule — honoured or refused, never ignored — is the increment's rule. Features
26–30 add fields and endpoints that must obey it.

## 2. User Stories & Scenarios

**US-1: a labelling job is never misled (miDataworks).** miDataworks sends `X-miLLM-Strict: true`
on every request (R-04.2). It sends a field miLLM does not know.
*Acceptance:* the request returns `400` naming the field. Without the header, the response carries
`X-miLLM-Ignored-Fields` naming it, and the server logs a warning (BRD-04 acceptance 1).

**US-2: an output-changing field is never ignored (any client).** A client sends `tools` to
`/v1/chat/completions`, or `n: 3` to `/v1/completions`.
*Acceptance:* the request is honoured or returns `400` naming the field, with or without strict
mode. It never returns a 200 that ignored it (R-04.3, R-04.4).

**US-3: score a chat-format classifier (miDataworks, miStudio MCP).** A client sends `messages`,
`logprobs: true`, `top_logprobs: 2` and `allowed_token_ids` for the answer tokens.
*Acceptance:* the response's `choices[0].logprobs.content[0]` holds the next-token log-probabilities
over the allowed set. They equal completion scoring of the same rendered prompt: identical token IDs
and logprobs within 1e-5 absolute (BRD-04 acceptance 3).

**US-4: score many conversations in one request.** The client adds `extra_messages`.
*Acceptance:* one choice per conversation, `index` in input order (R-04.10).

**US-5: scoring ignores steering.** A steering profile is active while a client scores.
*Acceptance:* logprobs equal those with no SAE attached (BRD-04 acceptance 4).

**US-6: structured judge output.** A client sends `response_format: {"type": "json_schema", ...}`.
*Acceptance:* the output parses and validates against the schema. `X-miLLM-Constrained` names the
format applied. If `max_tokens` cut it short, `finish_reason` is `"length"` (R-04.11, R-04.13).

**US-7: reproducible sampling.** A client sends the same sampled request twice with `seed: 7`.
*Acceptance:* byte-identical text on the serial path. The response echoes `X-miLLM-Seed` and carries
a `system_fingerprint` (BRD-04 acceptance 6).

**Edge cases and error scenarios:**
- **GGUF model and scoring or structured output** → `400` before any auto-load (R-04.7, R-04.12;
  checkpoint default "structured output on GGUF models is refused in v1").
- **Scoring with `max_tokens` ≠ 1, `n` ≠ 1, `stream: true`, or a temperature below the floor** →
  `400` (R-04.7).
- **`top_logprobs` without `logprobs: true`** → `400` (OpenAI's rule; FR-25.5.3).
- **Scoring with `profile`, `steering_intensity` or `steering`** → `400`. Scoring is unsteered, so a
  steering field would be silently ignored (R-04.3, R-04.8).
- **A schema keyword outside the supported subset** → `400` naming the keyword (R-04.12).
- **`response_format` with scoring fields** → `400`. Scoring returns one token, not a document.
- **Streaming chat with `n > 1` or `extra_messages`** → `400`. Today the streaming path reads neither
  field (FR-25.3.5).
- **A seed on batched rows** → honoured, with the scope "batch shape" stated (R-04.15).
- **Strict mode and an unknown field inside a message** → `400` naming `messages[i].<field>`.

**User journey (miDataworks labelling run):**
1. Send each row as a chat scoring request with `X-miLLM-Strict: true`.
2. Read P(answer) from `logprobs.content[0].top_logprobs`.
3. Record `system_fingerprint` and the headers against the row.
4. Any `400` stops the job with the named field, before rows are produced with the wrong settings.

## 3. Functional Requirements

Each FR-25.x below is the PPRD v1.5 requirement. The numbered items under it refine it into
testable statements. "Refused" always means HTTP `400` with the OpenAI error envelope
(`millm/api/routes/openai/errors.py:246-254`), `param` naming the field.

### 3.1 Request validation

**FR-25.1 Report every unused field.** Every `/v1` endpoint SHALL report the request fields it did
not use, top-level and inside `messages`, in an `X-miLLM-Ignored-Fields` response header and a
logged warning. A field SHALL never be dropped without a trace. (R-04.1)

- **FR-25.1.1** The endpoints covered are `/v1/chat/completions`, `/v1/completions` and
  `/v1/embeddings`. `GET /v1/models` takes no body. Any `/v1` endpoint added later (Feature 26's
  `/v1/files` and `/v1/batches`) SHALL use the same mechanism.
- **FR-25.1.2** A field is *unused* when (a) the endpoint's schema does not declare it, or (b) the
  schema declares it but the engine path serving the request does not consume it. Case (b) exists
  today: on llama.cpp, `chat_template_kwargs` is logged as ignored at `info` level
  (`millm/services/inference_service.py:3911-3920`) and the response says nothing.
- **FR-25.1.3** Detection covers top-level fields, each message in `messages`, and each message in
  each `extra_messages` conversation.
- **FR-25.1.4** Each header entry names the field's location: `foo`, `messages[2].name`,
  `extra_messages[0][1].name`. Entries are comma-separated, in request order, with no duplicates.
- **FR-25.1.5** The header is absent when no field was ignored. It is never sent empty.
- **FR-25.1.6** The header SHALL be bounded in size. If entries are dropped to fit, the header
  SHALL say how many were dropped. The bound is set in the design.
- **FR-25.1.7** On a streaming response, the header is sent with the stream's headers. Detection
  completes before the first byte.
- **FR-25.1.8** The warning is one structured log event per request. It carries the endpoint, the
  request id and the field locations. It SHALL NOT carry field *values*, which may contain prompt
  text.
- **FR-25.1.9** An unused field never changes the response body. Reporting is additive.

**FR-25.2 Strict mode.** A request sending `X-miLLM-Strict: true` SHALL get `400` instead of a
warning when any field would be ignored, naming every such field. (R-04.2)

- **FR-25.2.1** The error message lists every unused field location, not only the first.
- **FR-25.2.2** Header values are case-insensitive. `true` and `1` enable strict mode. Any other
  value, or no header, leaves it off. An unrecognised value SHALL NOT silently mean strict.
- **FR-25.2.3** The refusal happens before any auto-load whenever it can be decided from the
  request and the model row. Case (b) of FR-25.1.2 is decidable from the row (`gguf_files`,
  `millm/db/models/model.py:126`).
- **FR-25.2.4** Default behaviour is unchanged: no header means report, not refuse (checkpoint
  default "miLLM reports unknown request fields by default and refuses them in a strict mode";
  PADR §10 "Report-then-opt-in refusal"; BRD-04 RSK-01).

**FR-25.3 Output-changing fields are never ignored.** A single named list of output-changing fields
— at least `response_format`, `seed`, `logprobs`, `top_logprobs`, `allowed_token_ids`, `n`,
`dimensions`, `tools`, `tool_choice`, `logit_bias` and `steering` — SHALL never be ignored, strict or
not. Each SHALL be honoured or refused with `400`. A test SHALL assert every entry on every endpoint.
(R-04.3)

- **FR-25.3.1** The list lives in one module. Every endpoint and engine path reads it from there.
  No endpoint keeps its own copy.
- **FR-25.3.2** For each (field, endpoint, engine path) the list records one outcome: *honoured* or
  *refused*. A refusal names the field and the reason.
- **FR-25.3.3** The v1 outcomes are:

  | Field | Chat (transformers) | Chat (llama.cpp) | Completions | Embeddings |
  |---|---|---|---|---|
  | `logprobs`, `top_logprobs`, `allowed_token_ids` | honoured in scoring mode (FR-25.5) | refused | honoured (`logprobs`, `allowed_token_ids`); `top_logprobs` refused | refused |
  | `response_format` | honoured (FR-25.10) | refused | refused | refused |
  | `seed` | honoured (FR-25.13) | per FR-25.14.3 | honoured | refused |
  | `n` | honoured, non-streaming | refused for `n > 1` (existing, `inference_service.py:3921-3931`) | per FR-25.4 | refused |
  | `dimensions` | refused | refused | refused | refused until Feature 30 |
  | `steering` | refused until Feature 28 | refused | refused until Feature 28 | refused |
  | `tools`, `tool_choice` | refused | refused | refused | refused |
  | `logit_bias` | refused | refused | refused | refused |

- **FR-25.3.4** A refusal fires even when the value is the field's neutral value, except where the
  OpenAI default is explicitly neutral: `n: 1`, `logprobs: false`, `response_format: {"type":
  "text"}`, `tools: []`, `logit_bias: {}`. A neutral value is honoured as "no change".
- **FR-25.3.5** Latent defects found while writing this PRD are in scope. `stream_chat_completion`
  (`inference_service.py:4288`) reads neither `n` nor `extra_messages`, so a streaming request with
  `n: 3` returns one choice today. Streaming with `n > 1` or `extra_messages` SHALL be refused.
- **FR-25.3.6** On the continuous batching manager (CBM) path, a request carrying a listed field
  the CBM cannot honour SHALL either be routed to the serial path or refused. It SHALL never be
  served by the CBM with the field dropped. The existing CBM gate already routes differing sampling
  parameters to serial (`inference_service.py:958-982`).
- **FR-25.3.7** Features 28 and 30 replace the `steering` and `dimensions` outcomes. Until each
  ships, the outcome is *refused*.
- **FR-25.3.8** The list is checked before any auto-load whenever the outcome depends only on the
  request and the model row.

**FR-25.4 `n` on `/v1/completions`.** `n` SHALL either produce `n` choices per prompt, indexed as
OpenAI indexes them, or return `400` for `n > 1`. (R-04.4)

- **FR-25.4.1** Today `create_text_completion` never reads `n` (`inference_service.py:4780-4928`).
  The only read is the scoring-mode check (`openai.py:269-270`). This silent drop SHALL end.
- **FR-25.4.2** If implemented: a list of P prompts with `n = N` returns P × N choices. Choice
  `index` = prompt position × N + completion position, as OpenAI indexes them.
- **FR-25.4.3** If refused: `n > 1` returns `400` naming `n`, on every engine path, before any
  auto-load.
- **FR-25.4.4** Which of the two ships in v1 is Open Question 2.

### 3.2 Scoring on chat completions

**FR-25.5 Chat scoring through the shared path.** `/v1/chat/completions` SHALL accept `logprobs`,
`top_logprobs` (0–20) and `allowed_token_ids`, render the chat template with the generation prompt,
and score through the same next-token scoring path `/v1/completions` uses, without adding the
rendered prompt's special tokens a second time. (R-04.6)

- **FR-25.5.1** `logprobs` is a boolean (OpenAI chat semantics). `top_logprobs` is an integer 0–20.
  `allowed_token_ids` is a list of 1–1024 non-negative ids, as on completions (`openai.py:241`).
- **FR-25.5.2** Scoring mode is on when `logprobs` is `true` or `allowed_token_ids` is present.
  `logprobs: false` alone is ordinary generation.
- **FR-25.5.3** `top_logprobs` without `logprobs: true` is refused, naming `top_logprobs`.
- **FR-25.5.4** The prompt is rendered by the same function generation uses
  (`_format_chat_messages`, `inference_service.py:5580`) with `add_generation_prompt=True`. It
  honours `chat_template_kwargs`, and fails loudly when a template rejects them
  (`inference_service.py:5636-5640`).
- **FR-25.5.5** The rendered text is tokenised with `add_special_tokens=False`, because a rendered
  template already carries its beginning-of-sequence (BOS) token (PADR §10 "One scoring path").
- **FR-25.5.6** Scoring then runs the same per-prompt forward and arithmetic as completion scoring:
  `_unsteered_next_token_logits` (`inference_service.py:5037`) and `next_token_scores`
  (`millm/services/next_token_scores.py`). The same checks apply: empty prompt, context length,
  out-of-vocabulary ids, non-finite logits (`inference_service.py:4964-5007`).
- **FR-25.5.7** The chat path SHALL call the shared scoring code, not a copy. A test SHALL assert
  the call, its arguments and its call count (BRD-04 acceptance 3; global reachability rule).
- **FR-25.5.8** Scoring is routed **first** in `create_chat_completion`, before the llama.cpp,
  batched (`extra_messages`) and CBM branches (`inference_service.py:3595-3610`). This mirrors
  `create_text_completion` (`inference_service.py:4795-4796`). Otherwise an `extra_messages`
  scoring request would be generated by the batched path with its scores dropped.
- **FR-25.5.9** If the loaded model has no chat template, the request is handled per Open
  Question 1. Today `_format_chat_messages` falls back to a generic Gemma-style format
  (`inference_service.py:5653-5679`), which would score a prompt the model was never trained on.

**FR-25.6 Chat scoring limits.** Chat scoring SHALL carry the completion-scoring limits (`max_tokens`
1, `n` 1, no streaming, the temperature floor) and SHALL refuse a GGUF model before any auto-load.
(R-04.7)

- **FR-25.6.1** The limits are those of `validate_scoring_mode` (`openai.py:258-282`): `max_tokens`
  must be 1, `n` must be 1, and temperature must be 0 or at least `MIN_SCORING_TEMPERATURE` (1e-3,
  `openai.py:209`). Chat adds: `stream` must be false.
- **FR-25.6.2** The limits are enforced at schema validation, so they hold on every engine path.
- **FR-25.6.3** A GGUF model row is refused in the route before auto-load, as completions does
  (`millm/api/routes/openai/completions.py:86-94`). A llama.cpp engine already resident is refused
  in the service, as completion scoring does (`inference_service.py:4945-4949`).
- **FR-25.6.4** `response_format` with scoring is refused (FR-25.3.3).

**FR-25.7 Chat scoring is unsteered and unrecorded.** Every attached sparse autoencoder (SAE) SHALL
be suppressed, and probes and sensing SHALL record nothing for it. (R-04.8)

- **FR-25.7.1** Suppression uses `_unsteered` (`inference_service.py:1176-1199`), entered in the
  worker thread that runs the forward pass, covering every attached SAE.
- **FR-25.7.2** `profile`, `steering_intensity` or `steering` on a scoring request is refused, naming
  the field. Scoring cannot honour it, and ignoring it is the silent drop R-04.3 forbids.
- **FR-25.7.3** No probe context, sensing context or circuit-sensing context is opened. No probe
  event or sensing event is written.
- **FR-25.7.4** Feature 27's discovery test (FR-27.9) must be able to classify chat scoring as a
  path that never reaches generation. This feature SHALL keep scoring free of any `generate()` call
  so that classification holds.

**FR-25.8 OpenAI's chat logprobs shape.** The response SHALL use `choices[].logprobs.content[]` with
`token`, `logprob`, `bytes`, `top_logprobs[]`, and `return_tokens_as_token_ids` SHALL behave as on
completions. (R-04.9)

- **FR-25.8.1** `content` holds exactly one entry: the scored next token.
- **FR-25.8.2** `token` is the chosen token, decoded. With `return_tokens_as_token_ids: true` it is
  `"token_id:<id>"`, the key completions uses (`inference_service.py:4959-4962`).
- **FR-25.8.3** `logprob` is the chosen token's log-probability, renormalised over
  `allowed_token_ids` when given (vLLM's `processed_logprobs` semantics, `next_token_scores.py:1-14`).
- **FR-25.8.4** `top_logprobs` holds `min(top_logprobs, candidates)` entries, most probable first,
  each with `token`, `logprob` and `bytes`. With `top_logprobs: 0` or absent it is an empty list.
- **FR-25.8.5** `bytes` is the UTF-8 encoding of the decoded token text, a list of integers, in every
  case. The id form changes only `token`.
- **FR-25.8.6** `message.role` is `assistant`. `message.content` is the decoded chosen token.
  `finish_reason` is `"length"`, as completion scoring reports (`inference_service.py:5011`).
- **FR-25.8.7** `allowed_token_ids` without `logprobs: true` returns `logprobs: null` and still
  constrains the chosen token, as completions does (`inference_service.py:5012-5014`).
- **FR-25.8.8** `usage.prompt_tokens` is the sum over conversations. `usage.completion_tokens` is the
  number of choices.
- **FR-25.8.9** The field `return_tokens_as_token_ids` is added to the chat schema.

**FR-25.9 One choice per conversation.** A scoring request carrying `extra_messages` SHALL score each
conversation and return one choice per conversation, `index` in input order. (R-04.10)

- **FR-25.9.1** Index 0 is `messages`; index i is `extra_messages[i-1]`, as batched generation
  numbers them (`openai.py:84-86`).
- **FR-25.9.2** Conversations are scored **one at a time**, one forward pass each, in input order.
  This matches completion scoring's per-prompt loop and the checkpoint default "probe scoring runs
  one input at a time". bfloat16 is not batch-invariant (PADR §10 "Packed scoring"), so packing
  would break FR-25.5's equality with completion scoring. Packing belongs to Feature 26.
- **FR-25.9.3** The whole request holds one admission slot (`_admit`, `inference_service.py:649`).
- **FR-25.9.4** One failing conversation (empty render, context length, non-finite logits) fails the
  request with `400`, naming its index. No partial response is returned.
- **FR-25.9.5** `X-miLLM-Batch` reports the number of choices, as for batched generation
  (`millm/api/routes/openai/chat.py:270-272`).

### 3.3 Structured output

**FR-25.10 Constrained decoding on transformers.** `response_format` on `/v1/chat/completions` SHALL
support `json_object` and `json_schema` on the transformers engine through constrained decoding, so
the output parses and validates. (R-04.11)

- **FR-25.10.1** Accepted shapes: `{"type": "text"}` (no constraint), `{"type": "json_object"}`, and
  `{"type": "json_schema", "json_schema": {"name", "schema", "description"?, "strict"?}}`.
- **FR-25.10.2** `json_object`: the output parses as one JSON object.
- **FR-25.10.3** `json_schema`: the output parses and validates against `schema`.
- **FR-25.10.4** `strict: false` is accepted and the schema is still enforced. Enforcing more than
  asked is not a silent drop.
- **FR-25.10.5** The constraint is applied during decoding, not by retrying or repairing output.
- **FR-25.10.6** After generation, miLLM validates the output itself. If a complete generation fails
  to validate, the response is an error, never a 200 with invalid JSON.
- **FR-25.10.7** Structured output composes with steering (`profile`, `steering_intensity`). The
  constraint restricts tokens; steering changes activations.
- **FR-25.10.8** The constrained-decoding library is chosen in the FTDD (PADR §10 "Constrained
  decoding on the transformers engine, library chosen at design"). It must work with
  `transformers>=5.15.1,<6` (`pyproject.toml:52`) and the served tokenizers (LFM2, Gemma, Llama,
  Granite).
- **FR-25.10.9** The supported JSON Schema subset is declared in one place, published in the API
  reference, and checked before generation.
- **FR-25.10.10** Each combination with `n > 1`, `extra_messages` and streaming is honoured or
  refused per FR-25.3. Streaming is Open Question 5.

**FR-25.11 Refusal where unsupported.** Where structured output is unsupported, the request SHALL
return `400` naming `response_format` and the reason — before any auto-load when decidable from the
model row. (R-04.12)

- **FR-25.11.1** A GGUF model is refused in v1 (checkpoint default "structured output on GGUF models
  is refused in v1", closing BRD-04 open question 3). The check uses `gguf_files` on the row and
  runs before auto-load.
- **FR-25.11.2** A schema keyword outside the subset is refused, naming the keyword and its JSON path.
- **FR-25.11.3** The CBM path is refused (R-04.12).
- **FR-25.11.4** `response_format` on `/v1/completions` and `/v1/embeddings` is refused (FR-25.3.3).
- **FR-25.11.5** `response_format` with `stop` is refused. A stop string can cut a document and
  report `finish_reason: "stop"`, which is truncated JSON reported as complete (R-04.13).

**FR-25.12 Truncation is never complete.** A constrained generation stopped by `max_tokens` SHALL
report `finish_reason: "length"` and never return truncated JSON as complete; the response SHALL carry
`X-miLLM-Constrained` naming the format applied. (R-04.13)

- **FR-25.12.1** `finish_reason: "length"` is reported whenever the token budget ended the
  generation, whether or not the partial text happens to parse.
- **FR-25.12.2** `finish_reason: "stop"` means the output is complete and validated (FR-25.10.6).
- **FR-25.12.3** `X-miLLM-Constrained` is present whenever a constraint was applied, absent otherwise.
  Its value names the type and, for `json_schema`, the schema `name`, as a structured header in the
  style of `X-miLLM-Circuit-Rung` (`chat.py:209`).
- **FR-25.12.4** On a streaming response (if Open Question 5 allows it) the header is sent with the
  stream's headers; the final chunk carries the `finish_reason`.

### 3.4 Reproducibility

**FR-25.13 Seed applied and echoed.** `seed` SHALL be accepted on chat and text completions and
applied to sampling. On the serial transformers path the same seed, request, model and batch shape
SHALL give identical output, echoed in `X-miLLM-Seed`, with a `system_fingerprint` naming model,
revision, precision and engine. (R-04.14)

- **FR-25.13.1** `seed` is an integer. Values outside the range the design accepts are refused, not
  wrapped or clamped.
- **FR-25.13.2** The seed is applied inside the admission slot, so no other request's sampling
  consumes the seeded random stream.
- **FR-25.13.3** Identical means byte-identical text and identical `finish_reason`, for the same
  seed, request body, loaded model and batch shape.
- **FR-25.13.4** With `temperature: 0` (greedy) the seed is accepted and echoed. It changes nothing.
- **FR-25.13.5** In scoring mode the seed is accepted and echoed. Scoring is deterministic.
- **FR-25.13.6** `X-miLLM-Seed` echoes the seed applied. It is absent when no seed was sent (see
  Open Question 6).
- **FR-25.13.7** A seeded sampled request is never served by the CBM. It is routed to the serial
  path, as the CBM gate already routes differing sampling parameters
  (`inference_service.py:967-970`).
- **FR-25.13.8** `system_fingerprint` is present on every chat and text completion response, seeded
  or not, as OpenAI places it.
- **FR-25.13.9** It names the model, revision, precision and engine. A part miLLM does not know is
  written as unrecorded, never guessed. The row's `revision` is nullable
  (`millm/db/models/model.py:127`).
- **FR-25.13.10** The fingerprint is stable while the loaded configuration is unchanged, and changes
  when any named part changes.

**FR-25.14 Where a seed cannot promise identity.** Where a seed cannot promise identical output, the
response SHALL say so, and on llama.cpp the seed SHALL be forwarded to the engine or the request
refused. (R-04.15)

- **FR-25.14.1** Batched rows (`extra_messages`) are deterministic per batch shape only
  (`inference_service.py:3409-3420`). The seed header then states the scope "batch shape".
- **FR-25.14.2** On the serial single-conversation path the scope is "request".
- **FR-25.14.3** On llama.cpp the seed is forwarded to the engine, or the request is refused before
  auto-load. Which one is decided in the FTDD by measurement (Open Question 7). Today
  `_llamacpp_params` forwards no seed (`inference_service.py:3933-3954`).
- **FR-25.14.4** A response never claims a scope wider than what was measured.

### 3.5 Coverage of BRD-04 requirements

Every BRD-04 requirement this feature owns is covered. R-04.5 is listed for completeness.

| BRD-04 | Topic | PPRD FR | Refined in this PRD | Status |
|---|---|---|---|---|
| R-04.1 | Report unused fields | FR-25.1 | FR-25.1.1 – FR-25.1.9 | covered |
| R-04.2 | Strict mode | FR-25.2 | FR-25.2.1 – FR-25.2.4 | covered |
| R-04.3 | Output-changing list | FR-25.3 | FR-25.3.1 – FR-25.3.8 | covered |
| R-04.4 | `n` on completions | FR-25.4 | FR-25.4.1 – FR-25.4.4 | covered (choice: Open Question 2) |
| R-04.5 | `dimensions` | FR-30.1 | refusal only: FR-25.3.3, FR-25.3.7 | owned by Feature 30 |
| R-04.6 | Chat scoring, shared path | FR-25.5 | FR-25.5.1 – FR-25.5.9 | covered |
| R-04.7 | Scoring limits, GGUF refusal | FR-25.6 | FR-25.6.1 – FR-25.6.4 | covered |
| R-04.8 | Unsteered, unrecorded | FR-25.7 | FR-25.7.1 – FR-25.7.4 | covered |
| R-04.9 | Chat logprobs shape | FR-25.8 | FR-25.8.1 – FR-25.8.9 | covered |
| R-04.10 | `extra_messages` scoring | FR-25.9 | FR-25.9.1 – FR-25.9.5 | covered |
| R-04.11 | Constrained decoding | FR-25.10 | FR-25.10.1 – FR-25.10.10 | covered (library: FTDD) |
| R-04.12 | Refusal where unsupported | FR-25.11 | FR-25.11.1 – FR-25.11.5 | covered |
| R-04.13 | `finish_reason` and `X-miLLM-Constrained` | FR-25.12 | FR-25.12.1 – FR-25.12.4 | covered |
| R-04.14 | Seed, `X-miLLM-Seed`, `system_fingerprint` | FR-25.13 | FR-25.13.1 – FR-25.13.10 | covered |
| R-04.15 | Seed scope; llama.cpp | FR-25.14 | FR-25.14.1 – FR-25.14.4 | covered (llama.cpp: Open Question 7) |

Totals: 14 of 14 owned requirements covered, 14 PPRD FRs refined into 89 testable items.

## 4. User Experience Requirements

There is no Admin UI change. The user experience is the HTTP contract.

- Every refusal names the field in `param` and states the reason in plain words.
- Headers follow the existing `X-miLLM-*` family: `X-miLLM-Ignored-Fields`, `X-miLLM-Strict` (request),
  `X-miLLM-Constrained`, `X-miLLM-Seed`.
- OpenAI SDK clients work unchanged. New fields use OpenAI's names and shapes. Extensions are headers.
- The API reference documents every header, the output-changing list with its outcome table, the
  JSON Schema subset, and the seed scopes.

## 5. Data Requirements

- **No migration.** Nothing is persisted.
- **Schemas** (`millm/api/schemas/openai.py`):
  - `ChatCompletionRequest` gains `logprobs`, `top_logprobs`, `allowed_token_ids`,
    `return_tokens_as_token_ids`, `response_format` and `seed`.
  - `TextCompletionRequest` gains `seed`.
  - `ChatCompletionChoice` gains an optional `logprobs` in the chat shape.
  - Chat and text completion responses gain `system_fingerprint`.
  - Unknown-field detection replaces silent `extra="ignore"`. Whether the schemas capture extras or
    compare the raw body is a design choice. Validation SHALL stay `400` on `/v1`
    (`millm/api/exception_handlers.py:35`).
- **The output-changing list** is a module-level constant with its per-endpoint outcome table.
- **The JSON Schema subset** is a declared constant.

## 6. Technical Constraints

- PADR §10 "Report-then-opt-in refusal of unknown fields": warning by default, strict on request.
- PADR §10 "One scoring path for chat and completions": chat calls the existing scorer.
- PADR §10 "Constrained decoding on the transformers engine": library chosen at design.
- `MAX_CONCURRENT_REQUESTS` stays 1. Every forward pass goes through `_admit()`.
- The CBM is off in Kubernetes (BRD-04 §3) but SHALL still never drop a listed field (FR-25.3.6).
- Every refusal is a `MiLLMError` subclass with a code in `ERROR_STATUS_MAP`
  (`millm/api/routes/openai/errors.py:61`), following `InvalidScoringRequestError`
  (`millm/core/errors.py:59`).
- Refusals decidable from the request and the model row happen before auto-load. An auto-load
  evicts the resident model and its SAEs.

## 7. API/Integration Specifications

**Request headers:** `X-miLLM-Strict: true` (FR-25.2).

**Response headers:**
- `X-miLLM-Ignored-Fields` — unused field locations (FR-25.1).
- `X-miLLM-Constrained` — the format applied (FR-25.12).
- `X-miLLM-Seed` — the applied seed and its scope (FR-25.13, FR-25.14).
- Existing headers unchanged: `X-miLLM-Backend`, `X-miLLM-Batch`, `X-miLLM-Steering-Intensity`,
  `X-miLLM-Circuit-Rung`, `X-miLLM-Probe-Verdicts`.

**Body changes:** chat `logprobs` in OpenAI's chat shape; `system_fingerprint` on chat and text
completions.

**Integration:**
- **miDataworks** (BRD-03 R-03.22, R-03.28): chat scoring replaces the hand-rendered prompt in
  `jev_client.py`. It sends strict mode, records the seed and fingerprint, and uses structured output
  where supported.
- **miStudio MCP** (BRD-MIS-DATAWORKS-001): `millm_score_chat` calls chat scoring;
  `millm_generate` sends `response_format` and `seed`.
- **Features 26–30** add fields and endpoints and SHALL register each output-changing field on the
  list.
- **Feature 27** (FR-27.9) classifies the scoring path as non-generating (FR-25.7.4).

**Authentication:** none. miLLM runs on the local network (BRD-04 §3, §7).

## 8. Non-Functional Requirements

- **Correctness:** chat and completion scoring agree to 1e-5 absolute on 200 prompts, with identical
  token IDs (BRD-04 acceptance 3).
- **Overhead:** unknown-field detection adds no forward pass and no database read. Its cost is
  measured on a request and recorded.
- **Structured-output throughput:** the per-token cost of constrained decoding is measured on the
  reference model and published.
- **Privacy:** field values never reach the log (FR-25.1.8). A test pins it.
- **Compatibility:** a client sending no new field and no strict header gets the same body as before.
  Only additive headers and `system_fingerprint` appear.

## 9. Feature Boundaries (Non-Goals)

- **`dimensions` honoured** — Feature 30 (R-04.5). Only its refusal is here.
- **`steering` honoured** — Feature 28.
- **Function calling** (`tools`, `tool_choice`) — refused, not implemented (BRD-04 §7).
- **`logit_bias`** — refused in v1; no BRD-04 requirement implements it.
- **Structured output on GGUF** — refused in v1 (checkpoint default).
- **Packed scoring** — Feature 26 (R-04.20).
- **Logprobs on multi-token generation** — refused; scoring scores one token (`openai.py:258-268`).
- **Semantic no-ops of declared fields** (e.g. `top_p` under greedy decoding) — not "unused" fields
  under FR-25.1.
- **Authentication** — BRD-04 §7.

## 10. Dependencies

- **Existing code:** `_score_text_completion`, `_unsteered_next_token_logits`, `_next_token_logits`
  (`inference_service.py:4930-5066`); `next_token_scores`; `_unsteered`; `_format_chat_messages`;
  `_admit`; the CBM gate; `validate_scoring_mode`; the `/v1` validation handler.
- **Feature 23:** GGUF detection on the model row and refusal before auto-load.
- **External library:** one constrained-decoding package, chosen in the FTDD. `pyproject.toml` has
  none today.
- **Hardware:** mcs-lnxhost02 and the RTX 3090; JEV-9B-decision at bfloat16 for acceptance 3.
- **Consumers waiting on this feature:** miDataworks phases that score (BRD-03 R-03.22), miStudio's
  `millm_score_chat`. Features 26–30 inherit FR-25.1–FR-25.3.

## 11. Success Criteria

- **SC-1 (unknown fields):** a chat request with `foo: 1` returns `X-miLLM-Ignored-Fields: foo`. The
  same request with `X-miLLM-Strict: true` returns `400` naming `foo`. A message-level field is
  reported as `messages[i].<field>` (BRD-04 acceptance 1).
- **SC-2 (output-changing list):** every list entry, sent where its outcome is *refused*, returns
  `400` with or without strict mode, on every endpoint (BRD-04 acceptance 1).
- **SC-3 (`n`):** `n: 2` on `/v1/completions` returns two choices per prompt or `400`, never one
  (BRD-04 acceptance 2). Streaming chat with `n: 2` returns `400`.
- **SC-4 (chat scoring parity, hardware):** JEV-9B-decision, bfloat16, 200 prompts: identical token
  IDs and every logprob within 1e-5 absolute against completion scoring of the same rendered prompt
  (BRD-04 acceptance 3).
- **SC-5 (unsteered):** with a profile active, chat scoring equals scoring with no SAE attached
  (BRD-04 acceptance 4).
- **SC-6 (structured output):** 100 `json_schema` requests on transformers all parse and validate.
  The same request on a GGUF model returns `400` with no model load (BRD-04 acceptance 5).
- **SC-7 (seed):** the same sampled request with `seed: 7` twice gives byte-identical text, and the
  response echoes the seed (BRD-04 acceptance 6).
- **SC-8 (reachability):** each wiring line — the list check per endpoint, the chat-scoring route
  branch, the shared-scorer call, the constraint application, the seed application, each header —
  has a test that fails when the line is removed, asserting payload and call count.

## 12. Testing Requirements

- **Unit:**
  - unknown-field detection: top-level, `messages[i]`, `extra_messages[j][i]`, engine-path case (b)
  - header format, bound and absence
  - strict-mode values (`true`, `1`, others, absent)
  - the log event carries locations and never values
  - the output-changing table: one parametrised test over every (field, endpoint, engine path),
    enumerating the endpoints from the live application routes, not a hand-kept list
  - chat scoring schema limits and `top_logprobs` without `logprobs`
  - chat logprobs shape: `token`, id form, `bytes`, `top_logprobs` length, `logprobs: null`
  - `extra_messages` order and failure by index
  - steering fields refused in scoring
  - JSON Schema subset check and refusal paths
  - `finish_reason` under truncation and stop
  - seed range, application inside the slot, scope header, fingerprint parts and unrecorded parts
- **Integration** (tiny real transformer, the style of `tests/unit/services/test_scoring_completions.py`):
  - chat scoring equals completion scoring of `_format_chat_messages` output with
    `add_special_tokens=False`, bit for bit on CPU
  - scoring is routed before the batched and CBM branches
  - scoring with an attached SAE equals scoring without one
  - no probe or sensing context opens during scoring
  - a constrained generation parses and validates; a truncated one reports `"length"`
  - a seeded sampled generation repeats exactly
- **HTTP:** every refusal above through the route, including GGUF refusals with no auto-load
  (the pattern of `tests/unit/api/test_gguf_refused_before_load.py`).
- **Mutation controls:** per the reachability rule, delete each wiring line and require a red. The
  privacy line (values never logged) and the "scoring routed first" line are mutated first.
- **Hardware acceptance:** SC-4, SC-6 and SC-7 on mcs-lnxhost02.

## 13. Implementation Considerations

- **Complexity:** medium-high. `inference_service.py` has several chat paths (serial, streaming,
  batched, CBM, llama.cpp), and the list must hold on each.
- **Suggested order:**
  1. the output-changing list and unknown-field detection, with strict mode (unblocks every
     consumer's safety);
  2. `n` on completions and the streaming `n`/`extra_messages` defect;
  3. chat scoring;
  4. seed and `system_fingerprint`;
  5. structured output (library spike first).
- **Main risks:**
  - Special-token handling in chat scoring. A double BOS would shift every score while looking
    plausible. SC-4's token-ID equality catches it.
  - Constrained decoding may not support every served tokenizer (BRD-04 RSK-05). Refuse, naming the
    reason.
  - Detecting case (b) unused fields needs engine knowledge before the route answers. The design
    must decide it from the model row where possible.
  - Strict mode could break a miDataworks request that sends a harmless extra. That is the intended
    behaviour; the error names the field.

## 14. Open Questions

Batched for the operator. Each carries the default this PRD assumes until answered.

1. **Chat scoring on a model with no chat template.** Refuse with `400` (default), or score the
   generic Gemma-style fallback (`inference_service.py:5653-5679`)? The fallback scores a prompt the
   model was not trained on, and the response would not say so.
2. **`n > 1` on `/v1/completions` in v1.** Refuse with `400` (default; R-04.4 allows it and BRD-03
   R-03.28 accepts "implemented or refused"), or implement `n` choices per prompt now?
3. **`user`.** It is declared and never read in `millm/`. Report it as unused, so strict mode
   refuses it (default, the literal reading of R-04.1), or treat it as accepted metadata?
4. **`max_completion_tokens`.** Newer OpenAI clients send it instead of `max_tokens`. Today it is
   silently ignored and generation runs to 512 tokens. Add it to the output-changing list and honour
   it as `max_tokens` (default), or only report it?
5. **Streaming structured output.** Honour `response_format` on streaming chat in v1, or refuse it
   (default: refuse until the FTDD shows the chosen library works with the streamer)?
6. **Server-chosen seed.** When no seed is sent, should miLLM pick one, apply it and echo it, so any
   sampled row can be reproduced later? Default: no; `X-miLLM-Seed` is absent.
7. **Seed on llama.cpp** (technical, resolved in the FTDD). Forward it if two runs on the reference
   GGUF model are measured identical; otherwise refuse. R-04.15 allows either.
8. **Constrained-decoding library** (technical, resolved in the FTDD; PADR §10). Candidates are
   judged on transformers 5 compatibility, served tokenizers, the declarable schema subset and
   per-token overhead.

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is pre-answered from a cited source. Questions with no
source are Open Questions above.

| # | Question | Answer | Source |
|---|---|---|---|
| D1 | Unknown fields: reject or report by default? | Report by default; refuse under `X-miLLM-Strict: true` | Checkpoint default 2026-10-06; R-04.1, R-04.2; PADR §10 "Report-then-opt-in refusal" |
| D2 | Which fields can never be ignored? | BRD-04's list, in one module, with a per-endpoint outcome table | R-04.3; PADR §10 |
| D3 | Does "unused" include declared fields an engine ignores? | Yes (FR-25.1.2 case b) | R-04.1 "fields it did not use"; `inference_service.py:3911-3920` |
| D4 | Separate chat scorer or shared path? | Shared: render, then the completion scorer | R-04.6; PADR §10 "One scoring path" |
| D5 | Special tokens on the rendered prompt? | `add_special_tokens=False` | R-04.6; PADR §10 |
| D6 | Are `extra_messages` conversations packed? | No, one at a time | Checkpoint default "probe scoring runs one input at a time"; PADR §10 "Packed scoring"; BRD-04 acceptance 3 |
| D7 | Steering fields on a scoring request? | Refused | R-04.3 + R-04.8 |
| D8 | Structured output on GGUF? | Refused in v1 | Checkpoint default; closes BRD-04 open question 3 |
| D9 | Structured output on the CBM? | Refused | R-04.12 |
| D10 | `strict: false` in `json_schema`? | Accepted, schema still enforced | R-04.11 "validates against the schema" |
| D11 | `response_format` with `stop`? | Refused | R-04.13 (truncated JSON never complete) |
| D12 | Constrained-decoding library? | Chosen in the FTDD | PADR §10 "library chosen at design" |
| D13 | Seeded request on the CBM? | Routed to serial | R-04.14 (serial path promise); existing CBM gate `inference_service.py:967-970` |
| D14 | Unknown fingerprint parts? | Written as unrecorded, never guessed | R-04.14; `revision` nullable, `model.py:127` |
| D15 | `tools`, `tool_choice`, `logit_bias`? | Refused in v1 | R-04.3; BRD-04 §7 (function calling not implemented) |
| D16 | `dimensions`, `steering`? | Refused until Features 30 and 28 | PPRD v1.5 split note; FR-25.3 |
| D17 | Streaming with `n > 1` or `extra_messages`? | Refused | R-04.3; `stream_chat_completion` reads neither (`inference_service.py:4288`) |
