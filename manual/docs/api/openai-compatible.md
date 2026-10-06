---
sidebar_position: 2
title: OpenAI-Compatible API
---

# OpenAI-Compatible API

miLLM exposes an OpenAI-compatible API at `/v1`, making it a drop-in replacement backend for the OpenAI SDK, Open WebUI, LangChain, LlamaIndex, and anything else that speaks the OpenAI protocol. When steering is active on the server, it applies transparently to every completion. It never applies to embeddings or to [scoring-mode](#scoring-mode-next-token-log-probabilities) completions.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Text embeddings — always **unsteered** |
| `/v1/models` | GET | List available models |
| `/v1/models/{id}` | GET | Model metadata |

## Chat completions

### Request parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must match the loaded model's name (see `/v1/models`) |
| `messages` | array | required | Roles: `system`, `user`, `assistant`, `tool`, `function` |
| `stream` | bool | `false` | SSE streaming |
| `temperature` | float | `1.0` | `0` = greedy/deterministic. Range 0–2 |
| `top_p` | float | `1.0` | Nucleus sampling, 0–1 |
| `n` | int | `1` | Number of choices. Refused with `stream: true` |
| `max_completion_tokens` | int | — | OpenAI's newer name for `max_tokens`; honoured as it. Sent with a different `max_tokens`, refused naming both |
| `seed` | int | — | 0 to 2³²−1. Applied and echoed in `X-miLLM-Seed` — see [Seeds](#seeds-and-system_fingerprint) |
| `response_format` | object | — | `text`, `json_object` or `json_schema` — see [Structured output](#structured-output) |
| `logprobs`, `top_logprobs`, `allowed_token_ids`, `return_tokens_as_token_ids` | — | — | Chat scoring — see [Chat scoring](#chat-scoring) |
| `max_tokens` | int | server default | Validated against the model's context window |
| `stop` | string \| string[] | — | Stop sequences, enforced in both streaming and non-streaming |
| `frequency_penalty` | float | `0.0` | −2 to 2; mapped to repetition penalty internally |
| `presence_penalty` | float | `0.0` | −2 to 2; mapped to repetition penalty internally |
| `profile` | string | — | **miLLM extension**: apply a saved [steering profile](/features/profiles) for this request only |
| `steering_intensity` | float \| string | — | **miLLM extension** (chat completions only): per-request steering dial — a λ in `0`–`2`, or `"off"` / `"min"` / `"max"` |

:::note Intensity coupling
When the steering base is an imported **cluster**, its stored strengths are scaled by an intensity dial (λ) before applying. Without `steering_intensity`, the cluster's persistent dial (set on the Clusters page) applies; with it, the request's λ **overrides** the stored one for that request only. Symbolic `"min"`/`"max"` resolve to the cluster's declared `intensity_range` bounds (intersected with the `[0, 2]` dial envelope), and numeric λ is capped at the range's **maximum** (or the server's configured maximum for clusters without a declared range) — dialing *down* below the declared floor (toward off) is always honored, matching the management API's bounds of `[0, max]`. The base is the named `profile` if given, else the active profile, else the live steering values. `0`/`"off"` disables steering for the request without validating the base (a profile that would 400 at λ=0.01 still turns steering off at λ=0).
:::

When the steering base is an imported **circuit** (a multi-layer intervention spanning several SAEs), one λ scales **every layer together** — each member through its own layer's SAE. Two differences from the cluster rule above:

- **Both ends are clamped.** Numeric λ is clamped into the circuit's declared `intensity_range` intersected with the configured envelope — the floor as well as the ceiling. `0.1` against an authored `[0.5, 1.5]` resolves to `0.5`, not `0.1`. Only an exact `0`/`"off"` is honored below the floor.
- **The default floor is 0, not 0.5.** Circuits use `CIRCUIT_INTENSITY_MIN` (default `0.0`) where clusters use `CLUSTER_INTENSITY_MIN` (default `0.5`). A circuit whose document declares no `intensity_range` therefore makes `"min"` identical to `"off"`.

Members are re-derived from the strengths the circuit was **authored** with, so the dial is absolute rather than compounding on the circuit's stored intensity. Each member is clamped to miLLM's ±200 steering range at apply time, so at a high λ a strong member can reach the ceiling while weaker ones keep scaling, compressing their relative proportions.

A circuit serving in `slice_fallback` mode is steered by its backing **cluster profile**, so the cluster rule above applies to it — including the 0.5 floor.

### `X-miLLM-Circuit-Rung`

Responses carry `X-miLLM-Circuit-Rung` when a circuit is genuinely steering generation, in [RFC 8941](https://www.rfc-editor.org/rfc/rfc8941) structured form:

```
X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"
```

The rung is a bare integer and the phrase a quoted-string, so ladder punctuation cannot break a naive parser. The phrase is rendered server-side from the evidence ladder and **never composed per-request** — a circuit below rung 2 is never described as causal:

| Rung | `language` | Meaning |
|------|-----------|---------|
| 0 | `associated` | Mined co-occurrence only — unvalidated |
| 1 | `suggested (attribution-supported)` | Attribution evidence, not causal |
| 2 | `causally validated (edge)` | Each edge causally validated |
| 3 | `faithfulness-tested (circuit)` | The whole circuit was faithfulness-tested |

The header is **omitted** whenever the circuit is not actually steering — no active circuit, a slice-fallback serve, an unparseable definition, or no SAE attached on any member layer. Its absence never means "rung 0"; it means "no circuit-attributable steering on this response". Clients displaying evidence language must read this header (or the `steering` field on `GET /api/circuits/active`) rather than deriving it from whether a circuit row is active.

An operator changing steering through the management API **while a request is generating** wins: the request's restore is skipped rather than reverting them, and the management response reports whether the value is actually live. A per-request dial therefore never silently undoes a concurrent operator action.

Responses to dialed requests carry an `X-miLLM-Steering-Intensity` header echoing the resolved λ. The echo is best-effort: it is omitted whenever the dial will not change steering (no SAE attached, unknown profile, steering disabled with a dial-only request, or an empty steering base), and a concurrent profile switch while the request queues can in rare cases make a symbolic echo differ from the applied λ.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2-2b",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "What is a sparse autoencoder?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

```json title="Response"
{
  "id": "chatcmpl-9f3a2b...",
  "object": "chat.completion",
  "created": 1783761600,
  "model": "gemma-2-2b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "A sparse autoencoder is..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 24, "completion_tokens": 87, "total_tokens": 111}
}
```

`finish_reason` is `"stop"` for EOS or a stop sequence, `"length"` when `max_tokens` was reached.

### With the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # miLLM doesn't require auth
)

response = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Streaming uses standard SSE (`data: {json}\n\n` frames, terminated by `data: [DONE]`). The first chunk carries the `role`, the final chunk carries `finish_reason` and `usage`. If generation fails mid-stream, an SSE error event is emitted before `[DONE]` (the HTTP status is already 200 by then — check for `error` objects in-stream). When a stop sequence matches during streaming, generation is cancelled promptly rather than running to `max_tokens`.

### Per-request steering with `profile`

```python
response = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "What is truth?"}],
    extra_body={"profile": "honesty-amplification"},
)
```

The named profile's steering replaces the global configuration for this one request, then the previous state is restored. Unknown profile → `404`; profile invalid for the attached SAE → `400`; no SAE attached → the request runs unsteered. Requests with `profile` always use the serial backend. Details: [Profiles](/features/profiles#per-request-profiles-api).

## Text completions

`POST /v1/completions` accepts `prompt` (string or list of strings; each list entry produces a choice) plus the same sampling parameters as chat. No chat template is applied — the prompt goes to the model verbatim, which is often preferable for base-model steering experiments.

### Scoring mode: next-token log-probabilities

Classification models in the Jev style (for example `autotrust/JEV-9B`) don't generate an answer: you read the probability of a few answer tokens at the next position. Add these fields to a `/v1/completions` request to get those probabilities back. The names match OpenAI's legacy completions and vLLM, so their clients work unchanged.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `logprobs` | int | — | Alternatives to return for the next token, 0–20 |
| `allowed_token_ids` | int[] | — | Restrict the next token to these ids. Log-probabilities are normalised over this set only, so every listed token is always returned. |
| `add_special_tokens` | bool | `true` | Set `false` when the prompt must be tokenised exactly as written (no BOS) |
| `return_tokens_as_token_ids` | bool | `false` | Key `top_logprobs` by `"token_id:<id>"` instead of the decoded token text, which can be ambiguous |

Scoring mode needs `max_tokens: 1` (or `max_completion_tokens: 1`) and `n: 1`; any other value is refused with a 400. One forward pass runs per prompt and nothing is generated.
- The returned `text` is the most probable allowed token.
- `temperature` divides the logits before normalising; `0` means no scaling. `top_p`, `stop` and the penalties don't apply in scoring mode.
- Each choice carries an OpenAI-shaped `logprobs` object: `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset` (the character offset of the token within prompt plus completion, which is OpenAI's convention). With `allowed_token_ids` but no `logprobs`, the token is constrained and `logprobs` is `null`, as in vLLM.
- When tokens are keyed by text, two ids that decode to the same text share one entry. Use `return_tokens_as_token_ids` when that matters.

These requests are refused instead:
- A token id outside the loaded model's vocabulary: 400.
- A prompt that tokenises to nothing: 400.
- A GGUF (llama.cpp) model: 400, before the model is loaded, because llama.cpp exposes no per-token distribution here.
- NaN or +inf logits where they reach the answer (on an allowed token, or anywhere when the vocabulary is unrestricted), or a reported token whose logit is −inf: 500 `non_finite_logits`. A −inf on a token outside the allowed set is fine; some heads mask padded vocabulary that way.
- A temperature between 0 and 0.001, which would overflow the logits: 400.
- `profile`, `steering_intensity`, `steering` or `response_format` on a scoring request: 400 naming the field (scoring is always unsteered).
- Running out of GPU memory: the same typed error as generation.

:::caution Behaviour change
Before scoring mode existed, `logprobs` was silently ignored, so `logprobs: 5, max_tokens: 100` returned 100 tokens of text with no probabilities. That request is now refused with a 400 rather than answered without what it asked for.
:::

```bash
curl http://localhost:8000/v1/completions -H 'content-type: application/json' -d '{
  "model": "JEV-9B-decision", "prompt": "[kind] noul\n[state] ...\n[question] ...\n[options]\nfalse\ntrue\n[decision]:",
  "max_tokens": 1, "temperature": 1.0, "logprobs": 2, "allowed_token_ids": [3721, 1802],
  "add_special_tokens": false, "return_tokens_as_token_ids": true}'
```

Scoring requests are **never steered and never monitored**. Every attached SAE is suppressed for the forward pass, as for embeddings, because a judge's output is a probability that a steering profile left on the model would silently bias. Probes and sensing record nothing, because a judge's prompt isn't user traffic. Suppression applies only to the scoring pass's own forward computation, so a generation running at the same moment (with continuous batching enabled) is still steered.

## Per-request SAE activations

Add `return_sae_activations` to a chat or text completion to receive **this request's** activations
of an attached SAE, under a `millm` object in the response:

```json
"return_sae_activations": {"sae_id": "sae_…", "features": [12, 99], "top_k": 8,
                           "positions": "completion", "read_point": "post_steering"}
```

- `positions`: `last`, `prompt`, `completion`, `all`, or `{"start": int, "end": int}` (half-open),
  absolute over prompt-then-generated tokens. The final sampled token is never fed back to the
  model, so it has no activation; the response lists the positions actually read.
- `features` restricts the candidates; `top_k` returns the largest among them per position.
- `read_point`: `post_steering` (default — after this layer's steering delta, what the model
  computed) or `pre_steering` (before it, where monitoring reads). In scoring mode the read point is
  reported as `unsteered`, and the response carries `X-miLLM-Steering: none`.

```json
"millm": {"sae_activations": {"sae_id": "sae_…", "layer": 11, "read_point": "post_steering",
  "positions": [{"position": 17, "token_id": 345, "features": [{"index": 12, "value": 3.1}]}],
  "note": "post_steering is what the model computed at this layer for this request; it is not an unsteered counterfactual …"}}
```

:::warning Neither read point is a counterfactual
Steering at earlier layers, and tokens generated under steering, still shape the residual at this
layer. `pre_steering` removes only this layer's own delta.
:::

When streaming, the activations arrive in one final chunk with `choices: []` and the `millm` object,
after any probe-verdict chunk and before `[DONE]`. A response that did not ask carries no `millm`
key at all.

Refused with `400` before anything is generated: `n > 1`, `extra_messages`, several prompts (there
is no single position axis); no matching SAE attached; `sae_id` omitted while several SAEs are
attached (the error names the candidates); a feature index past the SAE's width; `top_k` over
`SAE_ACTIVATIONS_MAX_TOP_K` (64); and a worst case — positions × `top_k`, counting `max_tokens` in
full for `completion` and `all` — over `SAE_ACTIVATIONS_MAX_ENTRIES` (65,536). A GGUF model is
refused before any load. These requests are served on the serial path, never continuous batching.

## Embeddings

`POST /v1/embeddings` with `input` (string or list) returns mean-pooled last-hidden-layer embeddings. `encoding_format` may be `"float"` (default) or `"base64"`.

:::info Embeddings are never steered
The steering hook of **every** attached SAE is suppressed during embedding computation (each layer of a circuit, not just the first — fixed 2026-10-04), so embeddings always reflect the unmodified model — making them a neutral measuring stick for comparing steered vs. unsteered generations.
:::

## Models

```bash
curl http://localhost:8000/v1/models
```

```json
{"object": "list", "data": [{"id": "gemma-2-2b", "object": "model", "created": 1774046346, "owned_by": "google/gemma-2-2b"}]}
```

By default `/v1/models` lists **all available models** (READY, LOADED, LOADING). When a model is **locked for steering**, only that locked model is listed — so a steering-locked server presents a single stable model id to OpenAI clients. Use the [Management API](/api/models) to see everything on disk (including states not surfaced here).

## Serving GGUF models

GGUF models answer the same OpenAI-compatible surface as HuggingFace checkpoints, with three behaviours worth knowing.

**Model names carry a quantization tag.** A GGUF model is `repo:QUANT` — several quantizations of one repository can be served side by side. A bare repository name works while only one exists, and otherwise returns 400 `AMBIGUOUS_MODEL_NAME` naming the tags. See [Models API](/api/models#gguf-model-names).

**An oversized prompt is a 400, not a 500.** llama.cpp signals this as a bare error that would otherwise surface as an internal server error — which tells a client to *retry*, and a prompt that exceeds the window can never succeed. miLLM returns `context_length_exceeded` with the requested and available token counts, the same shape OpenAI uses, so a client shortens the prompt instead of burning three attempts on it.

**`chat_template_kwargs` is accepted and ignored.** A GGUF file's baked-in template cannot take arbitrary variables, so keys like `enable_thinking` are logged and dropped rather than refused. Refusing turned an entire class of client — anything that always sends the field — into a 400 on every call, which is a total failure rather than a degraded result.

### Continuing a truncated answer

Clients that offer a **Continue** action resend the conversation with the truncated answer as a trailing assistant message. A GGUF file's chat template *closes* that final turn, so the model can only begin a new answer — which is why continuing used to restart the response from the beginning, sometimes saying so out loud, and why reasoning models re-opened their thought block each time.

miLLM detects a trailing assistant message and completes it instead: everything before the partial is rendered with a generation prompt, the partial is appended raw, and generation continues from there. Nothing is required of the client; **Continue** in Open WebUI now resumes mid-sentence.

## Request validation

**Every field is honoured, reported or refused — never dropped silently.**

- A field the request path does not use is named in the **`X-miLLM-Ignored-Fields`** response
  header: an RFC 8941 list of strings giving each field's location — `"foo", "messages[2].name",
  "extra_messages[0][1].weight"`. It is absent when nothing was ignored, is sent on streaming
  responses too, and never changes the body. The server logs one `request_fields_unused` warning
  carrying the locations, never the values. The header is capped at 1,024 bytes
  (`IGNORED_FIELDS_HEADER_MAX_BYTES`); past that its last entry is `"+N more"`. Characters outside
  printable ASCII in a field name are percent-encoded.
- Send **`X-miLLM-Strict: true`** (or `1`) to have the same request refused with `400
  unused_fields_refused` naming every unused location. `false`, `0` or no header keeps reporting.
  Any other value (`yes`, say) is refused with `400`, naming the header.
- `user` is not read by miLLM, so it is reported, and strict mode refuses it.
- On a GGUF model, `chat_template_kwargs` is reported: llama.cpp applies the template baked into the
  file and takes no variables.

**Output-changing fields are refused with `400 field_not_honoured` wherever they cannot be
honoured, with or without strict mode**, before the requested model is loaded:

| Field | Chat (transformers) | Chat (GGUF) | Completions | Embeddings |
|---|---|---|---|---|
| `logprobs`, `allowed_token_ids` | honoured (chat scoring) | refused | honoured on transformers, refused on GGUF | refused |
| `top_logprobs` | honoured (chat scoring) | refused | refused (use `logprobs`) | refused |
| `response_format` | honoured | refused | refused | refused |
| `seed` | honoured | refused (not yet measured to reproduce) | honoured on transformers, refused on GGUF | refused |
| `n` | honoured, not streaming | refused above 1 | refused above 1 | refused above 1 |
| `max_completion_tokens` | honoured as `max_tokens` | honoured | honoured | refused |
| `profile`, `steering_intensity` | honoured, refused on scoring requests | refused | refused | refused |
| `steering` | refused (not yet implemented) | refused | refused | refused |
| `dimensions` | refused | refused | refused | refused (not yet implemented) |
| `tools`, `tool_choice`, `logit_bias` | refused | refused | refused | refused |

The values `n: 1`, `logprobs: false`, `response_format: {"type": "text"}`, `tools: []`,
`logit_bias: {}` and an explicit `null` mean "no change" and are accepted everywhere.

## Chat scoring

`/v1/chat/completions` scores the next token like `/v1/completions` does. Send `logprobs: true`
(and optionally `top_logprobs`, 0–20) and/or `allowed_token_ids`; scoring mode needs
`max_tokens: 1`, `n: 1`, no streaming, and a temperature of 0 or at least 0.001. `top_logprobs`
without `logprobs: true` is refused.

The chat template is rendered with the generation prompt and scored through the same function
`/v1/completions` uses, **without adding special tokens again** (a rendered template already has
its BOS). So chat scoring of `messages` equals completion scoring of the rendered prompt with
`add_special_tokens: false`. A model with no chat template is refused with `400
no_chat_template` — generation keeps its generic fallback; scoring would score a prompt the model
was never trained on.

```json
"choices": [{"index": 0, "finish_reason": "length",
  "message": {"role": "assistant", "content": " true"},
  "logprobs": {"content": [{"token": "token_id:1802", "logprob": -0.21, "bytes": [32, 116, 114, 117, 101],
    "top_logprobs": [{"token": "token_id:1802", "logprob": -0.21, "bytes": [32, 116, 114, 117, 101]},
                     {"token": "token_id:3721", "logprob": -1.66, "bytes": [32, 102, 97, 108, 115, 101]}]}]}}]
```

`content` holds one entry, the scored token. `bytes` is always the UTF-8 of the decoded token;
`return_tokens_as_token_ids` changes `token` only. `top_logprobs` has `min(top_logprobs,
candidates)` entries, empty for 0 or absent. `allowed_token_ids` without `logprobs: true` constrains
the token and returns `logprobs: null`. With `extra_messages`, each conversation is scored one at a
time and returns its own choice, `index` in input order; a failing conversation fails the request,
naming its index. Chat scoring is unsteered and unmonitored, as completion scoring is, and is
refused on GGUF models before loading.

## Structured output

`response_format` constrains generation on the transformers engine so the output parses (and, for
`json_schema`, validates):

```json
"response_format": {"type": "json_schema",
  "json_schema": {"name": "judge_v1", "strict": true,
    "schema": {"type": "object", "properties": {"label": {"enum": ["humor", "not_humor"]}},
               "required": ["label"], "additionalProperties": false}}}
```

- `{"type": "json_object"}` constrains to one JSON object. `strict: false` is accepted, and the
  schema is still enforced.
- The response carries **`X-miLLM-Constrained`**: `json_object` or `json_schema;name="judge_v1"`.
- `finish_reason: "stop"` means the model ended the document itself and miLLM validated it.
  `"length"` means `max_tokens` ended it, so the text is incomplete, even if it happens to parse.
  A complete document that fails validation is a `500 constrained_output_invalid`, never a 200.
- Refused with `400` naming `response_format`, before loading: a GGUF model, `/v1/completions`,
  `/v1/embeddings`, streaming, `stop` (a stop string could cut a document and report it complete),
  scoring fields, and a server with continuous batching enabled.
- `n > 1` and `extra_messages` are constrained too, one constraint per choice or row.

**Supported JSON Schema subset.** `type` (all seven), `properties`, `required`,
`additionalProperties` (boolean or schema), `items`, `minItems`, `maxItems`, `enum`, `const`,
`minimum`, `maximum`, `minLength`, `maxLength`, `pattern`, `anyOf`, and local `$ref` into `$defs`;
the annotations `title`, `description`, `default` and `examples` are accepted and ignored.
Anything else is refused with `400 response_format_unsupported`, naming each keyword and its JSON
pointer — including `multipleOf`, `not`, `uniqueItems`, `allOf`, `oneOf`, `if`/`then`/`else`,
`format`, `patternProperties`, `dependentRequired`, `$schema` and remote `$ref`. Some of these are
compiled by the constraint library without being enforced, which is why the subset is miLLM's own
list. A schema is limited to 64 KB, nesting depth 32 and 256 properties per object.

## Seeds and `system_fingerprint`

`seed` (an integer from 0 to 2³²−1; anything else, including `true`, is refused) is applied inside
the request's slot and echoed in **`X-miLLM-Seed`** with the scope of the promise:

| Header | Meaning |
|---|---|
| `7;scope="request"` | The same seed, body and loaded model give byte-identical text and `finish_reason` |
| `7;scope="batch-shape"` | `extra_messages` rows: identical only for the same batch composition |
| `7;scope="best-effort"` | Continuous batching runs in this process and draws from the same random generator |

The header is absent when no seed was sent; miLLM never picks one. The seed is accepted and
echoed on greedy (`temperature: 0`) and scoring requests, where it changes nothing. A seeded request
is never served by the batching manager. On a text completion with several prompts, each prompt is
seeded separately. Seeds are refused on GGUF models until llama.cpp's repeatability is measured.

Every chat and text completion response (not streamed chunks) carries
**`system_fingerprint`**: `millm:<model>@<revision>:<dtype>/<quantization>:<engine>`, for example
`millm:LFM2.5-1.2B-Instruct@0f604ada:bfloat16/FP16:transformers`. A part miLLM does not know is
written `unrecorded`, never guessed.

## Errors & backpressure

`/v1` errors use OpenAI's format so SDK exception handling works unchanged. Notable cases:

| Situation | Status | `code` |
|-----------|--------|--------|
| No model loaded | 503 | `model_not_loaded` |
| Unknown model name in request | 404 | `model_not_found` |
| Prompt + `max_tokens` exceeds the model's context window, on any engine and any route (an embeddings input past it too). The message gives the limit and what was asked for. A streamed request is refused with this 400 before the stream starts; if the stream has already started, it ends with this error event and `[DONE]` | 400 | `context_length_exceeded` |
| Steering error on an in-flight stream (mismatched cluster, bad index) | SSE `error` event, then `[DONE]` | `invalid_feature_index` |
| Request queue full (backpressure) | 503 | `queue_full` |
| An output-changing field this endpoint or engine cannot honour | 400 | `field_not_honoured` |
| `X-miLLM-Strict: true` and an unused field | 400 | `unused_fields_refused` |
| `response_format` that cannot be honoured (schema keyword, engine, combination) | 400 | `response_format_unsupported` |
| Chat scoring on a model with no chat template | 400 | `no_chat_template` |
| A complete constrained output that fails validation | 500 | `constrained_output_invalid` |
| A scoring token id outside the vocabulary; an empty prompt | 400 | `invalid_scoring_request` |
| Unknown `profile` | 404 | `profile_not_found` |
| Invalid `steering_intensity` (outside 0–2 / unknown symbol) | 400 | `invalid_parameter` |
| The named model has to be loaded and its quantization cannot be (a Q2 transformers checkpoint) | 400 | `unsupported_quantization` |
| The named model has to be loaded and no card, or split across cards, holds it | 503 | `insufficient_memory` |
| The named model has to be loaded and the load failed | 500 | `model_load_failed` |
| The named model is being unloaded, or another load is in progress: retry once it finishes. Type `server_error`; a streamed request is refused before its stream starts | 503 | `model_busy` |
| Generation runs a card out of memory (the prompt and its KV cache do not fit beside the model); type `invalid_request_error`, the message names the card. A stream ends with this error event and `[DONE]` | 503 | `insufficient_memory` |

A request that names a model other than the one loaded loads it first. When that load is refused before the loaded model is unloaded, the response carries the refusal's own status and message, with the figures for each card. A load that fails after the unload, for example because free memory changed in between, answers `500 model_load_failed` with the failure's message.

**Requests during an unload.** From the moment an unload begins, before any weight is moved, a request for that model is refused with `503 model_busy` and told to retry. Requests already running when the unload begins finish first: the unload waits for them for up to [`GRACEFUL_UNLOAD_TIMEOUT`](/reference/configuration) seconds before it moves anything. A request for a different model is refused the same way while the unload runs, rather than starting a second one. Unloading a split model takes several seconds (8.5 s for Qwen2.5-7B, 15 s for OLMo-2-13B on the node), and before this a request in that window ran on a half-moved model and answered `500`.

## Behavior under continuous batching

If the opt-in [CBM backend](/concepts/architecture#continuous-batching-opt-in) is enabled, requests matching the server's fixed sampling parameters are batched for throughput; requests with different `temperature`/`top_p`, a frequency or presence penalty, `n > 1`, a `seed`, a `profile` or `steering_intensity` parameter, or (optionally) active monitoring fall back to the serial path automatically. `response_format` is refused while it is enabled. `GET /api/health/inference` shows which backend is active.

:::tip Integration with Other Tools
- **Open WebUI:** set the OpenAI API base URL to `http://<host>:8000/v1` — [tutorial](/tutorials/open-webui)
- **miStudio:** use the "OpenAI Compatible" method pointed at miLLM's `/v1`
- **LangChain / LlamaIndex:** use the OpenAI provider with a custom base URL
:::
