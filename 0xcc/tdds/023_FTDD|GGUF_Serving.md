# Technical Design: GGUF Serving
## miLLM Feature 23

**Source:** BRD-MILLM-GGUF-001 · `023_FPRD|GGUF_Serving.md`

---

## 1. Where GGUF diverges

Exactly one place: the loader. Model rows, the download flow, `/v1/*`, locking,
graceful unload, and the admin UI are shared unchanged.

```
POST /api/models ──► ModelService.download_model ──► GGUF? ─┬─ yes ─► llama.cpp loader
                                                            └─ no  ─► transformers loader
                                    both produce a Model row and a loaded engine
```

SAE attachment does **not** cross this line. llama.cpp exposes no hook points,
so steering, probes and interpretability stay on the transformers path — the
real boundary of the feature, and the one worth defending.

## 2. Naming

`ModelService.download_model` names a GGUF row `f"{name}:{gguf_label}"`. A
`custom_name` is left untouched.

`ModelRepository.find_by_name` resolves in three steps:

1. exact match on `name`
2. otherwise a bare repository name, **while exactly one** row carries it —
   which keeps existing callers and saved client selections working
3. otherwise raise `AmbiguousModelNameError` naming every matching tag

`AmbiguousModelNameError` maps to **400 / `invalid_request_error`**, so a client
edits the request rather than retrying it as a server fault.

Migration `015` backfills existing rows. Data-only and idempotent: it tags only
rows whose name is still the bare repository tail — exactly the set the fixed
service would have produced.

## 3. Context sizing

Three inputs, in order of authority:

| input | source | role |
|---|---|---|
| `n_ctx_train` | ~1.2 s CPU metadata probe, no VRAM | what the model was trained for |
| predicted max | VRAM arithmetic (below) | what this card can hold |
| `GGUF_CONTEXT_LENGTH` | config, default 32768 | policy ceiling |

The loader starts at `min(declared, ceiling, predicted)`.

```
bytes_per_token = 2 (K,V) × n_layer × n_head_kv × head_dim × bytes_per_element
```

Every term comes from the same metadata probe. For `gemma-4-31b`:
`2 × 60 × 16 × 168` = **630 KB/token** at f16, halved at `q8_0` — which is
precisely why quantizing the cache doubles the usable window.

Two terms are empirical and named as such in the code: **~2 GiB** of CUDA
context and compute buffers, and planning against **94%** of the card because
the last few percent go to fragmentation.

**The prediction seeds the ladder; it does not replace it.** The load attempt
still decides, and a prediction that cannot be made falls back to the ceiling
exactly as before. Validated against four RTX 3090 measurements — f16@4096
loaded, f16@8192 failed, q8_0@12288 loaded, q8_0@16384 failed — all four
predicted correctly. The test asserts the prediction *brackets* the measurements
rather than asserting the formula against itself.

## 4. KV cache and flash attention

`GGUF_KV_CACHE_TYPE` sets `type_k`/`type_v`. `GGUF_FLASH_ATTENTION` is a hard
dependency of a quantized cache, not a companion optimisation — `q8_0` without
it fails at 8192 where `q8_0` with it reaches 12288 — so the loader refuses the
invalid pairing at construction rather than producing a confusing failure later.

## 5. Embeddings

`GGUF_ENABLE_EMBEDDINGS` is decided at construction; llama.cpp offers no way to
enable it afterwards. Measured cost: 113.5 → 105.9 tok/s (6.7%) on
`zora-v1.13 Q5_K_M`, no VRAM change.

Some architectures refuse `pooling_type=MEAN` outright and fail context creation
at **every** length. The loader retries the whole ladder without embeddings and
records `supports_embeddings` on the row, so a caller is told rather than
discovering it from a failure at `/v1/embeddings`. The transformers path sets it
`True` — it computes embeddings from hidden states at request time.

## 6. Request-path behaviour

**Oversized prompt.** llama.cpp raises a bare `ValueError`; miLLM maps it to
400 `context_length_exceeded` with the requested and available counts. A 500
here tells a client to retry something that can never succeed — miStudio's
labeling loop retried each oversized prompt three times.

**`chat_template_kwargs`.** A GGUF file's baked-in template cannot take
arbitrary variables. Unknown keys are ignored and logged rather than refused;
refusing turned any client that always sends the field into a 400 on every call.

**Continuation.** An OpenAI client's *Continue* resends the conversation with
the truncated answer as a trailing assistant message, and every GGUF chat
template CLOSES that turn:

```
no trailing assistant : …<|turn>model\n<|channel>thought\n<channel|>
trailing assistant    : …<|turn>model\nPARTIAL<turn|>\n
```

Sealed, the model can only start again. miLLM applies HuggingFace's
`continue_final_message` semantics: render `messages[:-1]` with
`add_generation_prompt=True`, append the partial raw, and use `create_completion`
rather than `create_chat_completion`.

## 7. Rejected alternatives

**A second server for GGUF** — duplicates model management, `/v1/*`, locking and
the UI to avoid one branch at load time.

**Keeping one row per repository** — makes two quantizations collide on `name`,
which is the defect that produced a 500 on every request for a loaded model.

**Choosing a quantization silently on a bare name** — makes the answer depend on
insertion order, with nothing on the wire to say which model replied.

**Searching for the context window by halving** — six model loads and ~30 s to
find a number the metadata already determines.

**`q4_0` as the default** — a third more window at a real accuracy cost, which
is the wrong trade when the model's job is discrimination.

**Refusing to load when embeddings are unsupported** — trades a serving model
for a clean capability matrix.
