# Implementation Notes: GGUF Serving
## miLLM Feature 23

Written after the fact. Every entry is something that actually went wrong here.

---

## 1. Widening a constraint is not widening a name

Migration 014 widened uniqueness to `(repo_id, quantization, gguf_label)` so
several quantizations could coexist — the whole point of the picker it shipped
with — while `name` stayed `repo_id.split("/")[-1]`. The second download of a
repository produced a second row with the **same name**,
`find_by_name`'s `scalar_one_or_none()` raised *"Multiple rows were found when
one or none was required"*, and **every OpenAI request naming that model
returned 500 while the model was loaded and serving.**

Three review rounds missed it because **no fixture ever held two rows from one
repository**. When you widen a uniqueness constraint, ask what else was unique
because of it.

## 2. A naming test can be true by construction

The first version asserted over two pre-named rows and **survived deleting the
naming code entirely**. It now runs the real `download_model`. If a test builds
the thing it is checking, it is checking your fixture.

## 3. "Try a smaller quantization" was advice that could not work

`ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF` failed at every rung of the ladder on
a card that was 300 MiB used of 24576, with a 7.8 GiB file. The next
quantization would fail identically, because **the context size was never the
problem** — `pooling_type=MEAN` is refused outright by that architecture.

When a failure recurs identically at every setting of a parameter, that
parameter is not the cause. A/B on the same file: with the flag, 8192/4096/2048
all fail; without it, 2048 loads.

## 4. Compute what you were about to search for

The ladder halved down from a fixed ceiling — one full model load per rung, six
of them, ~30 seconds — to find a number arithmetic gives in milliseconds from
metadata the loader already probes. The prompt that started it was simply
*"can't the ladder just look at the remaining VRAM and calculate the range?"*

Two terms stay empirical (CUDA overhead, a 94% planning fraction), so the
prediction **seeds** the ladder rather than replacing it, and the test asserts
the prediction brackets the measurements rather than asserting the formula
against itself.

## 5. Question the per-token cost, not just the token count

The ladder searched for a context that fit while the cost of a token went
unexamined at llama.cpp's f16 default. `gemma-4-31b` spends 630 KB per token at
f16 — 2.6 GiB at 4096 tokens, a third of the free VRAM, for about three thousand
words. Quantizing the cache to `q8_0` tripled the window on the same card.

## 6. Some flags are dependencies, not preferences

Quantized KV cache **requires** flash attention: `q8_0` without it fails at 8192
where `q8_0` with it reaches 12288. Refuse the invalid pairing at construction —
a config that fails later looks like a hardware limit.

## 7. A 500 makes clients retry the impossible

An oversized prompt surfaced as `ValueError → 500`, and miStudio's labeling loop
retried each one **three times** — three model calls on a request that could
never succeed, with nothing an operator could act on. OpenAI returns 400
`context_length_exceeded` and clients understand it.

## 8. Refusing an ignorable field is a total failure

miStudio's labeling service sends `chat_template_kwargs={"enable_thinking":
False}` on every request, on the reasonable premise that a template not
referencing the variable ignores it. True for transformers, false for this
engine — so refusing turned "labeling with a GGUF judge" into a 400 on
**every** call. Ignore and log.

## 9. The chat template seals the turn

Continuation did not fail; it silently restarted. A GGUF template closes the
final turn, so a resent partial answer leaves the model no option but to begin
again — gemma said so out loud (*"It looks like your previous message had a
technical glitch… Let's start fresh"*) and the Qwen reasoning model re-opened
its `<think>` block three times, because its template ends every fresh turn by
opening the thought channel.

Reproduce a template's *rendered output* before theorising about behaviour. The
diff between the two renderings is the whole diagnosis.

## 10. Verify against the layer that owns the thing

Two diagnostic probes in this work both failed the same way as the bug they were
chasing: one returned silently on the anomalous state, the other measured at
teardown instead of at the point of failure. A third read a stale CI run. Prove
a probe can see a known-true value before believing its "absent".

## 11. A phantom defect, recorded so it is not re-derived

A CI failure was attributed to "module-state pollution" from a `sys.modules`
swap in `test_gguf_engine.py`, and an entire known-issues entry was written
about it. Nothing was broken: since **FastAPI 0.141 / Starlette 1.6**,
`include_router` no longer flattens routes into `app.routes` — it appends one
lazy `_IncludedRouter` with no `.path`. Read `app.openapi()["paths"]`.

The sibling repo had already documented the real cause in three tests. Check the
other repo before inventing a defect.
