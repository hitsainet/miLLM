---
sidebar_position: 5
title: Probe Monitors
description: "Run a linear detector trained in miStudio against live traffic, and read what it says without over-reading it"
---

# Probe Monitors

A **probe monitor** is a small linear readout trained in miStudio: a vector, a threshold, and a
record of how well it actually worked. miLLM imports one, checks it was built for exactly the model
you have loaded, proves it scores the way miStudio measured, and then reads it on live traffic.

:::note Not the Feature Monitor page
[Feature Monitor](/features/probe-monitoring) watches **SAE feature activations**. This page runs
**trained detectors**. The older page was called "Probe" until this feature arrived; two pages with
that name was a coin flip for anyone reading them.
:::

## What it does, and what it deliberately does not

It **records, shows and annotates**. It does not stop a generation, re-route it, or escalate to a
judge. That is a later phase, and the reason it is later is that thresholds have to be proven on
real traffic before anything acts on them.

So a verdict is evidence for a human or a downstream process, not a gate.

## Importing

Upload a `.probe.json` exported from miStudio, or import from HuggingFace (repos tagged
`mistudio-probe-definition`). The whole document is stored verbatim, so re-export is lossless.

`on_conflict` is `rename` or `fail`. There is deliberately **no `replace`**: overwriting a
definition in place while its probe is armed would change the detector underneath a running
monitor, while every event before and after kept the same probe id — the history would describe two
different detectors as one. Re-importing a rebuilt probe goes **disarm → delete → import**.

> **MOVING A BAR IS NOT REPLACING A DETECTOR.**
>
> A probe definition carries two kinds of fact. The DETECTOR is everything that determines what
> number the probe produces: `head.weights`, `bias`, `norm_mean`, `norm_std`, `attention_query`,
> `read.layer`, `read.hook_point`, `scope`, `basis`, the `sae` block and its `feature_indices`,
> `aggregation.rule` and its `params`, `model`, and the `evidence` that says what the number is
> evidence of. The BAR is everything that determines only where that number is cut:
> `decision.threshold`, `target_fpr`, `realised_fpr`, `threshold_source`, `calibration`,
> `windows` and `length_bands`.
>
> `replace` was refused because it replaces the first kind, and the objection above stands exactly
> as written. It turns on a specific property of `probe_events`: the row records a `score` whose
> MEANING comes from the detector, and nothing on the row records which detector produced it.
> Change the weights and event #1's `score = 2.9` and event #900's `score = 2.9` are measurements
> of different quantities under one id, with nothing to tell them apart.
>
> A moved bar is not that, for one concrete reason: the event row ALREADY records the bar it was
> judged against, per verdict, at judgement time — including the length-band override — and nothing
> joins an event back to `probes.threshold`. After a re-cut, event #1 still says it was judged at
> 2.8786 and event #900 says 2.4011; both are true and both remain comparable, because the score
> beneath each was produced by the same weights at the same layer under the same scope with the
> same rule. The score is the measurement; the bar is the line drawn across it.
>
> THE RULE: a probe's identity is everything that determines its SCORE; its bar is everything that
> only determines the CUT. The first may never change in place under a probe id. The second may,
> through `POST /api/probes/{probe_id}/recalibrate`, which is `extra="forbid"` and therefore
> structurally incapable of carrying a detector, refuses any cut it cannot match to
> `provenance.probe_id`, never stores the incoming object as the definition, and refuses a
> threshold with no budget and no named source. `on_conflict` remains `rename|fail`.

## Arming: four gates

They run in the order they are cheapest to fail.

| # | Gate | Refuses when |
|---|---|---|
| 1 | **Limit** | `PROBE_MAX_ARMED` (4, measured; the cost budget is the real limit) are already armed |
| 2 | **Identity** | the loaded model is not the one the probe was fitted on |
| 3 | **Evidence** | the probe is below rung 2 and you have not acknowledged it |
| 4 | **Parity** | this build does not reproduce miStudio's recorded scores |

### Identity

Four fields must match and they **refuse** on mismatch: `hf_id`, `d_model`, `n_layers`, and the
**chat-template hash**. Every mismatch is named, not just the first — otherwise you fix one field,
retry, and meet the next, never learning whether you loaded the wrong model or imported the wrong
probe.

The template hash is the least obvious and the most valuable. Same weights plus a different chat
template is a different token stream, so the probe reads positions that do not mean what it
learned — and every other field would match.

The **revision warns rather than refusing** when miLLM cannot establish which commit is on disk.
But two *known* revisions that differ do refuse: not knowing and knowing-they-differ are different
situations.

:::warning Why a probe refuses where a circuit binds
A circuit can be bound across an identity mismatch, because it names features a human can reason
about. A probe cannot. Its weights are a direction in one specific model's residual space; read in
another model's space they produce numbers that are plausible, stable, well-behaved, and about
nothing. There is no symptom.
:::

Arming, disarming and checking parity all happen on this page. A refusal names the gate it came
from, because "wrong model" and "needs your acknowledgement" call for entirely different actions.

### Parity

Before arming, the definition's test vectors are re-scored here and compared with the scores
miStudio recorded, within 1e-3. If they differ, the probe is refused — the alternative is a
detector reporting under a measurement taken of something slightly different.

**The gate is the SCORE, not the per-token trace.** Parity passes or fails on the combined
score — what the probe actually decides with — and reports per-token divergence beside it
without gating on it.

That is deliberate, and measured. miStudio scores probes in float16; miLLM serves bfloat16,
because float16 overflows on models trained in bfloat16. Over sixteen real vectors that costs a
worst case of **0.098 on the score** against a threshold of 2.879 — 3.4% — while per-token
traces diverge by up to 6.9, because the recorded tolerance is *absolute* against values
reaching 55. A gate at that tightness is reachable only by computing bit-identically, and an
independent implementation is exactly what parity exists to verify.

It remains a real gate: the wrong dictionary, the wrong layer, the wrong hook or the wrong
feature selection all move the score by far more than precision does. A basis error found during
acceptance moved it to **1.68**, seventeen times the tolerance.

Parity scores from the recorded **`token_ids`**, never by re-rendering `messages`. miStudio measured
the ids reproducing exactly while re-rendered `messages` missed by over 1.1, because `messages` is a
reconstruction whose re-render adds template tokens. Whether `messages` re-renders to the same ids
is reported **separately**, as tokenization drift, and never affects pass or fail.

**Parity can also be checked without arming** — *Check parity* on a probe's row. That is the useful
thing to do after reloading a model: the same definition against a differently-loaded model is
exactly the case parity exists to catch, and the report is stored either way so you can read it
later. Checking never arms, and never marks the probe armed.

:::note Scope and reproducibility
Only `scope: all` is exactly reproducible. For `prompt` and `response`, miStudio recorded scores
under a narrower internal mask and the contract does not carry which positions those were — so
parity reports that it cannot check, and the probe is refused rather than armed unverified.
:::

### The evidence rung

| rung | what it means |
|---|---|
| 0 | trained |
| 1 | detects on held-out data |
| 2 | detects on unseen tasks |
| 3 | detects on unseen tasks, **compared with** a judge |

Arming below rung 2 opens a dialog asking why, and records the answer. The acknowledgement is
stored separately from the one in the definition — the person who exported a weak probe and the person arming it against live traffic are
not necessarily the same, and only the second is choosing to monitor with it.

:::warning "Compared with", not "beats"
Rung 3 means a judge scored the same data. It may have won. On miStudio's reference run it did: the
judge averaged 0.8744 AUROC against the 1.2B probe's 0.7938, on all five sets. A probe's case on a
small model is cost and latency, not accuracy.
:::

## Getting a probe here

Two paths, and the page offers both:

- **Import definition** — a `.probe.json` file exported from miStudio.
- **Browse Hub** — search Hugging Face for repos tagged `mistudio-probe-definition`, open one, and
  import a single definition. Anonymous and read-only.

A listing shows each definition's rung where the repo's manifest carries one, and says **"rung not
stated"** where it does not. That distinction matters: rung 0 means *trained, and nothing further
measured*, which is a claim about the probe's evidence — not the same as an unknown.

## Reading a verdict

**Non-streaming** responses carry a header:

```
X-miLLM-Probe-Verdicts: "high-stakes";score=2.31;threshold=1.07;verdict=?1;rung=3;window=all;behaviour="high-stakes";probe="pr_0a1b2c3d4e5f"
```

The header is an RFC 8941 list: every string is quoted and escaped, every number a decimal with at
most three fractional digits (the body field and the stream chunk carry the full-precision score).
Each member names its `behaviour` and `probe`, and one behaviour's members are always next to each
other. The header is **bounded** (`PROBE_VERDICT_HEADER_MAX_BYTES`, 2,048 bytes): past it, whole
behaviours are left out and a last member `"...";truncated;total=<n>` says how many results there
were. Nothing is lost — the body field, the stream chunk and the events list always carry every
result.

**Streaming** responses carry a final chunk before `[DONE]`, shaped like OpenAI's usage chunk:

```json
{"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[],
 "millm_probe_verdicts":[{"name":"high-stakes","scored":true,"score":2.31,
   "threshold":1.07,"verdict":true,"rung":3,"rung_language":"detects on unseen tasks, compared with a judge"}]}
```

Both clients we tested handle this: the OpenAI SDK exposes the extension through `model_extra`, and
Open WebUI skips chunks with empty choices. **Open WebUI will not display the verdict** — its filter
surface has no outlet hook — so a verdict reaches a human through this page, not the chat.

### Three things a verdict does not say

- **`verdict` absent** means no threshold was placed. The probe ranks without deciding; it is not a
  "no".
- **`scored: false`** means the request was not scored, and always carries a reason code (the
  [list below](#outcomes-and-reasons)). It is not a "no" either.
- **A high score is not a cause.** A probe detects; it does not explain.

### Where the bar is

A verdict fires when **`score >= threshold`** — a score exactly on the bar fires. miStudio cuts the
bar at a negative's score and counts that negative as admitted, so this is the rule that makes the
definition's stated false-positive rate true here. It is the same everywhere a verdict is reported:
the header, the streaming chunk, the event, and stateless scoring below.

## The result contract

Since Feature 38, every place a probe result appears carries the **same object**, built once per
probe and window when the request finishes: `mi.probe.result/v1`, published as
`docs/schemas/probe-result-v1.json`. The places are the non-streaming response body, the streaming
terminal chunk, the stored event, the `probe:results` socket message, the events list and detail
routes, session timelines, `POST /api/probes/score` and Batch output lines (each keeps its older
keys beside the contract fields), and, when [telemetry export](./telemetry-export.md) is on, the
exported `mi.probe.result` log record.

**Non-streaming responses** now carry the results in the body too, beside `choices`:
`millm_probe_verdicts` is a list of contract objects. It is absent when nothing is armed, so an
unarmed response is byte-for-byte what it was.

The fields worth knowing:

| Field | Means |
|---|---|
| `result_id` | One id (a time-ordered UUID) for this result on every surface and in every export, retry included. Filter events by it: `GET /api/probes/events?result_id=` |
| `request_id`, `trace_id`, `span_id` | The request; the W3C trace it ran in (from a `traceparent` header, or minted); the probe-evaluation span. `GET /api/probes/events?trace_id=` returns that request's results |
| `outcome`, `reason` | `completed`, or `skipped`, `failed` or `interrupted` with a reason code (below). A reason is a code, never free text |
| `status` | `final` for a finished request. An interrupted one is `partial` (a score over the tokens seen, with no verdict) or `unavailable` (no score) |
| `persistence` | Whether miLLM stored it: `persisted`, `not_persisted` (stateless scoring, or a failed write) or `unknown` |
| `detector_digest`, `detector_digest_source`, `identity_algorithm` | What produced the score: a digest of the detector (weights, layer, scope, rule, model). `producer` when miStudio supplied it and miLLM verified it, `computed` when miLLM computed it, `unknown` otherwise |
| `decision_digest`, `decision_revision`, `threshold_revision` | Where the bar was, and which cut: a recalibration moves both revisions |
| `engine`, `execution_mode`, `precision`, `served_model`, `model_revision`, `model_load_id`, `template_sha256` | How the request was actually served |
| `score`, `threshold`, `verdict`, `fires`, `n_scored_tokens`, `rung`, `rung_language` | The reading, as before |

The header `X-miLLM-Probe-Verdicts` keeps its keys and carries no `result_id` (it has to stay
short); its `reason` is now the code.

### Interrupted requests

A stream that ends early (the client disconnects, the request is cancelled, generation raises, or
the generation thread hangs) still records a result for every armed probe and window, with
`outcome: interrupted`, the reason, and `status: partial` (a score over the tokens it saw, never a
verdict) or `unavailable`. It cannot reach the client, whose stream is
gone; it is stored and exported like any other.

### Outcomes and reasons

The published list. Codes may be added within v1 (a consumer reads an unknown code as "other");
none is removed or renamed.

<!-- reason-codes:begin (generated by tests/support/telemetry_signal_doc.py; do not edit) -->

| Reason | Outcome | Means |
|---|---|---|
| `batched_request` | skipped | more than one row in the forward pass (n > 1 or a batch) |
| `continuous_batching` | skipped | served by the continuous-batching engine, which has no hook |
| `engine_unsupported` | skipped | served by an engine with no forward hooks (llama.cpp) |
| `judge_traffic` | skipped | a judge's own call (X-miLLM-Origin: judge); never user traffic |
| `last_user_span_unknown` | skipped | the last_user window needs a span and none was stated |
| `last_user_span_unresolved` | skipped | the newest user message could not be located |
| `no_chat_template` | skipped | the tokenizer has no chat template to locate a turn with |
| `no_scored_tokens` | skipped | the window selected no position the probe observed |
| `no_user_header` | skipped | the template's user header could not be found in the ids |
| `no_user_turn` | skipped | the conversation has no user message |
| `prompt_boundary_unknown` | skipped | the window needs the prompt boundary and none was stated |
| `rule_worker_unavailable` | skipped | operator code is scored off the live path and no runner result is attached (D41) |
| `speculative_decoding` | skipped | a draft model proposed tokens; positions are not served |
| `text_completion_has_no_user_turn` | skipped | a text completion has no user message |
| `encoder_failed` | failed | the k-sparse encoder raised (detail in the log) |
| `head_mismatch` | failed | the head's width does not match the activations |
| `non_finite_score` | failed | the score was NaN or infinite, so no score is reported |
| `prompt_boundary_conflict` | failed | two different prompt lengths were stated (detail in the log) |
| `prompt_boundary_invalid` | failed | a negative prompt length was stated (detail in the log) |
| `rule_error` | failed | a validator raised while the signal ran (detail in the log) |
| `rule_set_vectors_failed` | failed | the rule set disagrees with its own cases under this build, so it is not evaluated |
| `rule_signals_disabled` | failed | rule signals are switched off (RULE_SIGNALS_ENABLED) |
| `rule_timeout` | failed | the request's rule time budget ran out before this signal ran |
| `cancelled` | interrupted | the request was cancelled before it finished |
| `client_disconnected` | interrupted | the client went away before the stream finished |
| `generation_error` | interrupted | generation raised before it finished |
| `generation_thread_hung` | interrupted | the generation thread did not stop in time |

<!-- reason-codes:end -->

### Digests on import

A definition exported by a recent miStudio carries an `identity` block with its own digests. Import
(file or Hub) recomputes both and refuses a mismatch with `409 PROBE_IDENTITY_MISMATCH`, naming the
field and both values, and an algorithm it does not know with `409
PROBE_IDENTITY_ALGORITHM_UNKNOWN`. Either way nothing is stored: re-export the definition from
miStudio. A recalibration that sends `decision_digest` is checked the same way against the
decision it would store.

## When a probe is not scoring

An armed probe that is quiet always says why, on the page and in `GET /api/probes/status`. Silence
would read as "nothing detected", which is a claim it never made.

Requests are not scored when they are batched (`n > 1` or `extra_messages`, reason
`batched_request`), under speculative decoding, served by continuous batching (chat, streaming chat
and text completions all record `continuous_batching`), or served by llama.cpp
(`engine_unsupported`). Every one of these still writes an event and still sends the header — on
`/v1/completions` too, which never sent `X-miLLM-Probe-Verdicts` before Feature 27.

## Scoring stored text

`POST /api/probes/score` asks imported probes about inputs you already have — rows in a dataset, a
transcript, a definition's test vectors — **without arming anything and without recording
anything**:

```json
{"probe_ids": ["prb_…"],
 "inputs": [{"token_ids": [1, 2, 3], "prompt_tokens": 2},
            {"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}],
 "windows": ["all", "prompt"]}
```

- **Nothing is persisted.** No event, no change to what is armed, no change to a stored parity
  report. An armed probe on the same layer sees nothing either.
- **Offline equals live.** The probe is built, run and decided by the same code live serving uses,
  on the same layer hook, unsteered (every attached SAE is suppressed for the scoring forward).
- **Same ids as a live chat.** A `messages` input is tokenized exactly as live serving tokenizes a
  chat — one BOS, never two. Before 2026-10-08 both carried a duplicate BOS on Llama 3, gemma and
  LFM2.5, so scores from that period differ slightly from today's (see *How a chat becomes token
  ids* in the OpenAI-compatible API page).
- **One input at a time.** Each input takes its own place in the request queue, so a chat request
  waits at most one input's forward behind a scoring batch. Inputs are never packed together:
  bfloat16 results change with batch shape.
- **Window boundaries are never guessed.** `token_ids` may carry `prompt_tokens`; a `messages` input
  ending in an assistant turn derives it; a `messages` input ending in a user turn is all prompt.
  Otherwise a `prompt` or `response` window says `prompt_boundary_unknown`, and `last_user` on bare
  `token_ids` says `token_ids_have_no_turns`.
- **`verdict` is three-valued**, `rung_language` is verbatim, and `provisional` is carried as
  recorded — a provisional verdict is a ranking, not a rate.
- **Parity is reported, not required.** Each probe says whether its stored parity passed, failed or
  never ran, and which model load it was checked against (`"unknown"` for older reports).
- **Wrong model:** with `probe_ids` given, the request is refused naming every mismatched field, as
  arming refuses. With `probe_ids` omitted, every matching probe is scored and the others are listed
  under `skipped` with their reason.
- **If the model changes mid-request**, the remaining inputs fail with `MODEL_CHANGED`; earlier
  results are kept. No input is scored on a different model from the first.

:::note `text` inputs
`text` is scored as one user turn, the way miStudio builds its training corpus from prose: the
result is identical to sending `messages` with that one user turn. This render was checked on the
GPU node against a miStudio probe trained on prose, and reproduced miStudio's AUROC on both of its
evaluation sets.
:::

Limits: `PROBE_SCORE_MAX_INPUTS` (64) inputs and `PROBE_MAX_ARMED` (the arm limit, 4 by default) probes per request;
token ids must lie inside the model's vocabulary; each input must fit the context window.

:::warning Arming costs throughput
While any probe is armed, continuous batching is disabled and requests run serially
(`PROBE_FORCE_SERIAL`, default true). On a busy deployment that is a capacity decision, not a
detail. Turning it off does not make probes score under continuous batching — those requests are
marked `not_scored` with a reason instead.
:::

## Sessions

While at least one probe is armed, miLLM groups the requests of one conversation into a **session**
and records which turn of it each verdict belongs to. The **Sessions** page shows each session as a
timeline: one row per turn, one column per behaviour (a probe whose definition names no behaviour is
its own column).

**How a conversation is recognised.**

- A client can name it with the `X-miLLM-Session` request header: 1 to 128 printable characters,
  stored as given. Do not put personal data in it. An invalid value is refused with 400
  `SESSION_ID_INVALID`. The response says which session it joined in `X-miLLM-Session-Id`.
- **Open WebUI:** in *Admin → Settings → Connections*, edit the miLLM connection and add the custom
  header `X-miLLM-Session` with the value `{{CHAT_ID}}`. Each Open WebUI chat then becomes one
  session. No filter change is needed.
- Without the header, a request that resends the conversation so far (as every chat client does)
  joins the session it continues. A client that rewrites the earlier reply still joins when exactly
  one conversation matches; when two match, the request opens its own session, marked as ambiguous.
- A session idle for 30 minutes (`PROBE_SESSION_IDLE_MINUTES`) ends; a later continuation opens a new
  session linked to it.

**Text recording is off by default** (`PROBE_TRANSCRIPTS`, Settings → Environment). When it is on,
each monitored chat turn keeps the newest user message and the visible reply (never the reasoning)
for 7 days (`PROBE_TRANSCRIPT_MAX_AGE_DAYS`, at most `PROBE_TRANSCRIPT_MAX_TURNS` turns in all). Every
response of a monitored request says whether its text was kept: `X-miLLM-Transcript: recorded` or
`off`. Text is served only by `GET /api/probes/sessions/{id}/transcript`, never on the live socket,
and `DELETE /api/probes/sessions/{id}` removes a session's text at once (`events=keep` keeps its
verdicts, detached; `events=delete` removes them too). A background pass removes expired text,
sessions and verdicts every `PROBE_RETENTION_INTERVAL_S` seconds.

## Behaviour sets

A behaviour is usually watched by several probes — different layers, rules and windows, each looking
at a different aspect — and an application blends their results however it wants. miLLM treats
every imported probe whose definition names one behaviour key as that behaviour's **set**. Nothing
stores the set; it is whatever is imported under the key (one version at a time: when probes of
two versions are imported, name one with `version`, or the probes with `probe_ids`).

- **Arm a behaviour in one call** — the **Arm** button on the behaviour's group on this page, or
  `POST /api/probes/behaviours/{key}/arm`. Every probe runs its own gates (identity, rung, scope,
  parity, cost), then the set is checked once against the behaviour limit, the probe backstop and
  the cost budget. **All or nothing:** if any probe fails, none is armed, and the refusal lists
  every probe with the gate that stopped it.
- **Dry run first.** The **Dry run** button (`dry_run: true`) runs every gate — parity and cost
  included — arms nothing, and shows each probe's result and the projected cost against both halves
  of the budget.
- **Partial sets say so.** A probe imported later into an armed behaviour stays disarmed, and the
  group reads "partial: 3 of 4 armed", naming the missing probe. Arming the behaviour again arms
  only the missing ones.
- **After an unload or a restart** every probe is disarmed (as before); the page names the
  behaviours that were armed, so each comes back with one click.
- **Disarm a behaviour** with **Disarm** (`POST /api/probes/behaviours/{key}/disarm`). A layer hook
  another behaviour still uses stays.

Every result a request produces — header, body field, stream chunk, stored event and the live
`probe:results` feed (one message per request) — is listed in one order, behaviour by behaviour,
and each result of a behaviour carries a `set` block (`armed_members`, `total_members`,
`complete`, `missing_probe_ids`, `members_digest`), so an application knows whether its blend's
inputs were all there.

`POST /api/probes/score` accepts `behaviours: [key]` to score a behaviour's probes on stored text
— armed or not — before arming them.

## Rule sets and rule signals

A **rule set** is a miDataworks rule-set document: patterns (a linear-time dialect — no
back-references or look-around, so no pattern can stall serving), proximity to a cue phrase,
validators (check digits, length and prefix tables), weighted word lists, and compositions of those.
Import one with `POST /api/rule-sets/import` (the body is the document as the producer published
it); `POST /api/rule-sets/validate` runs every check and writes nothing. Each set must pass its own
cases to be accepted, and re-proves them at every start: a set that a new miLLM build disagrees
with is marked `vectors_failed` and stops being evaluated.

**Attach** a set to a behaviour with `PUT /api/probes/behaviours/{key}/rule-signals`. While at
least one probe of that behaviour is armed, every named expression of the set runs beside the
probes on every request, on the windows those probes report, within `RULE_TIME_BUDGET_MS` (20 ms).
Each reports as its own result — `kind: "rule"`, with `fires` and a `count`, never the matched text
— next to the probe results in the header, the body, the stream, the events list, the session
timeline and Batch lines. A rule result never changes a probe result: a set's vetoes and output
columns are for building datasets and are not applied here.

- Judge traffic and requests the probes do not score report each signal skipped with the same
  reason; nothing runs.
- `RULE_SIGNALS_ENABLED=false` (Settings → Environment) reports `rule_signals_disabled` for every
  attached signal and keeps the attachments.
- **Operator code is never run during a request.** A set may name a `code_rule` signal; it is
  accepted, and its signals report `skipped` / `rule_worker_unavailable` until the code runner
  scores them after the response.
- `POST /api/rule-sets/{id}/evaluate` runs a set on text you send; `return_rule_positions: true`
  returns the matched spans of that text, and only of that text.

## Probe limits per model

**The limit counts behaviours** (Feature 39). A set of nine probes is one behaviour; the cost budget
still charges every probe. `PROBE_MAX_ARMED_BEHAVIOURS` (2) applies to a model with no approved
value of its own; `PROBE_MAX_ARMED` (now 64) stays as a backstop on probes.

Each model gets an **explained recommendation** in its **Probe capacity** panel: how many
behaviours of a reference set (nine probes over three layers, mixed rules) fit within 10% of the
model's measured decode step and 5% of its prefill step, with the budgets that admits and the
reasons in words and numbers. **Approve** writes exactly those values (the server checks the
recommendation has not changed since you read it); **Override** sets your own. The panel shows where
each limit came from and who set it, and marks it **stale** when the model, its precision or its
card has changed since the measurement — the approved values stay in force until you approve again.

How many probes a model can carry depends on the model and the card. Each probe costs a little on
every forward pass (a fixed part per hooked layer plus its own arithmetic), and the budget allows
`PROBE_MAX_OVERHEAD_MS_PER_PASS` per generated token and `PROBE_MAX_PREFILL_MS_PER_1K` per 1,000
prompt tokens, of which arming fills at most `PROBE_BUDGET_HEADROOM`. On the RTX 3090 (2026-10-21):

| Model | Dense probe, per generated token | Per 1,000 prompt tokens | Fits the default budget |
|---|---|---|---|
| LFM2.5-1.2B | 0.050 + 0.025 × N ms | 0.041 + 0.057 × N ms | 6 (the per-token budget binds) |
| Llama-3.1-8B | 0.076 + 0.024 × N ms | 0.054 + 0.106 × N ms | 3 (the prompt budget binds) |

A k-sparse probe costs about five times a dense one per token, so one fits the default budget.

Each model can carry its **own limits**: open the model's details on the **Models** page; the
**Probe capacity** panel shows the measured lines, the limits they imply, the limits in force and
where each comes from, and how close the armed set is to each. Set the most armed probes, and each
budget in milliseconds or as a percentage of the model's measured step time (empty = the global
default). **Measure on its card** times the loaded model's own step time and the cost of `mean`,
`max` and `attention` probes on the card(s) it runs on; it needs the model loaded and nothing armed, and it
uses the graphics card. Over the API: `GET /api/models/{id}/probe-capacity`,
`PUT /api/models/{id}/probe-limits`, `POST /api/models/{id}/probe-capacity/measure`.

## Privacy

The socket broadcast (`probe:results`) carries **no prompt or context text**. The decoded window around the top firing
position is stored on the event and served only by the single-event detail route, so a reviewer
asks for one conversation rather than a dashboard receiving a feed of them.

## Overhead

Under 5 ms per request at 4k tokens with two probes armed, with exactly one device-to-host copy per
forward pass however many probes are armed. `GET /api/probes/status` reports the last request's
measured overhead and the warning threshold.

## Limits worth knowing

- **Probes cannot run on GGUF models** — llama.cpp exposes no module tree to hook.
- **A probe can be suppressed.** Published work shows adversarial inputs pushing a probe down. It is
  the cheap first stage of a cascade, not a perimeter.
- **Recall at a tight false-positive budget is modest.** On miStudio's reference run the 8B probe
  caught 47% of positives at a 1% false-positive rate. It is a triage signal.
- **Probes read the pre-steering residual**, so steering cannot change what a probe sees.
