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
[Feature Monitoring](/features/probe-monitoring) watches **SAE feature activations**. This page runs
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

## Arming: four gates

They run in the order they are cheapest to fail.

| # | Gate | Refuses when |
|---|---|---|
| 1 | **Limit** | `PROBE_MAX_ARMED` (8) are already armed |
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
X-miLLM-Probe-Verdicts: "high-stakes";score=2.31;threshold=1.07;verdict=?1;rung=3
```

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
- **`scored: false`** means the request was not scored, and always carries a reason
  (`batched_request`, `speculative_decoding`, `continuous_batching`, `role_mask_unreliable`). It is
  not a "no" either.
- **A high score is not a cause.** A probe detects; it does not explain.

## When a probe is not scoring

An armed probe that is quiet always says why, on the page and in `GET /api/probes/status`. Silence
would read as "nothing detected", which is a claim it never made.

Requests are not scored when they are batched (`n > 1` or `extra_messages`), under speculative
decoding, or served by continuous batching.

:::warning Arming costs throughput
While any probe is armed, continuous batching is disabled and requests run serially
(`PROBE_FORCE_SERIAL`, default true). On a busy deployment that is a capacity decision, not a
detail. Turning it off does not make probes score under continuous batching — those requests are
marked `not_scored` with a reason instead.
:::

## Privacy

The socket broadcast carries **no prompt or context text**. The decoded window around the top firing
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
