---
sidebar_position: 2
title: "Open WebUI as a Frontend"
---

# Tutorial: Open WebUI as a Frontend

miLLM speaks the OpenAI API, so [Open WebUI](https://openwebui.com/) works as a chat frontend with zero glue code — and every conversation flows through whatever steering you have configured. This is the most comfortable way to *feel* a steering intervention: chat naturally while flipping features on and off.

## Step 1 — Point Open WebUI at miLLM

In Open WebUI: **Admin Panel → Settings → Connections → OpenAI API**:

| Setting | Value |
|---------|-------|
| API Base URL | `http://<millm-host>:8000/v1` |
| API Key | anything non-empty (miLLM does not check it) |

Save, then verify the model list refreshes — you should see the loaded model (e.g. `gemma-2-2b`) appear in the model selector. That list comes from miLLM's `GET /v1/models`.

Running Open WebUI in Docker or Kubernetes? Use an address reachable *from inside that container/cluster* — `localhost` will point at the container itself. For a K8s-hosted Open WebUI talking to a miLLM on your LAN, that's the host's LAN IP (e.g. `http://192.168.x.x:8000/v1`).

## Step 2 — Chat

Pick the miLLM model in a new chat and talk to it. Streaming, stop sequences, temperature, and system prompts all behave as with any OpenAI backend.

:::tip Base models chat awkwardly
If you loaded the **base** `gemma-2-2b` for SAE work, expect completions-style behavior rather than crisp assistant answers — base models aren't instruction-tuned. For chat-quality output use `gemma-2-2b-it`, accepting that GemmaScope features degrade somewhat off their training distribution; for clean steering experiments, prefer short factual prompts on the base model.
:::

## Step 3 — Steer mid-conversation

Leave the chat open, and in the miLLM Admin UI set a strong steering value (e.g. an *ocean* feature at 60). Ask the same question again in Open WebUI. The reply shifts topic-ward — no Open WebUI configuration changed, because steering lives server-side and applies to every request.

Flip steering off and ask once more to confirm the effect disappears.

## Step 4 — (Optional) Per-conversation profiles

Open WebUI can't send miLLM's custom `profile` parameter from the chat box, but you can approximate per-conversation steering two ways:

- **Server-side switching** — activate a different profile in the miLLM UI between conversations
- **Scripted clients** — anything that lets you set extra body parameters can pass `"profile": "<name>"` per request; see [Per-request profiles](/features/profiles#per-request-profiles-api)

## Step 5 — The Cluster Dial (Filter Function)

With a [cluster](/features/clusters) active, each Open WebUI user can dial the cluster's steering
intensity per chat — without touching the server-side state other users see.

**Install:** in Open WebUI go to **Admin Panel → Functions → Import Function** and paste the
contents of [`integrations/openwebui/millm_dial_filter.py`](https://github.com/Onegaishimas/miLLM/blob/main/integrations/openwebui/millm_dial_filter.py)
from the miLLM repo. Enable it per model on miLLM-served models. (Only enable it globally if
Open WebUI talks exclusively to miLLM — strict OpenAI-compatible providers may reject the extra
field with a 400.)

**Use (v1.3.0):** the dial appears as a **toggle chip with a sliders icon in the message
input bar** — chip off means the filter doesn't run at all (the server's stored steering
governs); chip on applies your dial. Each reply also shows a one-line **status** ("miLLM
steering: off for this reply" / "λ=1.5" / an idle hint) so you always see what was sent —
operators can silence it with the `show_status` valve. The dial itself is a per-user
**dial** valve (chat ⚙ → Valves) — a dropdown, so typos are impossible:

| Dial | Effect on this user's requests |
|------|-------------------------------|
| `default` | Use the operator's `default_dial` setting (which itself defaults to leaving steering as-is) |
| `server` | Always send nothing — the server's stored state governs, even when the operator set a default |
| `off` | Steering disabled for the request (λ = 0) |
| `min` / `max` | The steering base's declared `intensity_range` bounds (the active cluster's, or the named profile's when a request also carries `profile`) |
| `custom` | The exact λ from your `custom_lambda` valve (`0`–`2`, capped at the cluster's declared maximum; dialing below the declared floor is honored). **For circuits both ends clamp** — see Step 6. |

Operators get matching `default_dial` / `default_custom_lambda` valves that apply to users who
leave their dial at `default`.

The dial rides each request as miLLM's `steering_intensity` extension field: miLLM applies it
inside the request boundary and restores the previous steering afterwards, so **concurrent chats
with different dials never interfere**. Responses echo the resolved λ in the
`X-miLLM-Steering-Intensity` header (omitted when nothing can apply — e.g. no SAE attached).

:::info Global vs per-request
The Admin UI's intensity slider (and `PUT /api/clusters/active/intensity`) changes the **global**
dial — it persists and affects everyone. The OWUI dial is **per request**: it overrides the stored
λ for that request only and leaves the global state untouched.
:::

Older miLLM builds without the dial simply ignore the field — enabling the Function against them
is safe and has no effect.


## Step 6 — Dialling a Circuit (Feature 14)

When a **circuit** is serving (Circuits page → Activate), the same dial scales
**every layer of the circuit together** under one λ. You do not dial layers
individually — a circuit is one intervention spanning several SAEs, and each
member is re-derived from the strength it was authored with, so the dial is
absolute rather than compounding on the circuit's stored setting.

:::note
Individual members are clamped to miLLM's ±200 steering range at apply time, so
at a high λ a very strong member can reach the ceiling while weaker ones keep
scaling — compressing their relative proportions. The Circuits page reports
which members clamped.
:::

Nothing changes in how you use it: pick `off` / `min` / `max` / `custom` in
⚙ Valves exactly as with a cluster. What changes is what the status line tells
you:

```
miLLM steering: max (circuit's declared bound) · circuit "fear→threat" — causally validated (edge)
miLLM steering: λ=1.2 · circuit "hedging" — associated [UNVALIDATED]
```

### Reading the evidence rung

The phrase after the circuit name is its **evidence rung**, rendered by the
server from the evidence ladder — the filter never composes it:

| Rung | Phrase | Meaning |
|---|---|---|
| 0 | `associated` | a statistical association survived a null test |
| 1 | `suggested (attribution-supported)` | gradient attribution agrees |
| 2 | `causally validated (edge)` | a real intervention confirmed the edge |
| 3 | `faithfulness-tested (circuit)` | circuit-level necessity was tested |

A circuit **below rung 2 is marked `[UNVALIDATED]`** and is never described as
"causal". That is deliberate: a mined or attribution-supported circuit is an
*association*, and presenting it as causal at the point where it actually
influences generation would be an overclaim. Activating such a circuit in the
Admin UI requires an explicit acknowledgement for the same reason.

A circuit in **slice-fallback** mode is a special case: it is steered by its
backing *cluster profile*, not by the circuit path, so your λ is resolved
against the **cluster** envelope (floor `0.5`) and no `X-miLLM-Circuit-Rung`
header is emitted. If you see `(serving a per-layer SLICE, not the whole circuit)`, only some of
the circuit's SAEs are attached, so miLLM is serving a single-layer projection
rather than the full cross-layer intervention.

### The response header

Every reply also carries the rung as a header, so scripted clients can read it
without the status line:

```
X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"
X-miLLM-Steering-Intensity: 1.5
```

### Configuration

The circuit probe is read-only and optional. In the Filter's operator valves:

- `show_circuit_rung` (default on) — show the rung in the status line.
- `millm_base_url` (**empty by default — the probe is off until you set it**)
  — used **only** for the read-only `GET /api/circuits/active` probe. Note that
  `localhost` inside the Open WebUI container is Open WebUI itself, not miLLM
  (the same trap as Step 1):
  - Docker: `http://host.docker.internal:8000`
  - Kubernetes: `http://millm-backend.millm.svc.cluster.local:8000`

If miLLM is unreachable or running an older build without the route, the probe
degrades silently and the dial behaves exactly as it did for clusters — it
never blocks your message.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| Model list empty | No model loaded in miLLM, or wrong base URL — `curl http://<host>:8000/v1/models` from the Open WebUI host |
| Connection refused from container | `localhost` inside the container; use the host's LAN IP or cluster DNS name |
| Replies but no steering effect | SAE not attached, steering disabled, or strength too low — check `steering_apply_count` ([verification](/concepts/steering#verifying-steering-is-active)) |
| `503 queue_full` errors under load | Serial queue full (`MAX_PENDING_REQUESTS`) — miLLM returns HTTP `503` (`queue_full`) as backpressure. Open WebUI parallel title-generation requests can pile up; raise the limit or disable title generation |
| CORS errors (browser direct) | Set `CORS_ORIGINS` to include your Open WebUI origin — see [Configuration](/reference/configuration) |
| Dial has no effect | All by-design silent no-ops — check in order: Function enabled on *this model*? master valve on? SAE attached? steering enabled in miLLM (a dial never re-enables disabled steering)? an active cluster or live steering values to scale? miLLM build new enough (older builds ignore the field)? Scripted clients can check for the `X-miLLM-Steering-Intensity` response header — absent means the dial didn't apply Circuit-specific: is `min` resolving to `0`? (a circuit that declares no floor makes `min` identical to `off` — the status line says so); is the circuit in `slice_fallback` (dialled through its cluster profile, different floor); is an SAE attached on **every** member layer? |

### When `min` means `off`

Circuits floor at `0` by default, where clusters floor at `0.5`. A circuit whose
definition declares no `budget.intensity_range` therefore makes **`min`
identical to `off`** — the same output, byte for byte. The status line
discloses this explicitly (`min — this circuit declares no floor, so min is
OFF`) rather than implying a bound that is not being applied. To get a
meaningful `min`, author an `intensity_range` with a non-zero floor in miStudio
before exporting the circuit.

## Upgrading from an earlier filter version

As of **v1.4.1** the filter is titled **miLLM Steering Dial** (it was "miLLM
Cluster Dial"): it now dials whole circuits as well as clusters. Open WebUI
keys filters by their internal id, not their title, so **pasting the new
version over the existing filter upgrades it in place** — your valve settings
and per-model assignments are preserved. Creating a *new* filter instead would
leave the old one still enabled and both would apply.

One valve needs attention on upgrade: `millm_base_url` is **empty by default**,
so the circuit-rung disclosure stays off until you set it. See Step 6.
