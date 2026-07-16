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
from the miLLM repo. Enable it globally or per model.

**Use:** every user gets a **dial** valve (chat ⚙ → Valves):

| Dial | Effect on this user's requests |
|------|-------------------------------|
| `default` | Leave steering exactly as the server has it |
| `off` | Steering disabled for the request (λ = 0) |
| `min` / `max` | The active cluster's declared `intensity_range` bounds |
| a number `0`–`2` | That exact λ (e.g. `0.8`) |

The dial rides each request as miLLM's `steering_intensity` extension field: miLLM applies it
inside the request boundary and restores the previous steering afterwards, so **concurrent chats
with different dials never interfere**. Responses echo the effective λ in the
`X-miLLM-Steering-Intensity` header.

:::info Global vs per-request
The Admin UI's intensity slider (and `PUT /api/clusters/active/intensity`) changes the **global**
dial — it persists and affects everyone. The OWUI dial is **per request**: it overrides the stored
λ for that request only and leaves the global state untouched.
:::

Older miLLM builds without the dial simply ignore the field — enabling the Function against them
is safe and has no effect.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| Model list empty | No model loaded in miLLM, or wrong base URL — `curl http://<host>:8000/v1/models` from the Open WebUI host |
| Connection refused from container | `localhost` inside the container; use the host's LAN IP or cluster DNS name |
| Replies but no steering effect | SAE not attached, steering disabled, or strength too low — check `steering_apply_count` ([verification](/concepts/steering#verifying-steering-is-active)) |
| `429` errors under load | Serial queue full (`MAX_PENDING_REQUESTS`); Open WebUI parallel title-generation requests can pile up — raise the limit or disable title generation |
| CORS errors (browser direct) | Set `CORS_ORIGINS` to include your Open WebUI origin — see [Configuration](/reference/configuration) |
