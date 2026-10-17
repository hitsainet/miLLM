---
sidebar_position: 10
title: Weights (attach and step barrier)
---

# Weights: attachment and the step barrier

These routes let one trainer process (for example miForge) **attach** to the weights miLLM is
serving, write to them in place, and have miLLM stop serving for exactly as long as a write is in
progress. It is an optional hot-update mode: nothing here is needed to serve a model, and with
`WEIGHTS_TOKENS` empty every gated call is refused.

All routes are under `/api/weights` and answer in the management envelope
`{"success": ..., "data": ..., "error": ...}`. Checkpoint save, load and lineage are on
[Weights: checkpoints](weights-checkpoints.md); adapters are on the
[Management API](management-api.md).

## Deployment: sharing the IPC namespace (Kubernetes)

The handles `POST /attach` returns are CUDA inter-process handles. A trainer in **another pod**
can only open them when both pods share the host's IPC namespace, so on Kubernetes both pods set
`hostIPC: true` (operator decision D1, 2026-10-16). That is not in the default deployment:
`hostIPC` lets a pod see every IPC object on the node, so it is opt-in, used only while hot-update
is in use.

- miLLM's half is the kustomize Component `k8s/components/hostipc`, applied by the overlay
  `k8s/overlays/hostipc` (`k8s/base` plus that one field on `millm-backend`). ArgoCD deploys
  `k8s/base`, which never references it; `tests/unit/test_k8s_hostipc_component.py` fails if it
  ever does.
- **To enable:** add `components: [../components/hostipc]` to `k8s/base/kustomization.yaml` in
  a commit (the path ArgoCD deploys; do not repoint the Application, whose path Image Updater keys
  its digest pins on). The backend restarts (Recreate): check for running GPU work first.
- **The same uid on both sides.** A shared IPC namespace is necessary and not sufficient: CUDA and
  torch create their IPC files in `/dev/shm` owner-only (0600, whatever the umask), so the trainer
  must run as miLLM's uid. miLLM's image user is pinned to uid/gid 1000 and the component states
  it on the backend container; miForge's component runs its training worker as uid 1000 while it
  is on. Without that, attach succeeds and the first handle open fails with
  `cudaErrorMapBufferObjectFailed` ("Failed to open/create interprocess memory file") — measured
  on 2026-10-17. Security note: while the component is on, a process with uid 1000 in the trainer
  pod can open miLLM's shared-memory files; that is the access CUDA IPC needs, and why the mode is
  opt-in and switched off after the session.
- Enable miForge's half as well (its own overlay in miForge's `gitops/`); one pod with `hostIPC`
  on its own changes nothing.
- **To disable:** remove the `components:` line from `k8s/base/kustomization.yaml` again.

Run with Docker Compose on one host, the same need is met with `ipc: host` on both services.

## Authentication and headers

| Header | On | Meaning |
|---|---|---|
| `Authorization: Bearer <token>` | every route except `GET /status` | A token from `WEIGHTS_TOKENS` (see [Configuration](../reference/configuration.md)). Each token has a `label` and a `role`, `trainer` or `operator` |
| `X-miLLM-Attachment: <attachment_id>` | the holder routes | The ID `POST /attach` returned. Never put it in a path or a query |

A **holder** call needs a valid token, the attachment ID, **and** the token whose label attached.
Checks run in this order: token (`401 WEIGHTS_UNAUTHORIZED` with `WWW-Authenticate: Bearer`, then
`403 WEIGHTS_FORBIDDEN` for the wrong role), nothing attached (`409 NOT_ATTACHED`), a missing or
wrong attachment ID or label (`403 NOT_ATTACHMENT_HOLDER`), then the state checks below. A missing
header is never a `422`.

`GET /status`, every `/v1` route, `/api/health/*` and every other management route stay
unauthenticated. A token is never logged, returned or echoed in an error.

## Routes

| Route | Who | Body returned |
|---|---|---|
| `POST /attach` | `trainer` or `operator` token (no attachment header) | `attachment_id` (once), `model {model_id, name, manifest_hash}` (`name` is the base model's served name, also while an adapter is applied), `weights_state`, `steps_since_load`, `heartbeat {timeout_s, interval_hint_s}`, `producer {torch, cuda, driver}`, `storages[]`, `parameters[]`, `release {pinned_bytes, contract}` (see [Releasing the handles](#releasing-the-handles-what-stays-pinned)), `config` (the served model's architecture config exactly as loaded, every attribute stated; `null` when it has none — rebuild the graph from this, not from a copy) |
| `POST /detach` | holder | `{weights_state, barrier}` |
| `POST /heartbeat` | holder | `{attachment_state, timeout_s, interval_hint_s, weights_state}`: the staleness window and the suggested interval (`WEIGHTS_HEARTBEAT_INTERVAL_HINT_S`) |
| `POST /quiesce` | holder | `{held_requests, held_batch_chunks, weights_state, quiesced_at, hold_deadline}` |
| `POST /resume` | holder | `{barrier, weights_state}` |
| `POST /commit-begin` | holder | `{weights_state, began_at, watchdog_ms}` |
| `POST /commit-end` | holder | `{weights_state, steps_since_load, commit_ms, overran}` |
| `POST /clear-stale` | `operator` token only (a `trainer` token is `403 WEIGHTS_FORBIDDEN`) | `{cleared {role, label, stale_since}, weights_state, barrier}` |
| `GET /status` | anyone, no token | see [Status](#status) |

Every gated call writes one `weights_audit` log line, accepted or refused: operation
(`weights_attach`, `weights_detach`, `weights_heartbeat`, `weights_quiesce`, `weights_resume`,
`weights_commit_begin`, `weights_commit_end`, `weights_clear_stale`; a token refusal on the checkpoint
routes logs `weights_save` or `weights_load`, while their accepted and other refused lines still log
`save` and `load`), outcome, code and reason, role, label, an 8-character reference to the attachment
(never the ID), the resident model's served name, the weights state after the call
(`weights_state`) and before it (`weights_state_before`), and `elapsed_ms` and `steps` for a
commit-end. A token refusal (`401`/`403`) is audited too.

`POST /api/weights/save` is also a holder call; it is documented with load and lineage on
[Weights: checkpoints](weights-checkpoints.md).

## The write cycle

```
attach ─▶ quiesce ─▶ commit-begin ─▶ (write) ─▶ commit-end ─▶ resume ─▶ … ─▶ detach
```

1. **Attach** (`trainer` or `operator` token). One attachment at a time: a second is
   `409 ALREADY_ATTACHED`, naming the holder's role, label, state (`active` or `stale`) and
   `attached_at`. With nothing resident it is `400 MODEL_NOT_LOADED`. While attached, every model
   load, unload and swap (including a `/v1` auto-load) is refused with `409 ALREADY_ATTACHED`,
   stale or not. Attaching to a `torn` set is allowed: that is how a restarted trainer finds out
   (the response's `weights_state` says `torn`).
2. **Quiesce.** miLLM finishes the requests already running, holds new ones, and returns once no
   forward pass runs. A drain that takes longer than `WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S` is
   `409 MODEL_BUSY` (`reason: drain_timeout`) and holds nothing; a full queue is `503 QUEUE_FULL`.
   Quiesce also drops any cross-request cache, so nothing computed on the old weights is reused.
   It is refused while the weights are not `clean` (`409 WEIGHTS_NOT_CLEAN`), when this attachment
   already holds the barrier (`409 ALREADY_QUIESCED`), while a save, a weights load or an adapter
   change holds it (`409 PAUSE_IN_PROGRESS`), and when continuous batching is on or
   `MAX_CONCURRENT_REQUESTS` is not 1 (`409 BARRIER_UNAVAILABLE`, `reason` `cbm` or
   `concurrency`; `GET /status` reports the same condition as `pause_unavailable`).
3. **Commit-begin.** The weights are now `writing`. Allowed only while quiesced (`409 NOT_QUIESCED`)
   and `clean` (`409 WEIGHTS_NOT_CLEAN`); refused with `409 PAUSE_IN_PROGRESS` while a save or an
   adapter change holds the barrier.
4. **Write**, then synchronize your CUDA stream, then **commit-end**: the weights are `clean` again
   and `steps_since_load` goes up by one. Only the attachment that began the commit may end it
   (`403 NOT_ATTACHMENT_HOLDER`); with no commit open it is `409 NO_OPEN_COMMIT`. Serving stays
   quiesced after commit-end, and the hold timer restarts in full.
5. **Resume** releases the held requests. Resume while serving is a no-op `200`. Resume sent while
   quiesce is still draining in-flight work is `409 NOT_QUIESCED` (`reason: quiesce_in_progress`):
   wait for quiesce to return, then resume.

> **miLLM cannot detect a write outside the bracket.** Every write to resident tensors must sit
> between `commit-begin` and `commit-end`. A write made at any other time is served, half-written,
> to whoever asks next. (BRD-01 R-01.21.)

## States

| `weights_state` | Served? | How it is entered | How it is left |
|---|---|---|---|
| `clean` | yes | start-up; commit-end; a load | commit-begin |
| `writing` | no: requests wait (the barrier holds the one slot) and are refused if the set tears | commit-begin; a weights load's write | commit-end; the watchdog; detach |
| `torn` | **no**: every `/v1` call is `503 WEIGHTS_TORN` with `Retry-After` | the watchdog; detach while writing; a parameter's address moved | a late commit-end from the attachment that began the commit (recorded as an overrun); a load through `POST /api/weights/load` |

`barrier` is `serving` or `quiesced`. While `torn`, a management load of another model and the
`/v1` auto-load are refused with `503 WEIGHTS_TORN` naming `POST /api/weights/load`; a management
load of the resident model itself does not recover a torn set either. Recover with
`POST /api/weights/load`, giving a checkpoint, or `{"model_id": N}` to reload the model row inside
the barrier. A management **unload** is allowed: with nothing resident the state is `clean` by
construction, and the held slot is released. Batch rows do not fail on `torn`: the batch waits
with `waiting_reason: weights_torn` and resumes once the weights are restored.

`torn` survives detach, clear-stale and re-attach within one process. A **restart** starts
`clean` and `serving`, with no attachment, no `last_save` and no recorded overrun or hold
timeout; `steps_since_load` is `null` until a model is resident and `0` once one is.

## Unsaved commits and the auto-load

Detaching keeps the trained values in place. While those values carry commits that no checkpoint
records, a `/v1` request that names **another** model is refused instead of auto-loading it, since
the load would discard the commits: `409 WEIGHTS_UNSAVED` (on `/v1`, `409 weights_unsaved`, typed
`invalid_request_error`). Nothing is loaded or unloaded. The message and `details` name the
resident's served name, the number of unsaved commits and the ways forward.

"Unsaved" is `steps_since_load` minus the count recorded when the resident was last saved: a save
records the counter, and any load resets it to 0. Commits made after a save are therefore counted
again from the save. The refusal ends when either of these happens:

- **A save.** `POST /api/weights/save` records the commits. It is a holder call, so re-attach
  first.
- **An explicit operator load** replaces the weights: `POST /api/weights/load` with a checkpoint
  or `{"model_id": N}`, or the management load `POST /api/models/{id}/load`. These are never
  refused for unsaved commits. They are the deliberate way to discard them.

A request naming the resident model itself, by its served name or its base name, is served as
usual. A request naming the catalogue row a save renamed is refused separately: see
[the served name after a save](weights-checkpoints.md#the-served-name-after-a-save-or-a-load).
`X-miLLM-Load-Policy: refuse` still refuses any swap first. A batch never loads a model, so this
rule does not apply to it. (Decision D-45.)

## Timers

| Setting | Default | What it does |
|---|---|---|
| `WEIGHTS_HOLD_TIMEOUT_S` | 5 s | A quiesce left open without commit-begin or resume resumes on its own (only while `clean`); status records `last_hold_timeout_at`. Suspended while a commit is open or a save or adapter change has taken over the hold; restarted in full by commit-end |
| `WEIGHTS_COMMIT_WATCHDOG_MS` | 200 ms | A commit open longer than this becomes `torn`. A commit-end that arrives later still returns `clean` with `overran: true`, recorded in `last_overrun`. Measured on the RTX 3090 (2026-10-16): 10,000 full-parameter in-place commits of a 1.2B model under concurrent chat load took 12.6 ms median, 15.9 ms at the 99.9th percentile and 55.8 ms at most; the default is ten times the 99.9th percentile, rounded up. A trainer whose step (write plus stream synchronise) takes longer must raise it |
| `WEIGHTS_HEARTBEAT_TIMEOUT_S` | 30 s | No heartbeat for this long: the attachment is `stale`. Stale changes nothing about serving; it still blocks attach and model loads. A heartbeat from the same attachment revives it |
| `WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S` | 60 s | How long quiesce (and a save or adapter pause) waits for running requests |
| `WEIGHTS_HEARTBEAT_INTERVAL_HINT_S` | 5 s | The interval the attach and heartbeat responses suggest (`heartbeat.interval_hint_s`, `interval_hint_s`); must be below the timeout |

An operator clears a stale attachment with `POST /clear-stale`. It is refused for an active one
(`409 ATTACHMENT_NOT_STALE`), because the holder may be mid-commit. Clearing releases a `clean`
quiesce; a commit still open is left for the watchdog, and `torn` stays `torn`.

## Status

`GET /api/weights/status` takes no token and no slot. It never returns the attachment ID or a
token.

| Field | Meaning |
|---|---|
| `attached` | Whether an attachment exists (active or stale) |
| `attachment` | `{state, role, label, attached_at, last_heartbeat_at, stale_since}` or `null` |
| `barrier` | `serving` or `quiesced` |
| `quiesced_at` | When the current hold began, or `null` |
| `held_requests` | Interactive requests waiting while quiesced; `null` while serving |
| `weights_state` | `clean`, `writing` or `torn` |
| `resident_model` | `{model_id, name, manifest_hash}` or `null`. `name` is the base model's served name, also while an adapter is applied (the attached process sees the base tensors; the adapter is in the `adapter` block). `manifest_hash` is `null` unless the model was loaded from a checkpoint manifest |
| `steps_since_load` | Completed commits since the parameter set became resident; `null` when nothing is resident |
| `last_save` | `{checkpoint_sha256, at, path}` or `null` |
| `commit` | `{began_at, open_ms, watchdog_ms, overran}` while a commit is open or torn, else `null` |
| `commit_overran`, `last_hold_timeout_at`, `last_overrun` | The last watchdog expiry, hold timeout and late commit-end |
| `pause_unavailable` | Why serving cannot be paused right now: `cbm` (continuous batching is on) or `concurrency` (`MAX_CONCURRENT_REQUESTS` is not 1); `null` when a pause is possible. While it is set, quiesce, a weights load and an adapter change are refused with `409 BARRIER_UNAVAILABLE` and the same `reason`, so a trainer can check before it attaches. It is the same check the refusal makes |
| `adapter` | The adapter block, the same object `GET /api/weights/adapter` returns ([Management API](management-api.md)); `null` only when it could not be read |
| `cuda_memory` | `[{device_index, allocated_mb, reserved_mb}]`: what miLLM's CUDA allocator reports per card; `null` without CUDA. With nothing resident, `allocated_mb` well above zero is memory a trainer never released ([Releasing the handles](#releasing-the-handles-what-stays-pinned)) |
| `exports_since_start` | How many times this process has exported handles (one per successful attach) |

`/api/health/detailed` carries a short `weights` summary: `weights_state`, `barrier`, `attached`,
`attachment_state` (`active`, `stale` or `null`), or `error` when they could not be read. The
names and values are the status route's.

## Which weights answered

Every `/v1` response says which parameter set produced it. Non-streaming responses and Batch API
output lines carry the header

```
X-miLLM-Model: "<served name>";origin=<model|checkpoint|adapter>;commits=<n>
```

and a stream carries the same value as `millm_model` on its terminal chunk. `commits` counts the
commit-ends since the served name was set (after a commit on a model loaded from the catalogue,
`"my-model";origin=model;commits=1`); it is omitted when it is not known, never written as 0.
A save or a checkpoint load changes the served name ([Weights: checkpoints](weights-checkpoints.md)).

## Which models can be attached

Attach is refused with `400 NOT_ATTACHABLE` and a `reason`:

| `reason` | Why |
|---|---|
| `gguf` | The model is served by llama.cpp |
| `quantized` | Any quantization other than none (bitsandbytes, GPTQ, AWQ, …) |
| `dtype` | A parameter is not float16 or bfloat16 (models that keep some modules in float32 are refused) |
| `not_on_gpu` | A parameter is not on a CUDA card |
| `allocator` | The model was not loaded as an attachable load. An unquantized 16-bit (`FP16` row, float16 or bfloat16 weights) transformers load is attachable when at least one `trainer` token is configured at load time; reload the model after adding one. `FP32` rows are never attachable |
| `continuous_batching_running`, `concurrency` | Continuous batching is on, or `MAX_CONCURRENT_REQUESTS` is not 1: the barrier could not stop serving |
| `model_transition` | A load, an unload, or a save, weights load or adapter change is in progress |
| `weights_writing` | A commit is still open: clear-stale removed a holder mid-commit and left the commit for the watchdog. Attach again once the watchdog has torn the set; attaching to `torn` is allowed, and the response's `weights_state` says `torn` |

## Rebuilding the tensors in the trainer

The full response schema of every route is published as `docs/schemas/millm-weights-api-v1.json`, generated from the routes themselves. Each storage entry carries the fields PyTorch's own reducer emits, with byte fields base64-encoded;
each parameter names its storage, shape, stride and offset. Parameters that share a storage (tied
embeddings) point at one storage entry. Devices are named by **UUID** (`device_uuid`) as well as
index: an index can differ between processes, a UUID cannot. Refuse to rebuild when `producer`
does not match your own torch and CUDA build.

**Rebuild each storage exactly once, then take every parameter as a view of it.** miLLM shares
each storage once, which opens one CUDA IPC reference count for it. Rebuilding a storage a second
time (for example once per tied parameter name) makes PyTorch's receiving side release a count
that was never opened, the counts no longer balance, and that storage is never freed: measured on
the RTX 3090, LFM2.5-1.2B's tied embedding (256 MiB) stayed reserved after every attach cycle that
rebuilt per parameter, and returned when each storage was rebuilt once.

```python
import base64, torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

def rebuild(body):
    storages = []
    for s in body["storages"]:          # ONE rebuild per storage entry
        b = lambda k: base64.b64decode(s[k]) if s[k] is not None else None
        whole = rebuild_cuda_tensor(
            torch.Tensor, (s["storage_size_bytes"],), (1,), 0, torch.UntypedStorage,
            torch.uint8, s["device_index"], b("handle"), s["storage_size_bytes"],
            s["storage_offset_bytes"], False, b("ref_counter_handle"), s["ref_counter_offset"],
            b("event_handle"), s["event_sync_required"],
        )
        storages.append(whole.untyped_storage())
    tensors = {}
    for p in body["parameters"]:        # a VIEW per parameter (tied names share one storage)
        st = storages[p["storage"]]
        t = torch.empty(0, dtype=getattr(torch, p["dtype"]), device=st.device)
        t.set_(st, p["storage_offset"], tuple(p["shape"]), tuple(p["stride"]))
        tensors[p["name"]] = t.requires_grad_(p["requires_grad"])
    return tensors
```

Write under `torch.no_grad()`: the views keep the parameters' `requires_grad`, and an in-place
write to a leaf that requires grad raises.

Keep the attachment alive with a heartbeat every `heartbeat.interval_hint_s` seconds. Detach when
you are done: detach keeps the current values, and miLLM reclaims the shared blocks you released.

## Releasing the handles (what stays pinned)

The attach response's `release` block states it: `pinned_bytes` is the GPU memory the exported
storages hold, and it stays reserved in miLLM until every handle has been rebuilt once **and**
released by your process. To release: drop every rebuilt tensor (and anything viewing them), then
call `torch.cuda.ipc_collect()`, then detach. A process that ends while still holding them —
killed, crashed, or exiting with the tensors still referenced (module globals are not reliably
freed at interpreter exit) — never releases them, and neither does an attach whose handles were
never rebuilt at all (a lost response, a check that attached only to look). **That memory stays
reserved in miLLM until miLLM restarts**: detach, clear-stale, a reload and an unload do not
return it. Measured on the RTX 3090 with LFM2.5-1.2B: about 2.2 GB stranded per such attach.

`GET /api/weights/status` reports what miLLM can see of it: `cuda_memory` (allocated and reserved
MiB per card in miLLM's process) and `exports_since_start`. With nothing resident, `allocated_mb`
well above zero is memory a trainer never released. After a crashed trainer, recover with an
**in-place** load (`POST /api/weights/load` of a checkpoint of the same model): it writes into the
existing storages and allocates nothing, whereas a `{"model_id": N}` reload allocates a second copy
beside the stranded one. Restart miLLM to return stranded memory.
