---
sidebar_position: 10
title: Weights Checkpoints API
---

# Weights Checkpoints API

Save the resident model's weights as a checkpoint, load a checkpoint back into the resident model, and read a checkpoint's lineage. The routes live at `/api/weights` beside the attachment routes. All responses use the [management envelope](/api/overview#the-management-envelope); the examples below show the `data` payload.

| Route | Who may call it |
|---|---|
| `POST /api/weights/save` | the trainer that holds the attachment: a `trainer` or `operator` bearer token **and** `X-miLLM-Attachment` |
| `POST /api/weights/load` | a `trainer` or `operator` bearer token |
| `GET /api/weights/lineage` | anyone (no token) |

Tokens are configured in `WEIGHTS_TOKENS` (see [Configuration](/reference/configuration)). There is no route to create a model with randomly initialised weights: that capability (instantiate) is deferred.

## Where checkpoints live

Each checkpoint is a standard Hugging Face directory:

```
<WEIGHTS_CHECKPOINT_DIR>/<run_id>/step-000500/
    model.safetensors   config.json   generation_config.json
    tokenizer.json   tokenizer_config.json ...   manifest.json
<WEIGHTS_CHECKPOINT_DIR>/<run_id>/latest -> step-000500
```

- miLLM writes only to `WEIGHTS_CHECKPOINT_DIR`. A directory is written under a temporary name, synced to disk, then renamed, so a checkpoint is either whole or absent. `latest` moves only after the rename.
- `WEIGHTS_CHECKPOINT_READ_ROOTS` is a comma-separated list of extra directories (for example miForge's checkpoint volume, mounted read-only). Load and lineage search them; save never writes there, and startup cleanup never touches them. A run present under two roots is refused (`CHECKPOINT_AMBIGUOUS`) rather than guessed.
- The step is padded to six digits in the path (`step-000500`) and unpadded in the name: `<base_model>:<run_id>:step-500`.
- `manifest.json` follows the vendored schema `docs/schemas/checkpoint-manifest-v1.json`. Only `model.safetensors` is pinned by a hash; the configuration and tokenizer files are not.

## Save

```bash
curl -X POST http://localhost:8000/api/weights/save \
  -H "Authorization: Bearer $TRAINER_TOKEN" \
  -H "X-miLLM-Attachment: $ATTACHMENT_ID" \
  -H "Content-Type: application/json" \
  -d '{"manifest": {"run_id": "house-sql", "step": 500, "domain": "sql style",
                    "signal_source": "corpus", "parent": "LiquidAI/LFM2-1.2B:house-sql:step-400",
                    "optimizer": {"name": "adamw-8bit", "state": "restored", "config_hash": "9f2c"}}}'
```

`201`:

```json
{
  "name": "LiquidAI/LFM2-1.2B:house-sql:step-500",
  "path": "house-sql/step-000500",
  "weights_sha256": "…64 hex…",
  "manifest_sha256": "…64 hex…",
  "bytes": 2470000000,
  "snapshot_ms": 410.2,
  "write_ms": 2900.5,
  "parent_check": "match",
  "latest": "step-000500",
  "identity_set": true
}
```

- **The caller writes the trainer fields only.** `base_model`, `name`, `created_at` and `weights_sha256` are written by miLLM; sending any of them is `400 MANIFEST_FIELD_NOT_CALLERS`. `kind` and `schema_version` may be sent only with their constant values. Every schema error is listed by JSON path in `details.errors`.
- **`run_id`** must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` (`400 INVALID_RUN_ID`), and the rendered name must fit the model table's 255-character name column.
- **`base_model`** is the resident model's lineage root: the base of the checkpoint it was loaded from, or, for a model loaded from the catalogue, its Hugging Face repository ID (its row name when it has none).
- **Serving pauses only for the snapshot.** The weights are copied to host memory while serving is paused and the state is checked to be `clean`; the disk write and the hash run after serving resumes. `snapshot_ms` is the pause.
- **The resident model is renamed to the checkpoint** (`/v1/models` and every completion then report the new name), unless a load replaced the weights while the files were being written. Then the checkpoint is still published but the resident is not renamed, and `identity_set` is `false`.
- **`parent_check`** compares your `parent` with the checkpoint the resident model was last named after (`null` for a catalogue model). A mismatch is reported and logged, never refused.
- Saving changes neither `weights_state` nor `steps_since_load`, and records `last_save` in [`GET /api/weights/status`](/api/weights). Because the served name moves to the checkpoint, the `commits=` count in `X-miLLM-Model` starts again from 0 (see [The served name after a save or a load](#the-served-name-after-a-save-or-a-load)).
- A holder that had quiesced stays quiesced through the save; otherwise the save takes the pause itself.
- An applied adapter is **not** part of the checkpoint: the save writes the base tensors, which an adapter never changes. Merge the adapter into the weights (an attached process can) if the checkpoint should include it.

Refusals, in order: token (`401`/`403`) → `INVALID_MANIFEST` with `reason: schema_ownership` (the vendored schema cannot be used; logged at startup) → `NOT_ATTACHED` / `NOT_ATTACHMENT_HOLDER` → `MANIFEST_FIELD_NOT_CALLERS` / `INVALID_MANIFEST` / `INVALID_RUN_ID` → `MODEL_NOT_LOADED` / `ENGINE_UNSUPPORTED` → `CHECKPOINT_EXISTS` → `409 CHECKPOINT_AMBIGUOUS` (the `run_id` already exists under a read-only checkpoint root; saving it under the primary root would make both runs unloadable by name, so save under a different `run_id`) → `INVALID_RUN_ID` (the rendered name is longer than the name column) → `INSUFFICIENT_STORAGE` (free disk below the estimate × `WEIGHTS_SAVE_FREE_SPACE_FACTOR`; the estimate adds `WEIGHTS_SAVE_OVERHEAD_MB`) → `INSUFFICIENT_MEMORY` (free host memory below the snapshot size plus 10%; an unmeasurable figure only warns) → `PAUSE_IN_PROGRESS` (another save is running) → `BARRIER_UNAVAILABLE` (continuous batching on, or more than one request slot) → `PAUSE_IN_PROGRESS` (a weights load or an adapter change holds the barrier) or `WEIGHTS_NOT_CLEAN` (a commit is open, or the weights are `torn`) → `MODEL_BUSY` (`reason: drain_timeout`). A failure while writing is `500 CHECKPOINT_WRITE_FAILED` with `details.stage` and leaves no directory behind. If only `latest` could not be moved, the error says `details.published: true`: the checkpoint exists and is valid.

## Load

```bash
curl -X POST http://localhost:8000/api/weights/load \
  -H "Authorization: Bearer $TRAINER_TOKEN" \
  -H "X-miLLM-Attachment: $ATTACHMENT_ID" \
  -H "Content-Type: application/json" \
  -d '{"path": "house-sql/latest"}'
```

The body names the checkpoint in exactly one way:

| Field | Meaning |
|---|---|
| `name` | a fully qualified name, `<base_model>:<run_id>:step-<n>` |
| `path` | a step directory (or `latest`) relative to a checkpoint root, or absolute under one |
| `model_id` | a catalogue model to reload inside the barrier: the way back from `torn` when no checkpoint exists. A checkpoint's own row is refused here; load it by name or path so its files are verified |

Optional: `gpu` (full reload only, as for `POST /api/models/{id}/load`) and `trust_remote_code` (see below). Headers: `X-miLLM-Attachment` identifies the attachment holder; `X-miLLM-Lease` lifts a [lease](/api/models) held by the caller.

A `model_id` reload answers with the row's name, `mode: "full_reload"`, `weights_sha256` and `manifest_sha256` `null` (nothing was verified against a manifest) and `verify_ms` `0`. It is a full reload, so it is refused while anything is attached: a trainer detaches (or an operator clears a stale attachment) first.

`200`:

```json
{
  "name": "LiquidAI/LFM2-1.2B:house-sql:step-400",
  "mode": "in_place",
  "weights_sha256": "…",
  "manifest_sha256": "…",
  "verify_ms": 1800.0,
  "write_ms": 900.0,
  "weights_state": "clean",
  "steps_since_load": 0
}
```

How a load works:

1. The manifest is validated (unknown keys refused; `kind` and `schema_version` may be absent, as miForge writes them), its `name` must match its own `base_model`, `run_id` and `step`, and its directory must be `<root>/<run_id>/step-<NNNNNN>`. A loaded `run_id` need only be a safe path segment (no `/`, NUL or `:`, not `.` or `..`); the stricter save pattern does not apply, because miForge names its own runs.
2. `model.safetensors` is hashed **before** serving is paused and compared with `weights_sha256` (`422 CHECKPOINT_HASH_MISMATCH` with both values). The file's size, inode and modification time are re-checked inside the barrier; a change refuses the load with nothing written.
3. **In place** when the resident model is a transformers model, the manifest's `base_model` is the resident's base, and every tensor name, shape and dtype matches (read from the file header alone). Each tensor is copied into the model's existing memory, so an attached trainer's handles stay valid and read the new values.
4. **Full reload** otherwise: the directory is registered as a checkpoint model row and loaded through the normal load path. Refused while anything is attached (`409 ALREADY_ATTACHED`). The weights must be 16-bit, all `F16` or all `BF16`, and are never cast (`400 CHECKPOINT_NOT_LOADABLE`, reason `dtype`). A `config.json` that names remote code (`auto_map`) is refused (reason `remote_code`) unless the request sets `"trust_remote_code": true` **with an operator token**. A directory already registered as an ordinary local model is refused (`409 LOCAL_PATH_REGISTERED`): delete or re-register that model first. The rest of the directory is checked before anything is changed, because the loader opens it only after the old model has been unloaded (a failure there would leave the weights `torn`). Each of these is `400 CHECKPOINT_NOT_LOADABLE`, with nothing unloaded:
   - `config.json` would make the loader cast the weights: it records a dtype other than the weights', or none over `F16` weights, where the loader's default is bfloat16 (reason `dtype`, with `load_dtype` in `details`). Record the weights' dtype in `config.json`.
   - `config.json` declares a `quantization_config` (reason `quantized`): the loader would serve it quantized, not as the verified bytes.
   - `config.json` is missing, unreadable or names no `model_type`, or there is no tokenizer file (reason `incomplete`, with `details.missing`).
   - The directory holds an entry that is not a regular file or directory (a FIFO, for example), or a link that leads out of its checkpoint root (reason `unsafe_file`, with `details.files`).
   - The directory's real path is longer than 500 characters, the most a model row holds (reason `path_length`).
5. On success the state is `clean`, `steps_since_load` is 0, and the resident model is named after the checkpoint before serving resumes, so the first answer after the load already carries the new name. An adapter applied to the old base is removed before the first tensor is written ([Management API](management-api.md)). If the holder had quiesced, the load takes over its hold and hands it back: serving is still quiesced afterwards, with the hold timer restarted.

**While a trainer is attached**, an in-place load is accepted from the holder (its token plus `X-miLLM-Attachment`), or from an operator token when the weights are `torn` or the attachment is stale. Anything else is `409 ALREADY_ATTACHED`; the same role's token without the header is not the holder.

**Failure after the first write** (a read error mid-copy, a reload that fails after the old model was unloaded) is `500 WEIGHTS_LOAD_FAILED` and leaves the weights `torn`: every `/v1` request is refused with `503 WEIGHTS_TORN` until a load succeeds. Loading is the recovery path: load any checkpoint, or `{"model_id": N}`.

Other refusals: `INVALID_CHECKPOINT_PATH` (outside every root), `CHECKPOINT_NOT_FOUND`, `CHECKPOINT_AMBIGUOUS`, `INVALID_MANIFEST` (including `reason: schema_ownership`), `INVALID_RUN_ID`, `CHECKPOINT_NOT_LOADABLE` (`reason` `safetensors_header`, `dtype`, `quantized`, `remote_code`, `incomplete`, `unsafe_file`, `path_length`, or `carrier_row` for a checkpoint's own row given as `model_id`), `MODEL_NOT_FOUND` (an unknown `model_id`), `WEIGHTS_NOT_CLEAN` (a commit is open, whoever holds it), `MODEL_LEASED`, `MODEL_BUSY` (a model load or unload is in progress, or the drain timed out), `PAUSE_IN_PROGRESS`, `BARRIER_UNAVAILABLE`.

Every save and load, accepted or refused, writes one `weights_audit` log line (fields as on [Weights](/api/weights)) under the operation name `weights_save` or `weights_load`. A token refusal (`401`/`403`) is logged under the same name, so one search finds every line for an operation. Lineage needs no token and is not audited.

## The served name after a save or a load

A save (with `identity_set: true`) or a checkpoint load gives the resident model the checkpoint's name, `<base_model>:<run_id>:step-<n>`. From then on:

- `/v1/models` lists that name, every completion reports it as its `model`, and the `X-miLLM-Model` header (and a stream's terminal `millm_model`) reads `"<name>";origin=checkpoint;commits=<n>`, where `commits` counts the commit-ends since the name was set. A commit after a save does not rename the model: the name stays the last checkpoint's and `commits` grows, so an answer never claims to be a step that was never written.
- Requests should name the checkpoint name. **A request naming the original catalogue model is no longer a request for the resident model, and it is refused**, attached or not: `404 model_not_found` (the same status, `code`, `type` and `param` as an unknown model), with a message naming the resident's served name and `POST /api/weights/load {"model_id": N}`. The resident weights are untouched. A request never reloads the row on demand, because that would discard the trained weights and any commits made since the save. The same applies to `/v1/completions`, `/v1/embeddings` and every line of a batch file that names the row. A checkpoint row (one the weights API registered) is never loaded on demand either: it is refused with `404 model_not_found` and a sentence pointing at `POST /api/weights/load`.
- A request naming **another** catalogue model would auto-load it over the resident. That is refused with `409 weights_unsaved` while the resident carries commits made since the last save (or since the last load), so a commit after a save is protected even though the name has not changed; once those commits are saved, the auto-load proceeds as usual. See [Unsaved commits and the auto-load](weights.md#unsaved-commits-and-the-auto-load).
- `POST /api/weights/load {"model_id": N}` is the deliberate way back: it reloads the catalogue model from its own files, after which it is served under its row name with `origin=model` and a request naming the row is answered again.

## Lineage

```bash
curl http://localhost:8000/api/weights/lineage
```

```json
{
  "resident": {"name": "LiquidAI/LFM2-1.2B:md-notary:step-200", "origin": "checkpoint",
               "manifest_sha256": "…", "commits_since_identity": 0},
  "base_model": "LiquidAI/LFM2-1.2B",
  "chain": [
    {"name": "LiquidAI/LFM2-1.2B:md-notary:step-200", "run_id": "md-notary", "step": 200,
     "path": "md-notary/step-000200", "root": "/data/checkpoints", "manifest_sha256": "…",
     "weights_sha256": "…", "parent": "LiquidAI/LFM2-1.2B:house-sql:step-400",
     "merged_adapter": null}
  ],
  "complete": false,
  "break": {"at": "LiquidAI/LFM2-1.2B:house-sql:step-400", "reason": "manifest_missing"},
  "weights_verified": false
}
```

- The walk starts at the checkpoint the resident model is named after and follows each manifest's `parent` across every checkpoint root. A catalogue model has an empty chain and `complete: true`.
- It stops at `parent: null` (`complete: true`) or at a break: `manifest_missing`, `manifest_invalid` (including a parent that would resolve outside its root), `name_mismatch`, `cycle`, `depth_limit` (`WEIGHTS_LINEAGE_MAX_DEPTH`) or `ambiguous` (the run exists under two roots).
- `commits_since_identity` counts commits since the resident was named, so a chain read after unsaved commits does not pass the resident off as its last checkpoint.
- Lineage reads manifests only and never hashes a weights file (`weights_verified: false`). Verification happens on load.
- With no model loaded: `400 MODEL_NOT_LOADED`.
