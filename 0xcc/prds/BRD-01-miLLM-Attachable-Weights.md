# BRD-01 · miLLM — Attachable Weights

**Owner:** Human in the Stream, LLC
**Status:** Draft v0.2 · 2026-09-11
**Companion:** BRD-02 (miForge — In-Place Optimizer and Domain Loop)

---

## 1. Why this exists

miLLM is an inference engine. It should stay one. But the weights it serves live in GPU memory, and there is no reason another process cannot hold a handle to that same memory and change it.

This BRD adds one capability to miLLM: let an authorized external process attach to the resident model's parameter tensors, update them in place, and detach. miLLM never runs an optimizer, never computes a gradient, and never knows what changed. It serves whatever the weights are right now.

The immediate consumer is miForge. Any future tool that wants to edit resident weights (merging adapters, baking a steering vector, patching a layer) uses the same door.

## 2. Goal in one sentence

Let an external process share, update, and snapshot the exact tensors miLLM is serving, without a reload and without miLLM becoming a trainer.

## 3. Scope

In scope: attach and detach, tensor handle export, quiesce and resume, checkpoint save from resident tensors, checkpoint load with lineage, and the identity rules for those calls.

Out of scope: any optimizer, loss, gradient, or batch logic. Adapter training. Deciding what a change means.

## 4. Requirements

### 4.1 Attach and detach

**R-01.1** `POST /api/weights/attach` returns, for every parameter tensor of the resident model: module path, shape, dtype, device, and a CUDA IPC handle (the mechanism PyTorch uses to share CUDA tensors across processes). The response also includes the resolved model name and the manifest hash of the loaded checkpoint.

**R-01.2** Only one attachment at a time. A second attach while one is active returns `409 ALREADY_ATTACHED`.

**R-01.3** `POST /api/weights/detach` releases the attachment. miLLM continues serving the current weight values. Detach does not revert anything.

**R-01.4** If the attached process disappears (heartbeat lost for a configured interval), miLLM logs it, marks the attachment stale, and continues serving. Stale attachments can be cleared by an operator.

**R-01.5** miLLM keeps the parameter tensors at fixed addresses for the lifetime of the load. It never reallocates, re-quantizes, or moves them while an attachment is active. Attach is refused on quantized models; the tensor layout must be plain FP16 or BF16.

*Why:* the whole design depends on both processes seeing the same memory. Anything that moves the tensors silently breaks the sharing.

### 4.2 Quiesce and resume (the step barrier)

**Weight states.** The resident parameter set is always in exactly one of three states, reported by R-01.20:

| State | Meaning | Serves `/v1/*`? |
|---|---|---|
| `clean` | Every tensor is at the same version. | Yes |
| `writing` | An attached process has declared a write in progress. | No — held |
| `torn` | A write began and did not complete. Tensor versions may be mixed. | **No — refused** |

**R-01.6** `POST /api/weights/quiesce` drains in-flight requests, holds new ones, and returns when no forward pass is running. Response includes the number of requests held and the current `weights_state`. Quiesce is refused unless the state is `clean`.

**R-01.7** `POST /api/weights/resume` releases held requests. Any request served after resume uses whatever the weights are at that moment. Resume is refused unless the state is `clean`; attempted while `writing` or `torn` it returns `409 WEIGHTS_NOT_CLEAN`.

**R-01.8** Quiesce has a maximum hold time (configurable, default a few seconds). If the attached process neither resumes nor begins a commit within it, miLLM resumes on its own and records the timeout. **This self-resume applies only while the state is `clean`** — that is, only when no resident tensor has been written.

**R-01.21** `POST /api/weights/commit-begin` and `POST /api/weights/commit-end` bracket the window in which the attached process writes to resident tensors. `commit-begin` is accepted only while quiesced and only from the holding attachment; it moves the state to `writing` and **suspends the R-01.8 hold timeout**. `commit-end` returns the state to `clean` and increments the steps-since-load counter. Writing to a resident tensor outside a commit bracket is a contract violation; miLLM cannot detect it, which is exactly why the bracket is mandatory rather than advisory.

**R-01.22** A commit has its own watchdog, configurable and separate from the R-01.8 hold time. If `commit-end` does not arrive within it, miLLM marks the state `torn`, logs it with the attaching identity and the elapsed time, and **does not resume serving**. `/v1/*` returns `503 WEIGHTS_TORN` until the state is `clean` again. miLLM never serves a `torn` parameter set and never self-resumes out of one.

**R-01.23** A `torn` state is recoverable, by exactly two paths:

  a. The attached process completes its write and calls `commit-end`. miLLM accepts a late `commit-end` from the same attachment, returns to `clean`, and records that the commit overran.
  b. An operator or trainer loads a checkpoint (R-01.14). A successful load replaces the entire parameter set and returns the state to `clean`.

A detach does **not** clear `torn` — the tensors are still mixed. The state survives detach and re-attach and is reported by R-01.20, so the next process to attach sees what it inherited.

**R-01.9** While quiesced, KV caches from prior requests are invalidated. Prefix-cache reuse across a weight change is not permitted.

*Why:* a request that starts on old weights and finishes on new ones is a torn read. The barrier makes every request see exactly one version of the weights.

*Why the commit bracket:* R-01.8's timeout exists so that a hung or dead trainer cannot take serving down indefinitely — it protects **availability**. But an in-place parameter update is not atomic; it writes tensor by tensor. A self-resume partway through leaves miLLM serving a model whose layers sit at two different steps — a model that was never tested, has no checkpoint, and cannot be reproduced. Without the bracket, the timeout that protects availability silently destroys **correctness**, reintroducing across the parameter set exactly the tearing this section exists to prevent.

The bracket separates the two waits the original timeout conflated. Waiting on a trainer that is *thinking* is unbounded and must be cut short. Waiting on a trainer that is *writing* is bounded — a fixed number of bytes, a VRAM-to-VRAM copy of the parameter set, expected in the tens of milliseconds against a hold time measured in seconds. Cutting that short buys nothing and costs correctness. So the hold time governs the first wait, the watchdog the second, and the watchdog fails closed: a `503` is visible and recoverable, a silently wrong answer is neither.

### 4.3 Checkpoint save from resident tensors

**R-01.10** `POST /api/weights/save` writes the resident tensors to a checkpoint directory:

```
checkpoints/<run_id>/step-<NNNNNN>/
  model.safetensors
  config.json
  tokenizer/
  manifest.json
checkpoints/<run_id>/latest -> step-<NNNNNN>
```

**R-01.11** Save is performed while quiesced, or miLLM quiesces internally for the duration of the write. Save is refused unless `weights_state` is `clean` (`409 WEIGHTS_NOT_CLEAN`). Checkpointing a `writing` or `torn` parameter set would persist a model that never existed at any step, breaking R-01.14's verification and BRD-02's R-02.21.

**R-01.12** The manifest's field list is defined by `docs/schemas/checkpoint-manifest-v1.json`, vendored byte-identical into this repo and miForge's. That schema is normative; this requirement does not restate it. Each field is marked as written by the caller or written by miLLM, and miLLM rejects a save whose caller-supplied fields do not validate against it.

*Why a schema rather than a sentence:* this requirement previously listed the caller's fields in prose, and BRD-02 R-02.20 listed them in prose too. The lists disagreed — BRD-02 sent `selector_hash`, `grader_hash`, `test` and `retention`, which this requirement did not accept, while this requirement expected an optimizer identity BRD-02 did not send. Two prose descriptions of one wire format drift silently. One file that both repos vendor can be diffed in CI.

**R-01.24** The step number is a single canonical integer, carried in the manifest as `step`. Every rendered form derives from it:

- **Checkpoint and trainer-state paths** use `step-` plus the integer zero-padded to six digits (`step-000500`), so that directories sort lexicographically.
- **The fully qualified name** `<base_model>:<run_id>:step-<step>` uses the integer unpadded (`step-500`). This is the display and API form — what `/v1/models` lists under R-01.15 and what a completion reports as its resolved model.

Padding is a path convention, not part of the step's identity. Nothing carries the step as a string of its own.

**R-01.13** Writes are atomic: temporary directory, fsync, rename, then update `latest`.

### 4.4 Checkpoint load with lineage

**R-01.25** `POST /api/weights/load` loads a checkpoint directory into the resident model.

- It verifies `weights_sha256` per R-01.14 **before touching a resident tensor**, and on mismatch refuses having changed nothing.
- **When the checkpoint matches the resident architecture** — same `base_model`, identical parameter shapes — the values are written into the existing tensors **in place**. Addresses do not move, so an attachment's CUDA IPC handles stay valid across the load (R-01.5). This is the case torn-set recovery uses.
- **Otherwise** it is a full reload, which reallocates and therefore invalidates every handle an attached process holds. It is refused while an attachment is active (`409 ALREADY_ATTACHED`); the attached process detaches first.
- A load is itself a multi-tensor write, so it runs inside the same barrier as any other: miLLM quiesces, marks the state `writing`, and returns it to `clean` on success. **A load that fails partway leaves the set `torn`,** exactly like an interrupted commit — the recovery path is not exempt from the rule it exists to serve.
- It is permitted while the state is `torn`; that is its recovery role (R-01.23b). It is refused while another process holds an open commit (`409 WEIGHTS_NOT_CLEAN`), because two writers to one parameter set is what the barrier exists to prevent.

*Why this is called out:* R-01.14 described what a load verifies without defining a call that loads, and this section's title promised the capability. R-01.23's only trainer-independent recovery path, BRD-02 R-02.21, R-02.22 and R-02.30, and the mockup's "Load in miLLM" all route through it.

**R-01.14** Loading a checkpoint directory verifies `weights_sha256` against `model.safetensors` and refuses on mismatch.

**R-01.15** `/v1/models` lists the resident model under its fully qualified name. Every completion response includes the resolved name in its metadata so a test run always knows which step it queried.

**R-01.16** `GET /api/weights/lineage` returns the parent chain of the resident model back to `base_model` by reading manifests.

### 4.5 Identity

**R-01.17** `attach`, `detach`, `quiesce`, `commit-begin`, `commit-end`, `resume`, `save`, `load` require a `trainer` or `operator` identity. `/v1/*` is unaffected. `commit-begin` and `commit-end` are further restricted to the holding attachment (R-01.21): an identity that is merely valid cannot close another process's commit.

**R-01.18** Every call in 4.1 through 4.3 is logged with identity, timestamp, resident model name, and (for save) the resulting checkpoint hash.

### 4.6 Test and probe surface

**R-01.19** Nothing new is required for testing: miForge tests through the existing `/v1/*` endpoints and, when it wants activations, the existing SAE attach and probe endpoints. Those continue to work while a weight attachment is active; probe hooks read the same resident tensors.

**R-01.20** miLLM exposes `GET /api/weights/status`: attached or not, attaching identity, quiesced or serving, **`weights_state` (`clean` | `writing` | `torn`)**, resident model name, steps-since-load counter (incremented by each completed commit), last save, and — when not `clean` — when the current commit began and how long it has been open. This endpoint is how a restarting trainer discovers a `torn` set it did not create.

## 5. Acceptance criteria

1. Attach returns a handle for every parameter; a second process reconstructs the model from those handles and its forward pass matches miLLM's logits for the same prompt.
2. The attached process writes zeros into one layer's bias between quiesce and resume; the next miLLM completion reflects it; no completion that started before quiesce is affected.
3. Attached process is killed; miLLM keeps serving; status shows `stale`.
4. Save produces a checkpoint whose `weights_sha256` matches a fresh hash of the file; loading it into a clean miLLM instance gives identical logits to the resident model at save time.
5. Attach on a quantized model returns `400 NOT_ATTACHABLE`.
6. Quiesce with a slow attached process that has **not** called `commit-begin` times out and miLLM resumes on its own; the parameter set is unchanged.
7. A trainer calls `commit-begin`, writes part of the parameter set, and is killed. miLLM does **not** resume: `/v1/*` returns `503 WEIGHTS_TORN`, status reports `torn`, and the state survives detach and re-attach. Loading the last checkpoint returns it to `clean` and serving resumes.
8. A commit that overruns the watchdog but completes: the late `commit-end` is accepted, the state returns to `clean`, and the overrun is recorded.
9. `save` is refused while `writing` or `torn`.
10. No completion is ever served from a parameter set that is not `clean` — verified by driving a commit that never ends and confirming every `/v1/*` call is refused rather than answered.
11. `load` of a checkpoint with matching architecture, while a process is attached, writes in place: the attached process's CUDA IPC handles remain valid, and a forward pass in that process after the load matches miLLM's logits for the same prompt. `load` of a differently-shaped checkpoint while attached is refused.
12. A `load` interrupted partway leaves the state `torn` and serving refused — the recovery path obeys the same rule as a commit.

## 6. Non-goals

- Running an optimizer, computing gradients, or holding optimizer state in miLLM.
- Multiple simultaneous attachments.
- Attach to quantized or GGUF-backed models (llama.cpp path is not attachable; use the tensor path).
- Serving two checkpoints at once.

## 7. Open questions

- GPU sharing between miLLM and the attached process: CUDA MPS, or accept time-slicing?
- Should `save` be callable by miLLM alone (operator snapshot of whatever is resident), or only by the attached process?
- Heartbeat transport for R-01.4: gRPC stream, WebSocket, or a file lock?
- R-01.22's commit watchdog default: the write is a VRAM-to-VRAM copy of the parameter set, so it should land in the tens of milliseconds, but the real distribution under a shared GPU (miLLM serving alongside) needs a benchmark before a default is fixed. Too tight and a healthy commit is declared torn; too loose and a dead trainer holds serving down for that long. BRD-02 R-02.14 records the per-step commit duration this benchmark reads.
