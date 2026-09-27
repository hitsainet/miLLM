# Feature 024 — Probe Monitor Runtime: mutation controls and findings

**Date:** 2026-09-27 · **Branch:** `feat/probe-monitor-runtime` · **PR:** #3
**Co-release:** miStudio 033 phase 7 (MCP contract v1.6)

This is the SC-5 record (task 10.5) plus the defects the phases found. It is organised by
*what the mutation proved*, not by phase, because the useful pattern in it is which kinds of
line have no test behind them.

---

## 1. SC-5: the wiring sweep (FTID §6)

Every control: back up the file by bytes, apply one edit, **verify the anchor matched exactly
once**, run the affected suite, restore, `cmp` the restore. Baseline for the sweep was
**464 passed, 0 failed**.

| # | Wiring line deleted | Result |
|---|---|---|
| S1 | `register_forward_hook(..., prepend=True)` → without `prepend` | **1 failed** — the probe would read post-steer |
| S2 | `app.include_router(probes_router)` | **17 failed** |
| S3 | one byte of `docs/schemas/probe-definition-v1.json` (`resid_post` → `resid_pos_`) | **2 failed** — byte-identity and the structural check |
| S4 | one word of the rung language (`compared with` → `better than` a judge) | **2 failed** — cross-repo identity |
| S5 | `_probe_begin` removed from the non-streaming chat path | **1 failed** |
| S6 | the CBM serial clause (`PROBE_FORCE_SERIAL` + `has_armed()`) | **2 failed** |
| S7 | `strip_context(payload)` → `payload` on the socket emit | **1 failed** — the privacy test asserts ABSENCE |

**S4 is the one worth reading twice.** Changing "compared with a judge" to "better than a
judge" is a one-word edit that would ship a false claim about every rung-3 probe, and it goes
red across repos. On miStudio's own reference run the judge won.

⚠ **S6 first ran with an anchor that matched twice, so the edit did not land and the suite
reported `44 passed`.** Recorded because it is the failure mode of mutation testing itself: an
unlanded mutation looks exactly like a surviving one. Every control here was re-run with a
verified-unique anchor.

---

## 2. Defects found, by how they were found

### 2.1 Found by writing a test that had never existed

**⚠ The arming service had no caller.** The FPRD §7 specifies twelve `/api/probes` paths; the
module served seven. Absent: `POST /{id}/arm`, `POST /{id}/parity`, `GET /hub/search`,
`GET /hub/{repo_id:path}/definitions`, `POST /hub/import`. So `ProbeArmingService.arm` (all
four gates and the acknowledgement), `check_identity`, `resolve_revision`,
`ProbeParityEngine.run`, `SaeFeatureSlice` (the entire k-sparse path) and `ProbeHubService`
had, between them, **no production caller**. A probe could be imported and never armed.

The reachability test did not catch it because it was a **subset** assertion —
`for expected in (…): assert expected in paths` over seven literals, every one served — and
task 7.5's own text read "(7 paths)" while the module had exactly seven. **A count is not a
set.** `test_probe_route_surface.py` now asserts the served set *equals* the set parsed out of
the FPRD itself.

**⚠ `millm_download_model` has always called a route that does not exist.** Parametrising
miLLM's `test_mcp_tool_paths_are_real.py` over surfaces — it had been pointed at
`millm_circuits` alone since it was written — went red immediately on four more modules that
had never been path-checked, and found that the tool posts to `/api/models/download`, which
miLLM serves as POST on the collection. The tool 404'd for every agent that ever called it,
and miStudio's own caller assertion **pinned the wrong path**, so tool and test agreed and
both were wrong. A caller assertion can only prove a tool matches its own documentation.

**⚠ miLLM's contract-consistency guard had never run.** It reported "10 skipped / UNVERIFIED"
for its whole life, because building miStudio's registry needs `mcp`, then psycopg2, then
structlog. `mcp>=1.9.0,<2` is now a test dependency, and the last check that still could not
run — the evidence-ladder phrase parity — reads `evidence_ladder.py` **by AST** rather than
importing it, which is what the file's own docstring had already identified as the answer for
the path half. **17 of 17, 0 skipped.**

### 2.2 Found by measuring

**⚠ The score path copied 33.6 MB to the host on every forward pass.**
`ProbeRequestContext.observe` did `hidden[0].detach().to(torch.float32).cpu()` and scored on
the CPU, single-threaded, in float32: **25–37 ms for two probes at 4k tokens**, against a 5 ms
budget. The comment above the line read *"⚠ THE ONE DEVICE-TO-HOST COPY"* — true, and beside
the point, because one copy of 33.6 MB is the cost. No test caught it: none measured, and no
correctness test could, because the scores were right.

Fixed by making the head and the SAE slice follow the activations' device. **16 kB instead of
33.6 MB.**

**Two faster forms were measured and rejected**, and both look right:

| Form | Speed | Error | Verdict |
|---|---|---|---|
| Fold standardisation into the weight | 14.1 → 5.4 ms | **1.9e-06** | **Rejected** — breaks bit-exactness with miStudio |
| fp16 matvec | a further ~4x | **8.5e-02** | **Rejected** — 85x the 1e-3 parity tolerance |

The fold is *algebraically exact*; it fails only at float32 rounding. That is enough, because
`test_probe_head_matches_mistudio.py` requires **bit-exact** agreement — the property 033's
acceptance recorded as "0.000e+00, all sixteen vectors". Exactness is a stronger guarantee
than "within tolerance" and is what makes a parity pass mean something. Both forms are now
**pinned as tests carrying their measured error**, because a comment does not fail.

### 2.3 Found by the fixtures being real

Three definitions were copied byte for byte off the node rather than hand-written. Against a
tiny random model they are **refused at the parity gate** — which is the guarantee the entire
export contract exists for, and which I had not set out to test. It is now asserted directly,
with the recorded numbers untouched.

---

## 3. My own defects, and what caught each

Recorded because the pattern is more useful than the list: **not one was caught by re-reading.**

| Mine | Caught by |
|---|---|
| A module-wide AST scan for `loaded_identity` / `build_probe_encoder` — satisfied by `check_parity`'s calls to them, so bypassing either in `arm_probe` went unnoticed | running the mutation |
| `arm.assert_not_awaited()` passed against a parity route setting `armed=True` on the ROW — a probe counted against the limit with no hook and no `paused_reason` | running the mutation |
| Two control runs named a test file that does not exist: pytest exited 4 with **0 collected**, and I read the empty result as a survival | reading the exit status |
| One control ran against a baseline already red from a pending contract regeneration, so a mutation appeared to *fix* the suite | noticing the baseline |
| S6's anchor matched twice, so the edit never landed | the `count == 1` assertion |
| `kind == "mistudio.probe-definition"` — circuits and clusters carry no `/v1`; probe definitions do | the real fixture |
| Narrowed `weights` without `norm_mean`/`norm_std` | `ProbeHead` refusing it, deliberately |
| Tried to invert sha256 to satisfy the chat-template gate | it not terminating |
| Asserted the 5 ms budget on CPU arithmetic as a "floor" — it is a ceiling, by orders of magnitude | the measurement |
| `_repo` vs `repo` in a new assertion, so it failed on the clean tree | the clean-tree run |

**The `0 collected` one is the worst**, and it is the reason every control in §1 reports a
baseline: a sweep whose runs collect nothing reports silence, and silence reads as survival.

---

## 4. Also fixed while here (pre-existing, in the code this touched)

- **miStudio `TestTheGatherScalesLinearly` flaked on CI and passed against the quadratic
  algorithm it was written to catch.** Single wall-clock samples at n=250; and at those sizes
  the linear per-index work dominates the quadratic filter by enough of a constant that no 4x
  step reached the 12x bar. Now a log-log exponent fit at n = 2k/8k/32k requiring < 1.5:
  clean code passes 4/4, the documented defect measures **n^1.97** and fails.
- **`npm run typecheck` was `tsc --noEmit`, a no-op** against a root tsconfig of
  `{"files": [], "references": [...]}` — and CI ran it as a gate. It had been masking a
  missing barrel export.
- **`e2e/navigation.spec.ts` clicked `text=Monitoring`, which matched neither page.**
- **`MILLM_REQUIRE_CROSS_REPO_CHECKS=1` was unread by the path guard**, so the switch meant to
  make the cross-repo gate mandatory left it optional: with the sibling repo deleted it
  reported `6 passed, 13 skipped`.

---

## 5. Hardware acceptance (10.3 / 10.3b): four defects in the first twenty minutes

Everything below was found by running the feature against a real model on the node. None of it
was findable any other way, and each was invisible to a suite that is green.

### 5.1 ⚠ EVERY PROBE WRITE WAS SILENTLY ROLLED BACK

`POST /api/probes/import` answered `{"success": true}` with a real id and the full serialised
row. `GET /api/probes` returned **zero rows** a second later; arming then failed
`PROBE_NOT_FOUND` for the probe just created.

`get_db` yields a session and closes it — it does not commit — and `ProbeRepository` only
called `flush()`. Imports, the armed flag, parity reports, acknowledgements and events: all of
it, with the route having already returned success. `circuit_repository.py` has committed since
it was written; this file was the outlier and the difference was invisible to the suite.

**No test could have caught it.** Unit tests assert inside the transaction that made the write,
where a flush is sufficient. The integration tests mock the repository. It needs a *second
session* — which is to say a second HTTP request. `tests/unit/db/test_probe_writes_persist.py`
is built entirely around that. Control: reverting all eight sites to `flush()` turns all 8 red.

### 5.2 ⚠ `sae.path` IS A DIRECTORY

miStudio writes `ExternalSAE.hf_filepath` into it (`layer_11`), naming the directory holding
`cfg.json` and `sae_weights.safetensors`. The loader branches on `.suffix == ".safetensors"`
and would have handed a directory to `np.load`. **The contract's `path` has no description, so
the producer is the authority** — the field name is not.

### 5.3 ⚠ `sae.normalization` IS AN OBJECT, AND THE SLICE REFUSING IS WHY THIS WAS QUICK

`{"mode": …, "source": …}` passed through `str()` reached the slice as a dict repr. All sixteen
vectors failed. A lenient loader would have defaulted, encoded in the wrong basis, and produced
plausible features with different meanings — miStudio shipped exactly that once. Refusing an
unrecognised mode turned a silent wrong answer into a five-minute diagnosis.

**The reported reason was still misleading:** the read hook swallows callback exceptions by
design, so the crash surfaced as `no_scored_tokens` — "the scope selected no positions", not
"the probe broke". Now `encoder_failed: <error>`.

⚠ **And the mutation survived first time, on the caller again.** Reverting to `str(...)` left
36 tests green because every one called `_normalization_mode` directly. **Five times in this
estate now: a well-tested helper whose call site is tested by nothing.**

### 5.4 ⚠ THE PRODUCER AND CONSUMER DISAGREE ON PRECISION, AND PARITY CORRECTLY REFUSED

With every defect above fixed, the dense probe still refuses at the parity gate — **and it is
right to.**

miStudio scores probes in **float16** (`model_loader.py` hardcodes it). miLLM serves in
**bfloat16**, deliberately: fp16 overflows on bf16-trained models and produces NaN logits. bf16
carries ~8 mantissa bits against fp16's 11.

Measured on the node, 16 vectors, all comparable, tolerance 0.05:

| | max | median | min |
|---|---|---|---|
| per-token | 6.875 | 0.958 | **0.706** |
| combined (the score) | 0.0981 | 0.0168 | 0.0014 |

Combined lands within 0.05 on **14 of 16** and within 0.10 on **16 of 16**; no vector's
per-token trace comes near 0.05. A standalone **fp16** run of the same vectors gives per-token
0.10–0.28 and scores agreeing to 0.007 — still outside 0.05 per-token, because 0.05 is an
ABSOLUTE tolerance against values reaching 55, i.e. 0.09% relative.

Alignment was ruled out by measurement, not argument: shifting the comparison by one position
makes the error **190x worse** (0.278 → 52.8).

**So the gate is working and the contract has a gap.** Parity as specified compares per-token
scores at a tolerance only reachable by bit-identical computation, which is what miStudio
measured when it recorded "0.000e+00" — re-scoring in the same process with the same model
object. An independent implementation cannot reach it, and an independent implementation is
exactly what the gate exists to verify. This is the mirror image of the defect miStudio already
fixed once, where the parity check told a correct consumer it was wrong on every vector.

**This needs a product decision and is not mine to take** — loosening a parity tolerance is
loosening the thing that stops a probe reporting under an AUROC measured on something else. The
options, with what each costs:

1. **Record the dtype in the contract and compare at a dtype-aware tolerance.** Honest and
   durable; needs a contract revision on both sides.
2. **Gate on the combined score, report per-token as informational.** Matches how the probe is
   actually used (a verdict against a threshold of 2.879, where the worst observed score error
   is 0.098 — 3.4%). Needs the tolerance raised to ~0.1 for all 16 to pass.
3. **miStudio re-records its vectors under bf16.** Correct for this consumer, wrong for the next
   one that serves fp16.
4. **miLLM serves probe-carrying models in fp16.** Rejected: unsafe for bf16-trained models.

### 5.4b The k-sparse residual, after centering

Fixing §5.3's basis collapsed the k-sparse divergence but did not close it:

| | before | after | dense probe, for scale |
|---|---|---|---|
| per-token median | 58.26 | **22.26** | 0.958 |
| combined median | 1.683 | **0.101** | 0.0168 |
| combined within 0.05 | 0 of 16 | **6 of 16** | 14 of 16 |

A 17x improvement on the statistic that matters, and still 6x the dense probe's. So **one more
k-sparse-specific difference remains.** The named suspect is threshold rescaling under
normalisation — miStudio's own `threshold-rescale-uncentred-basis` memory records its extraction
calibration reading a healthy SAE 2–3x sparser for a related reason. Ruled out already: the
`W_enc` orientation (shapes run and the encode returns (T, k)), the normalisation arithmetic
(both are `√d / ‖x‖`, one branch for both mode names), and precision (§5.3).

This sits behind §5.4, not beside it: the DENSE probe does not pass parity either, so the
k-sparse residual cannot be the next thing chased. Resolve the dtype question first, then
re-measure both.

### 5.5 What PASSED on hardware

- **Persistence across a pod restart.** Both probes survived a rollout and were listed after it.
- **The identity gate, in full.** Arming the LFM2 probe against Qwen2.5-7B refuses
  `PROBE_MODEL_MISMATCH` naming **all five** differing fields — `hf_id`, `d_model` (2048 vs
  3584), `n_layers` (16 vs 28), `chat_template_sha256` and `revision` — not the first.
- **The evidence rung and its language**, verbatim from the server: rung 3, "detects on unseen
  tasks, compared with a judge".
- **The SAE slice loads from a real 268 MB dictionary** whose sha256 matches the probe's pin
  byte for byte, resolved out of the cache by repo and path.
- **Import from file**, at 491,967 bytes, through the 2 MB cap.

### 5.6 Not verified on hardware, and why

- **GGUF refusal.** The only GGUF model on the node is in `error` status and will not load, so
  the `PROBE_HOOK_UNSUPPORTED` path is covered by unit test and code inspection only. Recorded
  as unverified rather than assumed.
- **Import from HuggingFace.** `mistudio/sae-…` and the probe repo are **private** (HTTP 401
  anonymously), so the Hub path needs a token miLLM does not hold. The SAE was staged onto the
  node directly and verified by hash instead, which tests the slice but not `hub/import`.
- **SC-4's absolute figure.** Needs an armed probe, which needs §5.4 resolved.

## 6. Still owed

- **10.3 / 10.3b / the SC-4 absolute figure** — hardware acceptance. All three need this branch
  deployed with a model loaded; they are not skippable and they are not done.
- `probe_head.token_scores` upcasts to float32 on the grounds of a **CPU** measurement of the
  fp16 error. Re-measure on the card before treating the 8.5e-02 figure as the GPU number.
