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

## 5. Still owed

- **10.3 / 10.3b / the SC-4 absolute figure** — hardware acceptance. All three need this branch
  deployed with a model loaded; they are not skippable and they are not done.
- `probe_head.token_scores` upcasts to float32 on the grounds of a **CPU** measurement of the
  fp16 error. Re-measure on the card before treating the 8.5e-02 figure as the GPU number.
