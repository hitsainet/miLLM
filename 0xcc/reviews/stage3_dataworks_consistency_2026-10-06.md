# Stage 3 Consistency Review — Dataworks Support (BRD-04, Features 025–030)

**Date:** 2026-10-06 · **Reviewer:** Stage 3 consistency agent · **Scope:** documents only; no code.
**Tree:** miLLM `aa25bc4` (2026-10-06). `git diff 61bed07 HEAD -- millm` and the same against
`7aa659c` and `f5c71b6` are empty, so every cited commit's code equals HEAD's.
**Inputs:** BRD-04 (R-04.1–47); PPRD and PADR v1.5; the 24 feature documents 025–030; miDataworks
`brd-decisions-2026-10-05.md` and `fprd-open-questions-2026-10-06.md` (read only).

Abbreviations: PADR = project architecture decision record; PPRD = project product requirements
document; FPRD, FTDD, FTID, FTASKS = feature PRD, technical design, implementation guide, task list;
GGUF = llama.cpp's model file format; SAE = sparse autoencoder; TTL = time to live.

## 1. Held amendments applied (8)

Every amendment carries "(Stage 3, 2026-10-06, requested by 0NN)".

| # | Where | Change | From |
|---|---|---|---|
| A1 | PADR §2.2, §5 | `xgrammar>=0.2.8,<0.3` and `jsonschema>=4.23,<5` added to the stack and the library table | 025 |
| A2 | PADR §10 "Constrained decoding" | "unless BRD-04 open question 3 extends support" removed; GGUF structured output **refused** in v1; library named | 025 |
| A3 | PADR §10 "One scoring path" | Shared function is `_score_prompts`, extracted from `_score_text_completion` | 025 |
| A4 | PADR §5 | `http-sfv==0.9.9` as a dev-only test dependency. §5 lists dev tools (pytest, ruff, mypy), so it belongs there | 028 |
| A5 | PADR §10 lease | Leases in memory (X-01); debts: management load/unload ignore `locked`; registry assumes one process | 029 |
| A6 | PADR §10 lease trade-off | Notes that C8 closed BRD-04 open question 1 | 029 |
| A7 | PADR §10 steering | First-SAE profile defect as tracked debt, plus its open question (fix both paths together; recommended yes) | 028 |
| A8 | PPRD increment section | `_score_prompts` dependency; FR ranges 26.1–26.10 and 27.1–27.10; BRD-04 §9 marked resolved; debts listed | 025–029 |

Also closed at source: 025 FTDD §14 amendment list, 028 FTDD/FTID open item 2, 028 FTASKS 0.2, and a
status note under BRD-04 §9.

## 2. Cross-feature consistency

| Pair | Finding | Fix |
|---|---|---|
| 026 ↔ 029 | 026 called `acquire(...)`, `renew`, `release`, `current`, `matches(lease_id)`, `ttl=`. 029 fixes `acquire_lease`, `renew_lease`, `release_lease`, `resolve_lease`, `get_lease`, `ttl_seconds=` | 026 FTDD §7, FTID §5, FTASKS 0.2 and 5.11 renamed to 029's API; error-to-`waiting_reason` mapping made explicit (`ModelNotResidentError`, `MODEL_BUSY` → `model_not_resident`; `ModelLeasedError` → `lease_unavailable`) |
| 026 ↔ 029 | `POST /v1/batches/{id}/lease` validation had no code for a lease on another model | `resolve_lease` → `None` or wrong model → `404 LEASE_NOT_FOUND`, matching 029 §5.1 |
| 026 ↔ 029 | `in_flight` counted interactive holders only, so a batch chunk in the slot was invisible to the wait estimate | `in_flight = holding_count + background_holding_count`; `queue_waiting = queue_pending − holding_count` (029 FPRD, FTDD §5.2, FTID; 026 FTDD §7) |
| 026 ↔ 029 | Backlog wiring unnamed in 026 | `register_backlog_provider(runner.backlog_rows)` |
| 029 internal | FPRD named the lease end time `released_at`; FTDD `ended_at` | FPRD aligned to `ended_at` |
| 025 ↔ 026 ↔ 030 | 030 used `evaluate(request, row, "embeddings")` and `evaluate(request, model, endpoint=...)` | Aligned to `evaluate(request, endpoint, engine, strict=...)`; 030 flow now shows the `X-miLLM-Strict` parse and its `400` |
| 027 ↔ 028 | Scoring header behaviour stated in 028 only | 027 FTDD §5.3: scoring-mode activations report `read_point: "unsteered"` and `X-miLLM-Steering: none`; `/api/probes/score` carries no steering header. 028 FPRD FR-28.3.1 says the same |
| 028 | Chat-scoring row named "Feature 25's chat scorer" | Names `_score_prompts` |
| 027 | `>=` claim | Verified: `probe_runtime.py:665` is `value >= threshold`, commit `0c3f3fe`, pinned at `test_probe_runtime.py:157`. No document claims miLLM uses `>` |
| Queue depth | 026 FR-26.4.7 and PPRD FR-29.7 both say `/api/health/detailed` | Consistent; no change |

## 3. Hash test vectors (028 FTDD §5.3)

Recomputed with `stage3-ml/tv.py`, following the spec literally: clamp to ±200, drop zero and `-0.0`,
lines `millm.steering-set/v1`, `sae=<id>`, `<index>:<struct.pack(">d").hex()>` ascending, each
LF-terminated, UTF-8, `sha256:` + hex.

| Vector | Canonical form reproduced | Hash reproduced |
|---|---|---|
| TV-1 | yes | yes `a4eae730…44105` |
| TV-2 | yes | yes `cc6c48fa…d78a7` |
| TV-3 | yes (`0.0` and `-0.0` dropped; order 5, 77) | yes `b8439122…2e710` |
| TV-4 | yes (500.0 → 200.0 = `4069000000000000`) | yes `3f51779e…b78df` |

**All four reproduce. No critical finding.**

## 4. Traceability

- **BRD → FR:** all 47 R-04.x appear in an FPRD coverage table. No orphan.
- **FR → FTASKS:** every top-level FR-25.1–FR-30.4 is covered by a parent task. Seven sub-FRs
  (FR-26.4.2, 26.4.3, 26.4.6, 26.6.2, 26.6.3, 26.6.7, 29.5.4) are not cited by number, but each sits
  inside a cited range (for example "FR-26.6.1 – FR-26.6.4"). No orphan.
- **FR → BRD (reverse):** three FRs trace to decisions, not to BRD-04. They are deliberate exceptions,
  now stated in the PPRD: FR-26.9 (retention, checkpoint default), FR-26.10 (provenance), FR-27.10
  (`>=` boundary, P-03).
- **Orphans fixed:** 0 (none found). PPRD FR ranges corrected: 2.

## 5. Contradictions fixed (10)

1–4. The four 026 ↔ 029 lease, holding and backlog naming items (§2).
5. 029 `released_at` vs `ended_at`.
6. 030 `evaluate` signature (two documents).
7. 027 ↔ 028 scoring header scope.
8. 028 chat-scoring function name.
9. PPRD "Requirements Covered" for 026 and 027 omitted FR-26.9/26.10 and FR-27.10.
10. PPRD and BRD-04 §9 still listed six resolved questions as open.

Headers (`X-miLLM-*`, 12 names), routes and config names were otherwise consistent across all documents.

## 6. Code citations

96 `path:line` citations checked against `aa25bc4`, across all six FTDD/FTID pairs and BRD-04.

| Result | Count | Detail |
|---|---|---|
| Correct | 92 | |
| Wrong line, corrected | 2 | 029 FTDD `inference_service.py:634` → `:635` (`unloading: True`); BRD-04 R-04.39 `completions.py:110-113` → `112-115` (110–111 are comments) |
| Ambiguous path, qualified | 2 | 028 FTID `errors.py:355-359` and 030 FTDD `errors.py:17-24` mean `millm/core/errors.py`, not `api/routes/openai/errors.py` |

## 7. Template conformance

All 24 documents carry every mandatory heading of `0xcc/instruct/004`–`007` (15 FPRD, 14 FTDD,
15 FTID sections; FTASKS Relevant Files, Notes, Category Checklist, Tasks, Feature Acceptance).

## 8. Cross-repo mismatches (for the orchestrator; not edited)

| # | Repo / document | Assumes | miLLM says | Suggested owner action |
|---|---|---|---|---|
| X1 | miDataworks `007_FTDD` §6.2 (lines 334–337), §12 | miLLM's hash FTDD "is not written yet"; canonical form is "SAE ID plus (index, strength) pairs sorted by index"; placeholder vector | 028 FTDD §5.3 is written: version line, `sae=` line, binary64 hex per feature, zeros dropped, clamped value hashed, `sha256:` prefix; TV-1–TV-4 published and reproduced | 007 adopts §5.3 verbatim and pins TV-1–TV-4; its hash precondition can close |
| X2 | miDataworks `005_FTDD` §6.5 step 3 | `GET /api/health/detailed` yields the resident model's ID for `POST /api/models/{id}/lease` | Health reports only a `"Model: <name>"` message string, no ID (`health.py:330-337`); 029 adds `lease` (an ID only when leased) | 005 resolves the ID from `GET /api/models` (status `loaded`), or miLLM 029 adds a resident `model_id` to the health block |
| X3 | miStudio `034_FTID` §3.5 | Renew, release and status shapes "fixed by miLLM's FTDD" | Now fixed in 029 FTDD §5.1: `POST …/lease/renew`, `DELETE …/lease`, `GET …/lease`; the lease ID travels **only** in `X-miLLM-Lease`, never in a body or path | 034 phase-4 precondition can record these; tools must send the ID as a header |
| X4 | miStudio `034_FTDD` §5.3 | `X-miLLM-Steering` read on inference tools | Present on chat and completion responses, `none` on scoring; absent on `/api/probes/score` and `/v1/embeddings` | 034 must not expect it from `millm_score_probes` |
| X5 | miStudio `034_FTDD` §5.3 | `X-miLLM-Strict: true` on every FR-16/FR-17 request | Parsed on `/v1` chat, completions, embeddings and batch create. `/api/probes/score` ignores the header but rejects unknown body fields (`extra="forbid"`, `422`) | Informational: no silent drop occurs |

## 9. Open for the operator

1. **028 FPRD §14 question 1:** fix both profile paths together (select by `sae_id` and `layer`,
   refuse when not attached)? Recommended yes, as one small increment. Recorded in PADR §10.
2. **X2:** choose whether miDataworks reads `/api/models` or miLLM 029 adds a resident model ID to
   `/api/health/detailed`.

No other item blocks implementation.
