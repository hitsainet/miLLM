# Multi-Agent Review — Feature 11: Co-Activation Sensing

**Date:** 2026-07-16 · **Scope:** feature (commits `9aff389..HEAD`, miLLM)
**Goal gate:** ≥10 findings per round, 3 rounds.

---

## Round 1 (multi-angle /code-review, 2 parallel finder agents) — 28 raw / 23 unique: 19 fixed, 4 documented

**Critical, fixed:**
1. **`MAX_CONCURRENT_REQUESTS` defaulted to 2 while the whole design assumes 1** — the "serial
   queue semaphore" every sensing/steering comment relies on admitted TWO generations at once:
   interleaved sensing boundaries corrupt each other, and (beyond 011) per-request steering
   apply/restore from Features 8/10 could interleave. → default 1 + loud init warning; the CBM
   backend is the supported concurrency path. Old default was pinned by a test — updated with the
   rationale.
2. **Hung generation thread poisons the next request** — the streaming join-timeout path collected
   and released the queue slot while the stale thread kept calling `_sense` into the NEXT request's
   freshly-begun buffer. → hang disarms sensing (lose observation until re-activation, never
   mis-attribute).
3. **`deactivate_profile`'s disarm condition was dead code** — it read `profile.is_active` AFTER
   `repository.deactivate` had mutated the identity-mapped row (always False); with
   `clear_steering=false` sensing stayed armed for a deactivated cluster. → `was_active` captured
   first. **`delete_profile` never disarmed at all** — armed runtime kept INSERTing against a dead
   FK (an IntegrityError per request, silently swallowed). → disarm on delete.
4. **Unvalidated member indices could fire a CUDA device-side assert at arm** (poisoning the
   process context) and `activate(apply_steering=false)` armed without the n_features gate. →
   arm applies the same declared-feature-space gate + explicit index-bounds check; the toggle
   route surfaces refusals with the real reason.
5. **Zero theta floor was degenerate** — members without `max_activation` fired on ANY positive
   activation (min_k inflation; floor_only mode saturated the cap instantly). → such members get
   an infinite threshold (never fire) unless a positive floor is configured; all-unsensable
   clusters refuse to arm with an actionable message.

**Also fixed:** per-element GPU→CPU syncs in the hot path (a hot prefill cost a CUDA sync per
fired member per position — now one transfer per pass + CPU threshold cache); speculative decoding
goes unsensed (verification passes re-run rejected positions — offset accounting diverges);
`begin` snapshots the profile_id so a mid-request re-arm cannot mis-attribute the flush; streaming
setup failures close the boundary (stale open boundary let later non-begin passes sense garbage);
batched-pass guard (observable skip instead of silently sensing row 0); `fired_count` = member
union on span merge (disagreed with `fired_members`); truncated marks only the LAST event (the cut
point was unrecoverable); event detail 404 (`SENSING_EVENT_NOT_FOUND`, was 422); prune-on-read
(the age cap is the documented privacy control — it must hold for idle clusters); status reports
`enabled_clusters` distinctly from `armed` (FTID pitfall 8); WS emission throttled (SEN-P4 parity
with monitoring); SAE attach re-arms when the active profile wants sensing (activate-then-attach
left sensing silently dark while the toggle showed on); `/v1/completions` wired for sensing
(single-prompt; was silently unsensed while status said armed); n>1/speculative skips logged;
frontend: `socket.off(event, handler)` (wholesale off killed sibling listeners), per-query-key
cache updates (a scoped hook dropped other clusters' events from the 'all' cache), toggle-only
hook (double subscription prepended every live event twice); API reference sensing section +
`sensing:event` in the WS docs.

**Fixed along the way (user-reported, pre-existing):** SAE attach 500'd at hook install —
`sae_hooker` logged with structlog kwargs on the stdlib logger (`TypeError: unexpected keyword
argument 'layer'`), reproduced live; same latent bug on both transposed-weight paths in
`sae_loader`. Third instance of this class today → `tests/unit/test_logging_conventions.py`
sweeps the whole package statically.

**Documented (not fixed this round):**
- Arm/disarm from the event loop is not synchronized with an in-flight generation's buffer
  (mid-request re-arm can reset offsets → wrong positions for that one request). Same class as
  the global-steering-mutation race documented in 010 R2; profile-id snapshot bounds the damage
  to positions, never attribution. Needs queue-serialized state mutations as its own work item.
- Multi-prompt `/v1/completions` and `n>1` chat go unsensed (position accounting would
  concatenate independent generations) — logged, documented v1 limitation.
- FPRD deviations accepted: route is `/api/sensing/{id}/enable` (not `/sensing/clusters/{id}`),
  PK is autoincrement int (not `sev_` hex) — recorded here as intentional.
- WS throttle drops beyond 5 events/flush at >10 flushes/sec — the DB is complete; UI reconciles
  on refetch.

Tests added this round: 30+ (wiring both generation paths, ambient rules matrix, force-serial-off
CBM eligibility, DB-outage resilience, lifecycle sync, logging-convention sweep). Suites: backend
1067 / frontend 203 / manual builds.

## Round 2 (fix verification + fresh angles — inline; the agent round was cancelled mid-run) — 14 items: 4 fixed, 3 documented, 7 verified

**Verified sound (round-1 fixes):** infinite thresholds survive fp16/bf16 arm-time casts (checked
empirically); attach re-arm runs AFTER `_sae_state.set` with the hook installed; the hung-thread
disarm/setup-close/flush paths all handle the (sae, profile_id) tuple consistently across all
three generation paths; the frontend live-prepend prefix key cannot poison the detail/status
caches; prune-on-read transaction semantics (commit only when rows deleted); the text-completion
reindent (structure + full suite).

**Fixed:**
1. `_ws_dropped` was tracked but surfaced nowhere → `ws_events_dropped` in the status API.
2. The WS throttle's initial timestamp of 0.0 would drop the FIRST flush on platforms where
   `monotonic()` starts near zero → initialized to `-inf` (first flush always emits).
3. Mid-request re-arm branded the snapshot profile's rows with the NEW cluster's display token
   and member labels → neutral formatting when the flush profile differs from the armed one.
4. An arm refusal at activation (unusable thresholds, mismatched SAE) was a log-only event —
   activation now returns `sensing_armed` so callers/UI see whether sensing engaged.

**Documented:** toggle-route arm failure persists the column then errors (intent-vs-runtime split
is deliberate; the 422 reason tells the user); WS throttle drops reconcile on refetch (stated in
the WS reference); `_sensing_batch_warned` is once-per-process by design.

## Round 3 (/review, 4 perspectives) — 19 findings: 13 fixed, 6 documented

**The round's genuine regression (fixed):** the R2 neutral-branding fix MUTATED singleton state —
a snapshot flush after a re-arm stomped the newly armed cluster's display token and labels for
every subsequent event. → local formatting inputs threaded through `_summary`; pinned by a test
that arms A, re-arms B, flushes A's snapshot, and asserts B's state is untouched.

**Also fixed:** `max_activation <= 0` treated as missing (same degenerate class as R1's zero
floor; negative `epsilon`/`theta_floor` overrides degrade to defaults); `to_device` migrates the
sensing tensors (a device move left member slices behind — sensing went silently dark);
attach-path arm refusals land in the attach response `warnings` (same silent class R2 fixed for
activation); armed-state duality self-heals (status() reconciles against the attached SAE — a
swallowed disarm can no longer report armed forever); prune-on-read throttled to once/10 min
(every list request was a DB writer racing the flush prune); frontend `SensingStatus` gained
`ws_events_dropped` + `enabled_clusters` and the panel shows both (the R2 observability fix ended
at the JSON boundary); Clear button scope matches the list (clear ALL + confirm; scoped clear
silently left other clusters' rows / deleted everything when disarmed); status invalidation
debounced (burst = one GET, not one per event); honest no-context message (decode failures were
mislabeled as K=0 config); manual speculative wording (sensing is DISABLED under speculative,
not "slightly noisy"); FPRD EC-11.4 amended (dated) to the shipped inf-threshold semantics;
R2 fixes + R1's 404 now pinned (truncated-last persist, throttle counts, re-arm mismatch,
reconcile, zero-max-activation, negative-epsilon, peak-decrease, post-cap tail, union
fired_count — the round's named mutation survivors all die).

**Documented:**
- The `sensing` override block lives OUTSIDE the frozen v1 schema (additive-unknown by design —
  the contract is miStudio-owned; producers can emit it once the contract revs). Manual documents
  the keys; a typo degrades to defaults by construction.
- begin/flush choreography is triplicated across chat/stream/text with per-path drift
  (hang-disarm exists only on streaming — the other paths use `asyncio.to_thread` without a
  timeout, so the hang class differs); a SensingRequestContext refactor is named debt.
- Toggle-route vs `_sync_sensing_arm_state` duplicate the arm/disarm decision with different
  error surfacing (deliberate: route raises, activation warns) — consolidation debt.
- Span highlighting in SensingEventDetail needs a persisted `context_start_pos` (schema change);
  the metadata line carries the absolute span meanwhile.
- FTASKS 6.2 (streaming early-stop context) + §9 criteria 3/4 ride the post-deploy E2E.
- Speculative-decoding exclusion is an accepted FPRD SEN-D1 deviation (amended note).

## Gate
**SHIP** — 56 findings across 3 rounds (36 fixed, 13 documented, 7 verified-sound), suites
backend 1083 / frontend 205 green. Post-deploy E2E (live co-firing traffic + overhead budget)
rides the increment's rollout.
