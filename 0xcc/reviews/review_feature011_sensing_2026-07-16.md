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
