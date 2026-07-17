# Multi-Agent Review — Sensing Enhancements (post-increment goal, 2026-07-17)

**Scope:** commits `37e771e` (four goal items) + follow-ups · **Goal gate:** 3 rounds, fix findings.
Items: span highlighting (context_parts), history dedup (LCP suppression), quorum=ALL-sensable
default, UI-adjustable min_k.

## Round 1 (2 finder agents + 1 inline catch) — 28 raw / 23 unique: 17 fixed, 6 documented

**Inline (fixed ahead of the agents, `69e2475`):** `context_parts` leaked into WS payloads — the
name-list content filter missed the new field → prefix-based `context_*` exclusion + pin.

**Critical, fixed:**
1. **Cluster switch kept the OLD cluster's dedup history** (`_sync_sensing_arm_state` re-arms
   without an intervening disarm) — cluster B's genuine first reports over an ongoing
   conversation were suppressed using history B never reported. → ANY (re)arm clears history;
   also resolves quorum-lowering keeping newly-reachable moments suppressed (both finders).
2. **Truncation × dedup: capped-away moments were permanently lost** — history advanced past
   positions the event cap prevented from ever being reported. → history stops at the last
   REPORTED position on truncated requests.
3. **Post-disarm history race** — `note_request_ids` ran after the hitless-return-guard for
   in-flight requests, repopulating history a disarm had just cleared. → armed-profile guard
   (begin-time snapshot id must match).
4. **SentencePiece gluing in the highlight** — independent per-segment decodes strip
   segment-leading space markers ('...told me**secret**plans...'). → PREFIX-decode slicing
   (monotonic appends; parts exactly reconstruct the window; specials kept consistently).
5. **Mid-request re-arm lent the flush the NEW cluster's config** — context window size and
   member denominator came from the armed config, so summaries could read "5/3 members" and
   context could vanish. → full config snapshot at begin rides the sensing ctx.

**Also fixed:** span-beyond-ids yields no context instead of an empty box + zero-width mark with
a contradictory "no context" message; prompt-only fallback history can never SHRINK the boundary
(failed generations re-reported once; prefix-of-history is discarded); `min_k` validated against
the SENSABLE ceiling (an unreachable quorum looked healthy while never firing) with both counts
in the error; status exposes `sensable_count` and the panel denominator/input-max use it; a
reset-to-default button (the null-clear path was unreachable from the UI — dead toast);
`from exc` on the re-arm failure; frontend tests for the quorum control, reset, highlight
rendering, and plain-text fallback (the mock lacked `setConfig`, so any future test would have
crashed); `context_parts` DB round-trip + old-rows-null pin; export-strips-`sensing_overrides`
pin; live re-arm route branch + sensable-refusal tests; management-api rows for the config
endpoint + `context_parts`; manual upgrade caution (the stricter default silences
previously-firing clusters BY DESIGN — say so); `millm_sensing_config` MCP tool (miStudio) +
contract row (agents hitting the all-members default can now tune quorum without a human);
typing nits.

**Documented (accepted v1 semantics/known limits):**
- Dedup is "first occurrence in the request stream", single history slot: cross-conversation
  identical openers suppress once; interleaved clients defeat dedup (bounded duplicates return)
  and can cross-suppress shared prefixes. Per-conversation keying is named future work;
  `SENSING_DEDUP_HISTORY=false` opts out.
- Boundary-straddling spans clip to their new-positions tail (peak/fired from the new part).
- Live re-arm (config/toggle) still races an in-flight generation's buffer — same documented
  class as steering mutations; queue-serialized arm mutations remain the named debt.
- No-stopping-criteria environments store prompt-only history (decode events re-report; context
  degraded) — inherent to the transformers fallback, mitigated by never-shrink.

## Round 2 (fix verification + fresh angles, 1 finder agent) — 11 findings: 9 fixed, 2 already-fixed/documented · 10 R1-fix verdicts (9 correct, 2 escalated)

**Verified correct:** arm-clears-history (all 5 call sites lifecycle-level — dedup intact);
truncation cap (hits proven ascending; truncated×prefix-guard proven safe — capped sequences are
never mistaken for prefixes); disarm-path profile guard; SP prefix-decode; config snapshot
(fresh object per arm — un-mutable by re-arms); sensable validation server-side; reset wiring;
docs substance; WS prefix filter; export strip.

**Fixed:**
1. **`sensable_count` never reached the API** — the status schema lacked the field, pydantic
   silently dropped it, and the UI's `?? member_count` fallback reverted the denominator/input
   cap to pre-fix behavior. → schema field + a route-body test (the class of "service returns
   it, schema eats it" now has a pin).
2. **Same-profile mid-request re-arm caused PERMANENT suppression** — the re-arm destroys the
   open boundary (hits dropped, documented) but the flush still wrote the full sequence into
   history: dropped moments became LCP-suppressed forever. → empty request_id (the
   destroyed-boundary signal) skips the history write.
3. **Byte-level BPE mid-character splits** could misplace the highlight (U+FFFD rewrite breaks
   length-slicing) → `startswith` consistency guard falls back to plain text.
4. **`millm_sensing_config` cleared the override when called WITHOUT min_k** — an agent
   "checking the config" wiped the operator's tuned quorum. → min_k-or-`reset=true` required;
   omission refused; WRITES-config docstring; 3 pins (the tool was also entirely unpinned).
5. Malformed admonition fence in the manual (caution block swallowed content); span-merge
   sentence restored to its paragraph.
6. Reset-vs-draft race (blur committed a stale draft against the null-clear) → mousedown
   preventDefault + draft cleared; draft also cleared on armed-profile switch (a draft typed for
   cluster A could commit against B); the invalidation debounce became a real `useRef`;
   `SENSING_MAX_EVENTS_PER_REQUEST` floored at 1 (cap=0 froze dedup history while looking armed).

**Already fixed mid-round:** CLAUDE.md session record (`d1a6f1e`, landed after the agent's
snapshot). **Documented:** history LCP cost (~ms at 32k ctx, off the hot path) and ~1 MB
worst-case history — acceptable.
