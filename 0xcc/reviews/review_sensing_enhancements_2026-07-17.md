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
