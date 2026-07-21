# Feature 20 — MCP Circuit Surface — Review Round 2

**Date:** 2026-07-21
**Scope:** attack R1's fixes. R1 closed 20 findings; this round asks which of
them were real, which were unpinned, and which reopened something.
**Suites:** miLLM 2124 passed / 8 skipped · miStudio unit suite green
**Findings:** 22 · **Fixed:** 22

## The shape of this round

R1's theme was *reachability applied to the miStudio registry*. R2's theme is
that **R1 applied that rigour to the registry and not to its own instruments**.
Nine of twenty findings are defects in the guards R1 built, four of them cases
where the guard reported green for precisely the condition it existed to detect.

Two findings are recorded as **NOT DEFECTS** after measurement. Both came from
reviewers, both were plausible, and both were wrong. They are kept in the record
because last feature's R3-12 was a fix made for a defect that did not exist, and
the cheapest way to not repeat that is to make disproof visible.

## Findings

### The reference material was not runnable (R2-11, R2-12)

| # | Finding | Fix |
|---|---|---|
| R2-11 | §7's circuit flow passed `activate=true` and `acknowledge_unvalidated` to `millm_import_circuit`, which accepts neither — an agent following the reference flow failed on LINE ONE. It also contradicted its own §4 row 200 lines above. | Rewritten against shipped signatures; now guarded against the LIVE registry (every tool must exist, every argument must be on its schema). Writing the guard immediately found two of my own errors. |
| R2-12 | The two recovery tools — the most dangerous failure modes in the surface — appeared in no end-to-end narrative. | Flow expanded to cover all three activation refusal branches, both intensity refusals, and the "enabled but no events" diagnostic. |

**Why R1 missed it:** R1-03 corrected the contract's status column. The guard it
wrote read the status COLUMN and never the EXAMPLE CODE beneath it — which is
where the surviving defect lived. An example is executable prose and nothing was
executing it.

### The contract guard was scoping its table by coincidence (R2-16..19)

| # | Finding | Control |
|---|---|---|
| R2-16 | `_rows()` scanned the whole document with no table scoping, capturing `millm_import_cluster` from the CLUSTERS table (an escaped pipe split a 2-column row into 3 cells, so its "status" was the prose fragment `` `fail`) ``). The other 14 non-circuit rows were excluded BY ACCIDENT — the arity filter was doing the scoping, by coincidence. | — |
| R2-17 | That phantom 17th row inflated the count past R1's `>= 14` "checking nothing" tripwire. Three real rows could be deleted with the alarm green. | delete 3 rows → **2 failures** (was 0) |
| R2-18 | `rows[name] = …` was last-write-wins across the document: a stale duplicate overwrote a correct status, or in the other order MASKED a wrong one. Both directions verified. | duplicate row → **4 failures** |
| R2-19 | `"MCP ✅" in status` is unanchored, so text after the mark is invisible. `REST ✅ · MCP ✅ REVOKED — do not call` passed as a clean claim: the contract told a human not to call the tool while the guard certified the row. **That is the exact over-claim divergence F20 exists to detect, in the one form the detector could not see.** | revocation qualifier → **1 failure** |

Also: the Status column is now located BY NAME, not `cells[-1]`. A trailing Notes
column moved every status onto it and failed all 16 rows as "your tools are
mismarked" — blaming the contract for a parser bug.

### Reachability (R2-14, R2-15, R2-20)

| # | Finding | Control |
|---|---|---|
| R2-14 | The increment's most transferable lesson — *a capability is not shipped until a test FAILS when its wiring is removed* — was written down NOWHERE. The next increment would have relearned it the same way. Now in both repos' CLAUDE.md and the global review discipline, as the strong form of the "grep for a caller" rule already there. A grep proves a call EXISTS, not that anything DEPENDS on it. | — |
| R2-15 | The path guard could skip a call and still report green. R1's floor guard only catches the set going EMPTY; 15-of-16 sails through, and the dropped call is the one nothing verifies. Four silent classes found by probing (single quotes, aliased receiver, path from a constant, unknown verb). Now counts call SITES against parsed PATHS. | unparseable call → **2 guards fail** (was silent) |
| R2-20 | `millm_delete_circuit` was the only irreversible operation in the module with no gating at all, while its sibling destructive tool requires explicit scope opt-in. The miLLM route deletes a live circuit unconditionally. **Nothing anywhere in the stack stood between a mistyped id and production steering being torn down** — first symptom: the model quietly behaving differently. | guard removed → **fail**; fail-closed → **fail**; cosmetic refusal → **fail** |

R2-20's fail-open branch is deliberate (cleanup must stay possible during an
outage) and therefore **pinned by its own test**, so a future "hardening" that
flips it to fail-closed breaks the build rather than silently making the tool
unusable when it is most needed.

### Documentation was unreachable too (R2-21, R2-22)

| # | Finding | Control |
|---|---|---|
| R2-21 | The "deliberately NOT served" rows are structurally invisible to `_rows()` (they open `\| _(`, since the point is that no tool name exists) — so the one class of row carrying a SAFETY decision was the one class nothing checked. EC-20.5 records hub-import as deliberately absent because a circuit references several SAEs by id; importing blind serves it against the WRONG FEATURE BASIS. Adding the tool later would be a one-line change that reads like filling a gap. | forbidden tool registered → **fail**; not-served rows deleted → **fail** (the guard refuses to go blind silently) |
| R2-22 | `features/circuits.md` — an entire increment's user documentation — was **absent from the sidebar**. The page builds, renders, and no reader can navigate to it; Docusaurus does not warn, because an orphaned doc is a valid doc. Found while adding the missing MCP page, and it is the SAME defect as the unregistered tools: everything that touched it touched it directly by path, never through the entry point a real reader uses. | circuits removed from sidebar → **fail**; sidebar naming a missing page → **fail** |

Also shipped: `manual/docs/features/mcp-circuits.md`, the human orientation to
the agent-facing surface (the three refusals and what each means, where the
tools deliberately stop and why, the guards on the two destructive operations,
the composition hazard WITH its one-model-one-fixture caveat, and the rung-2
rule for the word "causal"). The manual had no MCP documentation at all.

### Recorded as NOT DEFECTS (measured, not assumed)

| Claim | Measurement |
|---|---|
| The path-extraction regex misses multi-line calls | **False.** `\s*` spans newlines; all 16 sites parse. Recorded because accepting it unmeasured is how a fix gets made for a defect that does not exist. |
| `reapply` is accepted but never transmitted | **False.** It is not in this module at all — it is on `millm_set_intensity` in `millm_runtime.py`, correctly transmitted, with `reapply is not False` deliberately mapping `None`→`True`, and a test pinning it. |
| `intensity` lacks a local bound check | **Not a defect.** The bound is a per-circuit AUTHORED `intensity_range`, not a fixed 0..2. A local check would guess at a value the client cannot know and would drift from the authoritative server-side one. Deferring is correct. |

## Instrument failures in this round (recorded, not hidden)

Three times a harness lied about its own result:

1. A `RUN()` shell function `cd`'d inside a command substitution — a subshell —
   so pytest ran from the wrong directory and collected NOTHING. Three
   negative controls reported "0 failures" and I nearly recorded a working
   guard as unpinned.
2. `grep -c` on the summary counted the summary line itself, so pass and fail
   both read as "1".
3. A GPU-cleanup plugin prints after pytest's summary and the summary line is
   suppressed, so text-scraping the tail yields nothing.

**Resolved by judging controls on the pytest EXIT CODE**, which no plugin can
overwrite. This is the same class as the `pgrep` self-match from the previous
feature: *the monitoring was wrong, not the thing monitored.* When a control
reports "survived", verify the harness before recording the finding.

## Carried to R3

- The cross-repo CI job has never actually run — it is new in this round. Its
  first execution is itself a verification step, and until it goes green the
  guard is verified only by me running it locally.
- `test_manual_pages_are_reachable.py` uses a deliberately crude id extraction
  (over-collects, never falsely accuses). Worth checking in R3 whether it can
  be fooled by a page reached only from a category with no explicit id.
- R2-20's fail-open branch is a judgement call. R3 should challenge it: is
  "cleanup must work during an outage" worth a window where a serving circuit
  can be deleted without the guard firing?
