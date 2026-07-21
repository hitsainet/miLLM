# Feature 20 — MCP Circuit Surface — Review Round 3

**Date:** 2026-07-21
**Scope:** attack R2's fixes.
**Status:** IN PROGRESS — 19 findings

## The finding that matters most

**R3-04: my own R2 commit disabled the entire CI test workflow, and I did not
notice for four commits.**

`HF_HOME: ${{ runner.temp }}/hf` in a job-level `env:` block is a schema
violation — the `runner` context exists only inside steps. GitHub does not fail
that job and run the rest. It rejects the **whole workflow file**: no jobs, no
logs, no annotations, and an Actions UI that says only *"This run likely failed
because of a workflow file issue."*

    07:46 success 48d38c08   <- last green
    07:52 failure a955e21b   <- my cross-repo job added
    07:57 failure 02cb4b84
    08:02 failure 22ca2824
    08:17 failure bb0882d8

So the pre-existing backend suite stopped running four commits ago while I
reported it green from local runs each time. The failure is invisible from the
terminal in a way a normal red build is not: pytest passes, the push succeeds,
nothing in the working copy is wrong, and `gh run view --log-failed` returns
*"log not found"* — because there is no log.

**I found it only because R2's own record listed "the cross-repo CI job has
never actually run" as carried work.** Had I not written that down, four red
builds would have shipped behind a green local suite.

This is the second occurrence this session, and a memory file
(`check-ci-not-just-local-green`) already warned about it. **The lesson did not
stick as prose**, so R3-05 makes it a test: `test_workflow_contexts_are_legal.py`
rejects out-of-scope contexts in job `env:`/`if:` blocks and re-parses every
workflow file.

Fix verified end-to-end: CI now **green**, and the `Contract ↔ MCP registry`
job ran for the first time and passed — the guard is CI-verified rather than
verified only by me locally.

## Findings

| # | Finding | Control |
|---|---|---|
| R3-01 | R2-22's sidebar guard **could be fooled**. It took EVERY quoted string, on the reasoning that over-collecting could only make it permissive and never falsely accuse. Wrong: over-collecting means a doc id can be covered by an unrelated quoted word. Proved by attack — orphan `troubleshooting.md` and rename any category label to `'troubleshooting'`; the page is unreachable and the guard passes. Top-level pages (no slash in the id) are the most vulnerable shape and this repo has one. **The defect the guard exists to catch, inside the guard.** | the defeating attack → **fail**; nested orphan → **fail**; top-level entry naming a missing page → **fail** (the old filter structurally skipped these) |
| R3-02 | *Withdrawn.* I read a `sed` extract as showing a missing blank line between two functions. The blank line was there; `sed` had collapsed it. Recorded because acting on it would have been a change made for a defect that does not exist. | — |
| R3-03 | R2-20's fail-open delete was **silent**. When `/active` could not be read the delete proceeded WITHOUT the advertised protection, and the response was byte-identical to a clean delete — the operator whose steering just stopped had no way to connect the two. A guard that silently does not run is worse than no guard, because the tool description promises it did. Now returns `guard_skipped` + a warning. | warning never set → **fail**; warning on EVERY delete (noise nobody reads) → **fail** |
| R3-04 | CI disabled by a workflow-file schema error (above). | CI green, both jobs pass |
| R3-05 | The prose warning about checking CI had already failed once. Made a test. | the exact line that broke CI → **fail**; `env` in a job `if:` → **fail**; invalid YAML → **fail** |
| R3-06 | A contract row whose tool name is **not in `cells[0]`** was silently invisible — dropped before the duplicate check, the phantom check and `_claims_mcp`. Demonstrated: a row with one leading empty cell advertising `millm_hub_import_circuit` left the parse at 16 rows, unchanged, tripwire unmoved. | hidden forbidden row → **fail** (was silent) |
| R3-07 | R2-21's forbidden-tool guard checked only the **registry**, never the table. So the contract could advertise hub-import as MCP-registered — the EC-20.5 wrong-feature-basis hazard — and an agent trusting the document would try to call it. The over-claim direction is what this file exists to detect, and for the forbidden tools it checked one side. | (covered by R3-06's control, which now fails on the row) |
| R3-08 | The new manual page named tools **by hand** and drifted from the API immediately. Now checked against the live registry. | invented tool → **fail**; wrong group count → **fail**; wrong total → **fail** |
| R3-09 | **`millm_reconcile_circuit_claims` does not exist.** I invented it while writing R2-22's manual page; a reader would have asked an agent to call a nonexistent tool. Caught by R3-08's guard **on its first run**. | — |

### Manual page corrections (found by fact-checking against the code)

The page as first written carried seven wrong or overstated claims:

- **"5 tools"** for a group of six, and `millm_circuit_sensing_event` omitted entirely.
- **Contention "overridable via `allow_layer_overlap`"** — incomplete and actively
  misleading. `CIRCUIT_ALLOW_CONCURRENT=false` refuses contention *regardless* of the
  override, and is checked FIRST. A reader following the page would tell an agent to
  retry with the override and be refused with no idea why.
- **Header suppression understated.** It is whole-response and global, not per-layer:
  one composed layer strips the rung from a response steered by an unrelated
  validated circuit. And it **fails open** — presented as an unconditional guarantee.
- **"Enforced by a copy audit that fails the build"** — true of runtime/UI copy, false
  of the manual, and it appeared *on a manual page* where a reader reads it as
  covering what they are looking at.
- **"Enforced by a test"** for the hub prohibition — was not true when written. R3-07
  made it true.
- **Omitted the rung-2 activation gate** (`UNVALIDATED_CIRCUIT`), which is the refusal
  an agent hits FIRST, and the fact that the acknowledgement does not persist across
  the intensity dial.
- **Omitted `slice_fallback`**, `steering: null` meaning NOT EVALUATED, and
  `requests_sensed == 0` as the wiring-fault signal.


## The pattern R3 exposed: every earlier fix was applied to ONE representative

Two independent adversarial passes, both mutation-driven, converged on the same
structural defect. It is the most important thing this round found after R3-04.

| # | Finding | Surviving mutation → now |
|---|---|---|
| R3-10 | **The original F20 defect was reproducible TODAY, one category over.** The built-server test hardcoded `millm_circuits`, so the reachability rule this file exists to enforce covered one of four categories. `if category == "millm_sensing": continue` in the registration loop left **119/119 green while an entire tool category was unreachable** — the exact shape of the defect that started this feature. Now parameterized over the live registry. | SURVIVED → **fails**, naming all five unreachable tools |
| R3-11 | R1-06 fixed the hand-rolled gate on `millm_import_circuit` and **not on its two destructive siblings**. Deleting the `gate.check` block from `millm_circuit_sensing_clear` — irreversible and global in scope — left the suite green while the tool issued its DELETE against a miLLM known to be down. | SURVIVED → **fails**, printing the DELETE that would have gone out |
| R3-12 | `TestGateDegradation` used `millm_circuit_status` as sole representative, so **eleven `@gated` tools could lose the decorator silently**, turning a structured `unavailable` into an unclassifiable connection error. | SURVIVED → **fails** |
| R3-13 | miLLM's copy audit whitelisted by **whole-line substring**, so a marker anywhere licensed a claim elsewhere: *"This circuit is causally validated; we never cut corners."* passed. miStudio fixed exactly this in R1-09/10 by requiring same-sentence denial; **miLLM never adopted it**. | passed → **caught** |
| R3-14 | `UNRELATED_SENSE` contained **`architecture`** — an ordinary word in circuit copy — so *"Circuit architecture: this edge is causally validated by observation."* was exempted by its own subject matter. Sentence-scoping could not fix it (word and claim share a sentence); the term had to go. | passed → **caught** |
| R3-15 | Both audits keyed on the token **"causal"**, so plain-English overclaims were invisible: *"Ablation proves this edge causes the refusal, a confirmed mechanism."* and *"Verified effect size measured on live traffic."* assert rung-2/3 without the guarded word. New `PROOF_CLAIM` check. | passed → **caught** |
| R3-16 | miStudio's `SURFACES` was **hand-maintained at five files** — in a file whose own comment says a hand-maintained list "is only as good as the list" and globs the MCP tools for that reason. Sixteen circuit modules were unaudited, including all three REST endpoint modules whose responses reach the UI. An overclaim planted in an endpoint left the suite **26/26 green**. Now discovered (5 → 17 surfaces). | passed → **caught** |

**The honesty guarantee was enforced at different strengths on either side of
the export path.** miStudio's audit received three review rounds; miLLM's never
did — and miLLM is the SERVING side, the one a promoted circuit actually
reaches. R3-13/14/15 close that asymmetry.

Widening R3-16's scope surfaced three hits that were **legitimate**: the ladder
file that DEFINES "causally validated", and the intervention/validation modules
that ARE the rung-2 machinery. Exempted **by file with a stated reason each**
rather than by loosening the pattern, because loosening it would weaken the
audit everywhere to accommodate three places. The exemption `skip`s rather than
silently passing, so it stays visible in the test output.

## A fourth instrument failure

Shell quoting mangled a four-probe control loop; the probes were never written
to disk and all four reported "PASSED — gap" against fixes that were working.
Re-run from a Python driver that **asserts the mutation applied before judging
it**, all four came back CAUGHT.

That assertion is now the rule: *verify the mutation landed before recording a
survivor.* Five instrument failures this increment, and four of them would have
produced a change to correct code.


## The client every tool passes through had no test file at all

| # | Finding | Control |
|---|---|---|
| R3-17 | **A 200 whose body would not parse returned `{}` — indistinguishable from a genuine empty success.** A misrouted ingress or proxy splash page reaches the agent as an empty successful result: `millm_circuit_status` reports nothing steering, and the agent activates a circuit that contends with one already serving that layer. The `JSONDecodeError` was the only evidence the response had not come from miLLM at all, and it was discarded. `raw_get` ALREADY guarded this (009 R2) and the guard was never carried across — the same one-representative pattern as R3-10/11/12/13. | guard removed → **fail** |
| R3-18 | `str(e)` on an httpx timeout is frequently EMPTY, so the message read *"miLLM request timed out: "* with no diagnostic. The PHASE is what matters: a connect timeout means miLLM was never reached; a **read** timeout on a POST means the request may ALREADY have committed, and an agent that retries double-imports or double-activates. | warning removed → **fail**; connect-vs-read specificity → **fail** |
| R3-19 | Writing R3-18's shared helper reintroduced a defect in the other caller: `raw_get` has **no `method` variable in scope**, so the first version would have raised `NameError` on EVERY export timeout — masking the timeout with an unrelated crash. Caught by writing the test before trusting the edit. | `method` restored → **fail** |

`test_millm_client_failure_paths.py` is new: **there was no test file for this
client**, and it is the single component all 16 circuit tools pass through.

Also disproven by executed probe, and recorded so they are not re-investigated:
`raw_get` is the STRONGER path, not the weaker one (an HTML error page cannot be
returned as a valid exported circuit); timeouts exist and fire (60s default, no
hang-forever path); and there are **no retries anywhere**, so the POST/DELETE
replay hazard I hypothesised does not exist.


## Instrument failures (continued from R2)

R2 recorded three. R3 adds two — the `sed` misread below, and the shell-quoting failure above: a `sed` extract that collapsed a blank line, which
I read as a missing blank line and nearly "fixed". Withdrawn as R3-02.

The pattern across both rounds is consistent enough to name: **when a tool reports
something surprising about my own code, verify the tool before acting on the
report.** Four of the five instrument failures this increment would have produced a
change to correct code.
