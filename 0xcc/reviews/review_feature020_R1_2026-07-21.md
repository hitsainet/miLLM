# F20 MCP Circuit Surface & Reachability — Review Round 1

**20 findings, 20 fixed. miStudio 1380 → 1406, miLLM 2116 → 2121.**

Every finding was a defect in work built THIS SESSION, and most were the exact
failure mode F20 exists to prevent. That is the round's whole lesson: a feature
whose purpose is "prove the wiring works" is not exempt from its own rule.

## The contract said "not registered" on all 16 rows

**R1-02.** I updated the STATUS CORRECTION to say RESOLVED, defined a
three-state mark legend, and applied it to NOT ONE ROW. An agent reading
top-down is told the surface shipped, then told row by row that no tool exists
and it "must call REST directly" — so it bypasses 16 working tools.

Last increment those marks OVER-claimed. This time they UNDER-claimed. Same
column, same failure to check.

**R1-03 — my own guard could not see it.** It checked category HEADINGS against
the registry (the symptom already fixed) and never read the Status COLUMN,
where the defect lived both times. The reachability rule was applied rigorously
to the tool registry and not at all to the contract table — the artifact that
actually failed. Writing the corrected guard immediately found two more gaps.

## The harness verified the address, not the letter

**R1-04.** `EXPECTED_CALLS` recorded each call's payload and the assertion threw
it away. Three mutations passed 26/26 green — including changing the intensity
body key to `{"lambda": …}`, which 422s on EVERY production call.

**R1-06.** `millm_import_circuit` hand-rolls its gate check (so argument
validation runs first), and nothing tested that check. The one tool that opts
out of the decorator was the one tool whose gate was unverified.

**R1-07/08.** The recording client was MORE FORGIVING than the real client — an
optional `json_body` where the real one requires it, and an enveloped `raw_get`
where the real one returns a bare document. A stand-in that tolerates what
production rejects is the one thing a stand-in must never be.

## The copy audit certified the worst claim it could make

**R1-09.** `ALLOWED_CONTEXT` whitelisted TOPIC WORDS, so naming the topic
legitimised any claim about it. Six overclaims passed, verified by execution —
including **"Sensing performs a causal intervention on each edge"**, the single
most dangerous false statement this surface can make.

And I made it more permissive DURING implementation, adding
`causal\s+intervention` to clear a false positive. That is the loosening
pressure a false-positive-prone audit creates, and I walked into it.

**R1-11.** The audit was LINE-based, so wrapping at the right column hid an
overclaim — and Black wraps docstrings routinely. I patched the tokeniser twice
(rejoin lines; then break on code-looking lines) before recognising the UNIT was
wrong. Two patches should have been the signal after one. Now AST-extracted
prose: docstrings and comments, never code.

**R1-12.** Two of my own patterns licensed the claims they denied.
`n[o']?t?\s+causally\s+validated` degenerated to matching WHITESPACE — every
character after `n` optional — so it permitted "has BEEN causally validated". A
denial pattern that denied nothing.

## Guards that could not run, or could not fail

**R1-14.** The cross-repo guard skips unconditionally in CI: the workflow checks
out only miLLM, so `mcp` is never importable. A loud skip nobody reads IS the
vacuous green it was built to prevent. Now failable via
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1`.

**R1-15.** `test_the_status_correction_is_RESOLVED_not_deleted` greped two
strings that appear in prose nobody will remove.

## A destructive default, and guidance nobody would read

**R1-16.** `millm_circuit_sensing_clear` defaulted to GLOBAL scope — an agent
omitting `circuit_id` wiped every observation in the deployment. The
destructive path was the quiet one.

**R1-17..20.** The three cardinal semantics each reached exactly ONE tool, and
the module docstring stating them "because an agent reads those" is not
transported to `list_tools()` at all. An agent calls one tool and sees one third
of the picture.

## Round verdict

Every fix is pinned by a control that fails when reverted. The pattern that has
held all increment held hardest here: **the tests were the defect more often
than the code.** Of 20 findings, 11 were in the harness, the audit, or the
guards — the machinery built to prove correctness, which nothing was proving.
