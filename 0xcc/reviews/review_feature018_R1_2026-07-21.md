# Feature 018 — Review Round 1 (2026-07-21)

**Suite:** 1873 → **1904** green / 1 skipped. CI green.
**20 findings, 20 fixed.** Every fix negative-controlled.

## The critical one was mine, and it reproduced the bug it replaced

**R1-05.** `for_registry` — written to eliminate the `SAEService.__new__`
bypass's missing-attribute hazard — set `_repository`, `_emitter` and
`_cache_dir`. **None of those exist.** `__init__` sets `self.repository` and
`self.emitter` WITHOUT the underscore, and `self.repository` is read throughout
the class. So the two most-used fields were left ABSENT:

    >>> SAEService.for_registry().repository
    AttributeError: 'SAEService' object has no attribute 'repository'

Under a docstring asserting "every field `__init__` sets is set here" and "fails
on a value rather than on a missing attribute". Both claims false.

**R1-06.** And the totality test could not catch it: its regex was
`self\.(_[a-z_]+)`, requiring a leading underscore, so the two real names were
invisible. It passed against the broken version AND against the fix.

## Findings

| # | Finding |
|---|---|
| 01 | `serving_intensity` diverged from the pre-move expression on a null budget — recorded, not silent |
| 02/03 | a registry failure silently flipped a serving circuit to unserveable; the log test used `caplog`, which this structlog codebase never reaches |
| 04 | `for_registry`'s None fields are safe only while the dial ignores them — now asserted by tracing what the dial reaches |
| **05/06** | **the AttributeError hazard reproduced, and a guard blind to two thirds of the fields** |
| 07 | a comment claimed `bound_layers` could not drift; the code below it was unchanged and correct |
| 08 | the dial read the registry TWICE — a detach between them desynchronises snapshot from restore |
| 09 | `for_registry()` inside the try reported construction faults as apply failures |
| 10 | a dead stub patched a deleted method; 34 tests silently ran the real flattener |
| 11 | the characterization shim aliased the engine, making the parity claim unfalsifiable |
| 12/13 | a missing intensity basis returned 0.0 — "serve nothing" — indistinguishable from an authored off; negatives passed through |
| 14 | `_serve_full` reached through another service's private attribute |
| 15 | `unattached_layers` documented as the slice-fallback signal, with zero callers and a better signal already in use |
| 16 | **R1-08's fix orphaned `attached_layers()`** — found by mutation |
| 17–20 | echo parity across five shapes, the registry-raise delta, the sign rule end to end (FPRD crit. 5), claim-set identity on every fixture (crit. 3) |

## What this round says about the method

**Four of my own tests could not fail** — a `caplog` capture this codebase never
reaches, an underscore-only regex, an aliased shim, and an index comparison
against a string that also appears in a comment. Each was caught by the negative
control and by nothing else.

**Two of my fixes introduced the next finding.** R1-08 orphaned a method (R1-16);
my first attempt at R1-12 raised where a members-only derivation is legitimate
and broke ten tests.

**A "verbatim move" was verbatim** — the flattener's executable body diffed
24 lines against the pre-move original with zero differences, and 12 mutations
found zero survivors. That is the one part of F18 that behaved as advertised,
and it is the part I checked hardest before shipping.
