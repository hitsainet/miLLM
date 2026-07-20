# Technical Design Document: MCP Circuit Surface & Reachability Assurance

## miLLM Feature 20

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `020_FPRD|MCP_Circuit_Surface.md` · `009_FTDD|Unified_MCP.md` · `015_FTDD|Circuit_Edge_Sensing.md` · `docs/mcp-contract.md` (v1.1 → v1.2)

---

## 1. Executive Summary

F20 is a **registration and shaping exercise plus a test-discipline change**, and the two halves have
very different risk profiles. The tool surface is genuinely low-risk: every endpoint it exposes already
ships, is already tested, and already renders its own evidence language; the new `millm_circuits`
module is a fourth sibling to `millm_runtime` / `millm_clusters` / `millm_sensing`, following their
established shape exactly — `register(mcp, millm, gate)`, `@mcp.tool()` + `@gated(gate, "millm")`,
thin pass-through of the miLLM envelope, argument pre-validation only. The reachability half is where
the design work is, because "a test that fails when the wiring is cut" is a property of a test, not a
thing a test can assert about itself, and the obvious implementations degenerate into exactly the
box-ticking the rule exists to prevent.

The single most important design constraint is one the FPRD surfaces and the BRD does not: **the
code lands in the miStudio repo while the evidence-honesty audit that must govern it lives in miLLM,
and that audit is structurally incapable of reading across the repo boundary.** Everything else here
follows the shipped patterns.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Module shape | Fourth sibling `millm_circuits.py`; `register(mcp, millm, gate)` | Three working precedents; server.py wiring is already generic over the registry |
| Registration | Add to `MILLM_CATEGORY_MODULES` + `VALID_CATEGORIES`, NOT `DEFAULT_CATEGORIES` | Opt-in like every miLLM category; no deployment silently gains tools |
| Response handling | Pass the envelope through unchanged; validate ARGUMENTS only | Contract §2: unwrap in the client, never in tools. Re-shaping is how paraphrase enters |
| Evidence language | Tools never compose an evidence sentence; `rung_language` is transported, not rendered | The only structurally safe rule — a tool that never writes the words cannot overclaim |
| Copy audit scope | Mirror miLLM's audit into miStudio over the MCP tool modules; cross-repo constant-parity test | miLLM's audit cannot see miStudio (verified); a live-server audit is fragile |
| Description auditing | Audit `__doc__` of registered tools, not just source lines | A docstring IS the description the agent sees; source-line scanning alone misses composed ones |
| Hub tools | Not registered in v1 | Endpoints 404; registering them recreates the defect class being abolished |
| Reachability | Per-capability caller assertion + a build-server registration test + documented mutation proof | Registry membership and tool existence are each individually insufficient |
| Anti-pattern exclusion | Rule names `TestRingPruningIsWired`; meta-test forbids existence-only shapes | The rule must be self-defending or it decays |
| Contract | v1.2, additive; three-state marks; sensing paths corrected | The table is what tool authors read |

## 2. System Architecture

```
  miStudio repo (backend/src/mcp_server/)              miLLM repo (millm/)
 ┌──────────────────────────────────────┐            ┌────────────────────────────┐
 │ config.py  VALID_CATEGORIES          │            │ api/routes/management/     │
 │   + "millm_circuits"                 │            │   circuits.py  (:38 /api/  │
 │ tools/__init__.py                    │            │      circuits)             │
 │   MILLM_CATEGORY_MODULES             │            │   circuit_sensing.py (:38  │
 │   + "millm_circuits": [millm_circuits]│           │      /api/circuit-sensing) │
 │ tools/millm_circuits.py  (NEW)       │  HTTP      │                            │
 │   register(mcp, millm, gate)         │ ─────────► │ core/circuit_evidence.py   │
 │   13 @mcp.tool() @gated              │  envelope  │   rung_language() ← the    │
 │                                      │ ◄───────── │   ONLY renderer            │
 │ server.py :118-132 (unchanged —      │            │                            │
 │   generic over the registry)         │            │ tests/unit/core/           │
 │                                      │            │   test_circuit_evidence.py │
 │ tests/unit/                          │            │   TestCopyAudit (miLLM     │
 │   test_mcp_millm_circuits.py  (NEW)  │            │   tree ONLY — cannot see    │
 │   test_causal_language_audit.py (MOD)│◄ parity ──►│   miStudio)                │
 │   test_reachability.py        (NEW)  │  test      │                            │
 └──────────────────────────────────────┘            └────────────────────────────┘
                                                       docs/mcp-contract.md → v1.2
```

Note what does NOT change: `server.py` needs no edit. Its miLLM registration loop
(`:118-132`) iterates `MILLM_CATEGORY_MODULES` and calls `module.register(mcp, millm_client, gate)`
for any requested category — adding a registry entry is sufficient. That genericity is why the
original omission was so easy: **nothing in the wiring notices a missing category, it simply has one
fewer to loop over.** The reachability test exists precisely because absence here is silent.

## 3. Tool Module Design (`backend/src/mcp_server/tools/millm_circuits.py`)

Mirrors `millm_sensing.py` and `millm_clusters.py` exactly in structure. Thirteen tools, three groups.

```python
"""
miLLM circuit tools (category: millm_circuits) — Feature 20, Circuit Consolidation.

The cross-product loop this enables: miStudio export circuit →
millm_import_circuit → millm_activate_circuit(acknowledge_unvalidated=…) →
millm_set_circuit_intensity / millm_circuit_sensing_enable.

EVIDENCE RULE: every circuit and edge field carries `rung` and `rung_language`,
rendered by miLLM. Tools transport those values; they never compose, summarise
or paraphrase an evidence claim. See docs/mcp-contract.md §4a.
"""

def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    # --- lifecycle -------------------------------------------------------
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_status() -> Any:
        """The circuit currently serving, or null. …"""
        return await millm.get("/api/circuits/active")
    ...
```

### 3.1 Tool inventory

| Tool | Method + path | Notes |
|---|---|---|
| `millm_circuit_status` | `GET /api/circuits/active` | `steering` verdict field; `null` = not evaluated |
| `millm_list_circuits` | `GET /api/circuits` | `min_rung`, `serveable`, `limit`, `offset` |
| `millm_import_circuit` | `POST /api/circuits/import?on_conflict=` | inline document only; does NOT activate |
| `millm_activate_circuit` | `POST /api/circuits/{id}/activate?acknowledge_unvalidated=` | rung gate |
| `millm_deactivate_circuit` | `POST /api/circuits/{id}/deactivate` | |
| `millm_delete_circuit` | `DELETE /api/circuits/{id}` | deactivates first, server-side |
| `millm_export_circuit` | `GET /api/circuits/{id}/export` | RAW document, no envelope (OQ-2) |
| `millm_set_circuit_intensity` | `PUT /api/circuits/active/intensity` | `NO_ACTIVE_CIRCUIT` in-envelope |
| `millm_circuit_sensing_status` | `GET /api/circuit-sensing/status` | `unsensable_edges`, `paused_reason` |
| `millm_circuit_sensing_events` | `GET /api/circuit-sensing/events` | `circuit_id`, `edge_key`, `limit`, `since` |
| `millm_circuit_sensing_event` | `GET /api/circuit-sensing/events/{id}` | detail + context window |
| `millm_circuit_sensing_enable` / `_disable` | `POST /api/circuit-sensing/{id}/enable|disable` | intent + live arm |
| `millm_circuit_sensing_clear` | `DELETE /api/circuit-sensing/events?circuit_id=` | |

Thirteen tools against twelve documented rows: the contract's twelve rows collapse
enable/disable into one and omit `delete`, which the REST surface has shipped since F13 (:284).

### 3.2 Argument pre-validation (the only client-side logic permitted)

Mirrors the shipped precedents precisely — presence checks, not truthiness (`millm_clusters.py:44`),
so an empty-dict definition still reaches miLLM's validator for a real contract error:

```python
if (definition is not None) == (repo_id is not None):   # exactly one source
if on_conflict not in (None, "rename", "fail"):
if since is not None: <ISO-8601 + tz-aware check>       # millm_sensing.py:42-52
```

`since` naive-timestamp rejection is copied verbatim in behavior: a naive timestamp shifts the polling
window silently, which for an evidence surface means silently under-reporting observations.

### 3.3 What the tools deliberately do NOT do

- **No response transformation.** Not even convenience flattening. The moment a tool builds a
  `summary` field, it is authoring evidence copy, and RSK-003 is realised. The contract's own §2 rule
  ("unwrap in the client, never in tools") is the design constraint.
- **No local rung derivation.** The circuit rung is MIN over edges (contract §4a), and the server
  computes it. A tool that re-derives it can disagree with the server, and the disagreement will
  surface as an overclaim exactly when edges are heterogeneous.
- **No `is_active` → steering inference.** `GET /api/circuits/active` carries the `steering` field for
  this reason (circuits.py:101-129, added after R3 caught the OWUI filter doing precisely this). The
  description states that `null` means "not evaluated", not "not steering".
- **No hub tools.** EC-20.5.

## 4. Evidence Integrity Design

Three layers, in order of strength.

**Layer 1 — structural (strongest).** Tools transport `rung`/`rung_language` and never render. There
is no code path in which a tool chooses evidence words, so there is nothing for an audit to catch. This
is why §3.3 forbids response shaping: it converts an honesty property into a structural one.

**Layer 2 — the copy audit, extended to descriptions.** Docstrings are the one place a tool author
DOES write prose, and they are user-visible text — an agent reads them to decide what a tool means.
Two scanning modes are needed, and only having both is sufficient:

- *source scan* over `backend/src/mcp_server/tools/millm_*.py`, mirroring miLLM's
  `TestCopyAudit._scan` (`tests/unit/core/test_circuit_evidence.py:129-153`) including the
  `_code_only` comment strip (:157-166) — an R3 finding: a marker in a trailing comment previously
  exempted a claim in a string literal.
- *registered-description scan* over `mcp._tool_manager.list_tools()`, reading each tool's actual
  description as the MCP client receives it. This catches a description composed at runtime, or
  inherited, which no source-line grep sees.

**Layer 3 — the negative control.** A rung-0 circuit fixture driven through every tool path, asserting
"causal" appears nowhere in any response, description or error. Mandated by MCP-E3 and by RSK-003's
mitigation.

### 4.1 The cross-repo problem (FPRD OQ-1) — design of record

miLLM's audit computes `REPO = Path(__file__).resolve().parents[3]` (:26) and scans
`["millm", "admin-ui/src"]` beneath it. **It cannot see the miStudio repo**, and no path append fixes
that — the repos are independent checkouts with independent CI. miStudio's own
`backend/tests/unit/test_causal_language_audit.py` exists but uses a different, looser regex pair
(`CAUSAL` / `ALLOWED_CONTEXT`) than miLLM's three-stage `\bcausal` → `UNRELATED_SENSE` →
`EVIDENCE_CONTEXT` filter.

**Decision: option (a) with a parity guard.** Implement the audit in miStudio over
`backend/src/mcp_server/tools/millm_*.py`, ported from miLLM's stricter logic (all three regex stages,
`_code_only`, the marker allow-list). Then add, in EACH repo, a test asserting the OTHER's rule
constants are byte-identical to its own — the constants are duplicated deliberately, and drift fails
a build rather than silently widening an allow-list.

The honest cost, recorded rather than hidden: the parity test can only compare what it can read, so it
requires both checkouts present. Where CI has only one, the test must SKIP LOUDLY (an explicit skip
with a reason naming the missing repo) rather than pass vacuously — a silently-passing parity test is
the same defect class as `TestRingPruningIsWired`, one level up.

## 5. Reachability Assurance Design

This is the part that can degenerate, so the design is explicit about what counts.

### 5.1 The rule, as it will be written

> **Reachability rule.** A capability is accepted as shipped only when an automated test FAILS if its
> user-facing or agent-facing wiring is removed. A test that asserts an entry point *exists* — that a
> function is defined, a route is declared, a component is exported, or a module is importable — does
> NOT satisfy this rule. The excluded anti-pattern has a name: Feature 15 shipped
> `TestRingPruningIsWired`, which asserted a pruning entry point existed while nothing called it, and
> the defect it was named for survived two review rounds behind it.

### 5.2 Three test shapes, none sufficient alone

| Shape | Asserts | What it misses alone |
|---|---|---|
| Registry membership | `"millm_circuits" in MILLM_CATEGORY_MODULES` | A module that registers zero tools |
| Built-server presence | tool name ∈ `build_server(...)` tool manager | A tool registered but calling nothing / the wrong path |
| Caller assertion | the tool invokes method+path X on a mock client | Category not enabled ⇒ tool never reaches a user |

All three are required. Their conjunction is the reachability evidence; each individually is an
existence assertion of the kind the rule excludes. The built-server test must construct a server via
the real `build_server()` with `millm_api_url` set and the category enabled — not hand-call
`register()`, which would bypass the exact gating that failed originally.

### 5.3 Mutation proof, and where it is recorded

Per FPRD §9 criterion 5, each reachability test's failure mode is **demonstrated by cutting the wiring
and observing red**, not asserted by reading. This follows F15 R3's most transferable lesson: 14
mutation experiments found four load-bearing lines that two rounds of careful reading had missed,
including one that let the WebSocket broadcast leak prompt text while the suite stayed green — a line
R1 had recorded as "privacy holds — verified clean".

The mutations for this feature, each expected to turn a named test red:
1. Delete `"millm_circuits"` from `MILLM_CATEGORY_MODULES` → registry + built-server tests fail
2. Delete `"millm_circuits"` from `VALID_CATEGORIES` → config validation test fails
3. Re-point one tool's path (`/api/circuits/active` → `/api/circuits`) → that tool's caller test fails
4. Strip `@gated` from one tool → degradation test fails
5. Plant `"causally validated"` in a rung-1 tool description → copy audit fails
6. Remove the attach-set control's buttons → attach reachability test fails (RCH-5)
7. Remove ring pruning's caller (leaving the method) → pruning reachability test fails (RCH-5)

Mutations 6 and 7 are the audit's own findings, re-armed. They are the feature's proof that the rule
would have caught what review did not.

### 5.4 The meta-test (RCH-6), and its honest limits

A meta-test enumerates this increment's capabilities and asserts each maps to a registered
reachability test. It is a *Should*, not a *Must*, and the reason is worth stating: a meta-test over a
hand-maintained list is itself only as good as the list, so it can drift into the box-ticking it
polices. It is worth having as a checklist, and it must not be mistaken for the assurance — the
assurance is §5.3's demonstrated mutations.

## 6. Admin UI Design
None. F20 ships no UI (FPRD §6). RCH-5's attach-set test asserts an existing control's wiring; the
control itself belongs to BR-009.

## 7. Testing Strategy

### Unit (miStudio — `backend/tests/unit/`)
- `test_mcp_millm_circuits.py` (NEW): all 13 tools — method + path + query/body shaping against a mock
  `MiLLMClient`; envelope pass-through incl. a `success:false` body returned rather than raised;
  health-gate degradation (`{"unavailable": "millm", …}`) per tool; argument pre-validation
  (mutually-exclusive import sources, `since` tz-awareness, `on_conflict` enum); `export` returns the
  RAW document unmodified; no tool mutates a response dict.
- `test_reachability.py` (NEW): registry membership, `VALID_CATEGORIES`, built-server tool presence
  over the contract's row list, per-tool caller assertion, and the attach-set + ring-pruning
  regression tests (RCH-5).
- `test_causal_language_audit.py` (MOD): source scan extended over `mcp_server/tools/millm_*.py` with
  miLLM's three-stage filter + `_code_only`; registered-description scan; rung-0 negative control;
  constant-parity guard vs miLLM (skipping loudly when the sibling checkout is absent).

### Unit (miLLM)
- `tests/unit/core/test_circuit_evidence.py` (MOD): the reciprocal constant-parity guard, same
  loud-skip semantics.
- A contract-consistency test asserting every §4 row marked reachable names a tool the miStudio
  registry claims — the doc and the registry cannot silently diverge again.

### Integration
- End-to-end agent flow against a live-ish miLLM: import → activate without acknowledgement (expect
  `UNVALIDATED_CIRCUIT` in-envelope) → activate with it → set intensity → sensing enable → events →
  disable, asserting `rung_language` verbatim at every hop and no "causal" for a rung-1 fixture.

### Mutation (documented, per §5.3)
The seven mutations, each recorded with the test it turned red.

## 8. Risks
- **The reachability rule degenerating into box-ticking (RSK-004).** The primary risk, and the one the
  feature is least able to test its way out of — §5.2's three-shape conjunction and §5.3's demonstrated
  mutations are the mitigations, and §5.4 admits the meta-test's limit rather than leaning on it.
- **A tool description paraphrasing a rung (RSK-003).** Mitigated structurally (§4 Layer 1: tools never
  render evidence language) and caught by the extended audit's description scan. The residual risk is a
  description that is *technically* rung-free but implies validation — e.g. "confirms the circuit's
  edges fire", where "confirms" does the overclaiming without the audited word. No grep catches this;
  it is an explicit review-round Watch item.
- **Cross-repo audit drift.** The parity guard fails the build on divergence, but only where both
  checkouts exist; the loud-skip is the honest degradation.
- **Contract v1.2 re-diverging from the registry.** The consistency test in §7 makes the document and
  the code fail together rather than drift apart — which is what happened for an entire increment.
- **A rung on a composed serve (EC-20.7).** With Feature 19 landed, "the rung" has no single answer.
  The tools must report per-circuit rungs and never synthesise; the runtime's own header-omission rule
  is the precedent, and a Watch item covers it.
- **Sequencing.** F20 is last by locked decision. If it is pulled forward, tools get written against
  three serving derivations mid-move, and a tool bug becomes indistinguishable from a refactor bug —
  the exact confusion the sequencing exists to prevent.
