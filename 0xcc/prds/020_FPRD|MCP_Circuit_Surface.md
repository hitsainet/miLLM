# Feature PRD: MCP Circuit Surface & Reachability Assurance

## miLLM Feature 20

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `BRD-MILLM-CIRCUITS-002.md` · `000_PPRD|miLLM.md` (v1.3, FR-20.x) · `000_PADR|miLLM.md` (v1.3) · `docs/mcp-contract.md` (v1.1 → v1.2)

---

## 1. Feature Overview

### Feature Name
MCP Circuit Surface & Reachability Assurance — give agents the circuit runtime, and make
shipped-but-unreachable capability impossible to ship undetected.

### Brief Description
Two halves, deliberately paired. **(a) Agent reach:** a `millm_circuits` category on the existing
unified miStudio-hosted MCP server, exposing every circuit capability REST already exposes — list,
status, import, activate, deactivate, export, set intensity, plus edge-sensing
status/events/event-detail/enable/disable/clear — with every circuit- and edge-bearing response
carrying `rung` and the server-rendered `rung_language` VERBATIM, and the build-failing copy audit
extended to cover the MCP modules AND their tool descriptions. **(b) Reachability assurance:** an
acceptance rule with teeth — no capability is accepted as shipped without a test that FAILS when its
user- or agent-facing wiring is removed, plus documentation status marks that distinguish "endpoint
exists" from "reachable by a user or agent".

### Problem Statement
The agentic half of the ecosystem stops at clusters. `BRD-MILLM-CLUSTERS-001` delivered a unified MCP
surface so an agent could move a tuned cluster into production and watch it fire. The circuits arc
delivered strictly *more* capability — multi-layer interventions, an evidence ladder, edge observation
— and **none of it is agent-reachable**. Verified live:
`backend/src/mcp_server/tools/__init__.py:40-44` registers exactly `millm_runtime`, `millm_clusters`,
`millm_sensing`; there is no `millm_circuits` module in the package import at `:8-22`, and
`millm_circuits` is absent from `VALID_CATEGORIES` (`backend/src/mcp_server/config.py:8-15`). An agent
can steer a single-layer cluster and read its co-activations, but cannot import a circuit, cannot
activate one, cannot read an edge observation, and cannot see a rung. The honesty guarantees the whole
increment was built around are invisible to the consumer most likely to over-claim on their behalf.

Worse, `docs/mcp-contract.md` **listed those tools as shipped for an entire increment**. Its §4
`millm_circuits` table carries twelve rows now marked `REST ✅ · MCP not registered` behind a STATUS
CORRECTION block added 2026-07-20. The table read as a shipped tool surface, and nothing in the
document's own vocabulary could express the difference between "the endpoint exists and is tested" and
"an agent can invoke it".

That is not an isolated documentation slip. The post-close-out capability audit found **three**
shipped-but-unreachable capabilities, **all three discovered by an operator trying to use the system
rather than by any of twelve review rounds**:

1. **Feature 12's multi-SAE attach** — service, REST route, API client and React hook all present and
   tested, and `AttachmentPanel.tsx` destructured only the read fields and rendered **zero buttons**.
   No user could ever attach the second SAE that a cross-layer circuit requires.
2. **Eleven circuit MCP tools** documented as shipped while `MILLM_CATEGORY_MODULES` never registered
   them.
3. **Edge-sensing ring pruning** declared "request-level" in two consecutive review rounds and wired in
   NEITHER — the second time accompanied by a test named **`TestRingPruningIsWired`** that asserted an
   entry point EXISTED while nothing called it.

The common shape: reviews verified the *mechanism* and never asked whether anything *called* it. The
third case is the precise anti-pattern this feature's rule must exclude — which is why the rule is
worded "the test must FAIL when the wiring is cut", never "a test must exist". A test named for the
defect it fails to prevent is worse than no test, because it converts an open question into a closed
one.

### Feature Goals
1. A `millm_circuits` MCP category exposing every REST circuit capability, health-gated and degrading
   structurally (BR-004).
2. `rung` + `rung_language` verbatim on every circuit- and edge-bearing tool response; copy audit
   extended to the MCP modules and their tool DESCRIPTIONS (BR-004, RSK-003).
3. A reachability acceptance rule that fails when wiring is cut, and its enforcement across this
   increment's capabilities (BR-005).
4. `docs/mcp-contract.md` → v1.2, additive-only, with reachability-aware status marks (BR-005).

### User Value Proposition
"My agent exported a circuit from miStudio, imported it into miLLM, activated it with the
acknowledgement its rung-1 evidence requires, dialled it, armed edge sensing and read back three edge
observations — and every single answer said 'suggested (attribution-supported)', never 'causal',
because it never got to choose the words."

### Connection to Project Objectives
Completes the BRD's second business objective ("extend agent reach from clusters to circuits,
completing the ecosystem promise of BRD-MILLM-CLUSTERS-001 for the strictly richer artifact") and its
third ("make unreachable capability impossible to ship undetected").

### BRD Traceability

| BR | Requirement (abbrev.) | Coverage IDs |
|----|-----------------------|--------------|
| BR-004 | Agent SHALL do for circuits what it can for clusters, same unified server, rung + rung_language verbatim | MCP-T1..T5, MCP-E1..E4 |
| BR-005 | No capability accepted without a test proving a user/agent-facing caller invokes it; status marks distinguish exists vs reachable | RCH-1..RCH-6, DOC-1..DOC-3 |

---

## 2. User Stories & Scenarios

#### US-20.1: Move a circuit into production from an agent
**As an** agent with the unified MCP server
**I want to** import, activate and dial a circuit exactly as I already do for a cluster
**So that** the richer multi-layer artifact is reachable without dropping to raw HTTP.

**Acceptance Criteria:**
- [ ] `millm_import_circuit` accepts an inline `mistudio.circuit-definition/v1` document; import does
      NOT activate (the evidence gate stays an explicit separate step)
- [ ] `millm_activate_circuit` carries `acknowledge_unvalidated`; without it a rung<2 circuit is
      refused with `UNVALIDATED_CIRCUIT` in the envelope (200 + `success:false`), and the agent can
      read the rung from the refusal and re-send
- [ ] `millm_set_circuit_intensity` dials the active circuit; `millm_circuit_status` reports the
      active circuit with `serving_mode` and the `steering` verdict field
- [ ] Every response is the raw miLLM envelope, unwrapped by the client, never re-shaped by the tool

#### US-20.2: Read edge observations as an agent
**As an** agent investigating a served circuit
**I want to** arm edge sensing and read observations with their evidence language
**So that** I can report what fired without inventing a validation claim.

**Acceptance Criteria:**
- [ ] `millm_circuit_sensing_enable` / `_disable` / `_status` / `_events` / `_event` / `_clear` all
      present and health-gated
- [ ] Event rows carry `edge_rung` + `edge_rung_language` as stored AT THE MOMENT OF OBSERVATION
- [ ] `_status` surfaces `unsensable_edges` so an empty event list is never presented as absence of
      firing
- [ ] The tool DESCRIPTION states that an observation is not validation and never raises a rung

#### US-20.3: A capability cannot ship unreachable
**As a** maintainer
**I want** an acceptance rule that fails the build when a control or tool is disconnected
**So that** the next unreachable capability is caught by CI rather than by an operator.

**Acceptance Criteria:**
- [ ] Each capability in this increment has a reachability test that FAILS when its wiring is cut
- [ ] The rule explicitly excludes existence-only assertions, citing `TestRingPruningIsWired` by name
- [ ] A registry test asserts `millm_circuits` is in `MILLM_CATEGORY_MODULES` AND that each documented
      tool is actually registered on a BUILT server instance
- [ ] `docs/mcp-contract.md` v1.2 marks distinguish "endpoint exists" from "reachable"

#### US-20.4: Evidence language survives the agent surface
**As a** researcher whose circuit is rung 1
**I want** no tool, field or description to call it causal
**So that** an agent summarising my circuit cannot over-claim on my behalf.

**Acceptance Criteria:**
- [ ] The build-failing copy audit covers the `millm_circuits` MCP module and its tool descriptions
- [ ] A negative control proves a rung-0 circuit cannot be described as causal through any tool
- [ ] No tool composes its own evidence sentence; `rung_language` is passed through untouched

#### Edge Cases
**EC-20.1: miLLM unreachable** — **Trigger:** the health gate marks miLLM unavailable. **Behavior:**
every circuit tool returns `{"unavailable": "millm", "reason": …}`; tools are NEVER unregistered (MCP
clients cache tool lists) — identical to the shipped behavior at `health_gate.py:159-181`.
**EC-20.2: `MILLM_API_URL` unset** — **Behavior:** the `millm_circuits` category is skipped at
REGISTRATION with a single warning, exactly as `server.py:118-132` already does for the other three;
miStudio-only deployments are unaffected.
**EC-20.3: No active circuit** — **Trigger:** `millm_set_circuit_intensity` or a sensing call with
nothing serving. **Behavior:** `NO_ACTIVE_CIRCUIT` as a 200 + `success:false` envelope (house style);
the tool surfaces it verbatim and does NOT convert it to an exception.
**EC-20.4: Circuit serving in slice-fallback** — **Trigger:** not all referenced SAEs attached.
**Behavior:** `millm_circuit_status` reports `serving_mode: "slice_fallback"` and the bound layers; the
tool description states that a slice is never the whole circuit and that the dial then follows CLUSTER
rules (0.5 floor), not circuit rules. **A rung is NOT reported as the circuit's own evidence for a
slice serve** — see EC-20.7.
**EC-20.5: Hub rows not served** — **Trigger:** `millm_import_circuit(repo_id=…)` or a circuit hub
search. **Behavior:** those endpoints 404 today (contract §4 marks them "F15 — not served"); v1
therefore does NOT register hub tools. Registering a tool against an unserved endpoint would
manufacture a fourth unreachable capability inside the feature that exists to abolish them.
**EC-20.6: `since` without a UTC offset** — **Behavior:** rejected client-side in the tool with an
explanatory message, exactly as `millm_sensing.py:42-52` — a naive timestamp shifts the polling window
silently.
**EC-20.7: A rung requested for a composed or slice serve** — **Trigger:** an agent asks for "the rung"
while two circuits serve concurrently (Feature 19) or a slice-fallback is in effect. **Behavior:** no
single circuit's evidence describes a composed response, so the tool reports the per-circuit rungs and
an explicit `composed` / `serving_mode` note rather than synthesising one number. This mirrors the
runtime's own rule that `X-miLLM-Circuit-Rung` is OMITTED in exactly these cases, and that its absence
never means rung 0.

---

## 3. Functional Requirements

### Tool Surface (FR-20.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| MCP-T1 | A `millm_circuits` module registered in `MILLM_CATEGORY_MODULES` and added to `VALID_CATEGORIES`; NOT in `DEFAULT_CATEGORIES` (opt-in, mirroring the other three miLLM categories) | Must |
| MCP-T2 | Circuit lifecycle tools: `millm_circuit_status`, `millm_list_circuits`, `millm_import_circuit` (inline only), `millm_activate_circuit`, `millm_deactivate_circuit`, `millm_export_circuit`, `millm_set_circuit_intensity` | Must |
| MCP-T3 | Edge-sensing tools: `millm_circuit_sensing_status`, `_events`, `_event`, `_enable`, `_disable`, `_clear` — against the shipped `/api/circuit-sensing/*` prefix, NOT the `/api/circuits/…/sensing` paths the contract originally reserved | Must |
| MCP-T4 | All tools health-gated via `@gated(gate, "millm")`; structured `{"unavailable": "millm", "reason": …}`; never unregistered (EC-20.1) | Must |
| MCP-T5 | Tools pass the miLLM envelope through unmodified; argument pre-validation only (mutually-exclusive sources, ISO-8601 offsets, enum members) — never response re-shaping | Must |

### Evidence Integrity (FR-20.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| MCP-E1 | Every circuit- and edge-bearing response carries `rung` and `rung_language` verbatim as the server rendered them; no tool composes, paraphrases, summarises or re-derives evidence language | Must |
| MCP-E2 | The build-failing copy audit extends to the MCP modules AND their tool descriptions — a description is user-visible text and can overclaim exactly like UI copy | Must |
| MCP-E3 | A negative control asserts a rung-0 circuit cannot be described as causal through any tool, argument, description or response path | Must |
| MCP-E4 | Tool descriptions state the three semantics an agent can otherwise get wrong: an observation is not validation and never raises a rung; absence of rows is not absence of firing (`unsensable_edges`); a stored `edge_rung_language` is as-of-observation and is never re-rendered from today's rung | Must |

### Reachability Assurance (FR-20.3, FR-20.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| RCH-1 | The reachability rule is normative and worded as: *a capability is accepted only when a test FAILS if its user- or agent-facing wiring is removed*. A test asserting an entry point merely EXISTS does not satisfy it | Must |
| RCH-2 | The rule's wording cites `TestRingPruningIsWired` by name as the excluded anti-pattern | Must |
| RCH-3 | Registry reachability: a test asserts `millm_circuits` ∈ `MILLM_CATEGORY_MODULES` and ∈ `VALID_CATEGORIES`, and — separately — that every tool named in contract §4 is present on a BUILT server instance's tool manager. Registry membership alone is not sufficient evidence of registration | Must |
| RCH-4 | Caller reachability: for each tool, a test asserts the tool invokes the documented endpoint (method + path), failing if the call is removed or re-pointed | Must |
| RCH-5 | The audit's three findings each gain a regression test that fails when its wiring is cut: the attach-set control's buttons, the circuit category registration, and ring pruning's actual caller | Must |
| RCH-6 | A meta-test over this increment's capability list asserts each has a registered reachability test — the rule enforced mechanically, not by reviewer memory | Should |

### Documentation (FR-20.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| DOC-1 | `docs/mcp-contract.md` → v1.2, strictly additive; the STATUS CORRECTION block is RESOLVED rather than deleted (it becomes the historical record of why the marks changed) | Must |
| DOC-2 | Status marks distinguish three states, not two: `REST ✅ · MCP ✅` (reachable), `REST ✅ · MCP not registered` (exists, unreachable), `not served` (neither). Every row states which | Must |
| DOC-3 | §7's circuit agent flow updated to a runnable tool sequence; §6's deployment wiring gains `millm_circuits` in the `MCP_TOOL_CATEGORIES` example; §4's sensing rows corrected to the shipped `/api/circuit-sensing/*` paths | Must |

---

## 4. Data Requirements

**No schema changes. No migration.** This feature adds no tables, columns or persisted state. Every
tool is a thin, health-gated pass-through over an endpoint that already ships and is already tested —
which is precisely why the BRD calls it "a registration and shaping exercise, not new backend work".

The only new persistent artifact is documentation state: `docs/mcp-contract.md` v1.2 and its
reachability marks. The only new *runtime* state is the registry entry itself, which is the capability
under test.

## 5. API Specifications

**No new miLLM endpoints.** F20 consumes the shipped surface verbatim. Verified live:

**`millm/api/routes/management/circuits.py`** — `router = APIRouter(prefix="/api/circuits")` (:38):
- `GET /api/circuits` (:69) — `min_rung`, `serveable`, `limit`, `offset`; rows carry rung/layers
- `GET /api/circuits/active` (:96) — active circuit + `serving_mode`, **plus the `steering` verdict
  field** (:101-129), added post-hoc after R3 found the OWUI filter deriving steering locally from
  `is_active`, which overclaims for a slice-fallback, unparseable or unattached circuit. Clients — and
  therefore the tool description — must read `steering`, never infer from `is_active`. Note the field
  is computed in a `try/except` that swallows failures to `None`: `null` means "not evaluated", NOT
  "not steering", and the description must say so.
- `POST /api/circuits/import` (:130) · `POST /api/circuits/{id}/activate?acknowledge_unvalidated=`
  (:200) · `POST /api/circuits/{id}/deactivate` (:242) · `PUT /api/circuits/active/intensity` (:255,
  returning `NO_ACTIVE_CIRCUIT` in-envelope at :266-270) · `DELETE /api/circuits/{id}` (:284) ·
  `GET /api/circuits/{id}/export` (:296 — raw document, deliberately NO `response_model` so unknown
  additive fields from newer producers survive; the tool must not re-model it either)

**`millm/api/routes/management/circuit_sensing.py`** — `router = APIRouter(prefix="/api/circuit-sensing")`
(:38), six routes: `GET /status` (:62), `GET /events` (:88), `GET /events/{event_id}` (:131),
`DELETE /events` (:154), `POST /{circuit_id}/enable` (:254), `POST /{circuit_id}/disable` (:265).

**Contract-vs-code discrepancy, resolved in v1.2:** the contract's §4 table reserved
`/api/circuits/…/sensing/*`; the code shipped the flat `/api/circuit-sensing/*` prefix. The contract
documents this in prose but its TABLE still reads the old way — and the table is what a tool author
reads. v1.2 corrects the rows (DOC-3).

## 6. UI Requirements
**None — this feature has no UI tab** (PPRD: "UI Tab: none (MCP + process)"). Its user-facing surfaces
are tool descriptions, `docs/mcp-contract.md`, and CI failure messages. The reachability rule's *scope*
does include UI wiring — RCH-5 pins the Feature 12 attach-set control — but F20 ships the test, not the
control (BR-009 owns the control itself).

## 7. Non-Functional Requirements
- Registration adds no measurable startup cost: 13 `@mcp.tool()` decorators alongside the existing
  ~55, and one module import.
- No hot-path impact whatsoever — miLLM's inference path is untouched.
- Health-gate polling is shared with the three existing miLLM categories (one `HealthGate` instance,
  `server.py:116`); adding a category adds zero probes.
- Tool descriptions are the primary documentation surface for agents and are budgeted accordingly:
  each states what the tool does, what it does NOT prove, and the one mistake an agent is most likely
  to make with it.

## 8. Dependencies
- **Feature 18 (single serving derivation) — SEQUENCING DEPENDENCY, LOCKED.** F20 is sequenced LAST
  deliberately (BRD locked decision 1, `execution_order` step 8): the MCP surface is written against
  settled code rather than against three serving derivations about to move, so nothing is written twice
  and a tool bug is never confusable with a refactor bug.
- **Feature 13** (circuit rows, import/activate/export/intensity routes) — the endpoints being exposed.
- **Feature 15** (edge-sensing routes, `millm/core/circuit_evidence.py::rung_language`, and the
  build-failing copy audit at `tests/unit/core/test_circuit_evidence.py` being extended).
- **Feature 19** (concurrent serving) — informs EC-20.7: with two circuits serving, a single rung is no
  longer a meaningful answer, and the tools must not invent one.
- **miStudio repo** — hosts the MCP server. F20's code lands in `backend/src/mcp_server/` there; the
  contract and this doc chain live in miLLM. This cross-repo split is load-bearing for §10 and §13.

## 9. Success Criteria
1. `millm_circuits` registered; `mcp-contract.md` v1.2 carries **zero** rows marked
   `REST ✅ · MCP not registered` (baseline: 12).
2. Every circuit capability exposed by REST is invocable via MCP (BRD success metric), verified by a
   test that enumerates the contract table and asserts a registered tool per served row.
3. Every circuit- and edge-bearing response carries `rung` + `rung_language` verbatim; the copy audit
   covers the MCP module and its descriptions and FAILS on a planted violation in either.
4. The negative control proves a rung-0 circuit cannot be described as causal through any tool path.
5. Each capability in this increment has a reachability test that FAILS when its wiring is cut,
   demonstrated by actually cutting it (mutation, not assertion-by-reading).
6. Zero shipped capabilities without a reachability test (BRD baseline: 3).

## 10. Testing Requirements
- **Unit (miStudio):** per-tool endpoint/method assertions against a mocked `MiLLMClient`; health-gate
  degradation returning the structured unavailable shape; argument pre-validation (mutually-exclusive
  import sources, `since` offset rejection, `on_conflict` enum); envelope pass-through (a
  `success:false` body is RETURNED, not raised); registry membership and built-server tool presence.
- **Copy audit (cross-repo — the hard part):** miLLM's audit at
  `tests/unit/core/test_circuit_evidence.py:129-153` scans `REPO / root` over
  `["millm", "admin-ui/src"]`, where `REPO = Path(__file__).resolve().parents[3]` (:26) — **it cannot
  reach the miStudio repo at all**. miStudio has its own separate audit
  (`backend/tests/unit/test_causal_language_audit.py`) with different regexes and a different
  allow-list. Extending "the copy audit to the MCP modules" is therefore a decision, not a path append
  — see §13 OQ-1. Whichever resolution is chosen, the audit must strip comments before matching
  (miLLM's `_code_only`, :157-166) so a marker in a comment cannot exempt a claim in a description
  string, and must fail on a planted violation in a **tool description** specifically.
- **Reachability (both repos):** for each capability, a mutation test — remove/disconnect the wiring,
  assert the test goes red. Explicitly includes re-pointing a tool's endpoint path, deleting the
  category from `MILLM_CATEGORY_MODULES`, and stripping the attach-set buttons.
- **Integration:** an end-to-end agent flow against a live-ish miLLM — import → activate (with and
  without the acknowledgement) → set intensity → sensing enable → events → disable — asserting rung
  language verbatim at every hop.

## 11. Rollout & Migration
No migration. Opt-in by configuration: `millm_circuits` joins `VALID_CATEGORIES` but NOT
`DEFAULT_CATEGORIES`, so no existing deployment gains tools without an explicit `MCP_TOOL_CATEGORIES`
change. With `MILLM_API_URL` unset the category is skipped at registration with one warning (EC-20.2).
Contract v1.2 is additive-only; a v1.1 client is unaffected.

## 12. Out of Scope
Hub import / hub search tools (EC-20.5 — those endpoints 404 today). A new MCP server or repo
(miStudio remains the owner — BRD assumption). Authentication changes to miLLM's management API
(contract §6 posture unchanged). Any new interpretability capability. Re-litigating the evidence ladder
vocabulary or the rung<2 acknowledgement gate. The attach-set control itself and the compatibility
pre-filter (BR-009/BR-010 own those; F20 owns only their reachability tests).

## 13. Open Questions

**OQ-1 (blocking implementation, not this document): where does the extended copy audit live?**
The two repos have two independent audits, and F20's subject code sits in the repo whose audit is the
weaker of the two. Three options:
 **(a)** add a miStudio-side audit over `backend/src/mcp_server/tools/millm_*.py` mirroring miLLM's
 stricter `EVIDENCE_CONTEXT` + `_code_only` logic — self-contained, but the rule now exists twice and
 can drift;
 **(b)** extract the audit into a shared, vendored checker — no drift, but creates a cross-repo build
 dependency where none exists today;
 **(c)** have miLLM's audit fetch registered tool descriptions from a running MCP server —
 tests-against-a-live-service, rejected as fragile.
**Recommendation: (a)**, plus a test in EACH repo asserting the other's rule constants match, so drift
fails the build rather than silently widening an allow-list. Decide at FTDD time.

**OQ-2 (non-blocking): does `millm_export_circuit` return the raw document or the envelope?**
`GET /api/circuits/{id}/export` (:296) is deliberately envelope-free and deliberately un-modelled. The
cluster precedent (`millm_export_cluster`, contract §4) already passes the raw document through.
Resolution: mirror it, and say so in the description — an agent that unwraps `.data` on this one tool
gets `None`.

## 14. Documentation Requirements
`docs/mcp-contract.md` v1.2 (marks, corrected sensing paths, §6 wiring, §7 flow). The reachability rule
recorded as a standing project rule, not a feature note — it outlives this feature. A short section in
the manual's MCP page listing the circuit tools and the one semantic an agent must not get wrong
(observation ≠ validation).

## 15. Decisions from Clarifying Questions
1. **Refactor first, then agent reach** (BRD locked decision 1) — F20 is sequenced last so tools are
   written against settled code. Non-negotiable; re-sequencing invalidates the rationale.
2. **The rule is "the test must FAIL when the wiring is cut"**, never "a test must exist" (BRD RSK-004;
   PADR "Reachability as an acceptance gate vs relying on review"). `TestRingPruningIsWired` is cited
   by name in the rule's wording as the excluded anti-pattern.
3. **Tool descriptions are audited copy.** A description is user-visible text and can overclaim exactly
   like UI copy; it is in scope for the build-failing audit.
4. **Additive-only, opt-in.** Contract v1.2 breaks nothing; the category is not enabled by default.
5. **No hub tools in v1** — the endpoints are not served (EC-20.5).
6. **No rung for a composed or slice serve** (EC-20.7) — the tools mirror the runtime's own rule that
   the rung header is omitted rather than guessed, and its absence never means rung 0.
