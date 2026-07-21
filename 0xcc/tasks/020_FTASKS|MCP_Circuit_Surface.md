# Task List: MCP Circuit Surface & Reachability Assurance

## miLLM Feature 20

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** ✅ CLOSED 2026-07-21 — IMPLEMENTED + 3 review rounds (62 findings: R1 20, R2 22, R3 20; all fixed). Reachability is the durable deliverable: a capability is not shipped until a test FAILS when its wiring is removed. R3 found the ORIGINAL F20 defect reproducible one category over.
**References:** `020_FPRD|MCP_Circuit_Surface.md` · `020_FTDD|MCP_Circuit_Surface.md` · `020_FTID|MCP_Circuit_Surface.md` · `docs/mcp-contract.md` (v1.1 → v1.2)

## Relevant Files

**miStudio repo** (implementation — the MCP server lives here):
- `backend/src/mcp_server/tools/millm_circuits.py` — NEW, 13 tools; sibling of `millm_sensing.py`
- `backend/src/mcp_server/tools/__init__.py` — package import + `MILLM_CATEGORY_MODULES` entry
- `backend/src/mcp_server/config.py` — `VALID_CATEGORIES` (+1); `DEFAULT_CATEGORIES` unchanged
- `backend/src/mcp_server/server.py` — UNCHANGED (registration loop already generic over the registry)
- `backend/tests/unit/test_mcp_millm_circuits.py` — NEW, per-tool method/path/gate/validation
- `backend/tests/unit/test_reachability.py` — NEW, registry + built-server + caller + RCH-5 regressions
- `backend/tests/unit/test_causal_language_audit.py` — MOD, MCP module + description scan, parity guard

**miLLM repo** (contract + reciprocal guards):
- `docs/mcp-contract.md` — v1.2: three-state marks, corrected sensing paths, §6 wiring, §7 flow
- `tests/unit/core/test_circuit_evidence.py` — MOD, reciprocal constant-parity guard
- `tests/unit/test_mcp_contract_consistency.py` — NEW, doc rows vs registry claims

### Notes
- **Sequenced LAST by locked decision** (BRD `execution_order` step 8): refactor first, then agent
  reach, so tools are written against settled code rather than three serving derivations mid-move.
  Execute after Feature 18. Do not pull forward.
- **Cross-repo feature.** Implementation in miStudio, contract in miLLM. Both suites must be green;
  the parity guards are what keep them honest.
- No migration, no new endpoints, no runtime state, no UI. Every endpoint already ships and is tested.
- Test commands: `pytest` in miStudio `backend/`, `pytest` in miLLM, `npm test` in `admin-ui`.

### Category Checklist Results
- Data: n/a — no schema change, no migration (FPRD §4) ✓
- Backend/API: n/a in miLLM (no new endpoints); MCP tool surface 2.x ✓
- Frontend/UI: n/a — no UI tab; RCH-5 tests an EXISTING control (4.4) ✓
- Business logic: 2.x (tool shaping, argument pre-validation), 3.x (evidence integrity) ✓
- Integration wiring: 1.x (registration in three places + config), 5.2 (end-to-end agent flow) ✓
- Error handling & logging: 2.5 (envelope pass-through, in-envelope refusals, gate degradation) ✓
- Testing: paired throughout + 4.x reachability + 5.x integration/mutation ✓
- Performance & security: n/a hot path; auth posture unchanged (contract §6); opt-in category 1.3 ✓
- Config/deploy: 1.3 `VALID_CATEGORIES`; deployment wiring documented 6.1 — no migration ✓
- Documentation: 6.x contract v1.2 + manual + the standing reachability rule ✓

## Tasks

- [ ] 1.0 Registration & configuration (covers FR-20.1; MCP-T1)
  - [ ] 1.1 `tools/millm_circuits.py` module skeleton with `register(mcp, millm, gate)` matching the
        `millm_sensing.py` signature and module docstring conventions
  - [ ] 1.2 Register in BOTH places in `tools/__init__.py`: the package import (:8-22) and
        `MILLM_CATEGORY_MODULES` (:40-44) — a module imported but not registered fails silently
  - [ ] 1.3 `config.py`: add `"millm_circuits"` to `VALID_CATEGORIES`; confirm `DEFAULT_CATEGORIES`
        is NOT changed (no `millm_*` category is ever a default)
  - [ ] 1.4 Confirm `server.py` needs no edit; add a test pinning that the registration loop picks the
        category up generically (so a future special-case is caught)

- [ ] 2.0 Tool surface (covers FR-20.1; MCP-T2..T5)
  - [ ] 2.1 Lifecycle tools: `millm_circuit_status`, `millm_list_circuits`, `millm_activate_circuit`
        (`acknowledge_unvalidated`), `millm_deactivate_circuit`, `millm_delete_circuit`,
        `millm_set_circuit_intensity`
  - [ ] 2.2 `millm_import_circuit` (inline only) with argument validation BEFORE the gate check
        (`millm_clusters.py:44` precedent — an agent debugging its payload must not be told "millm is
        down"); NO hub tools (EC-20.5)
  - [ ] 2.3 `millm_export_circuit` returning the RAW document unmodified (no `response_model` mirror);
        description warns that unwrapping `.data` yields `None`
  - [ ] 2.4 Edge-sensing tools against `/api/circuit-sensing/*` (NOT the contract's reserved
        `/api/circuits/…/sensing`): `_status`, `_events`, `_event`, `_enable`, `_disable`, `_clear`;
        `since` tz-awareness rejection copied from `millm_sensing.py:42-52`
  - [ ] 2.5 Envelope pass-through everywhere: `success:false` bodies RETURNED not raised;
        `@gated(gate,"millm")` on every tool; structured unavailable never unregisters (EC-20.1/20.3)
  - [ ] 2.6 Unit tests: per-tool method+path, query/body shaping, gate degradation, argument
        pre-validation, export rawness, no tool mutates a response dict

- [ ] 3.0 Evidence integrity (covers FR-20.2; MCP-E1..E4)
  - [ ] 3.1 Verify structurally that NO tool composes, summarises or paraphrases evidence language —
        `rung`/`rung_language` transported only; no local rung re-derivation (it is MIN over edges,
        server-side); no `is_active` → steering inference
  - [ ] 3.2 Tool descriptions carry the three agent-facing semantics (MCP-E4): observation ≠ validation
        and never raises a rung; absence of rows ≠ absence of firing (`unsensable_edges`);
        `edge_rung_language` is as-of-observation. Plus `steering: null` = not evaluated, never "not
        steering"; and slice-fallback follows CLUSTER dial rules (0.5 floor)
  - [ ] 3.3 Decide OQ-1 and record it in the FTDD: port miLLM's stricter audit into miStudio
        (recommendation (a)) vs shared checker vs live-server scan
  - [ ] 3.4 Extend the copy audit: source scan over `mcp_server/tools/millm_*.py` with miLLM's
        three-stage filter (`\bcausal` → `UNRELATED_SENSE` → `EVIDENCE_CONTEXT`) AND `_code_only`
        comment stripping (miLLM :157-166)
  - [ ] 3.5 Registered-DESCRIPTION scan over `mcp._tool_manager.list_tools()` — catches a docstring
        composed at runtime or inherited, which no source grep sees
  - [ ] 3.6 Rung-0 negative control: a rung-0 circuit fixture driven through every tool path asserting
        "causal" appears in no response, description or error (MCP-E3)
  - [ ] 3.7 Reciprocal constant-parity guards in BOTH repos; each SKIPS LOUDLY (naming the missing
        checkout) when the sibling repo is absent — never passes vacuously

- [ ] 4.0 Reachability assurance (covers FR-20.3, FR-20.4; RCH-1..RCH-6)
  - [ ] 4.1 Write the rule text (FTDD §5.1) into the standing project rules — worded "a test that FAILS
        when the wiring is removed", explicitly excluding existence-only assertions and citing
        `TestRingPruningIsWired` by name (RCH-1, RCH-2)
  - [ ] 4.2 Registry reachability: `millm_circuits` ∈ `MILLM_CATEGORY_MODULES` ∈ `VALID_CATEGORIES`
        (RCH-3, shape 1)
  - [ ] 4.3 Built-server reachability via the REAL `build_server()` with the category enabled and
        `MILLM_API_URL` set — NOT a hand-called `register()`, which bypasses the gating that actually
        failed (RCH-3, shape 2)
  - [ ] 4.4 Per-tool caller assertion: each tool invokes its documented method+path against a recording
        client; fails if the call is removed or re-pointed (RCH-4, shape 3)
  - [ ] 4.5 RCH-5 regressions for the audit's own three findings: attach-set control buttons, circuit
        category registration, ring pruning's actual CALLER (not its existence)
  - [ ] 4.6 RCH-6 meta-test over this increment's capability list — recorded as a checklist, with its
        limit stated (a hand-maintained list is only as good as the list; the assurance is 5.3)

- [ ] 5.0 Integration & mutation verification (covers FR-20.1..20.3 end-to-end)
  - [ ] 5.1 End-to-end agent flow: import → activate WITHOUT acknowledgement (expect
        `UNVALIDATED_CIRCUIT` 200+envelope) → activate WITH it → set intensity → sensing enable →
        events → disable; `rung_language` verbatim asserted at every hop
  - [ ] 5.2 EC coverage: miLLM unreachable (EC-20.1), `MILLM_API_URL` unset (EC-20.2), no active
        circuit (EC-20.3), slice-fallback status (EC-20.4), naive `since` (EC-20.6), composed/slice
        rung refusal to synthesise (EC-20.7)
  - [ ] 5.3 **Mutation proof — the seven cuts** (FTDD §5.3), each recorded with the test it turned red:
        (1) drop the `MILLM_CATEGORY_MODULES` entry; (2) drop `VALID_CATEGORIES`; (3) re-point one
        tool's path; (4) strip one `@gated`; (5) plant "causally validated" in a rung-1 description;
        (6) remove the attach-set buttons; (7) remove ring pruning's caller. Mutations 6–7 are the
        audit's own findings re-armed — the feature's proof the rule would have caught what review did
        not
  - [ ] 5.4 Contract-consistency test (miLLM): every §4 row marked reachable names a tool the miStudio
        registry claims; row count reconciles with the 13 shipped tools

- [ ] 6.0 Documentation (covers FR-20.5; DOC-1..DOC-3)
  - [ ] 6.1 `docs/mcp-contract.md` → v1.2: three-state marks (`REST ✅ · MCP ✅` / `REST ✅ · MCP not
        registered` / `not served`); STATUS CORRECTION block RESOLVED not deleted (it is the record of
        why the marks changed); §4 sensing rows corrected to `/api/circuit-sensing/*`; §6 wiring gains
        `millm_circuits`; §7 circuit flow becomes a runnable tool sequence
  - [ ] 6.2 Manual MCP page: circuit tool list + the one semantic an agent must not get wrong
        (observation ≠ validation)
  - [ ] 6.3 Record the reachability rule as a STANDING project rule, not a feature note — it outlives
        this feature

- [ ] 7.0 Feature Acceptance (per instruct 008)
  - [ ] 7.1 Verify FPRD §9 criteria 1–6 + all US/EC boxes one-by-one
  - [ ] 7.2 Confirm zero `REST ✅ · MCP not registered` rows remain (baseline 12); zero capabilities
        without a reachability test (baseline 3)
  - [ ] 7.3 Both suites green (miStudio backend + frontend, miLLM backend); update CLAUDE.md Document
        Inventory + Current Status

## Coverage Audit
- FR-20.1→1.0/2.0/5.1; FR-20.2→3.0/5.1; FR-20.3→4.0/5.3; FR-20.4→4.1/6.1; FR-20.5→6.1/5.4 ✓
- US-20.1→2.1/2.2/5.1; US-20.2→2.4/3.2/5.1; US-20.3→4.x/5.3; US-20.4→3.4/3.5/3.6 — implementing +
  testing sub-tasks each ✓
- EC-20.1→2.5/5.2; EC-20.2→1.3/5.2; EC-20.3→2.5/5.2; EC-20.4→3.2/5.2; EC-20.5→2.2; EC-20.6→2.4/5.2;
  EC-20.7→3.2/5.2 ✓
- BRD: BR-004→MCP-T1..T5/MCP-E1..E4→1.0/2.0/3.0; BR-005→RCH-1..RCH-6/DOC-1..DOC-3→4.0/5.3/6.0 ✓
- TDD/TID sections mapped (module design→2.x, evidence integrity→3.x, reachability→4.x, contract→6.x) ✓
- Open questions: OQ-1 has a decision task (3.3) BEFORE its dependent work (3.4); OQ-2 resolved in the
  FPRD and implemented at 2.3 ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)

- [ ] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix criticals, document
      deferrals.
      **Watch for (feature-specific):**
      - **A tool description that PARAPHRASES a rung.** The audit greps for "causal"; it cannot catch
        "confirms the circuit's edges fire", "verified pathway", "proven to route" — words that do the
        overclaiming without the audited term. Read all 13 descriptions as an agent would.
      - **A tool that returns a rung for a COMPOSED or SLICE serve.** With Feature 19 landed, two
        circuits can serve at once and no single rung describes the response; in slice-fallback the
        served artifact is not the circuit. The runtime OMITS its header in exactly these cases —
        check no tool synthesises what the runtime refuses to.
      - **The reachability rule degenerating into box-ticking.** Any test whose failure mode is "the
        symbol was deleted" rather than "the wiring was cut" is `TestRingPruningIsWired` again. Ask of
        every reachability test: *if I disconnect this without deleting anything, does it go red?*
      - Registration asserted in one place only (three must agree, plus the deployment opt-in).
      - `success:false` bodies converted to exceptions, swallowing the rung the agent needs to re-send.
      - `steering: null` treated as `false` — the R3 overclaim from the other direction.
      - The parity guard passing vacuously when the sibling checkout is absent.
- [ ] Round 2 (post-fix verification + fresh angles): ≥10 findings — verify R1 fixes hold; hunt
      regressions. Specifically re-check that no R1 fix introduced a response-shaping convenience field.
- [ ] Round 3 (/review, 4 perspectives): ≥10 findings — fix, pin mutation survivors. Run the mutation
      practice over the tool module AND the audit itself: a copy audit that cannot fail is the same
      defect class it polices.
- [ ] Record: `.claude/context/sessions/review_feature020_R{1,2,3}_2026-07-*.md`.
- Directive: fix latent/pre-existing defects surfaced during review, not only regressions.

## Acceptance evidence
*[To be completed at acceptance — FPRD §9 criteria verified one-by-one, with the seven mutations
recorded against the tests they turned red.]*
