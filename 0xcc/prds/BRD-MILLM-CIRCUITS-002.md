# BRD-MILLM-CIRCUITS-002 — miLLM Circuit Runtime: Structural Consolidation, Agent Reach & Reachability Assurance

Incremental enhancement BRD produced via `0xcc/instruct/001_generate-brd.md`. Source material: the
**shipped BRD-MILLM-CIRCUITS-001 circuit runtime** (Features 12–15, closed 2026-07-20 — multi-SAE
attach and serving, circuit import with slice-fallback and the evidence ladder, the circuit-aware Open
WebUI dial, circuit edge sensing), treated here as a dependency rather than re-stated requirements;
the **twelve review rounds** conducted across that increment (349 findings, 135 fixed, 11 critical),
whose review records `0xcc/reviews/review_feature01{2,3,4,5}_*` are the primary evidence base for this
document; and the **post-close-out capability audit** (2026-07-20) run while attempting the GPU
close-out, which found three shipped-but-unreachable capabilities.

This is a **consolidation increment, not a feature increment.** It adds one genuinely new user-facing
capability (agent reach to circuits) and otherwise pays down structural debt that the 001 increment
incurred and recorded rather than hid. Its justification is empirical: across 001, **every review
round found a critical regression in the previous round's fix — twelve rounds, twelve for twelve.**
That is not variance; it is a structure in which correctness is maintained by convention rather than
enforced by construction, and the next feature to touch circuits will pay the same tax.

Clarifying-question round with the product owner is **PENDING** (see `next_steps`); the three themes
below are proposed, not locked.

```yaml
brd:
  metadata:
    brd_id: BRD-MILLM-CIRCUITS-002
    project_name: "miLLM"
    version: "0.1"
    author: "Sean"
    last_updated: "2026-07-20"
    status: "draft — clarifying round not yet held"
    increment_of: "miLLM (000_PPRD|miLLM.md)"
    successor_to: "BRD-MILLM-CIRCUITS-001"

  business_context:
    problem_statement: >
      The circuit runtime shipped and works, but it carries three classes of debt that are already
      costing real money and will cost more.

      FIRST — correctness is maintained by convention. Three of the eight criticals found in Feature
      15 alone share one root cause: N per-SAE position counters must agree on an absolute coordinate
      that no single component owns, and the shared EdgeFireRing's lifetime is managed by whoever
      remembers to call it. The code is correct today only because three separate code comments keep
      being obeyed. Similarly, "serve this circuit" is derived independently in THREE places
      (circuit_service.py:424 in _serve_full, circuit_service.py:799 in set_intensity, and
      inference_service.py:955 in the per-request dial); F14's two worst defects were both consequences
      of those derivations drifting. Neither CircuitService nor ProfileService takes the inference
      request queue (verified: zero references in either file), so an operator changing intensity
      mid-request has the change silently reverted by the request restore — while set_intensity returns
      "reapplied": true, an affirmative falsehood.

      SECOND — capabilities ship without a path to reach them. The post-close-out audit found three,
      all discovered only when a real operator tried to use the system: (a) Feature 12's multi-SAE
      attach had a service, a REST route, an API client and a React hook, and NO UI control — the
      AttachmentPanel destructured only the read fields and rendered zero buttons, so no user could
      ever attach the second SAE a circuit requires; (b) docs/mcp-contract.md listed eleven circuit
      MCP tools as shipped, while MILLM_CATEGORY_MODULES registers only millm_runtime, millm_clusters
      and millm_sensing — no agent can import, activate, dial or sense a circuit; (c) the edge-sensing
      ring's pruning was declared "request-level" twice, in two consecutive review rounds, and wired
      neither time, the second time accompanied by a test named for the defect it failed to prevent.
      The common shape is that reviews verified the mechanism and never asked whether anything called
      it.

      THIRD — the agentic half of the ecosystem stops at clusters. BRD-MILLM-CLUSTERS-001 delivered a
      unified MCP surface so an agent could move a tuned cluster into production and watch it fire.
      The circuits arc delivered strictly more capability — multi-layer interventions, an evidence
      ladder, edge observation — and none of it is agent-reachable. An agent can steer a single-layer
      cluster and read its co-activations, but cannot import a circuit, cannot activate one, cannot
      read an edge observation, and cannot see a rung. The honesty guarantees the increment was built
      around are invisible to the consumer most likely to over-claim on their behalf.
    vision_statement: >
      miLLM's circuit runtime becomes structurally sound rather than conventionally correct: one
      request-scoped context owns position accounting, one derivation serves a circuit, and an
      operator's action can never be silently undone by a request that was already in flight. Agents
      reach circuits through the same unified MCP server they already use for clusters, carrying the
      evidence rung verbatim into every answer. And "shipped" comes to mean "a user or an agent can
      actually invoke it" — enforced by an acceptance rule, not by hoping a reviewer asks.
    business_objectives:
      - "Eliminate the class of defect that produced 11 criticals across 001 by making the invariants
         unrepresentable rather than test-guarded."
      - "Extend agent reach from clusters to circuits, completing the ecosystem promise of
         BRD-MILLM-CLUSTERS-001 for the strictly richer artifact."
      - "Make unreachable capability impossible to ship undetected."
      - "Close the two 001 acceptance criteria that remain GPU-pending, and retire the alone-vs-within
         ambiguity."
    success_definition: >
      A subsequent feature touching circuit serving or edge sensing completes its three review rounds
      WITHOUT a critical regression in a prior round's fix — the first time that has happened in this
      arc. Every circuit capability is invocable by both a human and an agent. No shipped capability
      lacks a reachability test.

  stakeholders_users:
    primary:
      - name: "Interpretability engineer (miStudio author, miLLM operator)"
        need: >
          Attach a circuit's SAE set, serve it, dial it, and observe its edges — without needing to
          know which of three code paths applies, and without an operator action being silently
          reverted by an in-flight request.
      - name: "Agent (via the unified miStudio-hosted MCP server)"
        need: >
          Do for circuits what it can already do for clusters: import, activate, dial, sense, and
          report — with each answer carrying the evidence rung verbatim so the agent cannot
          accidentally over-claim causality on the user's behalf.
    secondary:
      - name: "End user in Open WebUI"
        need: "Unchanged from 001 — this increment must not regress the dial or the rung disclosure."
      - name: "Future maintainer"
        need: >
          Change circuit serving without re-deriving three copies of the same fact, and without the
          twelve-round review tax.

  scope_definition:
    in_scope:
      - "A request-scoped SensingRequestContext owning the position counter, the shared ring, and the
         per-request event budget; per-SAE counters retired."
      - "Extraction of the edge-sensing machinery out of sae_wrapper.py into its own module."
      - "A single circuit-serving derivation (CircuitSteeringEngine) consumed by activation,
         set_intensity and the per-request dial; the SAEService.__new__ bypass at
         inference_service.py:743 retired with it."
      - "A steering epoch on AttachedSAEState, bumped by every authoritative writer, compared under
         the lock at restore — closing the mid-request-mutation window on BOTH the circuit path and
         the Feature 10 profile path, with last-authoritative-writer-wins semantics and an honest
         set_intensity return value."
      - "A millm_circuits MCP category on the existing unified server: list, import, activate,
         deactivate, export, set intensity, status, plus edge-sensing status/events/enable/disable —
         every response carrying rung and rung_language verbatim."
      - "A reachability acceptance rule and its enforcement: no capability is accepted without a test
         proving a user-facing or agent-facing caller invokes it."
      - "Closing the two GPU-pending 001 criteria (F14 §9.1, F15 §9.1) against a live multi-SAE serve."
      - "Resolving alone-vs-within: either compute it per-event from the ring, or formally retire the
         requirement in favour of ambient_fired_count."
      - "Per-row truncation attribution plus truncated_layers in the edge-sensing status payload."
    out_of_scope:
      - "New interpretability capability — no new discovery, validation or steering mathematics."
      - "Changes to the frozen v1 cluster/circuit schemas (miStudio owns them; miLLM consumes)."
      - "Multiple concurrently-served circuits (F13's single-active invariant stands this increment)."
      - "Re-litigating the evidence ladder vocabulary or the rung<2 acknowledgement gate."
      - "The Open WebUI filter's UX, beyond not regressing it."
    assumptions:
      - "miStudio remains the owner of the MCP server; circuit tools are additive to it, mirroring how
         cluster tools were added (no new server, no new repo)."
      - "The 16-layer / 200-edge contract maxima remain the sizing envelope."
      - "The single-active-circuit invariant from F13 holds, so a request-scoped context has exactly
         one circuit to reason about."
    constraints:
      - "Additive-only at the API boundary; docs/mcp-contract.md moves to v1.2 with no breaking change."
      - "Refactors must be behaviour-preserving and provable as such — the existing 1597 backend / 272
         frontend tests are the floor, not the target."
      - "The evidence-honesty guarantees from 001 are inviolable: rung verbatim, no 'causal' below
         rung 2, the copy audit stays build-failing."

  business_requirements:
    - id: BR-001
      text: "Absolute token position, the shared edge ring, and the per-request event budget SHALL be owned by a single request-scoped context created at request start and passed to each participating SAE, so that per-SAE counter divergence, cross-layer prune races, and per-layer budget skew are structurally impossible rather than prevented by test coverage."
    - id: BR-002
      text: "Serving a circuit SHALL have exactly ONE derivation, consumed by activation, by intensity changes, and by the per-request dial; no caller SHALL construct a service instance by bypassing its constructor in order to reach it."
    - id: BR-003
      text: "An operator action that changes live steering state SHALL NOT be silently reverted by a request that was already in flight; the later authoritative writer SHALL win, the outcome SHALL be observable in the logs, and any API response reporting that steering was re-applied SHALL be truthful."
    - id: BR-004
      text: "An agent SHALL be able to do for circuits everything it can already do for clusters — list, import, activate, deactivate, export, dial, and read observations — through the SAME unified MCP server, with every circuit and edge response carrying its EvidenceRung and server-rendered rung_language verbatim."
    - id: BR-005
      text: "No capability SHALL be accepted as shipped without an automated test proving that a user-facing control or an agent-facing tool actually invokes it; documentation status marks SHALL distinguish 'endpoint exists' from 'reachable by a user or agent'."
    - id: BR-006
      text: "The edge-sensing truncation signal SHALL identify WHICH layer shed data rather than marking an entire request's observations truncated when any single layer did."
    - id: BR-007
      text: "The alone-vs-within distinction SHALL either be computed per-event and honestly gated, or SHALL be formally retired from the requirement set in favour of the existing ambient_fired_count — it SHALL NOT remain nominally required and unimplemented."
    - id: BR-008
      text: "The two acceptance criteria deferred from 001 for want of a live GPU serve SHALL be closed against a real multi-SAE circuit, or SHALL be restated as criteria the test suite can honestly discharge."

  success_metrics:
    - metric: "Regression-free review rounds"
      target: >
        The three review rounds for this increment complete with ZERO criticals that are regressions
        in a prior round's fix (baseline across 001: 12 of 12 rounds had one).
    - metric: "Derivation count for circuit serving"
      target: "Exactly 1 (baseline: 3, verified at circuit_service.py:424, :799 and inference_service.py:955)."
    - metric: "Position-accounting counters per request"
      target: "Exactly 1 request-scoped counter (baseline: N per-SAE counters)."
    - metric: "Agent-reachable circuit capability"
      target: >
        Every circuit capability exposed by REST is invocable via MCP; mcp-contract.md carries no row
        marked 'REST ✅ · MCP not registered' (baseline: 12 such rows).
    - metric: "Unreachable-capability defects"
      target: >
        0 shipped capabilities without a reachability test (baseline: 3 found post-hoc by an operator,
        not by review).
    - metric: "Behaviour preservation"
      target: "Backend ≥1597 and frontend ≥272 tests green throughout; no acceptance criterion regresses."

  feature_themes:
    - theme: "Structural consolidation"
      covers: [BR-001, BR-002, BR-003, BR-006]
      note: >
        The refactor core. Sequencing matters: the epoch (BR-003) is small, independent, and fixes an
        operator-visible falsehood, so it can land first and alone. The request-scoped context
        (BR-001) and the module extraction pair naturally, since both move the same code. The single
        derivation (BR-002) touches the most surface area and should land last.
    - theme: "Agent reach to circuits"
      covers: [BR-004]
      note: >
        The one genuinely new capability. Mirrors the millm_clusters and millm_sensing modules
        already in MILLM_CATEGORY_MODULES; the REST endpoints and their contracts already exist and
        are tested, so this is a registration and shaping exercise, not new backend work.
    - theme: "Reachability assurance"
      covers: [BR-005]
      note: >
        Process plus enforcement. The acceptance rule is cheap; the value is that it would have caught
        all three audit findings before an operator did.
    - theme: "Acceptance close-out"
      covers: [BR-007, BR-008]
      note: "Retires the two honest gaps left open at the end of 001."

  considerations:
    technical:
      - "The context refactor (BR-001) and the extraction are behaviour-preserving in intent, but the
         edge matcher is the single most defect-dense code in the arc. Characterization tests should be
         written BEFORE the move, and the mutation practice applied to the result."
      - "The epoch (BR-003) must cover the Feature 10 profile path in the same change; fixing only the
         circuit path leaves the identical window open one file over, which is how several 001 defects
         propagated."
      - "Three candidate epoch mechanisms were evaluated during F14 R3: extending the request semaphore
         to admin mutations (rejected — turns management calls into 503s behind long generations and
         inverts layering), a monotonic steering epoch (recommended), and per-layer version counters
         (rejected — produces a half-old/half-new state harder to reason about than either)."
      - "MCP tool responses must not re-phrase evidence language. The cluster tools already establish
         the pattern; the risk is a well-meaning summary field that paraphrases a rung."
    business:
      - "This increment produces little demo-able surface. Its value is that the NEXT increment costs
         less, which is real but not visible. Worth an explicit product-owner decision rather than an
         assumption."
      - "Agent reach (BR-004) is the exception — it is immediately demonstrable and completes a promise
         the cluster increment already made."
    ux:
      - "The attach-set control shipped during the audit is minimal: it cannot detach, infers layer from
         trained_layer, and cannot pre-filter incompatible SAEs because the list endpoint exposes no
         compatibility flag. Whether to invest further is a product call."
      - "Edge-sensing status should surface WHY an armed circuit is observing nothing (paused_reason
         shipped during the audit); the unsensable-edge list is currently uncapped and can push the
         event list off-screen under slice-fallback."

  risks:
    - id: RSK-001
      risk: "A behaviour-preserving refactor of the most defect-dense code in the arc silently changes behaviour."
      severity: "high"
      mitigation: >
        Characterization tests written and green BEFORE any move; mutation testing applied to the
        result per the standing practice; the three-round review cycle retained in full.
    - id: RSK-002
      risk: "The consolidation is deferred indefinitely because it produces no visible feature."
      severity: "high"
      mitigation: >
        The empirical cost is documented (12 of 12 rounds carried a regression). If deferred, that
        should be an explicit decision with the tax acknowledged, not a silent deprioritisation.
    - id: RSK-003
      risk: "MCP circuit tools leak or paraphrase evidence language, defeating the ladder at the surface most likely to over-claim."
      severity: "high"
      mitigation: >
        Extend the existing build-failing copy audit to the MCP tool modules and their descriptions;
        require rung + rung_language verbatim on every circuit-bearing response; add a negative
        control proving a rung-0 circuit cannot be described as causal through any tool.
    - id: RSK-004
      risk: "The reachability rule becomes box-ticking — a test that asserts a control exists rather than that it works."
      severity: "medium"
      mitigation: >
        The rule is specifically that a test must FAIL when the wiring is removed. Feature 15 shipped a
        test named TestRingPruningIsWired that asserted an entry point existed while nothing called it;
        that is the precise anti-pattern to exclude, and it should be cited in the rule's wording.
    - id: RSK-005
      risk: "The epoch's last-writer-wins semantics surprise an operator whose change lands mid-generation."
      severity: "medium"
      mitigation: >
        The in-flight generation necessarily finishes under the old value (the hook is already
        installed); what changes is that the operator's value SURVIVES afterwards instead of being
        reverted. Log the supersession explicitly so the behaviour is observable.
    - id: RSK-006
      risk: "GPU close-out (BR-008) blocks on host availability and stalls the increment."
      severity: "low"
      mitigation: >
        It is independent of every other theme and can close at any point. If the host stays
        unavailable, BR-008's fallback clause applies: restate the criteria as suite-dischargeable.

  next_steps:
    - "Hold the clarifying-question round with the product owner — this draft locks nothing."
    - "Decide the headline question: is this increment worth running before new capability? The case
       for is the 12-of-12 regression rate; the case against is zero demo surface outside BR-004."
    - "Decide whether BR-004 (agent reach) should be split into its own smaller increment that could
       ship immediately, leaving the refactor to follow."
    - "Resolve BR-007 by product decision: implement alone-vs-within, or retire it."
    - "On approval, proceed to PPRD/PADR updates and the per-feature FPRD/FTDD/FTID/FTASKS chain."
```

## Provenance note

Every factual claim in this document was verified against the code at the time of writing rather than
recalled: the three serving derivations (`circuit_service.py:424`, `:799`,
`inference_service.py:955`), the absent queue coordination (zero `_request_queue` references in either
`circuit_service.py` or `profile_service.py`), the constructor bypass (`inference_service.py:743`),
the edge machinery's footprint in `sae_wrapper.py` (91 `_edge` references in 1373 lines), and the MCP
registry contents (`MILLM_CATEGORY_MODULES` = `millm_runtime`, `millm_clusters`, `millm_sensing`).

The review records `0xcc/reviews/review_feature01{2,3,4,5}_*_2026-07-20.md` are the evidence base for
the regression-rate claim and for the three deferred designs carried into BR-001, BR-002 and BR-003.
