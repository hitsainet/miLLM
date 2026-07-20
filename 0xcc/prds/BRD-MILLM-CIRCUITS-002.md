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

Clarifying-question round completed with the product owner 2026-07-20. **Stated goal: "as mature and
bullet-proof a product as possible."** That answer settles the document's headline question — whether a
consolidation increment with little demo surface is worth running at all — in the affirmative, and
reframes every remaining decision as *how much assurance*, not *whether*. Locked decisions:
**(1)** **refactor first, then agent reach** — the MCP circuit surface is built on settled ground rather
than against three serving derivations that are about to move, so nothing is written twice and a tool
bug is never confusable with a refactor bug;
**(2)** **implement alone-vs-within properly** — computed per event from the ring's fired-position
sets, rather than retired or deferred a third time;
**(3)** **highest verification tier for the edge matcher** — characterization tests written and green
BEFORE any code moves, the mutation practice applied to the result, then the full three review rounds;
**(4)** **three previously out-of-scope gaps folded in** — detach from the attach-set dialog, SAE
compatibility pre-filtering, and concurrent multi-circuit serving. The third is a genuine design change,
not a cleanup: the single-active invariant is enforced by a partial unique index (`uq_circuits_active`)
at the database level, so lifting it requires a migration and a contention model for layers claimed by
more than one circuit.

```yaml
brd:
  metadata:
    brd_id: BRD-MILLM-CIRCUITS-002
    project_name: "miLLM"
    version: "0.1"
    author: "Sean"
    last_updated: "2026-07-20"
    status: "draft — clarifying round held 2026-07-20; decisions locked, awaiting approval to execute"
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
      - "Detach from the attach-set dialog — either a detach fan-out over deselected keys or
         set-semantics server-side, so the control's mental model matches its behaviour."
      - "SAE compatibility pre-filtering in the picker, which requires the SAE list endpoint to expose
         a compatibility verdict the client can act on before a round trip."
      - "Concurrent multi-circuit serving: lifting F13's single-active invariant, including the
         uq_circuits_active partial unique index, the migration to drop it, and a contention model for
         layers claimed by more than one active circuit."
    out_of_scope:
      - "New interpretability capability — no new discovery, validation or steering mathematics."
      - "Changes to the frozen v1 cluster/circuit schemas (miStudio owns them; miLLM consumes)."
      - "Re-litigating the evidence ladder vocabulary or the rung<2 acknowledgement gate."
      - "The Open WebUI filter's UX, beyond not regressing it."
    assumptions:
      - "miStudio remains the owner of the MCP server; circuit tools are additive to it, mirroring how
         cluster tools were added (no new server, no new repo)."
      - "The 16-layer / 200-edge contract maxima remain the sizing envelope."
      - "The request-scoped context (BR-001) must be designed for MORE than one active circuit from the
         start, since BR-011 lifts the single-active invariant in this same increment. Designing it
         around one circuit and generalising later would repeat the exact mistake this increment
         exists to correct."
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
      text: "Each recorded edge observation SHALL carry a per-event alone-vs-within classification, computed from the fired-position sets already held at match time, so the distinction does not depend on full-width monitoring co-running; where a classification genuinely cannot be made it SHALL be NULL and never estimated."
    - id: BR-008
      text: "The two acceptance criteria deferred from 001 SHALL be closed. F14 §9.1 was CLOSED at the 2026-07-20 GPU close-out. F15 §9.1 remains partial: sensing armed correctly across 5 layers with 4 sensable edges and ran with non-zero overhead, but a capture RATE requires a circuit built from miStudio-mined features known to co-fire — an authoring-side prerequisite this increment SHALL either satisfy or restate."
    - id: BR-009
      text: "The attach-set control SHALL be able to REMOVE an SAE from the attached set, not only add to it, so that the control's behaviour matches the mental model its multi-select presents; today unchecking a row does nothing because attach_set is purely additive."
    - id: BR-010
      text: "The SAE picker SHALL indicate which SAEs are compatible with the loaded model BEFORE submission, using a compatibility verdict exposed by the SAE listing rather than discovering incompatibility through a server rejection."
    - id: BR-011
      text: "miLLM SHALL serve MORE THAN ONE circuit concurrently, replacing the single-active invariant (enforced today by the uq_circuits_active partial unique index) with LAYER-EXCLUSIVE CLAIMS: a layer is claimed by at most one active circuit, activation is refused with CIRCUIT_LAYER_CONTENTION naming the incumbent when claim sets overlap, and an operator may override with an explicit allow_layer_overlap acknowledgement — under which the circuit-rung header is OMITTED, because no single circuit's evidence describes a composed response. The refusal that precedes any override SHALL carry the MEASUREMENT that motivates it, not merely the fact of contention: an override chosen in knowledge that two steered layers at individually-harmless strength destroyed generation in close-out testing is a research decision; one chosen blind is a footgun. Every use SHALL be echoed in the response, logged, and surfaced in the UI, mirroring acknowledge_unvalidated. Two circuits naming the same (layer, feature_idx) SHALL be refused unconditionally, since the merge would silently serve a strength belonging to neither author. Design of record: 0xcc/docs/circuit-contention-model.md."
    - id: BR-011a
      text: "CIRCUIT_ALLOW_CONCURRENT SHALL default to false for exactly ONE release, with the flip to true recorded as a dated commitment rather than deferred indefinitely — an unflipped flag makes a shipped capability unreachable, which is the precise defect class this increment exists to eliminate. While the flag is false, a second activation SHALL be refused LOUDLY, naming configuration as the reason; it SHALL NOT silently fall back to the single-active disarm behaviour that Feature 19 replaces."
    - id: BR-012
      text: "miLLM SHALL warn on circuit SHAPE — the number of steered layers and their aggregate strength — independently of miStudio-supplied effect sizes, because the GPU close-out measured generation collapsing at TWO steered layers at individually-harmless strengths, two orders of magnitude below the per-member clamp that is the only aggregate bound today."
    - id: BR-013
      text: "Runtime thresholds inherited from the single-SAE era SHALL be expressed with a per-layer denominator or scaled by the armed layer count; a constant that guarantees an alarm on every multi-layer circuit trains operators to ignore alarms (measured: 5.4-7.3 ms sensing overhead across 5 armed layers against a fixed 5 ms threshold)."

  success_metrics:
    - metric: "Regression-free review rounds"
      target: >
        The three review rounds for this increment complete with ZERO criticals that are regressions
        in a prior round's fix (baseline across 001: 12 of 12 rounds had one).
    - metric: "Derivation count for circuit serving"
      target: >
        Exactly 1. **Baseline corrected to 4 during F18 authoring**: beyond
        circuit_service.py:424, :799 and inference_service.py:955, a FOURTH
        independent derivation lives in _steering_circuit_uncached
        (inference_service.py:806-822), which flattens members and builds
        member_layers to gate the RUNG HEADER. That is a serving derivation on
        the evidence surface — the worst place to leave one, and the exact
        F14-R2-02 defect class. Scoped into Feature 18.
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
    - metric: "Concurrent circuits served"
      target: >
        ≥2 circuits serving simultaneously with a defined, tested outcome for a contended layer
        (baseline: 1; activating a second silently disarms the first).
    - metric: "Alone-vs-within coverage"
      target: >
        A classification present on every observation where one is derivable, independent of whether
        monitoring co-ran (baseline: NULL unless full-width monitoring happened to be running).
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
    - theme: "Aggregate-hazard awareness"
      covers: [BR-012]
      note: >
        Added after the GPU close-out measured 2-layer collapse. Distinct from the contention model:
        this hazard exists for a SINGLE circuit spanning several layers, so it is not solved by
        layer-exclusive claims. The runtime currently has no opinion about circuit shape independent
        of miStudio's authoring-side effect sizes.
    - theme: "Acceptance close-out"
      covers: [BR-007, BR-008]
      note: >
        Retires the two honest gaps left open at the end of 001. BR-007 is now an implementation
        rather than a decision: alone-vs-within is computed per event from the ring's fired-position
        sets, which are already in hand at match time, so it no longer depends on monitoring being on.
    - theme: "Operator-facing completeness"
      covers: [BR-009, BR-010, BR-013]
      note: >
        Folded in at the clarifying round; BR-013 added after the GPU close-out. Both are consequences of the attach-set control being built
        under time pressure during the capability audit: it can add but not remove, and it discovers
        incompatibility by round trip. Neither is deep, and both are the difference between a control
        that works and one that behaves as it looks.
    - theme: "Concurrent circuit serving"
      covers: [BR-011, BR-011a]
      note: >
        The one genuinely NEW design work in this increment, and the largest single risk in it. This
        is not a cleanup: the invariant is enforced at the database level by a partial unique index, so
        lifting it needs a migration, a contention model for layers claimed by more than one circuit,
        and a decision about what per-layer budgets mean when two circuits both steer L13. It also
        forces BR-001's context to be designed for N circuits from the outset — which is the right
        order, but only if the design work happens BEFORE the context lands, not after.

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
    - id: RSK-007
      risk: >
        Concurrent multi-circuit serving (BR-011) is scoped as a fold-in but is the largest design
        change in the increment, and it lands in the same code the refactor is moving.
      severity: "high"
      mitigation: >
        Sequence it as DESIGN-FIRST: the contention model must be settled before BR-001's context is
        implemented, so the context is built for N circuits rather than generalised afterwards.
        If the contention model proves contentious, BR-011 is the one item that can be split into a
        follow-on WITHOUT invalidating the rest of the increment — provided the context's interface
        is designed for N from the start regardless.
    - id: RSK-008
      risk: >
        Lifting a database-enforced invariant (uq_circuits_active) is irreversible in deployed data
        once two circuits have been active simultaneously.
      severity: "medium"
      mitigation: >
        The migration must be paired with a tested downgrade path, and the contention model must
        define a deterministic resolution for pre-existing rows. Treat the first concurrent activation
        as a one-way door and gate it behind explicit acceptance.
    - id: RSK-006
      risk: "GPU close-out (BR-008) blocks on host availability and stalls the increment."
      severity: "low"
      mitigation: >
        It is independent of every other theme and can close at any point. If the host stays
        unavailable, BR-008's fallback clause applies: restate the criteria as suite-dischargeable.

  next_steps:
    - "SETTLED 2026-07-20 (product owner): CIRCUIT_ALLOW_CONCURRENT ships false for ONE release with a
       dated flip commitment, NOT an open-ended default. The flag exists solely because the first
       concurrent activation is a one-way door in deployed data — trivial to enable, destructive to
       reverse — and not because the code is doubted. It protects OTHER deployments; if none exist at
       flip time, drop the flag rather than carry it."
    - "SETTLED 2026-07-20 (product owner): allow_layer_overlap IS retained. The close-out measurement
       is real but thin — one model, arbitrarily chosen feature indices, invented max_activation
       values — so it proves that fixture compounds destructively, NOT that all overlapping circuits
       do. Hard-refusing a legitimate research action on one unrepresentative data point is overreach,
       and deliberate compounding studies are a real use case for an interpretability tool. The
       honesty guarantee holds regardless, since the rung header is omitted when composed. The
       override is retained on the explicit condition that it is LOUD AND INFORMED: the refusal
       carries the measurement, and every use is echoed, logged and surfaced."
    - "Design the contention model for BR-011 FIRST — what happens when two active circuits claim the
       same layer, and what a per-layer budget means under two claimants. This gates BR-001, because
       the request-scoped context must be built for N circuits rather than generalised later."
    - "Write characterization tests for the edge matcher and get them green BEFORE any code moves;
       apply the mutation practice to the result (locked decision 3)."
    - "Proceed to PPRD/PADR updates, then the per-feature FPRD/FTDD/FTID/FTASKS chain in the locked
       order: structural consolidation → operator-facing completeness → concurrent serving →
       agent reach → acceptance close-out."
    - "Close BR-008 opportunistically whenever the GPU host is free; it is independent of every other
       theme and need not gate the increment."
    - "Re-run the capability audit at acceptance: every BR must have a reachability test, per BR-005."

  execution_order:
    rationale: >
      Locked at the clarifying round: refactor before agent reach, so the MCP surface is written
      against settled code. Within the refactor, smallest-and-independent first, largest-surface last.
    sequence:
      - step: 1
        item: "Steering epoch (BR-003)"
        why: >
          Small, independent of the other refactors, and it fixes an operator-visible falsehood today
          (set_intensity reporting "reapplied": true when the change was reverted). Covers the
          Feature 10 profile path in the same change.
      - step: 2
        item: "Contention model design for concurrent serving (BR-011, design only) — ✅ DONE 2026-07-20"
        why: >
          Gates step 3 — the context must be designed for N circuits, not retrofitted. Settled in
          0xcc/docs/circuit-contention-model.md: layer-exclusive claims, refuse-by-default with a
          named incumbent, explicit override that omits the rung header, unconditional refusal on
          same-key collision. Its section 4 specifies the context's required shape, including ONE
          RING PER (request, circuit) — a shared ring would let one circuit's upstream fire match
          another's downstream and record an edge that never fired in either.
      - step: 3
        item: "Request-scoped context + edge-machinery extraction (BR-001)"
        why: "The heart of the consolidation; highest verification tier applies here."
      - step: 4
        item: "Single serving derivation (BR-002)"
        why: "Largest surface area; lands once the context beneath it is stable."
      - step: 5
        item: "Concurrent serving implementation + migration (BR-011)"
        why: "Built on the settled context. The one item splittable into a follow-on if needed."
      - step: 6
        item: "Operator-facing completeness (BR-009, BR-010) and truncation attribution (BR-006)"
        why: "Independent of the refactor; can run in parallel if capacity allows."
      - step: 7
        item: "Alone-vs-within (BR-007)"
        why: "Touches the matcher, so it follows the extraction rather than preceding it."
      - step: 8
        item: "MCP circuit surface (BR-004)"
        why: "Written against settled ground — the locked sequencing decision."
      - step: 9
        item: "Reachability rule enforcement (BR-005) and acceptance close-out (BR-008)"
        why: "The rule applies to everything above; close-out lands whenever the GPU host is free."

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
