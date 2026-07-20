# BRD-MILLM-CIRCUITS-001 — miLLM Circuit Runtime: Multi-SAE Serving, Live Circuit Steering, Circuit-Aware Dial & Edge Sensing

Incremental enhancement BRD produced via `0xcc/instruct/001_generate-brd.md`. Source material: the
miLLM v1.0 baseline (Model Management, OpenAI-compatible API, single-SAE Management with residual-stream
hooking, Feature Steering, Feature Monitoring, Profile Management, Admin UI) plus the post-v1.0 shipped
runtime — GitOps/K8s deployment, CBM continuous batching + speculative decoding, hybrid/Mamba support,
Neuronpedia links, security hardening; the **shipped BRD-MILLM-CLUSTERS-001 cluster runtime**
(cluster-definition/v1 import, unified MCP server, Open WebUI live dial, cluster-scoped co-activation
sensing — the direct predecessor, treated here as a dependency, not re-stated requirements); and the
**miStudio circuits arc** (BRD-MIS-CIRCUITS-001 / BRD-MIS-CIRCUITS-002, features 015–018, closed
2026-07-20) whose `future_considerations` (BRD-MIS-CIRCUITS-001 §future_considerations, lines 106–107)
explicitly deferred the miLLM multi-SAE serving runtime — "attach multiple SAEs, import+serve circuit
definitions, circuit-aware per-request intensity dial" and "edge-level co-activation sensing" — to this
follow-on document (named there as "BRD-MILLM-CIRCUITS-001 or similar"), mirroring how
BRD-MIS-CLUSTERS-001 deferred the cluster runtime that became BRD-MILLM-CLUSTERS-001.
Clarifying-question round completed with the product owner 2026-07-20; locked decisions:
**(1)** **full multi-SAE now** — attach multiple SAEs at once and serve full `mistudio.circuit-definition/v1`
with live cross-layer edges (per-layer budgets under one global intensity), plus the per-layer
cluster-slice fallback for single-SAE deployments; this increment folds in the recorded two-SAE
GPU / VRAM<200 MB close-out;
**(2)** **honor the evidence ladder verbatim** — miLLM surfaces each circuit/edge EvidenceRung wherever
steering state is shown (MCP status, Open WebUI dial, Admin UI), never labels rung<2 "causal", and
requires an explicit acknowledgement to activate an unvalidated (rung<2) circuit;
**(3)** **extend existing surfaces** — new circuit MCP tools live on the SAME miStudio-hosted unified MCP
server, the live dial extends the SAME Open WebUI filter, and new miLLM circuit endpoints are
additive-only and tracked in `docs/mcp-contract.md` (→ v1.1); no new servers or repos.

```yaml
brd:
  metadata:
    brd_id: BRD-MILLM-CIRCUITS-001
    project_name: "miLLM"
    version: "0.1"
    author: "Sean"
    last_updated: "2026-07-20"
    status: "draft"
    increment_of: "miLLM (000_PPRD|miLLM.md)"
    successor_to: "BRD-MILLM-CLUSTERS-001"

  business_context:
    problem_statement: >
      miStudio can now discover cross-layer circuits, validate them with real causal interventions,
      grade each edge on an evidence ladder (mined → attribution-supported → causally-validated →
      faithfulness-tested), and export them as portable mistudio.circuit-definition/v1 artifacts —
      multi-SAE, with typed cross-layer edges, per-layer strength budgets, and validation manifests.
      But miLLM, the serving runtime, cannot run any of it. miLLM is hard single-SAE: exactly one SAE
      is attached at a time, so a circuit whose members live on layers L10 and L13 cannot be served —
      the L13 members would be steered through the wrong layer's feature basis, or not at all. miLLM
      has zero circuit awareness: it can import a cluster (one layer), but not a circuit (many layers,
      with edges). There is no way to dial a circuit's influence live in a real chat, no way for the
      end user or an agent to see that a circuit's steering is only "mined" (not causal), and no way to
      observe circuit EDGES firing in production — the upstream feature firing followed by its
      downstream partner. The ecosystem now discovers and validates circuits it cannot yet run, dial,
      grade honestly at the frontend, or sense.
    vision_statement: >
      miLLM becomes the RUNTIME half of the CIRCUIT ecosystem, exactly as it already is for clusters:
      miStudio discovers and validates circuits; miLLM attaches the SAEs a circuit references, serves
      the circuit live at its tuned per-layer budgets under a single global intensity dial, carries the
      evidence rung of every circuit and edge verbatim to the frontend (so a rung-1 circuit is never
      presented as "causal"), lets an end user dial a whole circuit's influence in a live Open WebUI
      chat, and closes the loop by sensing when a circuit's EDGES fire in real traffic — feeding
      edge-level observation back into authoring. A circuit discovered on Monday can be steering a live
      Open WebUI session on Tuesday, with its evidence status honestly attached.
    primary_objectives:
      - "Make the portable circuit definition executable: import a multi-SAE circuit and serve it live, every member steered through its own layer's SAE, zero manual tuning."
      - "Attach multiple SAEs at once, loading only the SAEs a circuit references, within a documented VRAM envelope."
      - "Keep the ecosystem honest at the point of influence: surface each circuit/edge's evidence rung verbatim and never call rung<2 steering 'causal'."
      - "Put a whole circuit's influence under the end user's hand in live chat (off/min/max dial) via Open WebUI."
      - "Close the authoring loop with EDGE-level co-activation sensing (upstream→downstream) in production traffic."
      - "Give agents a circuit surface on the SAME unified MCP server (import, activate, status, sensing) with no new server."
    success_criteria:
      - "A circuit validated in miStudio steers identically in miLLM after import, with every member applied through its own layer's SAE at the authored strength and no manual entry."
      - "On a single-SAE deployment, the same circuit still steers via its per-layer cluster-definition/v1 slice with zero runtime reconfiguration."
      - "The evidence rung of an active circuit (and its edges) is visible everywhere steering state is shown, and activating a rung<2 circuit requires an explicit unvalidated acknowledgement."
      - "An Open WebUI user can compare identical prompts at circuit influence off / min / max within one chat session."
      - "Circuit edge co-activation events are recorded (upstream firing followed by downstream partner) with the alone-vs-within distinction and are retrievable for analysis."
      - "Two SAEs serve a live circuit within the documented VRAM envelope (the two-SAE / VRAM<200 MB close-out reported as a measured number)."

  stakeholders_users:
    primary_users:
      - "Interpretability researchers who discover and validate circuits in miStudio and want them running in a serving stack."
      - "miLLM operators/self-hosters who want ready-made, evidence-graded circuit behaviors without doing discovery."
      - "AI agents (via the unified MCP) that discover, validate, promote, and now DEPLOY circuits across both products."
    secondary_users:
      - "End users chatting through Open WebUI who experience (and dial) circuit-steered generation and can see its evidence status."
      - "Community members exchanging circuit packs (consume-side; publishing stays in miStudio)."
    stakeholders:
      - "Product owner (Sean) — ecosystem vision: miStudio discovers/validates, miLLM serves/senses circuits."
      - "miStudio project — producer of the circuit-definition/v1 interchange artifact, the evidence ladder, and the unified MCP server."

  scope_definition:
    in_scope:
      - "Multi-SAE attachment: relax the single-attached-SAE constraint so miLLM can attach several SAEs simultaneously, keyed by (SAE, layer); a circuit's referenced SAEs are loaded together, and only the SAEs a circuit actually references are loaded (referenced-only loading)."
      - "Import of mistudio.circuit-definition/v1 documents from local JSON files: strict validation against the published v1 circuit schema, rejecting unknown kinds and incompatible schema major versions with actionable errors."
      - "Per-referenced-SAE compatibility evaluation at import mirroring the cluster matrix (bind / warn-bind / block / unbound), evaluated independently for each SAE the circuit references; a circuit is fully serveable only when all referenced SAEs bind."
      - "Live multi-SAE circuit serving: apply every member feature through ITS OWN layer's SAE, at the circuit's per-layer strength budgets, governed by a single global intensity (λ) — the multi-layer composition semantics miStudio authored and validated."
      - "Per-layer cluster-slice fallback: on a single-SAE deployment (or when only one referenced SAE binds), consume the circuit's per-layer mistudio.cluster-definition/v1 slice (the miStudio export-slices output) as an ordinary cluster import, so today's runtime is never left out."
      - "Evidence-rung surfacing: store and display each circuit's (and each edge's) EvidenceRung verbatim wherever active steering state is shown; the word 'causal' is forbidden below rung 2; activating a rung<2 circuit requires an explicit unvalidated acknowledgement; the rung travels to the MCP status surface and the Open WebUI dial."
      - "Circuit-aware live dial: an imported circuit exposed as a live influence control in Open WebUI (off / min / max, λ-scaled from the definition's intensity semantics; all layers scale together under the one λ) usable inside a real chat session against identical prompts, with per-request isolation and restore-on-completion."
      - "Edge-level co-activation sensing: extend cluster-scoped sensing from all-members-fire to EDGE-scoped — record when an upstream member firing is followed by its downstream partner firing, with the alone-vs-within-larger-set distinction and ±K token context; off by default, explicit per-circuit opt-in."
      - "New circuit tools on the existing unified MCP server (import, activate/deactivate, status, list, sensing readout), health-gated as a self-describing category, plus additive-only miLLM /api/circuits/* endpoints tracked in docs/mcp-contract.md (→ v1.1)."
    out_of_scope:
      - "Circuit discovery, validation, faithfulness testing, or authoring inside miLLM (miStudio's job; miLLM consumes graded artifacts)."
      - "Any change to the frozen mistudio.cluster-definition/v1 or mistudio.circuit-definition/v1 schemas (consumer-neutral contracts; miLLM consumes, never mutates)."
      - "Publishing circuit packs to Hugging Face from miLLM (authoring/producer side; stays in miStudio)."
      - "Gradient-based attribution consumption (attribution patching / integrated-gradients tier) — a future discovery tier, not a serving concern this increment."
      - "Cross-model circuit portability (a circuit binds to one model's SAE set)."
      - "Multi-user auth (unchanged v1.0 posture)."
      - "Automatic correction of cross-layer over-steering — hazards are surfaced, not auto-corrected (mirrors miStudio's detection-not-correction stance)."
    future_considerations:
      - "Joint cross-layer budget calibration (empirical γ-style fitting across layers) once per-layer v1 budgets have live field data."
      - "Publishing circuit packs to Hugging Face from miLLM (producer role) once consuming is proven."
      - "Attribution-tier consumption if/when miStudio adds gradient-based edge evidence."
      - "Circuit library / sharing / marketplace across the ecosystem."
      - "Sensing-driven authoring feedback: surface edge co-activation statistics back into miStudio to refine circuit membership, edges, and strengths."
      - "Feature-level (sub-cluster) hazard/steering granularity carried over from miStudio's recorded tech debt, once the circuit runtime is stable."
    dependencies:
      - "mistudio.circuit-definition/v1 interchange contract (published JSON Schema in the miStudio repo: docs/schemas/circuit-definition-v1.json; new kind mistudio.circuit-definition; per-layer SAE refs; members keyed to a layer/SAE; typed edges with evidence rung + validation status; per-layer budgets under one intensity; discovery provenance; no secrets, no filesystem paths)."
      - "mistudio.cluster-definition/v1 + the export-slices projection (miStudio renders each circuit layer as a valid single-SAE cluster-definition/v1 slice) — the fallback path for single-SAE deployments."
      - "EvidenceRung ladder as the single claims vocabulary (miStudio evidence_ladder.py: 0=mined, 1=attribution-supported, 2=causally-validated, 3=faithfulness-tested; 'causal' forbidden below rung 2) — carried verbatim, not reinterpreted."
      - "Shipped BRD-MILLM-CLUSTERS-001 runtime (dependency, not re-stated): cluster-definition/v1 import + compatibility matrix, materialization as a steering profile, the global/per-request intensity dial, cluster-scoped co-activation sensing, the unified miStudio-hosted MCP server, and the Open WebUI dial filter."
      - "miLLM v1.0 baseline + post-v1.0 runtime (shipped): single-SAE residual-stream hooking to be generalized to multi-SAE; OpenAI-compatible API with the steering_intensity extension; Feature Monitoring capture path (sensing piggybacks on it); K8s/ArgoCD GitOps; CBM continuous batching + speculative decoding (the latency-budget context for sensing)."
      - "miStudio MCP server (backend/src/mcp_server/) as the host for the new circuit tool category; docs/mcp-contract.md (v1.0 today) as the additive-only contract of record."
      - "Existing miStudio→miLLM coupling (miStudio uses miLLM as an OpenAI-compatible labeling backend) — must remain undisturbed."
    assumptions:
      - "The circuit's per-layer budgets and strengths are authoritative: imported budgets/strengths are FROZEN as authored, not recomputed against the local SAE set (mirrors miStudio's profile-load and the cluster-import semantics). Recompute-on-import is an explicit open question."
      - "A circuit is fully serveable only when all its referenced SAEs are compatible with attached SAEs (n_features match at minimum, per layer); a partially-compatible circuit degrades to the per-layer slice fallback rather than serving a wrong-decoder member."
      - "Open WebUI remains the reference chat front end; the circuit dial extends the existing OWUI filter and the steering_intensity OpenAI extension rather than introducing a new transport."
      - "Edge-level sensing piggybacks on the existing Feature Monitoring / cluster-sensing capture path; the upstream→downstream edge filter is a refinement, not a second inference pass."
      - "The multi-SAE VRAM envelope is bounded by loading only the referenced SAEs; the two-SAE case is the concrete close-out target (VRAM<200 MB, carried from miStudio's recorded FPRD §8 criteria)."

  business_requirements:
    - id: BR-001
      text: "miLLM SHALL attach multiple SAEs simultaneously, keyed by (SAE, layer), loading only the SAEs referenced by an imported circuit, and SHALL document and enforce a VRAM envelope for the attached set (with the two-SAE case measured as the concrete close-out number)."
    - id: BR-002
      text: "miLLM SHALL import mistudio.circuit-definition/v1 documents from user-provided JSON files, validating strictly against the published v1 circuit schema and rejecting unknown kinds or incompatible schema major versions with actionable errors."
    - id: BR-003
      text: "On import, miLLM SHALL evaluate compatibility per referenced SAE (bind / warn-bind / block / unbound, mirroring the cluster matrix) and SHALL treat a circuit as fully serveable only when all referenced SAEs bind; when they do not, miLLM SHALL fall back to the circuit's per-layer mistudio.cluster-definition/v1 slice rather than serve any member through a mismatched SAE."
    - id: BR-004
      text: "Activating a serveable circuit SHALL apply ALL member features simultaneously, each through ITS OWN layer's SAE, at the circuit's per-layer strength budgets under a single global intensity, with no manual tuning step — never applying members from multiple layers through one shared decoder."
    - id: BR-005
      text: "miLLM SHALL surface each circuit's and each edge's EvidenceRung verbatim wherever active steering state is shown (MCP status, Open WebUI dial, Admin UI), SHALL never describe rung-below-2 steering as 'causal', and SHALL require an explicit unvalidated acknowledgement to activate a circuit whose rung is below 2."
    - id: BR-006
      text: "An end user in a live Open WebUI chat session SHALL be able to control an imported circuit's influence with a simple dial (off / minimum / maximum, scaled per the definition's intensity semantics, all layers scaling together under one λ) and compare responses to identical prompts across dial positions within the same session, with per-request overrides isolated and prior steering restored on completion (including client disconnect)."
    - id: BR-007
      text: "miLLM SHALL record circuit EDGE-level co-activation events — the moments during inference when an upstream member's firing is followed by its downstream partner's firing — with each event carrying the alone-vs-within-larger-set distinction; edge sensing SHALL be off by default with explicit per-circuit opt-in."
    - id: BR-008
      text: "Recorded edge co-activation events SHALL be retrievable (API and UI) with enough context (timestamp, request association, upstream/downstream member activations, ±K token context, alone/within flag) to support the authoring-side question 'which edges actually fire in production?'."
    - id: BR-009
      text: "The existing unified MCP server SHALL expose a circuit tool category sufficient for an agent to inspect the attached-SAE set and circuit status, list circuits, import a circuit definition, activate/deactivate a circuit, and read edge co-activation sensing — health-gated and self-describing so a deployment without the circuit runtime presents no dead tools."
    - id: BR-010
      text: "Imported circuit definitions SHALL be treated strictly as data: miLLM SHALL never execute content from a definition, SHALL enforce size/count caps, and SHALL reject definitions containing filesystem paths or credential-like content, regardless of source (reusing the cluster-import posture)."
    - id: BR-011
      text: "On circuit activation miLLM SHALL surface cross-layer over-steering hazards (compounding and cancellation across layers), quantified from the circuit's validated effect sizes where present and labeled heuristic otherwise — surfacing the hazard for the user, not auto-correcting it (mirroring miStudio's hazards-v2 stance)."
    - id: BR-012
      text: "All new circuit endpoints and MCP tools SHALL be additive-only and tracked in docs/mcp-contract.md (advanced to v1.1), introducing no breaking change to the shipped cluster, profile, sensing, or health surfaces, and preserving the existing envelope and error-code conventions (with a circuit-specific error family, e.g. CIRCUIT_NOT_FOUND, SAE_SET_INCOMPLETE)."
    - id: BR-013
      text: "Multi-SAE serving and edge sensing SHALL NOT materially degrade the OpenAI-compatible inference path: attaching additional SAEs and enabling edge sensing on a circuit must keep serving within the increment's agreed latency budget (threshold to be fixed in the PRD; 'no user-perceivable degradation' is the business bar)."

  success_metrics:
    quantitative_metrics:
      - "Round-trip fidelity: 100% of a circuit's members applied through their own layer's SAE at exactly the authored per-layer strengths after miStudio→miLLM import (verifiable via the active-circuit surface)."
      - "Slice-fallback coverage: 100% of circuits remain steerable on a single-SAE deployment via their per-layer cluster-definition/v1 slice with zero runtime reconfiguration."
      - "Multi-SAE VRAM: two SAEs serve a live circuit within the documented envelope (two-SAE close-out reported as a measured number, framed as a measurement not a promise)."
      - "Evidence honesty: 0 instances of rung-below-2 steering labeled 'causal' anywhere in the runtime/UI/MCP; 100% of rung<2 activations gated behind the unvalidated acknowledgement."
      - "Edge sensing: 100% capture of a known upstream→downstream co-firing prompt set for an opted-in circuit with correct alone/within classification; inference latency overhead within the PRD-fixed budget."
      - "Unified MCP: correct tool set served in 3/3 topologies (both / miStudio-only / miLLM-only), verified by health-gated category lists; cluster suite + cluster-definition/v1 conformance unchanged."
    qualitative_indicators:
      - "A researcher describes moving a validated circuit from miStudio to miLLM as 'it just runs, and it still says how much I can trust it'."
      - "An Open WebUI user can feel and articulate the difference between circuit influence off / min / max on identical prompts, and can see whether the circuit is validated."
      - "Agent workflows (via the unified MCP) span discovery, validation, and serving of a circuit without switching servers."
    measurement_methods:
      - "E2E round-trip test: export a validated circuit from miStudio, import into miLLM, compare applied per-layer steering vectors/strengths against the definition."
      - "Multi-SAE VRAM harness: attach two SAEs, serve a circuit, record peak VRAM against the envelope."
      - "Playwright E2E for circuit import UI, activation with the unvalidated-rung acknowledgement, and the OWUI circuit dial; MCP topology matrix test (3 configurations)."
      - "Edge-sensing accuracy harness: scripted prompt panel with known upstream→downstream ground truth."
    # NOTE: the latency budget and the exact VRAM envelope figure are TBD pending PRD-level measurement.

  feature_themes:
    core_features:
      - "Multi-SAE Attach & Live Circuit Serving (referenced-only SAE loading; per-layer budgets under one global intensity; every member through its own layer's SAE)."
      - "Circuit Definition Import, Slice-Fallback & Evidence Ladder (circuit-definition/v1 import, per-referenced-SAE compatibility, per-layer cluster-slice fallback, verbatim rung surfacing with unvalidated-activation gate)."
    secondary_features:
      - "Circuit-Aware Open WebUI Dial (off/min/max live influence over a whole circuit, all layers under one λ, per-request isolation)."
      - "Circuit Edge-Level Sensing (upstream→downstream co-activation recording with alone/within side channel, opt-in)."
    future_features:
      - "Joint cross-layer budget calibration; HF circuit publishing from miLLM; attribution-tier consumption; circuit marketplace; edge-sensing→authoring feedback loop; feature-level hazard granularity."

  considerations:
    budget_constraints: "TBD"
    timeline_expectations: "TBD — sequencing preference: multi-SAE attach + VRAM spike first (proves the runtime + de-risks the highest-uncertainty item), then circuit import + slice fallback + evidence ladder, then the OWUI circuit dial and edge sensing."
    regulatory_or_policy_drivers:
      - "Evidence-integrity policy: honoring the EvidenceRung ladder verbatim (no 'causal' below rung 2) is a first-class product constraint, not a UI nicety — it prevents overclaiming at the point of live influence."
    technical_constraints:
      - "Multi-SAE attach relaxes the v1.0 single-SAE constraint: attachment state becomes per-(SAE, layer); only referenced SAEs are loaded, within a documented VRAM envelope (two-SAE close-out target VRAM<200 MB)."
      - "Steering semantics fixed to the validated contract: per-layer strength budgets travel inside the circuit definition and are not recomputed by miLLM; each member steers through its own layer's SAE decoder on the residual stream."
      - "The OpenAI-compatible API's per-request control channel (steering_intensity) is extended to a whole circuit under one λ; the OWUI filter is extended rather than replaced."
      - "Edge sensing must live inside the existing capture path (Feature Monitoring / cluster sensing / CBM) without a second inference pass, and stay within the CBM/speculative-decoding latency budget."
      - "All new surfaces are additive-only to preserve the shipped cluster/profile/sensing/health contract (docs/mcp-contract.md → v1.1)."
    integration_requirements:
      - "mistudio.circuit-definition/v1 (kind-keyed, versioned) as the primary interchange format, with mistudio.cluster-definition/v1 per-layer slices as the single-SAE fallback."
      - "EvidenceRung ladder (evidence_ladder.py vocabulary) carried verbatim to MCP + OWUI + Admin UI."
      - "MCP protocol (streamable HTTP, bearer auth, category gating) — the circuit category added to the inherited unified server; docs/mcp-contract.md advanced to v1.1 additively."
      - "Open WebUI as the reference chat client; the existing cluster dial filter is extended to circuits; existing miStudio→miLLM labeling integration must keep working."
    scalability_expectations: "Single-node, single-GPU serving posture retained; VRAM scales with the number of referenced SAEs (bounded by referenced-only loading and the documented envelope); edge-sensing storage bounded by opt-in scope and a retention policy (retention is an open question)."

  risks:
    - id: RSK-001
      description: "Attaching multiple SAEs exceeds the single-GPU VRAM ceiling, especially with CBM and the base model resident, making full multi-SAE circuit serving unaffordable on the target host."
      impact: "high"
      likelihood: "medium"
      mitigation: "Load only the SAEs a circuit references; document and enforce a VRAM envelope; make the two-SAE / VRAM<200 MB close-out an early spike and an acceptance gate; the per-layer slice fallback keeps circuits usable when the full set won't fit."
    - id: RSK-002
      description: "Cross-layer steering compounds (or cancels) unpredictably: applying budgets on multiple layers under one λ over-steers or produces incoherent generation."
      impact: "high"
      likelihood: "medium"
      mitigation: "Surface compounding/cancellation hazards at activation (BR-011), quantified from validated effect sizes where present; conservative default λ; per-request dial always able to reach off."
    - id: RSK-003
      description: "Edge-level co-activation sensing in the hot inference path (CBM/speculative decoding) degrades serving latency."
      impact: "high"
      likelihood: "medium"
      mitigation: "Opt-in per circuit, off by default; capture piggybacks on the existing monitoring/cluster-sensing hook; PRD fixes a hard latency budget with a measured baseline (BR-013)."
    - id: RSK-004
      description: "The evidence ladder is lost at the frontend: a mined (rung<2) circuit is presented to an end user or agent as if it were causally validated, overclaiming at the point of live influence."
      impact: "high"
      likelihood: "medium"
      mitigation: "Surface the rung verbatim everywhere steering state shows (BR-005); forbid 'causal' below rung 2; gate rung<2 activation behind an explicit unvalidated acknowledgement carried to MCP and OWUI."
    - id: RSK-005
      description: "Additive circuit endpoints/tools drift from the shipped cluster surface and strand or break the unified MCP server across the two products' release trains."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Contract-first, additive-only tool/endpoint definitions tracked in docs/mcp-contract.md (v1.1); per-product/per-category health gating degrades gracefully; the contract versions independently of both backends."
    - id: RSK-006
      description: "A circuit references an SAE that is not attached (or incompatible), and miLLM silently serves those members through the wrong SAE, producing meaningless steering."
      impact: "high"
      likelihood: "medium"
      mitigation: "Per-referenced-SAE compatibility hard-checked at import AND activation; an incomplete SAE set blocks full serving and degrades to the per-layer slice fallback (BR-003) — never a silent wrong-decoder path; SAE_SET_INCOMPLETE surfaced explicitly."
    - id: RSK-007
      description: "Terminology collision in one runtime: native profiles vs imported cluster profiles vs imported circuits confuses users and operators."
      impact: "low"
      likelihood: "high"
      mitigation: "PRD-level naming/UX decision with a copy audit before ship (precedent: miStudio's Clusters rename audit); the Admin UI surfaces circuit identity, layers, and rung distinctly."

  next_steps:
    open_questions:
      - "Multi-SAE VRAM: what is the target envelope and the eviction/retention policy when a new circuit references SAEs beyond what fits?"
      - "Per-request λ semantics across layers: one global intensity for the whole circuit (assumed) vs per-layer caps under the global dial — confirm before PRD."
      - "Edge-sensing granularity: what token-lag window defines 'upstream firing followed by downstream', and what is the retention/volume policy (per-token vs per-request)?"
      - "Frozen vs recomputed per-layer budgets on import: frozen is assumed (matches cluster-import semantics); confirm before PRD."
      - "Admin UI shape: a dedicated 'Circuits' page, or circuits surfaced inside the existing Profiles/Clusters pages?"
      - "Slice-fallback UX: how is a partially-serveable circuit (only some SAEs bound) presented so the user understands they are steering a per-layer projection, not the full circuit?"
    recommended_actions:
      - "Proceed to 0xcc/instruct/002_create-project-prd.md: add increment rows/sections to 000_PPRD|miLLM referencing this BRD."
      - "Run the multi-SAE attach + two-SAE VRAM spike early — it is the highest-uncertainty item and gates the core theme and the acceptance envelope."
      - "Define the circuit MCP tool category + /api/circuits/* contract jointly with the miStudio unified server (additive to docs/mcp-contract.md v1.1) before implementation."
    priority_for_clarification:
      - "Multi-SAE VRAM envelope + eviction policy (blocks the serving architecture)."
      - "Edge-sensing token-lag window + retention (blocks the sensing feature design)."
```
