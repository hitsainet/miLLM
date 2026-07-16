# BRD-MILLM-CLUSTERS-001 — miLLM Cluster Runtime: Import, Unified MCP, Live Dial & Co-Activation Sensing

Incremental enhancement BRD produced via `0xcc/instruct/001_generate-brd.md`. Source material: the
original `0xcc/docs/miLLM_BRD_v1.0.md` (the shipped v1.0 baseline — "everything we started with"),
the post-v1.0 delivery record ("everything we've added": GitOps/K8s deployment, CBM continuous
batching + speculative decoding, hybrid/Mamba model support, Neuronpedia links, security hardening),
and the miStudio Clusters increment (BRD-MIS-CLUSTERS-001, features 012–014, closed 2026-07-16) whose
`future_considerations` deferred the miLLM side of the cluster ecosystem to this document.
Clarifying-question round completed with the product owner 2026-07-16; locked decisions:
**(1)** all four capabilities in scope (import, unified MCP, Open WebUI dial, activation sensing);
**(2)** incremental BRD — the shipped baseline is context/dependencies, not re-stated requirements;
**(3)** Hugging Face marketplace participation is **consume-only** this increment.

```yaml
brd:
  metadata:
    brd_id: BRD-MILLM-CLUSTERS-001
    project_name: "miLLM"
    version: "0.1"
    author: "Sean"
    last_updated: "2026-07-16"
    status: "draft"

  business_context:
    problem_statement: >
      Tuned steering clusters are trapped inside miStudio. miStudio can now discover clusters,
      compute validated combined-strength allocations, author them as named/narrated profiles, and
      export them as portable mistudio.cluster-definition/v1 JSON — but miLLM, the serving runtime,
      cannot consume that artifact. miLLM steers raw feature indices only: a user must re-enter
      member strengths by hand, results carry no cluster identity, there is no agent (MCP) surface
      on the miLLM side, no way to modulate a cluster live during a real chat session, and no way to
      observe when a cluster's members actually fire together in production traffic. The ecosystem
      authors artifacts it cannot yet run, dial, or sense.
    vision_statement: >
      miLLM becomes the RUNTIME half of the cluster ecosystem: miStudio authors and validates
      clusters; miLLM imports them (from files or the Hugging Face community), serves them as
      one-click steering profiles at their tuned strengths, exposes them to agents through a single
      unified MCP server that spans both products, lets end users dial a cluster's influence live in
      real chat (Open WebUI), and closes the loop by sensing when a cluster's members co-fire in
      real traffic — feeding observation back into authoring.
    primary_objectives:
      - "Make the portable cluster definition executable: import → steer as one unit, zero manual tuning."
      - "Give the ecosystem ONE agent surface: a unified MCP server that serves whichever back ends are present."
      - "Put a cluster's influence under the end user's hand in live chat (off/min/max dial) via Open WebUI."
      - "Close the authoring loop with cluster-scoped co-activation sensing in production traffic."
      - "Participate in the community exchange as a consumer: browse and import cluster packs published to Hugging Face."
    success_criteria:
      - "A cluster tuned in miStudio steers identically in miLLM after a file or HF import, with no manual strength entry."
      - "A single MCP endpoint serves a correct, self-describing tool set in all three topologies (both products / miStudio only / miLLM only)."
      - "An Open WebUI user can compare identical prompts at cluster influence off / min / max within one chat session."
      - "Cluster co-activation events are recorded with the alone-vs-within-larger-set distinction and are retrievable for analysis."

  stakeholders_users:
    primary_users:
      - "Interpretability researchers who author clusters in miStudio and want them running in a serving stack."
      - "miLLM operators/self-hosters who want ready-made, community-tuned steering behaviors without doing discovery."
      - "AI agents (via MCP) that tune, validate, and now DEPLOY clusters across both products."
    secondary_users:
      - "End users chatting through Open WebUI who experience (and dial) cluster-steered generation."
      - "Community members exchanging cluster packs on Hugging Face."
    stakeholders:
      - "Product owner (Sean) — ecosystem vision: miStudio authors, miLLM serves/senses."
      - "miStudio project — producer of the interchange artifact and current MCP server."

  scope_definition:
    in_scope:
      - "Import of mistudio.cluster-definition/v1 single definitions and mistudio.cluster-bundle/v1 bundles from local JSON files."
      - "Import compatibility evaluation against the locally attached model+SAE (bind / warn-bind / block / unbound semantics mirroring miStudio's matrix), with member-index bounds checks against the actual SAE."
      - "Materialization of an imported definition as a miLLM steering profile: members+tuned strengths become the steering configuration; name/narrative/display token, budget block (B, formula id, constants, λ intensity), and provenance are retained."
      - "One-action activation of an imported cluster profile so ALL members steer together at their tuned strengths (Neuronpedia-scale, residual-stream — the same semantics miStudio validated)."
      - "Hugging Face consume-only integration: browse public cluster packs (tag convention: mistudio, mistudio-cluster-definition; filtered by the currently loaded base model where possible), preview name/narrative/member count, and import anonymously; record hub provenance (repo@revision/path)."
      - "Unified MCP server: a single MCP endpoint exposing both miStudio and miLLM tool sets, with per-product health checks that enable/disable product-specific tool categories so a single-product deployment still presents a coherent, self-describing tool set. miLLM-side tools cover at minimum: model/SAE status, profile list/activate, cluster-definition import, sensing readout."
      - "Open WebUI live dial: an imported cluster exposed as a live influence control (off / min / max, λ-scaled from the definition's intensity semantics) usable inside a real chat session against identical prompts."
      - "Cluster-scoped combined activation sensing: record ONLY the moments when ALL of a cluster's member features fire together during inference, with a side channel distinguishing 'this cluster alone fired' from 'this cluster fired within a larger activation set'; events retrievable via API/UI for analysis."
    out_of_scope:
      - "Publishing cluster packs to Hugging Face from miLLM (authoring-side; stays in miStudio per the research report's division of labor)."
      - "Marketplace commercialization (payments, ratings, moderation) — Hub-native mechanisms suffice for now."
      - "Cluster discovery/authoring/tuning inside miLLM (miStudio's job; miLLM consumes)."
      - "Multiple concurrent SAEs (unchanged v1.0 constraint; a definition binds against the single attached SAE)."
      - "Multi-user auth (unchanged v1.0 posture)."
      - "Any change to the mistudio.cluster-definition/v1 schema (consumer-neutral contract is frozen at v1; v2 wishes recorded below)."
    future_considerations:
      - "Publishing packs to Hugging Face from miLLM (producer role) once consuming is proven."
      - "Cluster-definition marketplace commercialization (trading/sharing economy across models and SAE sets)."
      - "Schema v2: structured provenance.hub_ref {repo_id, revision, path} replacing the source_note string convention."
      - "Sensing-driven authoring feedback: surface co-activation statistics back into miStudio to refine cluster membership and strengths."
      - "Multi-SAE support, which would relax the single-attachment binding constraint for imports."
      - "Registering the format as a Hub library integration ('Use in miLLM/miStudio' button on repo pages)."
    dependencies:
      - "mistudio.cluster-definition/v1 interchange contract (published JSON Schema in the miStudio repo: docs/schemas/cluster-definition-v1.json; kind-keyed, ≤20 members, ≤50 defs/bundle, no secrets, no filesystem paths; budget block carries formula_id/constants incl. the empirically fitted γ=0 model and λ intensity ∈ [0,2])."
      - "miLLM v1.0 baseline (shipped): Model Management, OpenAI-compatible API (verified with Open WebUI), single-SAE Management with residual-stream hooking, Feature Steering (−200..+200 Neuronpedia scale), Feature Monitoring (activation capture + WS), Profile Management (named steering configs with model/SAE/layer metadata), Admin UI."
      - "Post-v1.0 additions (shipped): Kubernetes + ArgoCD GitOps deployment with selective image builds, CBM continuous batching + speculative decoding + torch.compile, hybrid/Mamba (GraniteMoEHybrid) model support, Neuronpedia feature links, hardened security posture — the runtime this increment builds on."
      - "miStudio MCP server (Feature 010 lineage, 38 tools incl. the profiles category) as the starting point for the unified server."
      - "Hugging Face Hub API for anonymous public reads (browse/import); tag convention per the HF-marketplace research (0xcc/docs cross-ref in miStudio repo)."
      - "Existing miStudio→miLLM coupling (miStudio uses miLLM as an OpenAI-compatible labeling backend) — must remain undisturbed."
    assumptions:
      - "The definition's tuned strengths are authoritative: imported budgets/strengths are FROZEN as authored, not recomputed against the local SAE (mirrors miStudio's own profile-load semantics). Recompute-on-import is an explicit open question."
      - "A definition is only steerable when its SAE reference is compatible with the attached SAE (n_features match at minimum); incompatible imports are kept as unbound/inactive rather than rejected outright."
      - "Open WebUI remains the reference chat front end (compatibility verified during v1.0); the dial mechanism may use OWUI extension points rather than the raw OpenAI API."
      - "miLLM's existing Profile entity is the natural landing shape for imported clusters (members→steering dict, narrative→description); naming/UX disambiguation between native profiles and imported cluster profiles is a PRD-level decision."
      - "Sensing piggybacks on the existing Feature Monitoring capture path; the cluster-scoped filter is a refinement, not a new capture pipeline."

  business_requirements:
    - id: BR-001
      text: "miLLM SHALL import mistudio.cluster-definition/v1 documents (single) and mistudio.cluster-bundle/v1 documents (multi) from user-provided JSON files, validating strictly against the published v1 schema and rejecting unknown kinds or incompatible schema major versions with actionable errors."
    - id: BR-002
      text: "On import, miLLM SHALL evaluate compatibility against the locally attached model and SAE and SHALL communicate the outcome honestly per item: bind (silent), bind-with-warnings (model/layer mismatch), block (feature-space mismatch — member indices would be meaningless), or import-as-unbound (no compatible SAE present, steerable after later binding)."
    - id: BR-003
      text: "An imported definition SHALL materialize as a named miLLM steering profile preserving the author's name, narrative, display token, member set with tuned strengths and signs, budget metadata (total budget, formula identity and constants, intensity λ), and provenance (origin, export timestamp, and — for Hub imports — repository, revision, and path)."
    - id: BR-004
      text: "Activating an imported cluster profile SHALL apply ALL member features simultaneously at their stored tuned strengths with no manual tuning step, producing steering behavior equivalent to the authoring system's validated Blended steering for the same model+SAE."
    - id: BR-005
      text: "The user SHALL be able to verify what an active cluster profile is doing: which members are applied and at what strengths, and the cluster identity (name) SHALL be visible wherever the active steering state is surfaced."
    - id: BR-006
      text: "miLLM SHALL let users browse publicly published cluster packs on Hugging Face (filtered by the community tag convention and, where possible, by the currently loaded base model), preview a pack's clusters (name, narrative, member count), and import selected definitions anonymously — no Hugging Face account or token required for public packs."
    - id: BR-007
      text: "Imported definitions SHALL be treated strictly as data: miLLM SHALL never execute content from a definition, SHALL enforce size/count caps, and SHALL reject definitions containing filesystem paths or credential-like content, regardless of source (file or Hub)."
    - id: BR-008
      text: "A single unified MCP server SHALL expose both miStudio and miLLM capabilities through one endpoint, performing per-product health checks and enabling/disabling product-specific tool categories accordingly, such that a deployment with only one product still presents a coherent, self-describing tool set with no dead tools."
    - id: BR-009
      text: "The unified MCP server SHALL expose miLLM-side tools sufficient for an agent to: inspect model/SAE status, list and activate steering profiles (including imported clusters), import a cluster definition, and read cluster co-activation sensing results."
    - id: BR-010
      text: "An end user in a live Open WebUI chat session SHALL be able to control an imported cluster's influence with a simple dial (off / minimum / maximum, scaled per the definition's intensity semantics) and compare responses to identical prompts across dial positions within the same session."
    - id: BR-011
      text: "miLLM SHALL record cluster-scoped combined activation events: ONLY the moments during inference when ALL members of a designated cluster fire together, ignoring unrelated concurrent activations, with each event carrying a distinction between 'cluster fired alone' and 'cluster fired within a larger activation set'."
    - id: BR-012
      text: "Recorded co-activation events SHALL be retrievable (API and UI) with enough context (timestamp, request association, member activations, alone/within flag) to support the authoring-side question 'what patterns should we monitor for?', and sensing SHALL be off by default with an explicit per-cluster opt-in."
    - id: BR-013
      text: "Sensing SHALL NOT materially degrade serving performance: enabling it on a cluster must keep the OpenAI-compatible inference path within the increment's agreed latency budget (threshold to be fixed in the PRD; 'no user-perceivable degradation' is the business bar)."

  success_metrics:
    quantitative_metrics:
      - "Round-trip fidelity: 100% of members applied at exactly the authored strengths after miStudio→miLLM import (verifiable via the active-profile surface)."
      - "Time-to-first-steer from a public HF pack: under 5 minutes from browse to steered generation, zero manual strength entry."
      - "Unified MCP: correct tool set served in 3/3 topologies (both / miStudio-only / miLLM-only), verified by health-gated category lists."
      - "Sensing: 100% capture of a known co-firing prompt set for an opted-in cluster with correct alone/within classification; inference latency overhead within the PRD-fixed budget."
    qualitative_indicators:
      - "A researcher describes moving a cluster from miStudio to miLLM as 'it just runs'."
      - "An Open WebUI user can feel and articulate the difference between dial off / min / max on identical prompts."
      - "Agent workflows (via the unified MCP) span authoring and serving without switching servers."
    measurement_methods:
      - "E2E round-trip test: export a validated cluster from miStudio, import into miLLM, compare applied steering vectors/strengths."
      - "Playwright E2E for import UI, profile activation, and dial flows; MCP topology matrix test (3 configurations)."
      - "Sensing accuracy harness: scripted prompt panel with known co-activation ground truth."
    # NOTE: baseline latency figures and the sensing overhead budget are TBD pending PRD-level measurement.

  feature_themes:
    core_features:
      - "Cluster Definition Import & Profile Bridge (file + Hugging Face consume-only, compatibility matrix, provenance)."
      - "Unified MCP Server (both products, health-gated categories, miLLM tool set)."
    secondary_features:
      - "Open WebUI Live Cluster Dial (off/min/max λ-scaled influence in live chat)."
      - "Cluster Co-Activation Sensing (all-members-fire recording with alone/within side channel)."
    future_features:
      - "HF publishing from miLLM; marketplace commercialization; schema v2 hub_ref; sensing→authoring feedback loop; multi-SAE binding."

  considerations:
    budget_constraints: "TBD"
    timeline_expectations: "TBD — sequencing preference: import/profile bridge first (proves the interchange), then unified MCP, then OWUI dial and sensing."
    regulatory_or_policy_drivers:
      - "Hugging Face terms of service for Hub consumption (anonymous read rate limits; best-effort free storage)."
    technical_constraints:
      - "Single attached SAE at a time (v1.0 constraint retained): definition binding is against the one attached model+SAE."
      - "Steering semantics fixed to the validated contract: raw Neuronpedia-scale strengths on the residual stream; the fitted budget model (γ=0) travels inside the definition and is not recomputed by miLLM."
      - "The OpenAI-compatible API has no native per-session control channel; the OWUI dial will require an Open WebUI extension point (function/pipe/param) — mechanism is an open question, not a committed design."
      - "Sensing must live inside the existing capture path (Feature Monitoring / CBM) without a second inference pass."
    integration_requirements:
      - "mistudio.cluster-definition/v1 + bundle (kind-keyed, versioned) as the sole interchange format."
      - "Hugging Face Hub API (anonymous list/download; tag convention: mistudio, mistudio-cluster-definition, sae-layer:<n>, sae-source:<repo>, base_model card field)."
      - "MCP protocol (streamable HTTP, bearer auth, category gating) — contract inherited from the miStudio server."
      - "Open WebUI as the reference chat client; existing miStudio→miLLM labeling integration must keep working."
    scalability_expectations: "Single-node, single-GPU serving posture unchanged; sensing storage bounded by opt-in scope and retention policy (retention is an open question)."

  risks:
    - id: RSK-001
      description: "SAE/layer/feature-space mismatch between an imported definition and the locally attached SAE produces meaningless or misleading steering."
      impact: "high"
      likelihood: "medium"
      mitigation: "Enforce the compatibility matrix at import AND at activation time (n_features hard block; model/layer warn); unbound imports are inert until explicitly bound."
    - id: RSK-002
      description: "No clean control channel exists in the OpenAI-compatible API for a live per-session dial; a clumsy mechanism would undermine the end-user experience."
      impact: "medium"
      likelihood: "high"
      mitigation: "Treat the OWUI mechanism as a PRD-level design spike (Functions/pipe vs request param vs side channel); business requirement fixes the UX semantics (off/min/max on identical prompts), not the transport."
    - id: RSK-003
      description: "Co-activation sensing in the hot inference path (CBM/speculative decoding) degrades serving latency."
      impact: "high"
      likelihood: "medium"
      mitigation: "Opt-in per cluster, off by default; capture piggybacks on the existing monitoring hook; PRD fixes a hard latency budget with a measured baseline (BR-013)."
    - id: RSK-004
      description: "The unified MCP server couples two products' release trains; a breaking change in either backend strands the shared surface."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Contract-first tool definitions; per-product health-gated categories degrade gracefully; the server versions independently of both backends."
    - id: RSK-005
      description: "Terminology collision: miLLM's existing native 'profiles' vs imported 'cluster profiles' confuses users (echo of the miStudio semantic_clusters lesson)."
      impact: "low"
      likelihood: "high"
      mitigation: "PRD-level naming/UX decision with a copy audit before ship (precedent: miStudio's Clusters rename audit script)."
    - id: RSK-006
      description: "Community packs on HF are unvetted; a hostile or junk definition wastes user trust even if it can't execute."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Strict schema validation + caps + no-execution posture (BR-007); provenance always displayed; curated collection as a future quality signal."

  next_steps:
    open_questions:
      - "OWUI dial mechanism: Open WebUI Function/pipe, request parameter, or management-API side channel? (Design spike in the PRD phase.)"
      - "Where does the unified MCP server live — miStudio repo, miLLM repo, or a third repo — and which team/train owns its deployment?"
      - "Sensing retention and volume policy: how long are co-activation events kept, and at what granularity (per-token vs per-request)?"
      - "Frozen vs recomputed budgets on import: frozen is assumed (matches miStudio load semantics); confirm before PRD."
      - "Does the miLLM Admin UI grow a dedicated 'Clusters' page, or do imported clusters live inside the existing Profiles page?"
    recommended_actions:
      - "Proceed to 0xcc/instruct/002_create-project-prd.md: add increment rows/sections to 000_PPRD|miLLM referencing this BRD."
      - "Run the OWUI dial design spike early — it is the highest-uncertainty item and gates the secondary theme."
      - "Define the unified MCP tool contract (categories, health semantics) jointly with the miStudio server before implementation."
    priority_for_clarification:
      - "Unified MCP ownership/deployment home (blocks architecture)."
      - "OWUI dial mechanism (blocks the live-dial feature design)."
```
