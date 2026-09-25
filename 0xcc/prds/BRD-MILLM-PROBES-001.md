# BRD: Probe Monitor Runtime

**Specified in:** `~/app/enhance/specs/ENH-001-probe-monitors` (handoff 2026-09-25)

**Document ID:** BRD-MILLM-PROBES-001
**Status:** Draft v0.2, 2026-09-25. Split from the combined draft v0.1; clarifying rounds held;
decisions locked (§4).
**Specified in:** `~/app/enhance/specs/ENH-001-probe-monitors` (planning workspace)
**Sibling BRD:** **BRD-MIS-PROBES-001** (miStudio: Probe Monitors — Creation, Evaluation and Export),
which produces the artifact this BRD consumes.
**Depends on:** the `mistudio.probe-definition/v1` contract, **owned by miStudio**, vendored here as
`docs/schemas/probe-definition-v1.json`. This BRD cites the contract; it doesn't restate it.
**Becomes:** Feature 024 *Probe Monitor Runtime* (021 and 022 are reserved; confirmed at handoff)
**Related:** F11 Co-Activation Sensing, F15 Circuit Edge Sensing, F20 MCP Circuit Surface, F023 GGUF
Serving, `docs/mcp-contract.md`

---

## 1. Why

miStudio will train **probe monitors**: small classifiers on a model's residual activations that
detect a concept, such as a high-stakes request. It exports each one as a self-describing
definition. A probe is only useful where the traffic is, and the traffic is in miLLM.

Running one costs almost nothing, because the activations it reads are computed by the forward pass
anyway. The published systems use probes as the **cheap first stage** that screens every request
(Anthropic's Constitutional Classifiers++; DeepMind). They also show a probe is **not a guarantee**:
adversarial inputs can suppress it (Bailey et al.). So miLLM must report what a probe says honestly,
at the evidence level miStudio established.

miLLM already reads activations on live traffic (F11 sensing, F15 edge sensing), but every hook it
has requires an attached SAE, and nothing can run a probe.

## 2. What

**miLLM runs probe monitors on live traffic.** It:
- imports a probe definition, refusing one built for a different model
- scores requests with a read-only hook that doesn't depend on SAE attachment (dense probes need no
  SAE; SAE probes use a private encoder copy)
- records each verdict and shows it live
- puts the verdict on the response
- proves on import that it scores exactly as miStudio did (the definition's test vectors)

Acting on a verdict (stopping a generation, escalating to an LLM judge) is **not** in this BRD
(D2). It is the later phase that the contract already anticipates.

## 3. Feature and order

| Feature | Covers | Order |
|---|---|---|
| **024 Probe Monitor Runtime** | BR-001 – BR-010 | after miStudio Feature 033 has published the v1 schema; it can be built against a fixture definition before then |

## 4. Locked decisions (clarifying rounds, 2026-09-24/25)

| # | Decision |
|---|---|
| D2 | **Observe, then act.** This version records, shows and annotates. Stopping and escalating come later, once thresholds are proven on real traffic. |
| D7 | **Evidence rungs** (0 trained · 1 held-out · 2 unseen kinds of tasks · 3 judge-compared) travel in the definition. miLLM shows the rung and never claims more than it. Arming a probe below rung 2 requires the acknowledgement recorded in the definition, plus an operator acknowledgement, mirroring circuit activation. |
| D8 | **Named "Probe Monitors".** The existing Admin UI page labelled "Probe" (SAE feature monitoring) is renamed **"Feature Monitor"**. |
| D11 | **Parity runs on LFM2.5-1.2B**, the model miStudio and miLLM share. |
| D14 | **SAE probes run in miLLM too** (user, 2026-09-25). miLLM runs an SAE probe with a **private copy of the encoder columns it uses**. The SAE must be downloaded in miLLM, but it needn't be attached, and the probe never steers. |
| D13 | **Full chain handed off:** FPRD, FTDD, FTID and FTASKS are specified by the planning workspace (user, 2026-09-25). |

## 5. Business requirements

**BR-001: A probe is imported with a strict compatibility check.** miLLM imports a definition from a
file, from HF (browsing the `mistudio-probe-definition` tag), or through MCP. Import validates the
vendored v1 schema. **Arming** refuses a definition whose model id, revision, `d_model`, layer count
or chat-template hash doesn't match the loaded model, and the refusal names the mismatch. Circuits can
bind despite a mismatch; probes can't.

**BR-002: Probes don't depend on SAE attachment.** miLLM reads the residual stream at the probe's
layer through its own read-only hook. It reads the **pre-steering** residual, so steering can't
change what a probe sees. Dense probes need no SAE at all. SAE probes use their own encoder copy
(BR-010), never an attached SAE.

**BR-003: Scoring follows the sensing lifecycle.** Probes are armed, scored on each forward pass for
the definition's token scope, combined by its rule, compared with its threshold, collected, recorded,
pruned and emitted on the socket. This reuses F11's request-scoped context, per-request caps and
overhead accounting.

**BR-004: The verdict is visible.**
- Every scored request records an event: probe, score, threshold, verdict, rung, and the positions
  that fired.
- The Admin UI shows events live on a **Probe Monitors** page, and the old "Probe" page becomes
  "Feature Monitor".
- A non-streaming chat response carries the verdict in an `X-miLLM-Probe-*` header. A streaming
  response carries it in a **final SSE event** before `[DONE]`.

**BR-005: miLLM proves parity with miStudio.** On import and on arming, miLLM scores the definition's
test vectors and compares its results with miStudio's within a stated tolerance. A definition that
fails parity **can't be armed**, and the failure report names the largest deviation.

**BR-006: Unsupported serving paths say so.** GGUF (unhookable) models refuse to arm. Speculative
decoding and continuous batching either force the serial path, as sensing does, or report the probe
as paused. **A probe never goes silently quiet**: status always says why it isn't scoring.

**BR-007: The rung is shown.** The evidence rung, and any acknowledgement, appears wherever a probe is
armed, listed or reported, under the same no-overclaiming rule as the circuit rung ladder.

**BR-008: An MCP surface.** A `millm_probes` category (import, list, arm, disarm, status, events) is
added to `docs/mcp-contract.md` without changing anything existing, with new error codes, and is
covered by the cross-repo registry check. The matching tools are registered in miStudio's MCP server.

**BR-010: SAE probes run with a private encoder copy.** For an SAE probe, arming requires the
referenced SAE to be downloaded in miLLM (through the existing SAE management), and to match the
definition's repo, revision, weights hash, architecture, `d_model` and feature count. miLLM loads
**only the encoder columns of the probe's k features** onto the layer's card, and encodes the
pre-steering residual with the SAE's own activation function and normalization. The SAE is never
attached and never steers, so attaching or detaching SAEs for steering can't affect the probe.
Missing or mismatched SAEs refuse by name. Parity (BR-005) covers the SAE path.

**BR-009: Overhead stays inside budget.** Probe scoring stays within the sensing overhead budget
(NFR-1.4) at the reference context lengths, with one device-to-host copy per forward pass, and its
overhead is reported in status.

## 6. Contract

It is consumed, not defined here. See miStudio's `docs/schemas/probe-definition-v1.json`, vendored
byte-identical into `docs/schemas/`, with a pydantic mirror in `millm/api/schemas/probe.py` and a
sync test, following the cluster and circuit pattern: **fix the mirror, never the vendored file**.

## 7. Out of scope

- **Acting on a verdict** (stop, escalate, re-route): D2's later phase.
- **Probes on GGUF models:** they can't be hooked (F023 §3).
- **Training or evaluating probes:** that is miStudio.
- **Batched probe scoring under continuous batching:** the serial path is used.
- **Probe verdicts in the Open WebUI filter:** the filter has no outlet; a later enhancement.

## 8. Acceptance

The input is miStudio's Stage 2 export: an LFM2.5-1.2B probe at rung 2 or above, with test vectors.
- It imports from file and from HF, and **passes parity**.
- Arming against a different model is refused by name. Arming against a GGUF model is refused.
- Live requests record events, update the Probe Monitors page live, and carry the verdict (header on
  non-streaming, final SSE event on streaming).
- Overhead stays inside NFR-1.4 at about 4k-token contexts.
- miStudio's k-sparse **SAE probe** for the same model also imports, passes parity and arms with the
  SAE downloaded but not attached. It is refused, naming the SAE, when that SAE is missing.
- The MCP tools work end to end from miStudio's MCP server.

Every wiring requirement (hook installation, event recording, header and SSE emission, routes, MCP
registration, sidebar entry) is accepted only by a test that **fails when that wiring line is
removed**, asserting the payload, not just the call.

## 9. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| RSK-001 | miLLM's scores drift from miStudio's (template, revision, dtype, layer indexing) | high | medium | Parity gate (BR-005); strict identity check (BR-001) |
| RSK-002 | Probe scoring breaks latency budgets | medium | low | Sensing's overhead accounting; one device-to-host copy per pass; budget test |
| RSK-003 | A probe goes quiet on an unsupported path without anyone noticing | high | medium | BR-006 status reasons, and tests on every path |
| RSK-004 | The "Probe" → "Feature Monitor" rename breaks links or tests | low | medium | Route stays; the label changes; tests updated |
| RSK-005 | The v1 contract changes under miLLM | medium | low | Vendored byte-identical schema plus sync test; additive-only |

## 10. Open questions

1. Parity tolerance: an absolute score difference of 1e-3 at fp16 is proposed; confirm in the FPRD.
2. Whether probe events share the `sensing_events` table or get their own (FTDD decision).
