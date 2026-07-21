---
sidebar_position: 8
title: Circuits over MCP
---

# Circuits over MCP

Everything on the [Circuits](./circuits.md) page can also be driven by an AI agent, through
16 MCP tools in the `millm_circuits` category. The tools live in **miStudio's** MCP server —
miStudio is where circuits are authored, so it is also where the agent-facing surface lives —
and they call this runtime's REST API.

`docs/mcp-contract.md` in the miLLM repo is normative: every tool, its endpoint, and its
arguments. This page is the human orientation to what those tools let an agent do, and where
it will refuse.

:::info Why the tools live in the other repo
An agent that steers a model needs to reach both halves — the authoring side to find and
validate features, and the serving side to put them in production. Keeping one MCP server
avoids making the agent hold two connections and reconcile two vocabularies.
:::

## What an agent can do

| Group | Tools | Purpose |
|---|---|---|
| **Inspect** | `millm_circuit_status`, `millm_list_circuits`, `millm_circuit_claims` | What is serving right now, what is imported, which layers are claimed |
| **Serve** | `millm_activate_circuit`, `millm_deactivate_circuit`, `millm_set_circuit_intensity` | Put a circuit into production and dial it |
| **Move** | `millm_import_circuit`, `millm_export_circuit`, `millm_delete_circuit` | Bring a definition in, take it out, remove it |
| **Recover** | `millm_release_circuit_claims`, `millm_reconcile_circuit_claims` | Clear a stuck layer claim after an unclean shutdown |
| **Observe** | `millm_circuit_sensing_*` (5 tools) | Arm edge sensing and read co-firing events |

## The three refusals worth understanding

Agents encounter these constantly, and they mean different things.

### Contention — two circuits want the same layer

The unit of contention is the **layer**, not the feature. Steering is additive on the residual
stream, so two circuits touching layer 12 are writing to the same place regardless of which
features they name.

This is **overridable**. If you genuinely want both, activate with `allow_layer_overlap=true`
and the layer becomes *composed*.

:::warning Composition has a measured cost
Two steered layers at strength 5 destroyed generation entirely in our close-out test
(LFM2.5-1.2B-Instruct) — roughly two orders of magnitude below the ±200 clamp that nominally
bounds steering. **One model, one fixture: indicative, not exhaustive.** Treat composition as
something to measure on your model, not a setting to leave on.

A composed layer also **suppresses the `X-miLLM-Circuit-Rung` header**, because the evidence
rung of a circuit says nothing about what happens when another circuit is added on top of it.
:::

### Collision — the same layer AND the same feature

Two circuits steering feature 4211 on layer 12 is not contention, it is a collision, and it is
**never overridable**. The two strengths would silently sum into a value neither circuit
requested. An agent that retries a collision with `allow_layer_overlap=true` will be refused
again — this is deliberate, and the tool descriptions say so, because a retry loop is the
default failure mode otherwise.

### Stuck claims — an unclean shutdown

If the process died mid-serve, layer claims can outlive the circuit that made them. The symptom
is a contention refusal naming a circuit that is not actually serving. `millm_release_circuit_claims`
clears them.

## Where the tools deliberately stop

Some capabilities are **intentionally not exposed**, and this is enforced by a test rather than
left to convention:

- **Hub import and hub search have no tool.** A circuit references several SAEs by id. Importing
  one from a remote pack without checking those references would serve it against the wrong
  feature basis — the model steered by vectors that mean something else entirely. The REST
  endpoints exist; an agent must go through them deliberately, not stumble into it.

## Guards on destructive operations

Two tools can lose data, and both refuse by default:

- **`millm_delete_circuit`** refuses if the circuit is currently serving. Deleting it would stop
  live steering *and* destroy the definition. Export first if you may want it back, then pass
  `acknowledge_serving=true`. The check fails open — if the serving state cannot be read, the
  delete proceeds, so cleanup stays possible during an outage.
- **`millm_circuit_sensing_clear`** requires an explicit scope: either one `circuit_id` or
  `all_circuits=true`, never a bare call. There is no default, because the two meanings differ
  by everything and the wrong one is unrecoverable.

## Honest language about evidence

Circuit and edge claims carry an [evidence rung](./circuits.md#the-evidence-ladder), and the
word **"causal" is forbidden below rung 2** — enforced by a copy audit that fails the build, not
by reviewer vigilance. An agent relaying a rung-0 mined correlation must not describe it as a
causal finding, because the entire value of the ladder is that the distinction survives being
passed along.

Where a hazard is quantified from a heuristic rather than a measured effect size, it is labelled
`heuristic` and never presented as causal.

## Troubleshooting

**"Sensing is enabled but there are no events."** Expected for arbitrary feature indices — the
edges have to actually co-fire on your traffic. `millm_circuit_sensing_status` reports how many
edges are sensable and the observed overhead, which distinguishes "armed and quiet" from "not
armed".

**"The intensity dial returns `AMBIGUOUS_ACTIVE_CIRCUIT`."** More than one circuit is serving, so
"the active circuit" is not a single thing. Name the circuit explicitly.

**"The rung header is missing."** Either a layer is composed (see above), or the request did not
run an intervention at all.
