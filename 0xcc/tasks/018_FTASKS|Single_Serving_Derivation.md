# Task List: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** 📋 DOCS COMPLETE 2026-07-20 — implementation not started
**References:** `018_FPRD|Single_Serving_Derivation.md` · `018_FTDD|Single_Serving_Derivation.md` · `018_FTID|Single_Serving_Derivation.md`

## Relevant Files
- `millm/services/circuit_steering_engine.py` — NEW, the one derivation
- `millm/services/circuit_service.py` — sites at `:424` and `:799`; `_serving_members` (`:624`) moves out
- `millm/services/inference_service.py` — site at `:955`; `_sae_service_for_dial` (`:743`) DELETED
- `tests/unit/services/test_serving_characterization.py` — NEW, written FIRST
- `tests/unit/services/test_circuit_steering_engine.py` — NEW

## Tasks

- [ ] 1.0 Characterization FIRST (covers NFR-18.1) — **gate: do not start 2.0 until green**
  - [ ] 1.1 Pin `_serve_full`'s current behaviour: members applied, per-layer routing, edges passed
  - [ ] 1.2 Pin `set_intensity`'s re-serve behaviour
  - [ ] 1.3 Pin the dial's behaviour incl. the F14 R1-01 authored-basis rule and R2-01 snapshot source
  - [ ] 1.4 All three green against UNCHANGED code

- [ ] 2.0 The engine (covers FR-18.1, FR-18.2)
  - [ ] 2.1 `CircuitSteeringEngine(state)` — attachment registry only, no repository, no session
  - [ ] 2.2 Absorb `_serving_members` verbatim, preserving both-sources collection and dedupe
  - [ ] 2.3 `serve(definition, intensity)` calling `set_circuit_steering`; the sign rule stays where it is
  - [ ] 2.4 Unit tests incl. EC-18.3 (dupes), EC-18.4 (negative strength), EC-18.5 (both sources)

- [ ] 3.0 Migrate the call sites, ONE AT A TIME (covers FR-18.1)
  - [ ] 3.1 `_serve_full` (`:424`) → engine; suite green
  - [ ] 3.2 `set_intensity` (`:799`) → engine; suite green
  - [ ] 3.3 Dial (`:955`) → engine; suite green
  - [ ] 3.4 DELETE `_sae_service_for_dial` (`:743`) and the `SAEService.__new__` call — delete, do not wrap
  - [ ] 3.5 Structural test: `set_circuit_steering` has exactly ONE caller
  - [ ] 3.6 Structural test: no `__new__` constructor bypass in the serving path

- [ ] 4.0 Claim set (covers FR-18.3)
  - [ ] 4.1 `claim_set(definition)` derived from serving members, NOT `circuit.layers`
  - [ ] 4.2 Test that it matches the layers `serve` actually touches
  - [ ] 4.3 Test the column-vs-definition divergence case (the F14 R2-01 shape)

- [ ] 5.0 Verification
  - [ ] 5.1 Full backend suite green; F14 R1-01 and R2-01 regression tests still pass
  - [ ] 5.2 **Mutation (BR-005)**: removing dedupe, or both-sources collection, MUST fail a test
  - [ ] 5.3 EC-18.1: slice-fallback still routes through the cluster path

- [ ] 6.0 Feature Acceptance
  - [ ] 6.1 Verify FPRD §9 criteria 1–5 and all US/EC boxes one-by-one
  - [ ] 6.2 Update CLAUDE.md + PPRD Feature 18 status

## Coverage Audit
| FR | Tasks |
|---|---|
| FR-18.1 | 2.1–2.4, 3.1–3.3, 3.5 |
| FR-18.2 | 2.1, 3.4, 3.6 |
| FR-18.3 | 4.1, 4.2, 4.3 |

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle `/code-review`): ≥10 findings. **Watch:** `_serving_members` simplified during the move (dedupe or both-sources dropped); the sign rule relocated into the engine and double-negating; the claim set taken from `circuit.layers`; the bypass wrapped rather than deleted.
- [ ] Round 2 (attack Round 1's fixes + fresh angles): ≥10 findings. **Watch:** a call site migrated but its characterization test loosened to match; the engine acquiring a repository "just for one thing"; slice-fallback drifting into the engine; edges dumped differently at one site.
- [ ] Round 3 (`/review`, 4 perspectives): ≥10 findings. **Watch:** whether the structural tests actually fail when a second caller is added; whether the claim set is genuinely consumed by contention rather than recomputed; mutation coverage on the flattening rules.

## Acceptance evidence
_(to be completed at 6.0)_
