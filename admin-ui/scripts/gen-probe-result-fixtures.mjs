#!/usr/bin/env node
/**
 * Generate the Admin UI's `mi.probe.result/v1` fixtures (038 FR-38.28, FTASKS 9.1).
 *
 *   node scripts/gen-probe-result-fixtures.mjs          # write src/test/fixtures/probe-result/
 *   node scripts/gen-probe-result-fixtures.mjs --check  # fail if any committed file differs
 *
 * One fixture per shape the event detail must render: a completed thresholded result, a
 * ranking-only result (no threshold, so no verdict), a skip, a failure, an interrupted partial,
 * a row migrated from before 038, and a stateless result that was never persisted. Each is
 * VALIDATED against the published schema (`docs/schemas/probe-result-v1.json`, generated from
 * miLLM's contract model) before it is written; an invalid fixture stops the script. The Vitest
 * test `probeResultFixtures.test.ts` validates them again, so a schema change that breaks a
 * fixture turns the UI suite red.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';

const here = dirname(fileURLToPath(import.meta.url));
const SCHEMA = resolve(here, '../../docs/schemas/probe-result-v1.json');
const OUT = resolve(here, '../src/test/fixtures/probe-result');

const base = {
  schema: 'mi.probe.result/v1',
  kind: 'probe',
  result_id: '0192f2a0-6c1e-7a3b-8000-00000000c0de',
  event_time: '2026-10-23T12:00:00.000001Z',
  event_time_source: 'finalized',
  completed_at: '2026-10-23T12:00:00.000001Z',
  origin: 'live',
  producer: { name: 'millm', version: '0.5.0' },
  migrated: false,
  request_id: 'chatcmpl-0123456789abcdef01234567',
  trace_id: '4bf92f3577b34da6a3ce929d0e0e4736',
  span_id: '00f067aa0ba902b7',
  batch_id: null,
  batch_line: null,
  session_id: 'ses_0123456789ab',
  turn_index: 2,
  behaviour: { key: 'b-example', version: 3 },
  set: null,
  outcome: 'completed',
  reason: null,
  persistence: 'persisted',
  source_event_id: 41,
  window: 'all',
  status: 'final',
  choice_index: null,
  input_index: null,
  probe_id: 'pr_0123456789ab',
  probe_name: 'example probe',
  detector_digest: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  detector_digest_source: 'producer',
  identity_algorithm: 'mistudio.probe-identity/1',
  definition_schema: 'mistudio.probe-definition/v1',
  target: 'positive_class',
  score_semantics: 'raw_margin',
  layer: 12,
  hook: 'resid_post',
  basis: 'residual',
  rule: 'mean',
  served_model: 'org/model-8b',
  model_revision: 'main',
  template_sha256: 'unknown',
  precision: 'bf16',
  engine: 'transformers',
  execution_mode: 'serial',
  steering_ref: { kind: 'none', id: null },
  definition_scope: 'all',
  n_scored_tokens: 37,
  scored: true,
  score: 1.25,
  probability: null,
  calibration_ref: null,
  threshold: 0.5,
  verdict: true,
  fires: true,
  comparator: '>=',
  decision_digest: '60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752',
  decision_revision: 2,
  threshold_revision: 4,
  decision: { calibration_id: 'cal_01', target_fpr: 0.01, provisional: false },
  rung: 2,
  rung_language: 'detects on held-out data',
  parity_status: 'passed',
  model_load_id: '7d2c3f9e-1b4a-5c6d-8e9f-0a1b2c3d4e5f',
  evaluation_ms: 0.42,
};

const skippedFields = { scored: false, score: null, verdict: null, fires: null, n_scored_tokens: 0 };

export const FIXTURES = {
  completed: base,
  'ranking-only': { ...base, threshold: null, verdict: null, fires: null, decision_digest: 'unknown' },
  skipped: { ...base, ...skippedFields, outcome: 'skipped', reason: 'judge_traffic' },
  failed: { ...base, ...skippedFields, outcome: 'failed', reason: 'encoder_failed' },
  'interrupted-partial': {
    ...base, outcome: 'interrupted', reason: 'client_disconnected', status: 'partial',
    verdict: null, fires: null, scored: false, score: 0.31, n_scored_tokens: 9,
  },
  migrated: {
    ...base, migrated: true, event_time_source: 'inserted', detector_digest: 'unknown',
    detector_digest_source: 'unknown', identity_algorithm: null, decision_digest: 'unknown',
    decision_revision: null, trace_id: null, span_id: null, evaluation_ms: null,
    model_load_id: 'unknown', session_id: null, turn_index: null,
  },
  'stateless-not-persisted': {
    ...base, origin: 'stateless', persistence: 'not_persisted', source_event_id: undefined,
    input_index: 0, request_id: null, session_id: null, turn_index: null,
  },
};

function serialise(value) {
  return JSON.stringify(value, (_k, v) => (v === undefined ? undefined : v), 2) + '\n';
}

export function validator() {
  const ajv = new Ajv2020({ strict: false, allErrors: true });
  return ajv.compile(JSON.parse(readFileSync(SCHEMA, 'utf8')));
}

function main() {
  const check = process.argv.includes('--check');
  const validate = validator();
  mkdirSync(OUT, { recursive: true });
  let failed = false;
  for (const [name, fixture] of Object.entries(FIXTURES)) {
    const body = serialise(fixture);
    if (!validate(JSON.parse(body))) {
      console.error(`${name}: ${JSON.stringify(validate.errors)}`);
      failed = true;
      continue;
    }
    const path = resolve(OUT, `${name}.json`);
    if (check) {
      let current = '';
      try {
        current = readFileSync(path, 'utf8');
      } catch {
        // missing → differs
      }
      if (current !== body) {
        console.error(`${name}.json is stale; run scripts/gen-probe-result-fixtures.mjs`);
        failed = true;
      }
    } else {
      writeFileSync(path, body);
    }
  }
  process.exit(failed ? 1 : 0);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
