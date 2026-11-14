/**
 * The generated `mi.probe.result/v1` fixtures validate against the PUBLISHED schema and against
 * miLLM's STRICT test schema (038 FR-38.28, FTASKS 9.1), and match what the generator would write
 * today — so a contract change that breaks a fixture, or a hand edit to one, turns this red.
 *
 * MUTATION CONTROLS: corrupt a fixture value (e.g. `outcome: "done"` with a reason pattern the
 * strict schema refuses); change a field in the generator without regenerating.
 */

import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Ajv2020 from 'ajv/dist/2020.js';

// The repository root: admin-ui/src/test/__tests__ → four levels up.
const ROOT = resolve(import.meta.dirname, '../../../..');
const FIXTURES = resolve(import.meta.dirname, '../fixtures/probe-result');

function compile(path: string) {
  const ajv = new Ajv2020({ strict: false, allErrors: true });
  return ajv.compile(JSON.parse(readFileSync(resolve(ROOT, path), 'utf8')));
}

const files = readdirSync(FIXTURES).filter((f) => f.endsWith('.json')).sort();

describe('probe-result fixtures', () => {
  it('are the seven shapes the UI renders', () => {
    expect(files).toEqual([
      'completed.json', 'failed.json', 'interrupted-partial.json', 'migrated.json',
      'ranking-only.json', 'skipped.json', 'stateless-not-persisted.json',
    ]);
  });

  const open = compile('docs/schemas/probe-result-v1.json');
  const strict = compile('tests/fixtures/contracts/probe-result-v1.strict.json');

  it.each(files)('%s validates against the published and the strict schema', (name) => {
    const fixture = JSON.parse(readFileSync(resolve(FIXTURES, name), 'utf8'));
    expect(open(fixture), JSON.stringify(open.errors)).toBe(true);
    expect(strict(fixture), JSON.stringify(strict.errors)).toBe(true);
  });

  it('match what the generator writes', async () => {
    const gen = await import('../../../scripts/gen-probe-result-fixtures.mjs');
    for (const [name, value] of Object.entries(gen.FIXTURES as Record<string, unknown>)) {
      const committed = JSON.parse(readFileSync(resolve(FIXTURES, `${name}.json`), 'utf8'));
      expect(committed).toEqual(JSON.parse(JSON.stringify(value)));
    }
  });

  it('the strict schema refuses a skip dressed as a negative verdict', () => {
    const skipped = JSON.parse(readFileSync(resolve(FIXTURES, 'skipped.json'), 'utf8'));
    expect(strict({ ...skipped, verdict: false })).toBe(false);
  });
});
