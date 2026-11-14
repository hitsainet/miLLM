/**
 * T01b (040 FR-040.1, CF-16, CF-25 m5): every `entry` and `guard` record of the legacy-identifier
 * export compiles as a JavaScript RegExp with its `js_flags`, and matches every literal it lists.
 * The pytest parser checks the Python `pattern`; this checks the `pattern_js` the UI guards use.
 * Reads the full variant when present, else the public one (the mirror carries only that).
 */
import { existsSync, readFileSync } from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';

const REPO = path.resolve(import.meta.dirname, '../../../..');
const DOCS = path.join(REPO, 'docs');
// Repository paths spelled whole, so tests/unit/deploy/test_frontend_ci_covers_repo_reads.py sees them.
const FILE = process.env.LEGACY_FILE
  ?? [path.join(REPO, 'docs/legacy-identifiers.md'), path.join(REPO, 'docs/legacy-identifiers.public.md')]
    .find((p) => existsSync(p));

interface PatternRecord {
  section: string;
  id: string;
  pattern_js: string;
  js_flags: string;
  literals?: string[];
}

function records(): PatternRecord[] {
  if (!FILE) throw new Error(`no legacy-identifier export under ${DOCS}: it is required`);
  const text = readFileSync(FILE, 'utf8');
  const block = text.split('```jsonl\n')[1]?.split('\n```')[0];
  if (!block) throw new Error(`${FILE}: no jsonl block`);
  return block.split('\n').filter((l) => l.trim()).slice(1).map((l) => JSON.parse(l) as PatternRecord)
    .filter((r) => r.section === 'entry' || r.section === 'guard');
}

describe('legacy-identifier export (T01b)', () => {
  it('compiles every pattern_js with its js_flags and matches every literal', () => {
    const all = records();
    expect(all.length).toBeGreaterThan(5);
    const failures: string[] = [];
    for (const r of all) {
      let rx: RegExp;
      try {
        rx = new RegExp(r.pattern_js, r.js_flags);
      } catch (e) {
        failures.push(`${r.id}: does not compile: ${String(e)}`);
        continue;
      }
      for (const lit of r.literals ?? []) {
        if (!rx.test(lit)) failures.push(`${r.id}: ${JSON.stringify(lit)} is not matched`);
      }
    }
    expect(failures).toEqual([]);
  });
});
