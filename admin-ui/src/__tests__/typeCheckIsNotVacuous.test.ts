/**
 * The type-check must actually look at the source.
 *
 * `tsconfig.json` here is solution-style — `"files": []` plus project
 * references — so a bare `tsc --noEmit` type-checks ZERO files and exits 0
 * unconditionally. On 2026-09-07 that produced three "tsc clean" reports in a
 * row over code that did not compile: `ModelLoadForm` passed a `helperText`
 * prop that `Select` does not accept, and it was caught only by CI and by the
 * admin-ui image build, both of which run `tsc -b` via `npm run build`.
 *
 * A check that inspects nothing and reports success is worse than no check: it
 * converts "unverified" into "verified".
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * change build to `tsc --noEmit && vite build` -> "build type-checks" fails
 *   * empty the references array                   -> "references reach the app" fails
 */

import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = join(__dirname, '..', '..');

describe('the type-check is not vacuous', () => {
  it('build type-checks in BUILD mode, not bare --noEmit', () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf-8'));
    const build: string = pkg.scripts.build;

    expect(build).toContain('tsc -b');
    expect(build).not.toMatch(/tsc\s+--noEmit(?!\S)/);
  });

  it('the root config delegates to references, and they reach the app', () => {
    const cfg = JSON.parse(readFileSync(join(ROOT, 'tsconfig.json'), 'utf-8'));

    // If this ever gains its own `files`/`include`, bare tsc would work and the
    // reasoning above changes — so assert the shape we actually have.
    expect(cfg.files).toEqual([]);
    expect(Array.isArray(cfg.references)).toBe(true);
    expect(cfg.references.length).toBeGreaterThan(0);
    expect(cfg.references.map((r: { path: string }) => r.path)).toContain(
      './tsconfig.app.json',
    );
  });

  it('bare `tsc --noEmit` really does see nothing (why this file exists)', () => {
    // Pinning the trap itself. If a future tsconfig change makes bare tsc
    // meaningful, this fails and the comment above should be revised.
    const out = execFileSync(
      'npx',
      ['tsc', '--noEmit', '--listFiles'],
      { cwd: ROOT, encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'] },
    );
    expect(out.trim()).toBe('');
  }, 120_000);
});
