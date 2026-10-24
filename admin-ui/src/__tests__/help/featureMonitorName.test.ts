/**
 * The page at /monitoring has ONE name, "Feature Monitor" (ENH-004, D7): in the sidebar, the
 * header, the Help topic and the manual. It was "Feature Monitor" in the sidebar, "Feature
 * Probing" in the header and on the page's own control card, "View Probe" on the dashboard, and
 * "Feature Monitoring" / "Probe Monitoring" in the manual.
 *
 * Comment lines are skipped (they may record the old names); the manual's "Renamed" note, which
 * names the old titles on purpose, is the one allowed prose mention.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { navItems } from '@/components/layout/navItems';
import { helpTopic } from '@/config/help';

const REPO = resolve(__dirname, '../../../..');
const STALE = /Feature Probing|Probe Monitoring|Feature Monitoring|View Probe\b/;

function walk(dir: string, keep: (name: string) => boolean): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    if (e.name === 'node_modules' || e.name === '__tests__') return [];
    const path = join(dir, e.name);
    return e.isDirectory() ? walk(path, keep) : keep(e.name) ? [path] : [];
  });
}

function staleLines(files: string[]): string[] {
  const out: string[] = [];
  for (const file of files) {
    readFileSync(file, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        const t = line.trim();
        if (t.startsWith('//') || t.startsWith('*') || t.startsWith('/*') || t.startsWith('{/*')) return;
        if (/was called/.test(line)) return; // the manual's "Renamed" note
        if (STALE.test(line)) out.push(`${file.slice(REPO.length + 1)}:${i + 1}: ${t}`);
      });
  }
  return out;
}

const UI = walk(resolve(REPO, 'admin-ui/src'), (n) => /\.tsx?$/.test(n) && !/\.test\.tsx?$/.test(n));
const MANUAL = walk(resolve(REPO, 'manual/docs'), (n) => /\.mdx?$/.test(n));

describe('the Feature Monitor has one name', () => {
  it('the sidebar, the Help topic and the manual page title agree', () => {
    expect(navItems.find((i) => i.path === '/monitoring')?.label).toBe('Feature Monitor');
    expect(helpTopic('feature-monitor')?.title).toBe('Feature Monitor');
    const page = readFileSync(resolve(REPO, 'manual/docs/features/probe-monitoring.md'), 'utf8');
    expect(page).toMatch(/^title: Feature Monitor$/m);
    expect(page).toMatch(/^# Feature Monitor$/m);
  });

  it('no screen uses an old name', () => {
    expect(UI.length).toBeGreaterThan(50); // the walk found the source
    expect(staleLines(UI)).toEqual([]);
  });

  it('no manual page uses an old name outside the Renamed note', () => {
    expect(MANUAL.length).toBeGreaterThan(20);
    expect(staleLines(MANUAL)).toEqual([]);
  });

  it('the sweep bites: an old name on a code line is reported', () => {
    expect(STALE.test('<h3>Feature Probing</h3>')).toBe(true);
    expect(STALE.test('View Probe')).toBe(true);
    expect(STALE.test('Feature Monitor')).toBe(false);
  });
});
