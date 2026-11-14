/**
 * Help text is plain language (ENH-004): no internal ids, and every acronym spelled out at its
 * first use in a topic. Rules and reasons: ./plainLanguage.ts.
 *
 * ⚠ A guard over text fails OPEN: a checker that finds nothing and a checker that looks at
 * nothing both pass. So this file also proves the checker bites (a deliberately bad topic must
 * fail, and every rule must catch its own sample) and that it saw all of the real text.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { HELP_TOPICS } from '@/config/help';

import { EXPANSIONS, INTERNAL_RULES, plainLanguageProblems } from './plainLanguage';

describe('Help text is plain language', () => {
  it.each(HELP_TOPICS.map((t) => [t.id, t] as const))('%s', (_id, topic) => {
    expect(plainLanguageProblems(topic)).toEqual([]);
  });

  it('looked at all of it (a scan over nothing would pass too)', () => {
    expect(HELP_TOPICS.length).toBeGreaterThanOrEqual(13);
    const chars = HELP_TOPICS.flatMap((t) => [t.title, ...t.body, ...(t.steps ?? [])]).join(' ').length;
    expect(chars).toBeGreaterThan(5000);
  });
});

describe('the checker bites (negative controls)', () => {
  it('a deliberately bad topic fails: an internal requirement id and a bare acronym', () => {
    const problems = plainLanguageProblems({
      id: 'bad',
      title: 'Scores',
      body: ['As FR-009.1 requires, a probe reports its AUROC on held-out data.'],
    });
    expect(problems.some((p) => p.includes('FR-009.1'))).toBe(true);
    expect(problems.some((p) => p.includes('"AUROC"'))).toBe(true);
  });

  it('a known acronym used bare before it is spelled out fails', () => {
    expect(
      plainLanguageProblems({ id: 'bad', title: 'SAE', body: ['A sparse autoencoder (SAE) is attached.'] }),
    ).toHaveLength(1);
    expect(plainLanguageProblems({ id: 'bad', title: 'Cards', body: ['Each GPU is listed.'] })).toHaveLength(1);
  });

  it('both spelled-out forms pass, plurals included', () => {
    expect(plainLanguageProblems({ id: 'ok', title: 'Sparse autoencoders (SAEs)', body: ['An SAE.'] })).toEqual([]);
    expect(
      plainLanguageProblems({ id: 'ok', title: 'Cards', body: ['GPU (graphics processing unit) memory, per GPU.'] }),
    ).toEqual([]);
  });

  it.each(INTERNAL_RULES.map((r) => [r.id, r] as const))('rule %s catches its own sample', (_id, rule) => {
    expect(new RegExp(rule.pattern, rule.flags).test(rule.sample)).toBe(true);
    expect(plainLanguageProblems({ id: 'bad', title: 'x', body: [rule.sample] }).length).toBeGreaterThan(0);
  });

  it('every expansion is lower-case prose or a proper name, never another acronym', () => {
    for (const [acronym, expansion] of Object.entries(EXPANSIONS)) {
      expect(expansion, acronym).not.toMatch(/\b[A-Z]{2,}\b/);
    }
  });
});

describe("the internal-id rules are miForge's (one list for the suite)", () => {
  // Sibling found by relative path, overridable by MIFORGE_REPO (suite CLAUDE.md). This frontend's
  // CI checks out miLLM alone, so there the comparison skips, and says so.
  const forge = process.env.MIFORGE_REPO ?? resolve(__dirname, '../../../../../miForge');
  const rulesPath = resolve(forge, 'manual/publish-rules.json');
  const present = existsSync(rulesPath);
  if (!present) console.warn(`[plainLanguage] miForge not found at ${forge}; the rule comparison is SKIPPED`);

  it.skipIf(!present)('each miForge "internal" rule is copied here verbatim', () => {
    const forgeRules: { id: string; category: string; pattern: string; flags: string }[] = JSON.parse(
      readFileSync(rulesPath, 'utf8'),
    ).forbidden;
    const internal = forgeRules.filter((r) => r.category === 'internal');
    expect(internal.length).toBeGreaterThan(0);
    for (const r of internal) {
      const mine = INTERNAL_RULES.find((m) => m.id === r.id);
      expect(mine, r.id).toBeDefined();
      expect(mine?.pattern).toBe(r.pattern);
      expect(mine?.flags).toBe(r.flags);
    }
  });
});
