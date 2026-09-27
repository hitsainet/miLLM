/**
 * The sidebar must distinguish the two monitoring pages by NAME (D8).
 *
 * Feature 24 added "Probe Monitors" — trained linear detectors. The existing page at
 * `/monitoring` watches SAE feature activations and was called "Probe". Two entries both called
 * Probe is a coin flip for an operator, and the failure is silent: they read one page's numbers
 * believing them to be the other's.
 *
 * This lives in vitest rather than only in Playwright because it runs in CI on every push, and
 * the thing it guards is a one-word edit anyone could make.
 */

import { describe, expect, it } from 'vitest';
import { navItems } from '../navItems';

describe('sidebar naming', () => {
  const byId = (id: string) => navItems.find((item) => item.id === id);

  it('the SAE feature page is called Feature Monitor, not Probe', () => {
    const item = byId('monitoring');
    expect(item?.label).toBe('Feature Monitor');
  });

  it('keeps the /monitoring URL so bookmarks and runbooks still work', () => {
    expect(byId('monitoring')?.path).toBe('/monitoring');
  });

  it('Probe Monitors is its own entry', () => {
    const item = byId('probe-monitors');
    expect(item?.label).toBe('Probe Monitors');
    expect(item?.path).toBe('/probe-monitors');
  });

  it('no two entries share a label', () => {
    const labels = navItems.map((item) => item.label);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it('exactly one entry is called Probe Monitors and none is called just "Probe"', () => {
    expect(navItems.filter((i) => i.label === 'Probe')).toHaveLength(0);
    expect(navItems.filter((i) => i.label === 'Probe Monitors')).toHaveLength(1);
  });
});
