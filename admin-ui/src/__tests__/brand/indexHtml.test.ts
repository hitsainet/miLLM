/**
 * The page a browser loads is named miServer (040 FR-040.30, T27): the tab title, the description
 * and the icon, and the icon file the link names exists under `public/`.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const ADMIN_UI = resolve(__dirname, '../../..');
const html = readFileSync(resolve(ADMIN_UI, 'index.html'), 'utf8');

describe('index.html', () => {
  it('titles the page miServer and uses the miserver icon', () => {
    const page = new DOMParser().parseFromString(html, 'text/html');
    expect(page.title).toBe('miServer');
    expect(page.querySelector('meta[name="description"]')?.getAttribute('content')).toBe(
      'miServer - Mechanistic Interpretability LLM Server');
    const icon = page.querySelector('link[rel="icon"]')?.getAttribute('href');
    expect(icon).toBe('/miserver-icon.svg');
    expect(existsSync(resolve(ADMIN_UI, 'public', 'miserver-icon.svg'))).toBe(true);
    expect(existsSync(resolve(ADMIN_UI, 'public', 'millm-icon.svg'))).toBe(false);
  });
});
