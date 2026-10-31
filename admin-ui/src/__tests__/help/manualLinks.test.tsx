/**
 * Every link into the manual lands on a page and a heading that exist (ENH-004).
 *
 * Origin: miDataworks frontend/src/test/manualLinks.test.tsx @ 32cf406. miLLM's manual pages are
 * all hand-written under manual/docs (no generated pages, no allow-list), and the site is served
 * from the custom domain in manual/docusaurus.config.ts.
 *
 * These checks read the manual's own sources, so a renamed page or heading fails here rather
 * than as a 404 in a user's browser.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { HelpLink } from '@/components/help';
import { Sidebar } from '@/components/layout/Sidebar';
import { bottomNavItems, navItems } from '@/components/layout/navItems';
import { HELP_FOR_PATH, HELP_TOPICS, helpTopic, helpTopicForPath } from '@/config/help';
import { DEFAULT_MANUAL_BASE_URL, MANUAL_BASE_URL, manualBaseUrl, manualUrl } from '@/config/manual';
import { HelpPage } from '@/pages/HelpPage';

const REPO = resolve(__dirname, '../../../..');
const DOCS = resolve(REPO, 'manual/docs');
const read = (p: string) => readFileSync(resolve(REPO, p), 'utf8');

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : /\.mdx?$/.test(e.name) ? [join(dir, e.name)] : [],
  );
}

/** The source file Docusaurus serves at a manual path (routeBasePath "/"): a front-matter `slug`
 *  wins, otherwise the file's path under manual/docs without its extension. */
function sourceOf(page: string): string | undefined {
  const path = page.split('#')[0].replace(/^\/+|\/+$/g, '');
  for (const file of walk(DOCS)) {
    const front = readFileSync(file, 'utf8').match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
    const slug = front.match(/^slug:\s*(\S+)\s*$/m)?.[1];
    const route = slug
      ? slug.replace(/^\/+|\/+$/g, '')
      : file.slice(DOCS.length + 1).replace(/\.mdx?$/, '');
    if (route === path) return file;
  }
  return undefined;
}

/** Heading ids as Docusaurus makes them (github-slugger): lower case, punctuation dropped, each
 *  space to "-". An explicit {#id} wins. */
function slug(heading: string): string {
  return heading
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{M}\p{N}\p{Pc}\- ]/gu, '')
    .replace(/ /g, '-');
}

function headingSlugs(markdown: string): string[] {
  let fenced = false;
  const out: string[] = [];
  for (const line of markdown.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    const m = !fenced && line.match(/^#{2,6}\s+(.*?)\s*$/);
    if (!m) continue;
    const explicit = m[1].match(/\{#([^}]+)\}\s*$/);
    out.push(explicit ? explicit[1] : slug(m[1].replace(/&amp;/g, '&')));
  }
  return out;
}

describe('the manual address comes from config', () => {
  it("the default agrees with the published site (manual/docusaurus.config.ts url + baseUrl)", () => {
    const config = read('manual/docusaurus.config.ts');
    const url = config.match(/^\s*url:\s*'([^']+)'/m)?.[1];
    const baseUrl = config.match(/^\s*baseUrl:\s*'([^']+)'/m)?.[1];
    expect(url && baseUrl).toBeTruthy();
    expect(DEFAULT_MANUAL_BASE_URL).toBe(`${url}${baseUrl}`);
    expect(MANUAL_BASE_URL).toBe(DEFAULT_MANUAL_BASE_URL);
  });

  it('a configured base replaces the default host, with or without a trailing slash', () => {
    expect(manualBaseUrl('https://docs.example.test/millm')).toBe('https://docs.example.test/millm/');
    expect(manualBaseUrl('  ')).toBe(DEFAULT_MANUAL_BASE_URL);
    expect(manualBaseUrl(undefined)).toBe(DEFAULT_MANUAL_BASE_URL);
    expect(manualUrl('features/circuits#serving-a-circuit', 'https://docs.example.test/m/')).toBe(
      'https://docs.example.test/m/features/circuits#serving-a-circuit',
    );
  });

  it('the image build can pass the override through (admin-ui/Dockerfile)', () => {
    const dockerfile = read('admin-ui/Dockerfile');
    const arg = dockerfile.indexOf('ARG VITE_MANUAL_BASE_URL');
    expect(arg).toBeGreaterThan(-1);
    expect(dockerfile.indexOf('RUN npm run build')).toBeGreaterThan(arg);
  });

  it("the sidebar's book icon opens the configured manual, not a hard-coded host", () => {
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>,
    );
    const link = screen.getByTestId('sidebar-manual');
    expect(link).toHaveAttribute('href', MANUAL_BASE_URL);
    expect(link).toHaveAttribute('href', 'https://docs-millm.hitsai.net/');
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('the manual build fails on a link to a heading that does not exist', () => {
    expect(read('manual/docusaurus.config.ts')).toMatch(/^\s*onBrokenAnchors:\s*'throw',/m);
  });
});

describe('every help topic names a manual page and section that exist', () => {
  it.each(HELP_TOPICS.map((t) => [t.id, t.manual] as const))('%s -> %s', (_id, page) => {
    expect(page).toBeTruthy();
    const source = sourceOf(page);
    expect(source, `no manual page is served at ${page.split('#')[0]}`).toBeDefined();
    const anchor = page.split('#')[1];
    if (anchor) expect(headingSlugs(readFileSync(source!, 'utf8'))).toContain(anchor);
  });

  it('the page lookup is not vacuous: it finds known pages and refuses a made-up one', () => {
    expect(walk(DOCS).length).toBeGreaterThan(20);
    expect(sourceOf('')).toMatch(/intro\.md$/); // slug: /
    expect(sourceOf('troubleshooting')).toMatch(/troubleshooting\.md$/);
    expect(sourceOf('features/no-such-page')).toBeUndefined();
    expect(headingSlugs(read('manual/docs/features/circuits.md'))).not.toContain('no-such-heading');
  });

  it('the slug rule matches the built site on "&", "λ" and "↔"', () => {
    // Ids read from a real `npm run build` of the manual on 2026-10-20.
    expect(slug('Loading & Unloading')).toBe('loading--unloading');
    expect(slug('Activating & the intensity dial (λ)')).toBe('activating--the-intensity-dial-λ');
    expect(slug('SAE ↔ model compatibility')).toBe('sae--model-compatibility');
    expect(slug('Arming: four gates')).toBe('arming-four-gates');
  });
});

describe('every screen has a Help topic', () => {
  it('each nav path except Help itself maps to a topic that exists', () => {
    for (const item of [...navItems, ...bottomNavItems]) {
      if (item.path === '/help') continue;
      const topic = helpTopicForPath(item.path);
      expect(topic, `${item.path} has no Help topic`).toBeDefined();
      expect(helpTopic(topic), `${item.path} -> ${topic}`).toBeDefined();
    }
  });

  it('every mapped topic exists, and Help itself has none', () => {
    for (const [path, topic] of Object.entries(HELP_FOR_PATH)) expect(helpTopic(topic), path).toBeDefined();
    expect(helpTopicForPath('/help')).toBeUndefined();
    expect(helpTopicForPath('/help/models')).toBeUndefined();
    expect(helpTopicForPath('/')).toBe('dashboard');
  });

  // No screen links to these three, so the checks above would pass with any of them gone. They
  // are the spec's (ENH-004 §3.2, D6): a first-run topic, a "when something fails" topic, and the
  // "Words used here" glossary on the manual's vocabulary section. Linkable at /help/<id>.
  it.each([
    ['getting-started', 'Getting started', 'getting-started/quickstart'],
    ['errors', 'When something fails', 'troubleshooting'],
    ['glossary', 'Words used here', 'concepts/interpretability#vocabulary'],
  ])('the topic no screen links to is still there: %s', (id, title, manual) => {
    expect(helpTopic(id), id).toMatchObject({ id, title, manual });
  });

  it('topic ids are unique', () => {
    const ids = HELP_TOPICS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('the Help page links each summary to its manual page', () => {
  it('one "Read more in the manual" link per topic, to that topic\'s page, in a new tab', () => {
    render(
      <MemoryRouter>
        <HelpPage />
      </MemoryRouter>,
    );
    for (const t of HELP_TOPICS) {
      const link = screen.getByTestId(`help-manual-${t.id}`);
      expect(link).toHaveAttribute('href', `${MANUAL_BASE_URL}${t.manual}`);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link.getAttribute('rel')).toContain('noopener');
      expect(link).toHaveTextContent('Read more in the manual');
    }
    expect(screen.getAllByRole('link', { name: /Read more in the manual/ })).toHaveLength(HELP_TOPICS.length);
    expect(screen.getByTestId('manual-home')).toHaveAttribute('href', MANUAL_BASE_URL);
  });
});

describe('a Help link carries its topic and exactly one manual link', () => {
  it.each(HELP_TOPICS.map((t) => [t.id, t.manual] as const))('%s', (id, page) => {
    render(
      <MemoryRouter>
        <HelpLink topic={id} />
      </MemoryRouter>,
    );
    expect(screen.getByTestId(`help-link-${id}`)).toHaveAttribute('href', `/help/${id}`);
    const manual = screen.getByTestId(`manual-link-${id}`);
    expect(manual).toHaveAttribute('href', `${MANUAL_BASE_URL}${page}`);
    expect(manual).toHaveAttribute('target', '_blank');
    expect(screen.getAllByRole('link').filter((a) => a.getAttribute('target') === '_blank')).toHaveLength(1);
  });

  it('an unknown topic shows the Help link and no manual link', () => {
    render(
      <MemoryRouter>
        <HelpLink topic="no-such-topic" />
      </MemoryRouter>,
    );
    expect(screen.getAllByRole('link')).toHaveLength(1);
    expect(existsSync(DOCS)).toBe(true);
  });
});
