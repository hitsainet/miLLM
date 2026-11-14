/**
 * Help is reachable in the REAL app (ENH-004), not only when its module is imported.
 *
 * These render <App/> itself: its BrowserRouter, its route table, the sidebar and the header.
 * Each test goes red when one piece of wiring is removed:
 *   - the Help nav entry (navItems.ts)              -> "Help is the last main-nav item", "the nav entry opens Help"
 *   - the /help route (App.tsx)                     -> "the nav entry opens Help"
 *   - the /help/:topic route (App.tsx)              -> "the header's Help link ...", "a topic has its own address"
 *   - the header's <HelpLink> (Header.tsx)          -> "the header's Help link ..."
 * The network stays silent (fetch never answers) and the socket is a stub, so every page renders
 * in its loading state; nothing here depends on server data.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@services/socket', () => {
  const socketClient = { connect: vi.fn(), disconnect: vi.fn(), on: vi.fn(), off: vi.fn(), emit: vi.fn() };
  return { socketClient, default: socketClient };
});

import App from '@/App';
import { HELP_TOPICS } from '@/config/help';
import { MANUAL_BASE_URL } from '@/config/manual';

const scrolled: string[] = [];

function renderAppAt(path: string) {
  window.history.pushState({}, '', path);
  return render(<App />);
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
  scrolled.length = 0;
  Element.prototype.scrollIntoView = vi.fn(function (this: Element) {
    scrolled.push(this.id);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.history.pushState({}, '', '/');
});

// The app's header is the first banner; a page may draw a <header> of its own (Probe Monitors does).
const appHeader = () => screen.getAllByRole('banner')[0];
const sidebarNav = () => within(screen.getByRole('complementary')).getAllByRole('navigation')[0];

describe('Help in the real app', () => {
  it('Help is the last main-nav item, and the footer keeps Settings, the manual and the version', () => {
    renderAppAt('/models');
    const links = within(sidebarNav()).getAllByRole('link');
    const last = links[links.length - 1];
    expect(last).toHaveTextContent('Help');
    expect(last).toHaveAttribute('href', '/help');
    const aside = screen.getByRole('complementary');
    expect(within(aside).getByTestId('sidebar-manual')).toHaveAttribute('href', MANUAL_BASE_URL);
    // The footer, in order: Settings, the book-icon manual link, then the version, which is
    // package.json's (it was a hand-kept literal that nothing tied to the package).
    const pkg = JSON.parse(readFileSync(resolve(__dirname, '../../../package.json'), 'utf8'));
    const settings = within(aside).getByRole('link', { name: 'Settings' });
    const manual = within(aside).getByTestId('sidebar-manual');
    const version = within(aside).getByText(`v${pkg.version}`);
    expect(settings.compareDocumentPosition(manual) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(manual.compareDocumentPosition(version) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // Help is in the main list, not the footer: exactly one Help link in the sidebar.
    expect(within(aside).getAllByRole('link', { name: 'Help' })).toHaveLength(1);
  });

  it('the nav entry opens Help, with every topic and one manual link per topic', () => {
    renderAppAt('/models');
    fireEvent.click(within(sidebarNav()).getByRole('link', { name: 'Help' }));
    expect(window.location.pathname).toBe('/help');
    expect(screen.getByTestId('help-page')).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Help');
    for (const t of HELP_TOPICS) {
      const section = document.getElementById(`help-${t.id}`)!;
      expect(section, t.id).not.toBeNull();
      expect(within(section).getAllByRole('link', { name: /Read more in the manual/ })).toHaveLength(1);
    }
  });

  it("the header's Help link on a screen opens Help at that screen's topic", () => {
    renderAppAt('/models');
    const link = screen.getByTestId('help-link-models');
    expect(appHeader()).toContainElement(link);
    fireEvent.click(link);
    expect(window.location.pathname).toBe('/help/models');
    expect(screen.getByTestId('help-page')).toBeInTheDocument();
    expect(document.getElementById('help-models')).toHaveAttribute('data-current', 'true');
    expect(scrolled).toContain('help-models');
    expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(1);
  });

  it('a topic has its own address that can be opened directly', () => {
    renderAppAt('/help/probe-monitors');
    expect(screen.getByTestId('help-page')).toBeInTheDocument();
    expect(document.getElementById('help-probe-monitors')).toHaveAttribute('data-current', 'true');
    expect(scrolled).toEqual(['help-probe-monitors']);
    expect(screen.getByRole('link', { name: 'Probe Monitors', current: true })).toHaveAttribute(
      'href',
      '/help/probe-monitors',
    );
  });

  it('an unknown topic says so and still lists every topic', () => {
    renderAppAt('/help/no-such-topic');
    expect(screen.getByRole('status')).toHaveTextContent('no-such-topic');
    expect(screen.getAllByRole('link', { name: /Read more in the manual/ })).toHaveLength(HELP_TOPICS.length);
  });

  it('Help itself carries no header Help link', () => {
    renderAppAt('/help');
    expect(within(appHeader()).queryByRole('link', { name: 'Help' })).toBeNull();
  });

  it.each([
    ['/clusters', 'Clusters', 'clusters'],
    ['/circuits', 'Circuits', 'circuits'],
    ['/probe-monitors', 'Probe Monitors', 'probe-monitors'],
    ['/monitoring', 'Feature Monitor', 'feature-monitor'],
  ])('%s has one page title, "%s", and a Help link to its topic', async (path, title, topic) => {
    renderAppAt(path);
    await act(async () => {});
    const h1s = screen.getAllByRole('heading', { level: 1 });
    expect(h1s).toHaveLength(1);
    expect(h1s[0]).toHaveTextContent(title);
    expect(within(appHeader()).getByTestId(`help-link-${topic}`)).toHaveAttribute(
      'href',
      `/help/${topic}`,
    );
  });
});
