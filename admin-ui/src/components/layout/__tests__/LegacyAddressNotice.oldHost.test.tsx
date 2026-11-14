// @vitest-environment-options {"url": "http://k8s-millm.hitsai.local/dashboard"}
/**
 * Wiring (040 FR-040.31): the REAL app (<App/>: its router and `Layout`), loaded from an old
 * address, shows the notice between the header and the page, reading `window.location.hostname`
 * itself. `LegacyAddressNotice.test.tsx` holds the other half (a new address shows none).
 *
 * Control NC-85b (0xcc/reviews/040_P4_controls_2026-11-11.md): delete the `<LegacyAddressNotice />`
 * line in Layout.tsx -> red here.
 */
import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@services/socket', () => {
  const socketClient = { connect: vi.fn(), disconnect: vi.fn(), on: vi.fn(), off: vi.fn(), emit: vi.fn() };
  return { socketClient, default: socketClient };
});

import App from '@/App';

const OLD = 'k8s-millm.hitsai.local';
const NEW = 'k8s-miserver.hitsai.local';

beforeEach(() => {
  window.sessionStorage.clear();
  Element.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    if (String(input) === '/api/health/detailed') {
      return new Response(JSON.stringify({ legacy_names_meta: {
        transition_hosts: [OLD], new_host_for: { [OLD]: NEW }, retire_on: null, error: null } }),
      { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    return new Promise<Response>(() => {}); // every other request never answers
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe('the old-address notice in the real app', () => {
  it('is drawn on an old address, between the header and the page', async () => {
    expect(window.location.hostname).toBe(OLD);
    render(<App />);
    const notice = await screen.findByTestId('legacy-address-notice');
    expect(notice).toHaveTextContent(`This address is retiring. Use ${NEW}.`);
    const header = screen.getAllByRole('banner')[0];
    const main = screen.getByRole('main');
    expect(header.compareDocumentPosition(notice) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(notice.compareDocumentPosition(main) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // the link keeps the page the operator was on
    expect(screen.getByRole('link', { name: NEW })).toHaveAttribute('href', `http://${NEW}/dashboard`);
  });
});
