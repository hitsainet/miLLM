/**
 * The old-address notice (040 FR-040.31, T27; TID §6). The list of old hosts, each one's new host
 * and the retirement date all come from the server (`GET /api/health/detailed`,
 * `legacy_names_meta`); the UI keeps no copy. Its mount in `Layout` is pinned by
 * `LegacyAddressNotice.oldHost.test.tsx` (the real app on an old address) and by the last test here.
 *
 * Control NC-85 (0xcc/reviews/040_P4_controls_2026-11-11.md): invert the host condition in
 * `LegacyAddressNotice` -> "shows the notice on an old host only" red.
 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { LEGACY_NOTICE_DISMISSED_KEY, LegacyAddressNotice } from '../LegacyAddressNotice';

const OLD = 'k8s-millm.hitsai.local';
const NEW = 'k8s-miserver.hitsai.local';

function installHealth(meta: Record<string, unknown> | null) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path !== '/api/health/detailed') throw new Error(`unexpected request ${path}`);
    return new Response(JSON.stringify(meta === null ? {} : { legacy_names_meta: meta }), {
      status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const META = {
  transition_hosts: [OLD, 'millm.hitsai.net'],
  new_host_for: { [OLD]: NEW, 'millm.hitsai.net': 'miserver.hitsai.net' },
  retire_on: '2026-12-11',
  error: null,
};

function renderNotice(hostname: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <LegacyAddressNotice hostname={hostname} />
    </QueryClientProvider>,
  );
}

beforeEach(() => window.sessionStorage.clear());
afterEach(() => vi.unstubAllGlobals());

describe('LegacyAddressNotice', () => {
  it('shows the notice on an old host only', async () => {
    const fetchMock = installHealth(META);
    const old = renderNotice(OLD);
    const notice = await screen.findByTestId('legacy-address-notice');
    expect(notice).toHaveTextContent(
      `This address is retiring on 2026-12-11. Use ${NEW}. Preferences saved in this browser do not carry over.`);
    // the same page on the new host: scheme, port and path kept (the test page is localhost:3000)
    expect(screen.getByRole('link', { name: NEW })).toHaveAttribute(
      'href', `${window.location.protocol}//${NEW}:${window.location.port}/`);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    old.unmount();

    renderNotice(NEW);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    await Promise.resolve();
    expect(screen.queryByTestId('legacy-address-notice')).toBeNull();
  });

  it('hides after dismissal for the session', async () => {
    installHealth(META);
    const first = renderNotice(OLD);
    fireEvent.click(await screen.findByRole('button', { name: 'Dismiss the old-address notice' }));
    expect(screen.queryByTestId('legacy-address-notice')).toBeNull();
    expect(window.sessionStorage.getItem(LEGACY_NOTICE_DISMISSED_KEY)).toBe('1');
    expect(LEGACY_NOTICE_DISMISSED_KEY).toBe('miserver.legacyNotice.dismissed');
    first.unmount();

    // a new mount in the same session stays hidden; a new session shows it again
    const fetchMock = installHealth(META);
    const second = renderNotice(OLD);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await Promise.resolve();
    expect(screen.queryByTestId('legacy-address-notice')).toBeNull();
    second.unmount();
    window.sessionStorage.clear();
    renderNotice(OLD);
    expect(await screen.findByTestId('legacy-address-notice')).toBeInTheDocument();
  });

  it('gives no date before one is set, and no new address the server did not name', async () => {
    installHealth({ ...META, retire_on: null, new_host_for: {} });
    renderNotice(OLD);
    const notice = await screen.findByTestId('legacy-address-notice');
    expect(notice).toHaveTextContent(
      'This address is retiring. Preferences saved in this browser do not carry over.');
    expect(notice).not.toHaveTextContent('Use ');
  });

  it('draws nothing when the server sends no host list or the read fails', async () => {
    installHealth(null);
    const none = renderNotice(OLD);
    await Promise.resolve();
    expect(screen.queryByTestId('legacy-address-notice')).toBeNull();
    none.unmount();
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 503 })));
    renderNotice(OLD);
    await Promise.resolve();
    expect(screen.queryByTestId('legacy-address-notice')).toBeNull();
  });
});
