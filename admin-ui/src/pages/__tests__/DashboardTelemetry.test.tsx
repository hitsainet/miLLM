/**
 * The Telemetry card is on the Dashboard, fed by the real hook from the real
 * `GET /api/health/detailed` (038 FTASKS 9.3).
 *
 * MUTATION CONTROL: remove `<TelemetryCard {...telemetry} />` from DashboardPage → this fails;
 * point the hook at another route → this fails.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

vi.mock('@hooks/useModels', () => ({ useModels: () => ({}) }));
vi.mock('@hooks/useSAE', () => ({ useSAE: () => ({}) }));
vi.mock('@hooks/useSteering', () => ({ useSteering: () => ({}) }));
vi.mock('@hooks/useMonitoring', () => ({ useMonitoring: () => ({}) }));

import { DashboardPage } from '../DashboardPage';

describe('Dashboard Telemetry card', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('reads the telemetry block of the detailed health route', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === '/api/health/detailed') {
        return new Response(JSON.stringify({
          status: 'healthy',
          telemetry: {
            enabled: true,
            signals: {
              traces: { state: 'disabled', exported: 0, failed: 0, dropped: 0, last_success_at: null,
                        last_error: null, queue: { used: 0, capacity: 0 } },
              metrics: { state: 'disabled', exported: 0, failed: 0, dropped: 0, last_success_at: null,
                         last_error: null, queue: { used: 0, capacity: 0 } },
              logs: { state: 'degraded', exported: 9, failed: 1, dropped: 0, last_success_at: null,
                      last_error: { type: 'ConnectionError', at: null }, queue: { used: 1, capacity: 8192 } },
            },
            series_overflow: 0, attributes_dropped: 0,
            deletion_note: 'Deleting probes or events in miLLM does not erase copies already exported.',
          },
        }), { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return new Response(JSON.stringify({ detail: 'Not Found' }), {
        status: 404, headers: { 'content-type': 'application/json' },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <DashboardPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    const card = await screen.findByTestId('telemetry-card');
    await waitFor(() => expect(within(card).getByTestId('telemetry-state-logs')).toHaveTextContent('degraded'));
    expect(within(card).getByTestId('telemetry-error-logs')).toHaveTextContent('ConnectionError');
    expect(fetchMock.mock.calls.filter(([u]) => u === '/api/health/detailed')).toHaveLength(1);
  });
});
