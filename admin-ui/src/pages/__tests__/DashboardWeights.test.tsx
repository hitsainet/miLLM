/**
 * The Weights card is on the Dashboard, fed by the real hook and API client (FTASKS 031 8.2).
 *
 * MUTATION CONTROL: remove `<WeightsCard {...weights} />` from DashboardPage -> this fails.
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

describe('Dashboard Weights card', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('shows a torn set as serving refused', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        const data =
          url === '/api/weights/status'
            ? {
                attached: false, attachment: null, barrier: 'serving', weights_state: 'torn',
                resident_model: null, steps_since_load: 12, last_save: null, commit_overran: false,
                last_hold_timeout_at: null, last_overrun: null, quiesced_at: null,
                held_requests: null, commit: null, adapter: null,
              }
            : null;
        if (data === null) {
          return new Response(JSON.stringify({ detail: 'Not Found' }), {
            status: 404,
            headers: { 'content-type': 'application/json' },
          });
        }
        return new Response(JSON.stringify({ success: true, data, error: null }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <DashboardPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    const card = await screen.findByTestId('weights-card');
    await waitFor(() => expect(within(card).getByText('serving refused')).toBeInTheDocument());
    await waitFor(() => expect(within(card).getByText('not available')).toBeInTheDocument());
    expect(within(card).getByText('12')).toBeInTheDocument();
  });
});
