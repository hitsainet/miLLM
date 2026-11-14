/**
 * The header names the loaded model on EVERY page, not only the three that happened to fetch it.
 *
 * ⚠ REPORTED FROM THE RUNNING UI 2026-09-30. The header read "No Model" while
 * Llama-3.1-8B-Instruct was loaded, holding 15.7 GB, and scoring live traffic through an armed
 * probe. Everything downstream was right — the API said `loaded`, the probe was firing, the VRAM
 * was resident — and the one thing an operator glances at said the opposite.
 *
 * `loadedModel` was seeded ONLY by `useModels()`, and only ModelsPage, SteeringPage and
 * MonitoringPage called it. Land on Probe Monitors, or reload there, and nothing ever populated
 * it. The badge described WHICH PAGES YOU HAD VISITED, not the server's state — and it failed
 * in the direction that reads as "nothing is running", which is the direction that makes someone
 * go looking for a fault that does not exist.
 *
 * The header now fetches what it displays. The query key is shared, so pages that already call
 * the hook hit the same cache rather than issuing a second request.
 *
 * MUTATION CONTROL: remove `useModels()` from Header -> the first test goes red.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { Header } from '../Header';
import { useServerStore } from '@/stores/serverStore';

const list = vi.hoisted(() => vi.fn());
vi.mock('@/services/api', () => ({ modelApi: { list } }));

function renderHeader() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Header />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('the header names the loaded model wherever you are', () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => useServerStore.getState().reset());

  it('shows the model without any page having fetched it first', async () => {
    // The store starts empty, exactly as it does on a fresh load of /probe-monitors.
    expect(useServerStore.getState().loadedModel).toBeNull();
    list.mockResolvedValue([
      { id: 1, name: 'LFM2.5-1.2B-Instruct', status: 'ready' },
      { id: 9, name: 'Llama-3.1-8B-Instruct', status: 'loaded' },
    ]);

    renderHeader();

    await waitFor(() =>
      expect(screen.getByText('Llama-3.1-8B-Instruct')).toBeInTheDocument()
    );
    expect(screen.queryByText('No Model')).not.toBeInTheDocument();
  });

  it('says No Model when the server really has none loaded', async () => {
    // ⚠ Specificity. Without this, the test above passes against a header hard-wired to a name.
    list.mockResolvedValue([
      { id: 1, name: 'LFM2.5-1.2B-Instruct', status: 'ready' },
      { id: 9, name: 'Llama-3.1-8B-Instruct', status: 'ready' },
    ]);

    renderHeader();

    await waitFor(() => expect(screen.getByText('No Model')).toBeInTheDocument());
  });

  it('stops naming a model once the server unloads it', async () => {
    // The other half of the defect: the store was only ever SET, so a model unloaded elsewhere
    // kept its name in the header until a socket event happened to arrive.
    useServerStore.getState().setLoadedModel({ id: 9, name: 'Llama-3.1-8B-Instruct' } as never);
    list.mockResolvedValue([{ id: 9, name: 'Llama-3.1-8B-Instruct', status: 'ready' }]);

    renderHeader();

    await waitFor(() => expect(screen.getByText('No Model')).toBeInTheDocument());
    expect(screen.queryByText('Llama-3.1-8B-Instruct')).not.toBeInTheDocument();
  });
});
