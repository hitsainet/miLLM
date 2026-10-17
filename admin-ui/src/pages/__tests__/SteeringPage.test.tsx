/**
 * SteeringPage orientation UX: the page is a VIEW of currently-applied
 * steering — the empty state must say how to populate it (activate a
 * cluster/profile), and populated sliders attribute the active cluster.
 */

import { describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';

const listMock = vi.fn();

vi.mock('@/services/api', () => ({
  clusterApi: { list: (...args: unknown[]) => listMock(...args) },
}));

const serverState = {
  loadedModel: { id: 1, name: 'org/tiny-model' },
  attachedSAE: { id: 2, d_sae: 16384, trained_layer: 6 },
  steering: { enabled: false, features: [] as { index: number; strength: number }[] },
};

vi.mock('@stores/serverStore', () => ({
  useServerStore: () => serverState,
}));
vi.mock('@hooks/useModels', () => ({ useModels: () => ({}) }));
vi.mock('@hooks/useSAE', () => ({ useSAE: () => ({}) }));
vi.mock('@hooks/useProfiles', () => ({
  useProfiles: () => ({ createProfile: vi.fn(), isCreating: false }),
}));
vi.mock('@hooks/useSteering', () => ({
  useSteering: () => ({
    isLoading: false,
    setFeature: vi.fn(),
    isSetting: false,
    batchSetFeatures: vi.fn(),
    removeFeature: vi.fn(),
    isRemoving: false,
    clearFeatures: vi.fn(),
    isClearing: false,
    enableSteering: vi.fn(),
    disableSteering: vi.fn(),
    isEnabling: false,
    isDisabling: false,
  }),
}));

import { SteeringPage } from '../SteeringPage';

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <SteeringPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('SteeringPage orientation', () => {
  it('empty state explains activation and links to Clusters/Profiles', async () => {
    serverState.steering = { enabled: false, features: [] };
    listMock.mockResolvedValue({ clusters: [] });
    renderPage();
    expect(await screen.findByText('No active steering')).toBeInTheDocument();
    expect(screen.getByText(/Activate an imported cluster or a saved profile/)).toBeInTheDocument();
    expect(screen.getByText('Go to Clusters')).toBeInTheDocument();
    expect(screen.getByText('Go to Profiles')).toBeInTheDocument();
  });

  it('attributes populated sliders to the active cluster by name', async () => {
    serverState.steering = { enabled: true, features: [{ index: 100, strength: 1.2 }] };
    listMock.mockResolvedValue({
      clusters: [
        {
          id: 'prof_c1',
          name: 'fear cluster',
          is_active: true,
          member_count: 1,
          members: [[100, 'fear_of_water', 1.2]],
        },
      ],
    });
    renderPage();
    expect(
      await screen.findByText(/1 feature from cluster "fear cluster"/)
    ).toBeInTheDocument();
  });
});
