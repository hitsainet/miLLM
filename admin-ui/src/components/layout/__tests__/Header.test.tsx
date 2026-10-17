/**
 * Header (007 task 20.5): the page title follows the route, the model/SAE/steering badges follow
 * the server store, and the theme toggle flips the theme. GPU chips and the self-fetching model
 * badge have their own files (HeaderGpus, HeaderLoadedModel).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Header } from '../Header';
import { useServerStore } from '@/stores/serverStore';
import { useUIStore } from '@/stores/uiStore';

vi.mock('@/services/api', () => ({
  modelApi: { list: vi.fn().mockResolvedValue([]) },
}));

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Header />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Header', () => {
  afterEach(() => {
    useServerStore.getState().reset();
    useUIStore.setState({ theme: 'dark' });
  });

  it.each([
    ['/models', 'Models'],
    ['/sae', 'Sparse Autoencoders'],
    ['/steering', 'Feature Steering'],
    ['/nowhere', 'miLLM'],
  ])('titles %s as %s', (path, title) => {
    renderAt(path);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(title);
  });

  it('shows No Model / No SAE when nothing is loaded', () => {
    renderAt('/');
    expect(screen.getByText('No Model')).toBeInTheDocument();
    expect(screen.getByText('No SAE')).toBeInTheDocument();
  });

  it('shows the attached SAE and the steering count only when steering is on', () => {
    useServerStore.setState({
      attachedSAE: { name: 'res-jb-12' } as never,
      steering: { enabled: false, features: [{ index: 1, strength: 2 }] } as never,
    });
    const { unmount } = renderAt('/');
    expect(screen.getByText('res-jb-12')).toBeInTheDocument();
    expect(screen.queryByText(/Steering$/)).not.toBeInTheDocument();
    unmount();
    useServerStore.setState({
      steering: { enabled: true, features: [{ index: 1, strength: 2 }, { index: 2, strength: 3 }] } as never,
    });
    renderAt('/');
    expect(screen.getByText('2 Steering')).toBeInTheDocument();
  });

  it('the theme button toggles the theme', () => {
    renderAt('/');
    fireEvent.click(screen.getByTitle('Switch to light mode'));
    expect(useUIStore.getState().theme).toBe('light');
    expect(screen.getByTitle('Switch to dark mode')).toBeInTheDocument();
  });
});
