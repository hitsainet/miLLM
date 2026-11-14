/**
 * Layout (007 task 21.5): the shell renders the sidebar, header, the routed page in its outlet
 * and the toast container, and shifts the content with the sidebar's width.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from '../Layout';
import { useUIStore } from '@/stores/uiStore';

vi.mock('@/services/api', () => ({
  modelApi: { list: vi.fn().mockResolvedValue([]) },
}));

function renderLayout() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/models']}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/models" element={<p>models page body</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Layout', () => {
  afterEach(() => {
    useUIStore.setState({ sidebar: { collapsed: false, mobileOpen: false }, toasts: [] });
  });

  it('renders the sidebar, the header and the routed page', () => {
    renderLayout();
    expect(screen.getByText('models page body')).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Models');
    expect(screen.getByRole('complementary')).toBeInTheDocument(); // <aside>
  });

  it('offsets the content by the sidebar width, collapsed or not', () => {
    renderLayout();
    const content = screen.getByRole('main').parentElement!;
    expect(content.className).toContain('ml-64');
    act(() => useUIStore.getState().toggleSidebar());
    expect(content.className).toContain('ml-16');
  });

  it('mounts the toast container', () => {
    renderLayout();
    act(() => useUIStore.getState().addToast({ type: 'info', message: 'toast here', duration: 0 }));
    expect(screen.getByText('toast here')).toBeInTheDocument();
  });
});
