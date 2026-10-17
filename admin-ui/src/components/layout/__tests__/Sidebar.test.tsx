/**
 * Sidebar (007 task 19.7): every nav item is a link to its path, the current route is marked
 * active, and the toggle collapses to icons (labels become tooltips) and back.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Sidebar } from '../Sidebar';
import { bottomNavItems, navItems } from '../navItems';
import { useUIStore } from '@/stores/uiStore';

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <Sidebar />
    </MemoryRouter>,
  );

describe('Sidebar', () => {
  afterEach(() => useUIStore.setState({ sidebar: { collapsed: false, mobileOpen: false } }));

  it('links every nav item to its path', () => {
    renderAt('/');
    for (const item of [...navItems, ...bottomNavItems]) {
      expect(screen.getByText(item.label).closest('a')).toHaveAttribute('href', item.path);
    }
  });

  it('marks the current route active and no other', () => {
    renderAt('/models');
    const models = navItems.find((i) => i.path === '/models')!;
    expect(screen.getByText(models.label).closest('a')!.className).toContain('text-primary-400');
    const other = navItems.find((i) => i.path !== '/models')!;
    expect(screen.getByText(other.label).closest('a')!.className).not.toContain('text-primary-400');
  });

  it('collapses to icons with tooltips, and expands again', () => {
    renderAt('/');
    const first = navItems[0];
    expect(screen.getByText('miLLM')).toBeInTheDocument();
    const toggle = screen.getAllByRole('button').at(-1)!;
    fireEvent.click(toggle);
    expect(useUIStore.getState().sidebar.collapsed).toBe(true);
    expect(screen.queryByText(first.label)).not.toBeInTheDocument();
    expect(screen.queryByText('miLLM')).not.toBeInTheDocument();
    expect(document.querySelector(`a[title="${first.label}"]`)).not.toBeNull();
    fireEvent.click(toggle);
    expect(screen.getByText(first.label)).toBeInTheDocument();
  });
});
