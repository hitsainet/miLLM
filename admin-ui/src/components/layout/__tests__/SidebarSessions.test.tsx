/**
 * The Sessions page is reachable: a sidebar entry and a route (Feature 36, FR-36.16).
 * Removing the nav item or the `<Route path="/sessions">` turns one of these red.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { navItems } from '../navItems';

vi.mock('@/pages/SessionsPage', () => ({ SessionsPage: () => <div data-testid="sessions-route" /> }));

describe('Sessions navigation', () => {
  it('has a sidebar entry', () => {
    const item = navItems.find((i) => i.id === 'sessions');
    expect(item).toMatchObject({ label: 'Sessions', path: '/sessions' });
  });

  it('the app routes /sessions to the page', async () => {
    window.history.pushState({}, '', '/sessions');
    const { default: App } = await import('@/App');
    render(<App />);
    expect(await screen.findByTestId('sessions-route')).toBeInTheDocument();
  });
});
