/**
 * ENH-008 L12, L13 in the REAL app.
 *
 * L12: the Approvals nav entry opens the page (route + nav entry), the badge shows the waiting
 * count ("{n} waiting", hidden at 0), and the header titles the page.
 * L13: a decided request appears in History. One request waits; Reject with reason "x"; the fake
 * server now holds it `rejected`; History shows "Rejected", "operator", "x", and the decision time
 * in the browser's zone with the exact instant (ISO, UTC) as its tooltip.
 *
 * Controls (0xcc/reviews/enh008_controls_2026-10-28.md):
 * * delete the `/approvals` <Route> in App.tsx        -> "the nav entry opens the page" red
 * * delete the `approvals` entry in navItems.ts        -> "the nav entry opens the page" red
 * * drop `reason` from the reject call (approvalsApi)  -> "a decided request appears in History" red
 */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@services/socket', () => {
  const socketClient = { connect: vi.fn(), disconnect: vi.fn(), on: vi.fn(), off: vi.fn(), emit: vi.fn() };
  return { socketClient, default: socketClient };
});

import App from '@/App';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { approval, installApprovalsFake, many, resetApprovalsStore } from '@/test/approvalsFake';
import { formatLocal } from '@/utils/eventTime';

beforeEach(() => {
  resetApprovalsStore(useApprovalsStore);
  window.sessionStorage.clear();
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  vi.unstubAllGlobals();
  window.history.pushState({}, '', '/');
});

const sidebar = () => screen.getByRole('complementary');

describe('the Approvals page', () => {
  it('the nav entry opens the page, titled by the header, with both sections', async () => {
    installApprovalsFake([]);
    window.history.pushState({}, '', '/models');
    render(<App />);
    fireEvent.click(within(sidebar()).getByRole('link', { name: /Approvals/ }));
    expect(window.location.pathname).toBe('/approvals');
    expect(await screen.findByTestId('approvals-page')).toBeInTheDocument();
    expect(screen.getAllByRole('heading', { level: 1 })[0]).toHaveTextContent('Approvals');
    expect(await screen.findByRole('heading', { name: 'Waiting for you (0)' })).toBeInTheDocument();
    expect(screen.getByText('No agent requests are waiting.')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'History' })).toBeInTheDocument();
    expect(screen.getByText('No decided requests yet.')).toBeInTheDocument();
    expect(screen.getByText('Requests from AI agents that wait for your decision, and the ones already decided.'))
      .toBeInTheDocument();
  });

  it('the badge shows the waiting count, and is hidden at 0', async () => {
    const fake = installApprovalsFake(many(4));
    window.history.pushState({}, '', '/models');
    render(<App />);
    const badge = await within(sidebar()).findByTestId('nav-badge-approvals');
    expect(badge).toHaveTextContent('4');
    expect(badge).toHaveAttribute('aria-label', '4 waiting');
    expect(badge).toHaveAttribute('title', '4 waiting');
    // Solid, not a translucent tint: the same pill as the sibling apps (operator 2026-10-28).
    expect(badge).toHaveClass('bg-amber-500', 'text-white', 'h-5', 'min-w-5');
    expect(badge.className).not.toMatch(/bg-amber-500\/\d+/);
    fake.state.rows.forEach((r) => { r.status = 'rejected'; });
    await useApprovalsStore.getState().refresh();
    await waitFor(() => expect(within(sidebar()).queryByTestId('nav-badge-approvals')).toBeNull());
  });

  it('a decided request appears in History', async () => {
    const fake = installApprovalsFake([approval()]);
    window.history.pushState({}, '', '/approvals');
    render(<App />);
    const card = await screen.findByTestId(`approval-card-apr_${'a'.repeat(32)}`);
    fireEvent.click(within(card).getByRole('button', { name: 'Reject' }));
    fireEvent.change(within(card).getByPlaceholderText('Reason (optional)'), { target: { value: 'x' } });
    fireEvent.click(within(card).getByRole('button', { name: 'Confirm reject' }));
    const row = await screen.findByTestId(`history-row-apr_${'a'.repeat(32)}`);
    const posts = fake.calls.filter((c) => c.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0].path).toBe(`/api/approvals/apr_${'a'.repeat(32)}/reject`);
    expect(posts[0].body).toEqual({ reason: 'x' });
    const cells = within(row).getAllByRole('cell').map((c) => c.textContent?.replace(/^[^:]+: /, ''));
    const decided = formatLocal('2026-10-28T19:30:00.000Z');
    expect(cells).toEqual([
      'Rejected', 'Load a model in miLLM', 'mcp', formatLocal('2026-10-28T18:00:00.000Z').text,
      'operator', decided.text, 'x', '—', `apr_${'a'.repeat(32)}`,
    ]);
    expect(within(row).getByTitle('2026-10-28T19:30:00.000Z')).toHaveTextContent(decided.text);
    expect(screen.getByRole('heading', { name: 'Waiting for you (0)' })).toBeInTheDocument();
  });

  it('Show older loads the next page of History', async () => {
    const decided = Array.from({ length: 201 }, (_, i) =>
      approval({ approval_id: `apr_${String(i).padStart(32, '0')}`, status: 'expired' }));
    const fake = installApprovalsFake(decided);
    // the fake ignores limit/offset; answer the first History load with exactly 200 rows
    const real = fake.fetchMock.getMockImplementation()!;
    fake.fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes('status=decided') && url.includes('offset=0')) {
        return new Response(JSON.stringify({ success: true, data: { approvals: decided.slice(0, 200), count: 200, pending: 0 }, error: null }),
          { status: 200, headers: { 'content-type': 'application/json' } });
      }
      if (url.includes('status=decided') && url.includes('offset=200')) {
        return new Response(JSON.stringify({ success: true, data: { approvals: decided.slice(200), count: 1, pending: 0 }, error: null }),
          { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return real(input, init);
    });
    window.history.pushState({}, '', '/approvals');
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'Show older' }));
    expect(await screen.findByTestId(`history-row-apr_${'200'.padStart(32, '0')}`)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Show older' })).toBeNull();
    const row = screen.getByTestId(`history-row-apr_${'0'.repeat(32)}`);
    expect(row).toHaveTextContent('nobody (expired)');
    expect(row).toHaveTextContent('expired without a decision');
  });
});
