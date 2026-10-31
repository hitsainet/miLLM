/**
 * ENH-008 L11 (component half): the banner shows at most 3 cards, the soonest to expire first,
 * "and N more waiting", collapses (kept in sessionStorage; a new request expands it), is a labelled
 * region, hides on the Approvals page, reports a failed load, and Approve posts ONCE with no agent
 * header. The mount in `Layout` is pinned by `layout/__tests__/Layout.approvals.test.tsx`.
 */
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useApprovalsStore } from '@/stores/approvalsStore';
import { approval, installApprovalsFake, many, resetApprovalsStore } from '@/test/approvalsFake';

import { ApprovalsBanner } from '../ApprovalsBanner';

function renderBanner(onApprovalsPage = false) {
  return render(
    <MemoryRouter>
      <ApprovalsBanner onApprovalsPage={onApprovalsPage} />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  resetApprovalsStore(useApprovalsStore);
  window.sessionStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

describe('ApprovalsBanner', () => {
  it('shows at most three cards, soonest to expire first, and "and N more"', async () => {
    installApprovalsFake(many(5));
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner();
    const region = screen.getByRole('region', { name: 'Agent requests waiting for your approval' });
    expect(region).toHaveAttribute('data-testid', 'approvals-banner');
    expect(within(region).getByRole('heading', { level: 2 })).toHaveTextContent(
      '5 agent requests waiting for your approval');
    const cards = region.querySelectorAll('[data-approval-card]');
    expect(cards).toHaveLength(3);
    // many(): the LAST one expires soonest
    expect(within(cards[0] as HTMLElement).getByText('Action number 4')).toBeInTheDocument();
    expect(within(region).getByRole('link', { name: 'and 2 more waiting — Open Approvals' }))
      .toHaveAttribute('href', '/approvals');
    expect(within(region).getByRole('link', { name: 'Open Approvals' })).toHaveAttribute('href', '/approvals');
  });

  it('says one request in the singular and draws no "more" line', async () => {
    installApprovalsFake([approval()]);
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner();
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(
      '1 agent request waiting for your approval');
    expect(screen.queryByText(/more waiting/)).toBeNull();
  });

  it('draws nothing when nothing waits, and nothing on the Approvals page', async () => {
    installApprovalsFake([]);
    await act(() => useApprovalsStore.getState().refresh());
    const { unmount } = renderBanner();
    expect(screen.queryByTestId('approvals-banner')).toBeNull();
    unmount();
    installApprovalsFake([approval()]);
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner(true);
    expect(screen.queryByTestId('approvals-banner')).toBeNull();
  });

  it('says so when the queue cannot be read', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('network down'); }));
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner();
    expect(screen.getByTestId('approvals-banner')).toHaveTextContent(
      'Could not read the approval queue: network down');
  });

  it('floats: a sticky dock under the sticky header, height-capped with its own scroll, also when collapsed or failing', async () => {
    installApprovalsFake(many(5));
    await act(() => useApprovalsStore.getState().refresh());
    const { unmount } = renderBanner();
    const banner = screen.getByTestId('approvals-banner');
    const dock = screen.getByTestId('approvals-banner-dock');
    expect(banner.parentElement).toBe(dock);
    expect(dock).toHaveClass('sticky', 'top-16', 'z-20', 'bg-slate-950');
    expect(banner).toHaveClass('max-h-[40vh]', 'overflow-y-auto');
    fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
    expect(screen.getByTestId('approvals-banner-dock')).toHaveClass('sticky', 'top-16');
    unmount();
    resetApprovalsStore(useApprovalsStore);
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('network down'); }));
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner();
    expect(screen.getByTestId('approvals-banner').parentElement).toHaveClass('sticky', 'top-16');
    expect(screen.getByTestId('approvals-banner')).toHaveClass('max-h-[40vh]', 'overflow-y-auto');
  });

  it('collapses to its heading, remembers it, and a new request expands it again', async () => {
    const fake = installApprovalsFake([approval()]);
    await act(() => useApprovalsStore.getState().refresh());
    const { unmount } = renderBanner();
    fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
    expect(document.querySelectorAll('[data-approval-card]')).toHaveLength(0);
    expect(window.sessionStorage.getItem('millm.approvals.banner.collapsed')).toBe('1');
    unmount();
    renderBanner();
    expect(screen.getByRole('button', { name: 'Expand' })).toBeInTheDocument();
    fake.state.rows.push(approval({ approval_id: `apr_${'b'.repeat(32)}` }));
    await act(() => useApprovalsStore.getState().refresh());
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeInTheDocument();
    expect(document.querySelectorAll('[data-approval-card]')).toHaveLength(2);
  });

  it('Approve posts once, to the right request, with no agent header', async () => {
    const fake = installApprovalsFake([approval()]);
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner();
    fireEvent.click(screen.getByRole('button', { name: 'Approve load a model in miLLM' }));
    await waitFor(() => expect(screen.queryByTestId('approvals-banner')).toBeNull());
    const posts = fake.calls.filter((c) => c.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0].path).toBe(`/api/approvals/apr_${'a'.repeat(32)}/approve`);
    for (const call of fake.calls) expect(Object.keys(call.headers)).not.toContain('x-millm-agent');
  });

  it('Reject sends the typed reason, or the default when left blank', async () => {
    const rows = many(2);
    const fake = installApprovalsFake(rows);
    await act(() => useApprovalsStore.getState().refresh());
    renderBanner();
    const [first, second] = Array.from(document.querySelectorAll<HTMLElement>('[data-approval-card]'));
    fireEvent.click(within(first).getByRole('button', { name: 'Reject' }));
    fireEvent.change(within(first).getByPlaceholderText('Reason (optional)'), { target: { value: 'not now' } });
    fireEvent.click(within(first).getByRole('button', { name: 'Confirm reject' }));
    await waitFor(() => expect(fake.calls.filter((c) => c.method === 'POST')).toHaveLength(1));
    fireEvent.click(within(second).getByRole('button', { name: 'Reject' }));
    fireEvent.click(within(second).getByRole('button', { name: 'Confirm reject' }));
    await waitFor(() => expect(fake.calls.filter((c) => c.method === 'POST')).toHaveLength(2));
    const bodies = fake.calls.filter((c) => c.method === 'POST').map((c) => c.body);
    expect(bodies).toEqual([{ reason: 'not now' }, { reason: 'Rejected by the operator.' }]);
  });

  it('shows the expired message inside the card on approval_expired', async () => {
    installApprovalsFake([approval()]);
    await act(() => useApprovalsStore.getState().refresh());
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) =>
      String(input).includes('/approve')
        ? new Response(JSON.stringify({ success: false, data: null, error: { code: 'approval_expired', message: 'x' } }),
          { status: 409, headers: { 'content-type': 'application/json' } })
        : new Response(JSON.stringify({ success: true, data: { approvals: [approval()], count: 1, pending: 1 }, error: null }),
          { status: 200, headers: { 'content-type': 'application/json' } })));
    renderBanner();
    fireEvent.click(screen.getByRole('button', { name: 'Approve load a model in miLLM' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('This request expired. Ask the agent to try again.');
  });
});
