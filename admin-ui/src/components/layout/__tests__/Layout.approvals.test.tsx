/**
 * ENH-008 L11 (wiring half): in the REAL app (<App/>: its router, Layout, sidebar and header),
 * the banner sits on every page between the header and the page, and not on the Approvals page.
 *
 * Control (recorded in 0xcc/reviews/enh008_controls_2026-10-28.md): delete the
 * `<ApprovalsBanner …/>` line in Layout.tsx -> "the banner is on every page" red.
 */
import { render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@services/socket', () => {
  const socketClient = { connect: vi.fn(), disconnect: vi.fn(), on: vi.fn(), off: vi.fn(), emit: vi.fn() };
  return { socketClient, default: socketClient };
});

import App from '@/App';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { approval, installApprovalsFake, resetApprovalsStore } from '@/test/approvalsFake';

beforeEach(() => {
  resetApprovalsStore(useApprovalsStore);
  window.sessionStorage.clear();
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  vi.unstubAllGlobals();
  window.history.pushState({}, '', '/');
});

describe('the approvals banner in the real app', () => {
  it.each(['/dashboard', '/models', '/probe-monitors', '/settings', '/help'])(
    'the banner is on every page: %s', async (path) => {
      installApprovalsFake([approval()]);
      window.history.pushState({}, '', path);
      render(<App />);
      const banner = await screen.findByTestId('approvals-banner');
      const header = screen.getAllByRole('banner')[0];
      const main = screen.getByRole('main');
      // between the header and the page
      expect(header.compareDocumentPosition(banner) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(banner.compareDocumentPosition(main) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(within(banner).getByRole('button', { name: 'Approve load a model in miLLM' })).toBeInTheDocument();
    });

  it('floats just under the sticky header, with no scrolling ancestor between it and the window', async () => {
    installApprovalsFake([approval()]);
    window.history.pushState({}, '', '/dashboard');
    render(<App />);
    const dock = (await screen.findByTestId('approvals-banner')).parentElement!;
    expect(dock).toHaveAttribute('data-testid', 'approvals-banner-dock');
    expect(dock).toHaveClass('sticky', 'top-16');
    // The header it tucks under is sticky at top-0 and h-16 tall.
    expect(screen.getAllByRole('banner')[0]).toHaveClass('sticky', 'top-0', 'h-16');
    for (let el = dock.parentElement; el; el = el.parentElement) {
      expect(el.className, el.tagName).not.toMatch(/overflow-(auto|scroll|hidden|y-auto|y-scroll)/);
    }
  });

  it('is not drawn on the Approvals page, which shows the same card', async () => {
    installApprovalsFake([approval()]);
    window.history.pushState({}, '', '/approvals');
    render(<App />);
    expect(await screen.findByTestId(`approval-card-apr_${'a'.repeat(32)}`)).toBeInTheDocument();
    expect(screen.queryByTestId('approvals-banner')).toBeNull();
  });
});
