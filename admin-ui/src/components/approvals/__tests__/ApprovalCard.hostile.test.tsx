/**
 * ENH-008 L14: every piece of request text renders as TEXT. Hostile markup in every field —
 * action, requested_by, summary keys and values, payload, the code-rule source, and in History the
 * reason and the error — creates no <img> and no <script>, and the text is visible as typed.
 */
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { approval } from '@/test/approvalsFake';

import { ApprovalCard } from '../ApprovalCard';
import { ApprovalHistory } from '../ApprovalHistory';

const IMG = '<img src=x onerror=alert(1)>';
const SCRIPT = '<script>window.__pwned=1</script>';
const BAD = `${IMG}${SCRIPT}`;

const hostile = approval({
  action_label: BAD,
  requested_by: `agent:${BAD}`,
  summary: {
    [BAD]: BAD,
    Nested: { [BAD]: [BAD] },
    code_modules: [{ code_sha256: BAD, functions: [BAD], runner_contract: BAD, source: BAD }],
  },
  payload: { [BAD]: BAD, deep: { x: BAD } },
});

function noMarkup(container: HTMLElement) {
  expect(container.querySelectorAll('img')).toHaveLength(0);
  expect(container.querySelectorAll('script')).toHaveLength(0);
  expect((window as unknown as { __pwned?: number }).__pwned).toBeUndefined();
}

describe('hostile text renders as text', () => {
  it('in a waiting card, including the code source', () => {
    const { container } = render(<MemoryRouter><ApprovalCard approval={hostile} /></MemoryRouter>);
    noMarkup(container);
    expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent(BAD);
    expect(screen.getByText(`asked by agent ${BAD}`)).toBeInTheDocument();
    expect(container.querySelector('pre')).toHaveTextContent(BAD);
    expect(container.textContent).toContain(JSON.stringify({ [BAD]: [BAD] }));
  });

  it('in History, including the reason and the error', () => {
    const rows = [
      approval({ ...hostile, status: 'rejected', decided_by: BAD, reason: BAD }),
      approval({ ...hostile, approval_id: `apr_${'f'.repeat(32)}`, status: 'failed',
        decided_by: 'operator', error: { code: BAD, message: BAD } }),
    ];
    const { container } = render(<ApprovalHistory rows={rows} />);
    noMarkup(container);
    expect(container.textContent).toContain(`${BAD}: ${BAD}`);
  });
});
