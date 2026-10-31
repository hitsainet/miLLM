/**
 * ENH-006 D79: "Approve all settings changes (N)" in the banner and on the Approvals page, and
 * D81: "Approve all code rules (N)", a separate button and batch (the kinds never mix).
 *
 * * the button appears only when TWO or more waiting requests are settings changes (never at one,
 *   never counting a non-settings request);
 * * its confirm lists every request it will approve (action, what changes, who asked) and nothing
 *   else;
 * * confirming sends ONE `POST /api/approvals/approve-batch` with exactly those ids (no single
 *   approves, no agent header);
 * * afterwards the banner updates and History shows each decided request with the batch.
 *
 * Controls (0xcc/reviews/approve_all_settings_controls_2026-10-29.md):
 * * `APPROVE_ALL_MIN` 2 -> 1                                 -> "not at one" red (both kinds)
 * * the store's approveBatch posts only the first id         -> "ONE request with every id" red
 * * `batchOf` ignores the kind (kinds mixed)                 -> "never mixes" red
 * * the banner's <ApproveAllSettings> mount removed          -> "banner offers" red
 * * the page's <ApproveAllSettings> mount removed            -> "page offers" red
 */
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApprovalsPage } from '@/pages/ApprovalsPage';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { approval, installApprovalsFake, resetApprovalsStore } from '@/test/approvalsFake';
import type { Approval, CodeModuleCard } from '@/types/approvals';

import { ApprovalsBanner } from '../ApprovalsBanner';

const ID = (c: string) => `apr_${c.repeat(32)}`;

function limits(c: string, over: Partial<Approval> = {}): Approval {
  return approval({
    approval_id: ID(c),
    action: 'probe_limits_write',
    action_label: "Change a model's probe limits in miLLM",
    settings_change: true,
    batch_kind: 'settings',
    summary: {
      Model: `Model ${c}`,
      'Current limits': { max_behaviours: 1 },
      'Requested limits': { max_behaviours: 3 },
    },
    requested_by: `agent:bot-${c}`,
    ...over,
  });
}

function telemetry(c: string): Approval {
  return approval({
    approval_id: ID(c),
    action: 'telemetry_write',
    action_label: 'Change telemetry settings in miLLM',
    settings_change: true,
    batch_kind: 'settings',
    summary: { 'Trace sampling ratio (TELEMETRY_TRACES_SAMPLER_RATIO)': 'now 1.0; after approval 0.5' },
    requested_by: 'agent:mcp',
  });
}

const SHA = (c: string) => c.repeat(64);

function codeRule(c: string): Approval {
  const modules: CodeModuleCard[] = [
    { code_sha256: SHA(c), functions: ['f'], runner_contract: 'millm.rule-runner/v1', source: `def f(text):  # ${c}\n    return 1` },
  ];
  return approval({
    approval_id: ID(c),
    action: 'code_rule_import',
    action_label: 'Import a rule set with operator code into miLLM',
    settings_change: false,
    batch_kind: 'code_rules',
    summary: { 'Rule set': `rules-${c} (behaviour b-${c})`, Digest: `sha256:${SHA(c)}`, code_modules: modules },
    requested_by: `agent:coder-${c}`,
  });
}

const unload = () =>
  approval({ approval_id: ID('9'), action: 'model_unload', action_label: 'Unload a model in miLLM', settings_change: false });

function renderBanner() {
  return render(
    <MemoryRouter>
      <ApprovalsBanner onApprovalsPage={false} />
    </MemoryRouter>,
  );
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ApprovalsPage />
    </MemoryRouter>,
  );
}

async function load(rows: Approval[]) {
  const fake = installApprovalsFake(rows);
  await act(() => useApprovalsStore.getState().refresh());
  return fake;
}

const button = () => screen.queryByRole('button', { name: /^Approve all settings changes/ });
const codeButton = () => screen.queryByRole('button', { name: /^Approve all code rules/ });

beforeEach(() => {
  resetApprovalsStore(useApprovalsStore);
  window.sessionStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

describe('Approve all settings changes', () => {
  it('is not offered at one waiting settings change, nor counting other requests', async () => {
    await load([limits('1'), unload()]);
    renderBanner();
    expect(screen.getByTestId('approvals-banner')).toBeInTheDocument();
    expect(button()).toBeNull();
  });

  it('is not offered when nothing that waits is a settings change', async () => {
    await load([unload(), approval()]);
    renderPage();
    expect(button()).toBeNull();
  });

  it('banner offers it at two, named with the count (other requests not counted)', async () => {
    await load([limits('1'), telemetry('2'), unload()]);
    renderBanner();
    expect(within(screen.getByTestId('approvals-banner')).getByRole('button', {
      name: 'Approve all settings changes (2)' })).toBeInTheDocument();
  });

  it('page offers it in the Waiting list', async () => {
    await load([limits('1'), telemetry('2'), limits('3')]);
    renderPage();
    const waiting = screen.getByRole('region', { name: /Waiting for you/ });
    expect(within(waiting).getByRole('button', { name: 'Approve all settings changes (3)' }))
      .toBeInTheDocument();
  });

  it('lists every request in the confirm, then sends ONE request with every id', async () => {
    const fake = await load([limits('1'), telemetry('2'), unload()]);
    renderBanner();
    fireEvent.click(button()!);
    const list = screen.getByTestId('approve-all-list');
    const items = within(list).getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByText("Change a model's probe limits in miLLM")).toBeInTheDocument();
    expect(within(items[0]).getByText('asked by agent bot-1')).toBeInTheDocument();
    expect(within(items[0]).getByText('{"max_behaviours":3}')).toBeInTheDocument();
    expect(within(items[1]).getByText('now 1.0; after approval 0.5')).toBeInTheDocument();
    expect(within(items[1]).getByText('asked by agent mcp')).toBeInTheDocument();
    expect(within(list).queryByText('Unload a model in miLLM')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Approve these 2 settings changes' }));
    await waitFor(() => expect(button()).toBeNull());
    const posts = fake.calls.filter((c) => c.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0].path).toBe('/api/approvals/approve-batch');
    expect(posts[0].body).toEqual({ approval_ids: [ID('1'), ID('2')], kind: 'settings' });
    for (const call of fake.calls) expect(Object.keys(call.headers)).not.toContain('x-millm-agent');
    // The banner updated: only the unload still waits.
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(
      '1 agent request waiting for your approval');
  });

  it('Cancel sends nothing', async () => {
    const fake = await load([limits('1'), telemetry('2')]);
    renderBanner();
    fireEvent.click(button()!);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByTestId('approve-all-list')).toBeNull();
    expect(fake.calls.filter((c) => c.method === 'POST')).toHaveLength(0);
  });

  it('History shows each decided request with the batch, and a failure is reported', async () => {
    const fake = await load([limits('1'), telemetry('2'), limits('3')]);
    fake.state.failIds = new Set([ID('3')]);
    renderPage();
    fireEvent.click(button()!);
    fireEvent.click(screen.getByRole('button', { name: 'Approve these 3 settings changes' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      '1 of the approved changes failed; see History.');
    const history = await screen.findByTestId('approvals-history');
    for (const c of ['1', '2', '3']) {
      const row = within(history).getByTestId(`history-row-${ID(c)}`);
      expect(row).toHaveTextContent('operator (with "Approve all settings changes", batch cccccccc)');
    }
    expect(within(history).getByTestId(`history-row-${ID('3')}`)).toHaveTextContent('Approved, then failed');
    expect(within(history).getByTestId(`history-row-${ID('1')}`)).toHaveTextContent('Approved and run');
  });

  it('code rules: not offered at one, and settings changes are not counted', async () => {
    await load([codeRule('4'), limits('1'), telemetry('2')]);
    renderBanner();
    expect(codeButton()).toBeNull();
    expect(button()).toHaveTextContent('Approve all settings changes (2)');
  });

  it('code rules: a separate button and batch that never mixes with settings', async () => {
    const fake = await load([codeRule('4'), limits('1'), codeRule('5'), telemetry('2'), unload()]);
    renderPage();
    expect(codeButton()).toHaveTextContent('Approve all code rules (2)');
    expect(button()).toHaveTextContent('Approve all settings changes (2)');
    fireEvent.click(codeButton()!);
    const list = screen.getByTestId('approve-all-list');
    const items = within(list).getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByText('rules-4 (behaviour b-4)')).toBeInTheDocument();
    expect(within(items[0]).getByTestId('approve-all-code-hashes')).toHaveTextContent(`Code: ${'4'.repeat(12)} (f)`);
    expect(within(items[0]).getByText('asked by agent coder-4')).toBeInTheDocument();
    expect(within(items[0]).getByRole('region', { name: 'Operator code' })).toHaveTextContent('def f(text): # 4 return 1');
    expect(within(items[1]).getByTestId('approve-all-code-hashes')).toHaveTextContent(`Code: ${'5'.repeat(12)} (f)`);
    expect(within(list).queryByText("Change a model's probe limits in miLLM")).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Approve these 2 code rules' }));
    await waitFor(() => expect(codeButton()).toBeNull());
    const posts = fake.calls.filter((c) => c.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0].body).toEqual({ approval_ids: [ID('4'), ID('5')], kind: 'code_rules' });
    // The settings batch is untouched and still offered; History marks the code-rule batch.
    expect(button()).toHaveTextContent('Approve all settings changes (2)');
    const history = await screen.findByTestId('approvals-history');
    expect(within(history).getByTestId(`history-row-${ID('4')}`)).toHaveTextContent(
      'operator (with "Approve all code rules", batch cccccccc)');
  });
});
