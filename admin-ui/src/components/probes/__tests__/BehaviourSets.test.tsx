/**
 * Behaviour sets on the Probe Monitors page (Feature 39, FTASKS 7.2).
 *
 * The API client is mocked at its module; the component and its hooks run for real. Requests are
 * asserted by PAYLOAD and CALL COUNT — a test that only checks "was called" passes against a
 * request that sends the wrong body.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { BehaviourArmResult, BehaviourSummary } from '@/types/probe';

const api = { behaviours: vi.fn(), armBehaviour: vi.fn(), disarmBehaviour: vi.fn() };
vi.mock('@/services/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/services/api')>();
  return { ...real, probesApi: { ...real.probesApi, ...api } };
});
const { BehaviourSets } = await import('../BehaviourSets');

function behaviour(key: string, over: Partial<BehaviourSummary> = {}): BehaviourSummary {
  return {
    key, imported: 3, armed: 0, armed_probe_ids: [],
    versions: [{ version: 1, probe_ids: ['pr_a', 'pr_b', 'pr_c'], imported: 3, armed: 0,
                 missing_probe_ids: [] }],
    set: null, armed_before_unload: null, attached_rule_sets: [], ...over,
  };
}

const DRY: BehaviourArmResult = {
  behaviour: { key: 'k-one', version: 1 }, dry_run: true, armed: [], already_armed: [],
  probes: [
    { probe_id: 'pr_a', gate: 'all', outcome: 'passed',
      cost: { decode: 0.03, prefill: 0.05, first_on_layer: true, layer: 0, device: 'cpu' } },
    { probe_id: 'pr_b', gate: 'rung', outcome: 'refused', code: 'UNVALIDATED_PROBE' },
  ],
  projected: { decode: 0.03, prefill: 0.05 },
  budget: { decode_ms_per_pass: 0.25, prefill_ms_per_1k: 0.5, headroom: 0.9 },
  would_refuse: { code: 'UNVALIDATED_PROBE', details: {} },
  set: null,
};

function renderSets() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <BehaviourSets />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
});

describe('BehaviourSets', () => {
  it('renders a three-probe group and its partial state, and arms it with one request', async () => {
    api.behaviours.mockResolvedValue({ data: [behaviour('k-one', {
      armed: 2, set: { version: 1, armed_members: 2, total_members: 3, complete: false,
                       missing_probe_ids: ['pr_c'], text: 'partial: 2 of 3 armed' } })], next: null });
    api.armBehaviour.mockResolvedValue({ ...DRY, dry_run: false, probes: [], would_refuse: undefined });
    renderSets();
    expect(await screen.findByTestId('behaviour-k-one')).toHaveTextContent('2 of 3 armed');
    expect(screen.getByTestId('partial-badge')).toHaveTextContent('partial: 2 of 3 armed');
    await userEvent.click(screen.getByRole('button', { name: 'Arm' }));
    await vi.waitFor(() => expect(api.armBehaviour).toHaveBeenCalledTimes(1));
    expect(api.armBehaviour).toHaveBeenCalledWith('k-one', { dryRun: false });
  });

  it('shows a dry run’s per-probe outcomes and its projected cost, arming nothing', async () => {
    api.behaviours.mockResolvedValue({ data: [behaviour('k-one')], next: null });
    api.armBehaviour.mockResolvedValue(DRY);
    renderSets();
    await userEvent.click(await screen.findByRole('button', { name: 'Dry run' }));
    await vi.waitFor(() => expect(api.armBehaviour).toHaveBeenCalledTimes(1));
    expect(api.armBehaviour).toHaveBeenCalledWith('k-one', { dryRun: true });
    expect(await screen.findByTestId('dry-run')).toHaveTextContent(
      'nothing was armed. Arming would be refused: UNVALIDATED_PROBE');
    expect(screen.getByTestId('outcome-pr_b')).toHaveTextContent('refused (UNVALIDATED_PROBE)');
    expect(screen.getByTestId('probe-budget-meter')).toBeInTheDocument();
  });

  it('lists every probe of a refused set with its own cost', async () => {
    const { ApiError } = await import('@/services/api');
    api.behaviours.mockResolvedValue({ data: [behaviour('k-one')], next: null });
    api.armBehaviour.mockRejectedValue(new ApiError('PROBE_BUDGET', 'over budget', {
      probes: [
        { probe_id: 'pr_a', gate: 'cost', outcome: 'refused',
          cost: { decode: 0.11, prefill: 0.2, first_on_layer: true, layer: 0, device: 'cpu' } },
        { probe_id: 'pr_b', gate: 'cost', outcome: 'refused',
          cost: { decode: 0.12, prefill: 0.21, first_on_layer: false, layer: 0, device: 'cpu' } },
      ],
    }));
    renderSets();
    await userEvent.click(await screen.findByRole('button', { name: 'Arm' }));
    expect(await screen.findByTestId('set-refusal')).toHaveTextContent('PROBE_BUDGET');
    expect(screen.getByTestId('outcome-pr_a')).toHaveTextContent('0.110 ms');
    expect(screen.getByTestId('outcome-pr_b')).toHaveTextContent('0.120 ms');
  });

  it('renders one page of 500 behaviours and asks for the next with its cursor', async () => {
    const page = Array.from({ length: 500 }, (_, i) => behaviour(`k-${String(i).padStart(3, '0')}`));
    api.behaviours
      .mockResolvedValueOnce({ data: page, next: 'CURSOR-2' })
      .mockResolvedValueOnce({ data: [behaviour('k-500')], next: null });
    renderSets();
    expect(await screen.findByTestId('behaviour-k-499')).toBeInTheDocument();
    expect(screen.queryByTestId('behaviour-k-500')).toBeNull();
    expect(api.behaviours).toHaveBeenCalledTimes(1);
    expect(api.behaviours).toHaveBeenLastCalledWith({ key: undefined, cursor: null });
    await userEvent.click(screen.getByTestId('behaviours-more'));
    expect(await screen.findByTestId('behaviour-k-500')).toBeInTheDocument();
    expect(api.behaviours).toHaveBeenCalledTimes(2);
    expect(api.behaviours).toHaveBeenLastCalledWith({ key: undefined, cursor: 'CURSOR-2' });
  });

  it('names behaviours that were armed before an unload or a restart', async () => {
    api.behaviours.mockResolvedValue({ data: [behaviour('k-one', {
      armed_before_unload: { probe_ids: ['pr_a'], reasons: ['unloaded'] } })], next: null });
    renderSets();
    expect(await screen.findByTestId('armed-before')).toBeInTheDocument();
  });
});
