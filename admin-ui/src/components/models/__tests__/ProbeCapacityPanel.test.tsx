/** Probe capacity panel (operator 2026-10-21): what drives the limits, and per-model edits. */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ProbeCapacity } from '@/types/probeCapacity';

const api = {
  probeCapacity: vi.fn(), setProbeLimits: vi.fn(), measureProbeCapacity: vi.fn(),
  probeLimitRecommendation: vi.fn(),
};
vi.mock('@/services/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/services/api')>();
  return { ...real, modelApi: { ...real.modelApi, ...api } };
});
const { ProbeCapacityPanel } = await import('../ProbeCapacityPanel');

const fixture: ProbeCapacity = {
  measured: {
    id: 1, repo_id: 'LiquidAI/LFM2.5-1.2B-Instruct', quantization: 'FP16',
    device: 'cuda:GPU-247aa582', decode_ms_per_token: 6.69, prefill_ms_per_1k: null,
    dense: { decode_own_ms: 0.0253, decode_fixed_ms: 0.0496, prefill_own_ms_per_1k: 0.0569,
             prefill_fixed_ms_per_1k: 0.0411 },
    sparse: null, method: 'seeded from the hardware record', source: 'record',
    measured_at: '2026-10-21T13:30:00Z',
  },
  configured: { max_armed: null, decode_budget: null, prefill_budget: null, reason: null },
  effective: { max_armed: 4, decode_ms_per_pass: 0.25, prefill_ms_per_1k: 0.5, headroom: 0.9,
               sources: { max_armed: 'global', decode: 'global', prefill: 'global' } },
  implied: { dense: { decode: { fits_budget: 7, fits_with_headroom: 6 },
                      prefill: { fits_budget: 8, fits_with_headroom: 7 }, binding: 'decode', n: 4 } },
  armed: { count: 2, projected: { decode: 0.1, prefill: 0.15 },
           share: { count: 0.5, decode: 0.444, prefill: 0.333 } },
};

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ProbeCapacityPanel modelId={7} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  api.probeCapacity.mockResolvedValue(fixture);
  api.setProbeLimits.mockResolvedValue(fixture);
  api.probeLimitRecommendation.mockResolvedValue(RECOMMENDATION);
});

const RECOMMENDATION = {
  model_id: 7,
  values: { max_behaviours: 1, decode_budget_ms: 0.42, prefill_budget_ms: 0.71 },
  reasons: ['Measurement 5 of the model, taken on cuda:GPU-247aa582.',
            '2 set(s) would take 12.5% of this model’s 6.69 ms decode step.'],
  estimated: true,
  recommendation_digest: 'd'.repeat(64),
};

describe('ProbeCapacityPanel', () => {
  it('shows the measured lines, what they imply, what is in force and the armed share', async () => {
    renderPanel();
    expect(await screen.findByTestId('cost-line-dense')).toHaveTextContent('0.050 + 0.0253 × N');
    expect(screen.getByTestId('probe-implied')).toHaveTextContent(
      'dense: up to 6 by the per-token budget, 7 by the prompt budget — the decode budget binds');
    expect(screen.getByTestId('probe-effective')).toHaveTextContent('In force: 4 probes (global)');
    expect(screen.getByTestId('probe-armed-share')).toHaveTextContent('per-token budget 44%');
    expect(api.probeCapacity).toHaveBeenCalledWith(7);
  });

  it('saves this model’s own limits, once, with the units chosen', async () => {
    renderPanel();
    await screen.findByTestId('probe-capacity');
    await userEvent.type(screen.getByLabelText('Most armed probes'), '6');
    await userEvent.type(screen.getByLabelText('Per-token budget value'), '3');
    await userEvent.selectOptions(screen.getByLabelText('Per-token budget unit'), 'pct');
    await userEvent.click(screen.getByTestId('save-probe-limits'));
    await vi.waitFor(() => expect(api.setProbeLimits).toHaveBeenCalledTimes(1));
    // 039: an override names its source; `max_behaviours` left empty is the global default.
    expect(api.setProbeLimits).toHaveBeenCalledWith(7, {
      max_behaviours: null, max_armed: 6, decode_budget: { value: 3, unit: 'pct' },
      prefill_budget: null, source: 'operator', set_by: 'admin-ui',
    });
    expect(api.measureProbeCapacity).not.toHaveBeenCalled();
  });

  it('approves the recommendation by its digest and its own values, once (039 FR-39.8)', async () => {
    renderPanel();
    expect(await screen.findByTestId('probe-recommendation')).toHaveTextContent(
      'Recommended: 1 behaviour, 0.42 ms per generated token, 0.71 ms per 1,000 prompt tokens');
    // Typing into the override fields must not leak into an approval.
    await userEvent.type(screen.getByLabelText('Most armed behaviours'), '9');
    await userEvent.click(screen.getByTestId('approve-recommendation'));
    await vi.waitFor(() => expect(api.setProbeLimits).toHaveBeenCalledTimes(1));
    expect(api.setProbeLimits).toHaveBeenCalledWith(7, {
      source: 'recommended-approved', recommendation_digest: 'd'.repeat(64), set_by: 'admin-ui',
    });
  });

  it('says where the behaviour limit came from, and when it is stale', async () => {
    api.probeCapacity.mockResolvedValue({
      ...fixture,
      effective: { ...fixture.effective, max_behaviours: 2, stale: true,
                   sources: { ...fixture.effective.sources, max_behaviours: 'recommended-approved' },
                   approval: { source: 'recommended-approved', set_by: 'admin-ui',
                               approved_at: null, measurement_id: 5,
                               newer_recommendation_available: false } },
    });
    renderPanel();
    expect(await screen.findByTestId('probe-behaviour-limit')).toHaveTextContent(
      'Behaviours: 2 (recommended-approved, set by admin-ui)');
    expect(screen.getByTestId('limits-stale')).toBeInTheDocument();
  });
});
