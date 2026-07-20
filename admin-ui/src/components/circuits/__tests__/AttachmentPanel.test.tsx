/**
 * Tests for AttachmentPanel (Feature 12): plural chip render, VRAM-warning
 * badge, empty state, and total readout.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import type { AttachmentStatusSet } from '@/types';

const attachmentsMock = vi.fn();

vi.mock('@/services/api', () => ({
  saeApi: {
    attachments: (...args: unknown[]) => attachmentsMock(...args),
  },
}));

import { AttachmentPanel } from '../AttachmentPanel';

function makeStatus(overrides: Partial<AttachmentStatusSet> = {}): AttachmentStatusSet {
  return {
    is_attached: true,
    count: 2,
    entries: [
      {
        sae_id: 'sae-a',
        layer: 10,
        memory_usage_mb: 64,
        steering_enabled: true,
        monitoring_enabled: false,
        steering_apply_count: 3,
      },
      {
        sae_id: 'sae-b',
        layer: 13,
        memory_usage_mb: 64,
        steering_enabled: false,
        monitoring_enabled: false,
        steering_apply_count: 0,
      },
    ],
    total_memory_usage_mb: 128,
    vram_envelope_mb: 200,
    vram_warning: false,
    ...overrides,
  };
}

function renderPanel() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <AttachmentPanel />
    </QueryClientProvider>,
  );
}

describe('AttachmentPanel', () => {
  beforeEach(() => {
    attachmentsMock.mockReset();
  });

  it('renders one chip per (sae_id, layer) with the total readout', async () => {
    attachmentsMock.mockResolvedValue(makeStatus());
    renderPanel();
    await waitFor(() => {
      expect(screen.getAllByTestId('attachment-chip')).toHaveLength(2);
    });
    expect(screen.getByText('sae-a')).toBeInTheDocument();
    expect(screen.getByText('sae-b')).toBeInTheDocument();
    expect(screen.getByText('L10')).toBeInTheDocument();
    expect(screen.getByText('L13')).toBeInTheDocument();
    // total / envelope readout
    expect(screen.getByText(/128 MB/)).toBeInTheDocument();
    expect(screen.getByText(/200 MB/)).toBeInTheDocument();
  });

  it('shows the VRAM-warning badge when over envelope', async () => {
    attachmentsMock.mockResolvedValue(
      makeStatus({ total_memory_usage_mb: 256, vram_warning: true }),
    );
    renderPanel();
    await waitFor(() => {
      expect(screen.getByTestId('vram-warning')).toBeInTheDocument();
    });
  });

  it('does NOT show the warning badge within envelope', async () => {
    attachmentsMock.mockResolvedValue(makeStatus({ vram_warning: false }));
    renderPanel();
    await waitFor(() => {
      expect(screen.getAllByTestId('attachment-chip')).toHaveLength(2);
    });
    expect(screen.queryByTestId('vram-warning')).not.toBeInTheDocument();
  });

  it('renders the empty state when nothing is attached', async () => {
    attachmentsMock.mockResolvedValue(
      makeStatus({ is_attached: false, count: 0, entries: [], total_memory_usage_mb: null }),
    );
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText('No SAEs attached')).toBeInTheDocument();
    });
  });
});
