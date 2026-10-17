/**
 * Tests for AttachmentPanel (Feature 12): plural chip render, VRAM-warning
 * badge, empty state, and total readout.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { AttachmentStatusSet, SAEInfo } from '@/types';

const attachmentsMock = vi.fn();
const attachSetMock = vi.fn();

vi.mock('@/services/api', () => ({
  // Declared inside the factory: vi.mock is hoisted above any top-level
  // binding, so a class defined outside would be in its TDZ here.
  ApiError: class ApiError extends Error {
    code: string;
    details?: Record<string, unknown>;
    constructor(code: string, message: string, details?: Record<string, unknown>) {
      super(message);
      this.name = 'ApiError';
      this.code = code;
      this.details = details;
    }
  },
  saeApi: {
    attachments: (...args: unknown[]) => attachmentsMock(...args),
    attachSet: (...args: unknown[]) => attachSetMock(...args),
  },
}));

function makeSAE(id: string, layer: number): SAEInfo {
  return {
    id,
    repository_id: `repo/${id}`,
    revision: 'main',
    name: id,
    format: 'npz',
    d_in: 2304,
    d_sae: 16384,
    trained_on: 'gemma-2-2b',
    trained_layer: layer,
    width: '16k',
    average_l0: 50,
    file_size_bytes: 1024,
    status: 'cached',
    error_message: null,
    created_at: '2026-07-20T00:00:00Z',
    updated_at: '2026-07-20T00:00:00Z',
  };
}

// The panel derives its picker rows from the SAE list and the loaded model.
vi.mock('@/hooks/useSAE', () => ({
  useSAE: () => ({ saes: [makeSAE('sae-a', 10), makeSAE('sae-b', 13)] }),
}));

vi.mock('@/stores/serverStore', () => ({
  useServerStore: (selector: (s: unknown) => unknown) =>
    selector({ loadedModel: { name: 'gemma-2-2b' } }),
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

function renderPanel(props: { minCount?: number } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <AttachmentPanel {...props} />
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

  it('renders nothing at or below minCount (single-SAE case on SAEPage)', async () => {
    attachmentsMock.mockResolvedValue(
      makeStatus({ count: 1, entries: [makeStatus().entries[0]] }),
    );
    const { container } = renderPanel({ minCount: 1 });
    // Nothing rendered — the single-SAE case is covered by AttachedSAECard.
    await waitFor(() => expect(attachmentsMock).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText('No SAEs attached')).not.toBeInTheDocument();
  });

  it('renders above minCount (a real multi-SAE circuit)', async () => {
    attachmentsMock.mockResolvedValue(makeStatus()); // count = 2
    renderPanel({ minCount: 1 });
    await waitFor(() => {
      expect(screen.getAllByTestId('attachment-chip')).toHaveLength(2);
    });
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

  it('offers the Attach set control alongside the attached chips', async () => {
    attachmentsMock.mockResolvedValue(makeStatus());
    renderPanel();
    await waitFor(() => {
      expect(screen.getAllByTestId('attachment-chip')).toHaveLength(2);
    });
    expect(screen.getByTestId('open-attach-set')).toBeInTheDocument();
  });

  it('offers the Attach set control from the empty state too', async () => {
    attachmentsMock.mockResolvedValue(
      makeStatus({ is_attached: false, count: 0, entries: [], total_memory_usage_mb: null }),
    );
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText('No SAEs attached')).toBeInTheDocument();
    });
    expect(screen.getByTestId('open-attach-set')).toBeInTheDocument();
  });

  it('opens the picker, pre-selected from the attached set', async () => {
    const user = userEvent.setup();
    attachmentsMock.mockResolvedValue(makeStatus());
    renderPanel();
    await waitFor(() => {
      expect(screen.getByTestId('open-attach-set')).toBeInTheDocument();
    });

    await user.click(screen.getByTestId('open-attach-set'));

    // The dialog is open; both attached (sae_id, layer) keys are pre-selected,
    // so the control reads as "edit the set" rather than "start over".
    expect(await screen.findByText('Attach SAE set')).toBeInTheDocument();
    expect(screen.getByLabelText('sae-a at layer 10')).toBeChecked();
    expect(screen.getByLabelText('sae-b at layer 13')).toBeChecked();
  });
});
