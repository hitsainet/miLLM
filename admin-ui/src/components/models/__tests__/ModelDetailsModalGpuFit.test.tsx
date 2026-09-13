/**
 * The details modal judges GGUF fit against the node's LIVE cards.
 *
 * It passed a constant 24 GB. Found by mutation: feeding the picker an empty
 * card list left the whole suite green, because nothing rendered the modal.
 *
 * MUTATION CONTROL: pass `gpuTotalsBytes([])` instead of the store's cards ->
 * the heading reads "GPU fit" with no verdicts, and this fails.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';

import { ModelDetailsModal } from '../ModelDetailsModal';
import { useServerStore } from '@/stores/serverStore';
import type { GpuMetrics, ModelPreviewResponse } from '@/types';

const GB = 1024 ** 3;

const GPUS: GpuMetrics[] = [
  { index: 0, uuid: 'GPU-0', name: 'NVIDIA GeForce RTX 3080 Ti', utilization: 3, memory_used_mb: 1_000, memory_total_mb: 12_288, temperature: 41 },
  { index: 1, uuid: 'GPU-1', name: 'NVIDIA GeForce RTX 3090', utilization: 60, memory_used_mb: 1_000, memory_total_mb: 24_576, temperature: 66 },
];

const PREVIEW = {
  name: 'big-model-GGUF',
  params: null,
  architecture: null,
  requires_trust_remote_code: false,
  is_gated: false,
  estimated_sizes: null,
  downloads: 0,
  likes: 0,
  tags: null,
  pipeline_tag: null,
  model_type: null,
  architectures: null,
  license: null,
  language: null,
  revision: null,
  gguf_quants: [
    {
      label: 'Q8_0',
      files: [{ path: 'm-Q8_0.gguf', size_bytes: 26 * GB }],
      total_size_bytes: 26 * GB,
      is_split: false,
      quant_parsed: true,
    },
  ],
  gguf_architecture: null,
  gguf_context_length: null,
  gguf_total_params: null,
} satisfies ModelPreviewResponse;

describe('ModelDetailsModal GGUF fit hint', () => {
  afterEach(() => useServerStore.getState().reset());

  it('uses every live card, so a quant only the two cards hold says so', () => {
    useServerStore.getState().setSystemMetrics({ gpus: GPUS });
    render(<ModelDetailsModal previewData={PREVIEW} isOpen onClose={vi.fn()} />);

    expect(screen.getByText('On 2 cards (12 + 24 GB)')).toBeInTheDocument();
    expect(
      within(screen.getByTestId('gguf-quant-Q8_0')).getByText('fits across cards'),
    ).toBeInTheDocument();
  });
});
