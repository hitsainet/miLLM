/**
 * The load form's GPU selector offers each live card and sends the choice.
 *
 * It offered "Auto / CUDA (GPU) / CPU", named no card, and its value went
 * nowhere the backend read.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * go back to the static device options   -> "one option per card" fails
 *   * submit without `gpu`                    -> "submits the chosen card" fails
 *   * stop calling onGpuChange                -> "tells the page" fails
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ModelLoadForm } from '../ModelLoadForm';
import type { GpuMetrics } from '@/types';
import { LoadedModelCard } from '../LoadedModelCard';

const GPUS: GpuMetrics[] = [
  { index: 0, uuid: 'GPU-11111111-2222-3333-4444-555555555555', name: 'NVIDIA GeForce RTX 3080 Ti', utilization: 3, memory_used_mb: 1_288, memory_total_mb: 12_288, temperature: 41 },
  { index: 1, uuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', name: 'NVIDIA GeForce RTX 3090', utilization: 60, memory_used_mb: 1_576, memory_total_mb: 24_576, temperature: 66 },
];

describe('ModelLoadForm GPU selector', () => {
  it('offers Auto by default and one option per card', () => {
    render(<ModelLoadForm onSubmit={vi.fn()} gpus={GPUS} />);
    const select = screen.getByLabelText('GPU') as HTMLSelectElement;

    expect(select.value).toBe('auto');
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      'Auto (most free memory)',
      'GPU 0 · NVIDIA GeForce RTX 3080 Ti · 10.7 GB free',
      'GPU 1 · NVIDIA GeForce RTX 3090 · 22.5 GB free',
    ]);
    expect(select).not.toHaveTextContent('CPU');
  });

  it('tells the page and submits the chosen card', async () => {
    const onSubmit = vi.fn();
    const onGpuChange = vi.fn();
    render(<ModelLoadForm onSubmit={onSubmit} gpus={GPUS} onGpuChange={onGpuChange} />);

    await userEvent.selectOptions(screen.getByLabelText('GPU'), GPUS[1].uuid);
    expect(onGpuChange).toHaveBeenCalledWith(GPUS[1].uuid);

    await userEvent.type(screen.getByLabelText(/Hugging Face Repository ID/i), 'google/gemma-2-2b');
    await userEvent.click(screen.getByRole('button', { name: /Download & Load Model/i }));
    expect(onSubmit.mock.calls[0][0].gpu).toBe(GPUS[1].uuid);
  });

  it('submits auto when nothing was chosen', async () => {
    const onSubmit = vi.fn();
    render(<ModelLoadForm onSubmit={onSubmit} gpus={GPUS} />);
    await userEvent.type(screen.getByLabelText(/Hugging Face Repository ID/i), 'google/gemma-2-2b');
    await userEvent.click(screen.getByRole('button', { name: /Download & Load Model/i }));
    expect(onSubmit.mock.calls[0][0].gpu).toBe('auto');
  });
});

describe('LoadedModelCard placement', () => {
  const base = {
    id: 1, name: 'm', repo_id: 'o/m', source: 'huggingface' as const, quantization: 'FP16' as const,
    params: '8B', disk_size_mb: null, estimated_memory_mb: null, local_path: '', status: 'loaded' as const,
    created_at: '', updated_at: '',
  };

  it('shows the memory used on each card', () => {
    render(
      <LoadedModelCard
        model={{
          ...base,
          device: 'cuda:1',
          placement: {
            mode: 'single', reason: 'most_free_card_fits', requested: null, required_mb: 18_000,
            capacity_mb: 23_000, devices: ['cuda:1'], gpu_indices: [1],
            memory_by_device_mb: { 'cuda:1': 16_384 },
          },
        }}
        onUnload={vi.fn()}
      />,
    );
    expect(screen.getByTestId('placement-memory')).toHaveTextContent('cuda:1: 16.0 GB');
    expect(screen.queryByTestId('placement-split')).not.toBeInTheDocument();
  });

  it('says when the model is split across cards', () => {
    render(
      <LoadedModelCard
        model={{
          ...base,
          device: 'cuda:0, cuda:1',
          placement: {
            mode: 'all', reason: 'no_single_card_fits', requested: null, required_mb: 30_000,
            capacity_mb: 34_000, devices: ['cuda:0', 'cuda:1'], gpu_indices: [0, 1],
            memory_by_device_mb: { 'cuda:0': 9_000, 'cuda:1': 20_000 },
          },
        }}
        onUnload={vi.fn()}
      />,
    );
    expect(screen.getByTestId('placement-split')).toBeInTheDocument();
  });
});
