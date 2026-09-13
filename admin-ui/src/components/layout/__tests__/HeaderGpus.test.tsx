/**
 * The header shows each card's memory when there is more than one card.
 *
 * MUTATION CONTROL: drop the per-card chips -> "each card" fails.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { Header } from '../Header';
import { useServerStore } from '@/stores/serverStore';
import type { GpuMetrics } from '@/types';

const GPUS: GpuMetrics[] = [
  { index: 0, uuid: 'GPU-0', name: 'NVIDIA GeForce RTX 3080 Ti', utilization: 7, memory_used_mb: 11_776, memory_total_mb: 12_288, temperature: 58 },
  { index: 1, uuid: 'GPU-1', name: 'NVIDIA GeForce RTX 3090', utilization: 91, memory_used_mb: 2_048, memory_total_mb: 24_576, temperature: 81 },
];

function renderHeader() {
  return render(
    <MemoryRouter>
      <Header />
    </MemoryRouter>,
  );
}

describe('Header GPU display', () => {
  afterEach(() => useServerStore.getState().reset());

  it('shows each card beside the aggregate summary', () => {
    useServerStore.getState().setSystemMetrics({
      gpuMemoryUsed: 13_824,
      gpuMemoryTotal: 36_864,
      gpuUtilization: 49,
      gpuTemperature: 81,
      gpus: GPUS,
    });
    renderHeader();

    expect(screen.getByTestId('header-gpu-0')).toHaveTextContent('11.5/12.0');
    expect(screen.getByTestId('header-gpu-1')).toHaveTextContent('2.0/24.0');
    expect(screen.getByTestId('header-gpu-1')).toHaveAttribute(
      'title',
      expect.stringContaining('NVIDIA GeForce RTX 3090'),
    );
    // The summary is still there.
    expect(screen.getByText('13.5/36.0 GB')).toBeInTheDocument();
  });

  it('one card needs no per-card chip; the summary is that card', () => {
    useServerStore.getState().setSystemMetrics({
      gpuMemoryUsed: 2_048,
      gpuMemoryTotal: 24_576,
      gpus: [GPUS[1]],
    });
    renderHeader();
    expect(screen.queryByTestId('header-gpu-1')).not.toBeInTheDocument();
  });
});
