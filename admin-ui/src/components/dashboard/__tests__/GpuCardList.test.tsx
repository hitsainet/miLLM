/**
 * Each GPU gets its own tile; the aggregate cards cannot show a full card.
 *
 * MUTATION CONTROL: render only gpus[0] -> "a tile per card" fails.
 */

import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';

import { GpuCardList } from '../GpuCardList';
import type { GpuMetrics } from '@/types';

const GPUS: GpuMetrics[] = [
  { index: 0, uuid: 'GPU-0', name: 'NVIDIA GeForce RTX 3080 Ti', utilization: 7, memory_used_mb: 11_776, memory_total_mb: 12_288, temperature: 58 },
  { index: 1, uuid: 'GPU-1', name: 'NVIDIA GeForce RTX 3090', utilization: 91, memory_used_mb: 2_048, memory_total_mb: 24_576, temperature: 81 },
];

describe('GpuCardList', () => {
  it('renders a tile per card with its own name, memory and temperature', () => {
    render(<GpuCardList gpus={GPUS} />);

    const ti = screen.getByTestId('gpu-card-0');
    expect(within(ti).getByText('NVIDIA GeForce RTX 3080 Ti')).toBeInTheDocument();
    expect(within(ti).getByText(/11\.5\/12\.0 GB/)).toBeInTheDocument();

    const rtx = screen.getByTestId('gpu-card-1');
    expect(within(rtx).getByText('NVIDIA GeForce RTX 3090')).toBeInTheDocument();
    expect(within(rtx).getByText(/2\.0\/24\.0 GB/)).toBeInTheDocument();
    expect(within(rtx).getByText(/81°C/)).toBeInTheDocument();
  });

  it('says so when no card is reported', () => {
    render(<GpuCardList gpus={[]} />);
    expect(screen.getByText('No GPU reported.')).toBeInTheDocument();
  });
});
