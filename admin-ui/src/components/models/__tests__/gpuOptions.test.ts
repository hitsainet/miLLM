/**
 * The GPU selector's options come from the live card list.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * drop the per-card options (Auto only)          -> "one option per card" fails
 *   * keep a card that is no longer reported         -> "vanished card" fails
 *   * reset the selection when no list has arrived   -> "before the socket" fails
 */

import { describe, expect, it } from 'vitest';

import type { GpuMetrics } from '@/types';
import {
  gpuFreeMb,
  gpuOptions,
  gpuSelectionValue,
  gpuTotalsBytes,
  resolveGpuSelection,
} from '../gpuOptions';

const TI: GpuMetrics = {
  index: 0,
  uuid: 'GPU-11111111-2222-3333-4444-555555555555',
  name: 'NVIDIA GeForce RTX 3080 Ti',
  utilization: 3,
  memory_used_mb: 1_288,
  memory_total_mb: 12_288,
  temperature: 41,
};
const RTX: GpuMetrics = {
  index: 1,
  uuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
  name: 'NVIDIA GeForce RTX 3090',
  utilization: 60,
  memory_used_mb: 1_576,
  memory_total_mb: 24_576,
  temperature: 66,
};

describe('gpuOptions', () => {
  it('Auto first, then one option per card with name and free memory', () => {
    expect(gpuOptions([TI, RTX])).toEqual([
      { value: 'auto', label: 'Auto (most free memory)' },
      { value: TI.uuid, label: 'GPU 0 · NVIDIA GeForce RTX 3080 Ti · 10.7 GB free' },
      { value: RTX.uuid, label: 'GPU 1 · NVIDIA GeForce RTX 3090 · 22.5 GB free' },
    ]);
  });

  it('only Auto when no card is reported', () => {
    expect(gpuOptions([])).toEqual([{ value: 'auto', label: 'Auto (most free memory)' }]);
  });

  it('falls back to the index when a card has no UUID', () => {
    expect(gpuSelectionValue({ ...RTX, uuid: '' })).toBe('1');
  });

  it('free memory never goes negative', () => {
    expect(gpuFreeMb({ ...TI, memory_used_mb: 99_999 })).toBe(0);
  });
});

describe('resolveGpuSelection', () => {
  it('keeps a card that is reported, by UUID or index', () => {
    expect(resolveGpuSelection(RTX.uuid, [TI, RTX])).toBe(RTX.uuid);
    expect(resolveGpuSelection('1', [TI, RTX])).toBe('1');
  });

  it('a vanished card reads as auto', () => {
    expect(resolveGpuSelection('GPU-99999999-8888-7777-6666-555555555555', [TI, RTX])).toBe('auto');
  });

  it('keeps the choice before the socket has delivered any card', () => {
    expect(resolveGpuSelection(RTX.uuid, [])).toBe(RTX.uuid);
  });
});

describe('gpuTotalsBytes', () => {
  it('is each card, not a constant', () => {
    expect(gpuTotalsBytes([TI, RTX])).toEqual([12_288 * 1024 ** 2, 24_576 * 1024 ** 2]);
  });
});
