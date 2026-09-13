/**
 * The Models page stores the card chosen in either GPU selector.
 *
 * The page owns the selection so the list's Load/Switch buttons send it (via
 * useModels, which reads `loadGpu` from the store — pinned in useModels.test).
 * Found by mutation: removing either wiring line left the whole suite green.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * header selector's onChange no longer calls setLoadGpu  -> "header" fails
 *   * form rendered without onGpuChange={setLoadGpu}          -> "form" fails
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { useServerStore } from '@/stores/serverStore';
import type { GpuMetrics } from '@/types';

vi.mock('@hooks/useModels', () => ({
  useModels: () => ({
    models: [],
    isLoading: false,
    downloadModel: vi.fn(),
    isDownloading: false,
    load: vi.fn(),
    isLoadingModel: false,
    unloadModel: vi.fn(),
    isUnloading: false,
    delete: vi.fn(),
    isDeleting: false,
    previewModel: vi.fn(),
    isPreviewingModel: false,
    previewData: null,
    clearPreview: vi.fn(),
    lockModel: vi.fn(),
    unlockModel: vi.fn(),
    isLocking: false,
    isUnlockingModel: false,
  }),
}));

import { ModelsPage } from '../ModelsPage';

const GPUS: GpuMetrics[] = [
  { index: 0, uuid: 'GPU-11111111-2222-3333-4444-555555555555', name: 'NVIDIA GeForce RTX 3080 Ti', utilization: 3, memory_used_mb: 1_288, memory_total_mb: 12_288, temperature: 41 },
  { index: 1, uuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', name: 'NVIDIA GeForce RTX 3090', utilization: 60, memory_used_mb: 1_576, memory_total_mb: 24_576, temperature: 66 },
];

describe('ModelsPage GPU selection', () => {
  beforeEach(() => {
    useServerStore.getState().setSystemMetrics({ gpus: GPUS });
  });
  afterEach(() => useServerStore.getState().reset());

  it('the header selector stores the card for the next Load or Switch', async () => {
    render(<ModelsPage />);
    await userEvent.selectOptions(screen.getByLabelText('GPU for loading'), GPUS[1].uuid);
    expect(useServerStore.getState().loadGpu).toBe(GPUS[1].uuid);
  });

  it('the load form selector stores the card too, and both show it', async () => {
    render(<ModelsPage />);
    await userEvent.selectOptions(screen.getByLabelText('GPU'), GPUS[0].uuid);
    expect(useServerStore.getState().loadGpu).toBe(GPUS[0].uuid);
    expect((screen.getByLabelText('GPU for loading') as HTMLSelectElement).value).toBe(GPUS[0].uuid);
  });
});
