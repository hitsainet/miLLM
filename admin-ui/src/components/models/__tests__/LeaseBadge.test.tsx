/**
 * Feature 29 task 9.5: the lease badge, its placement, and the 10-second poll.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * LeaseBadge stops hiding past expiry (`if (!(ms > 0)) return null` removed) -> "hides"
 *   * ModelsPage no longer renders <LeaseBadge> beside the lock               -> "page"
 *   * modelsRefetchInterval drops the 10 s lease branch                        -> "poll"
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';

import type { LeaseSummary, ModelInfo } from '@/types';
import { useServerStore } from '@/stores/serverStore';
import { LeaseBadge } from '../LeaseBadge';
import { formatRemaining } from '../leaseTime';
import { modelsRefetchInterval } from '@/hooks/useModels';

const NOW = new Date('2026-10-06T12:00:00Z');

function lease(overrides: Partial<LeaseSummary> = {}): LeaseSummary {
  return {
    model_id: 1,
    model_name: 'm1',
    holder: 'midataworks',
    reason: 'label run 7',
    acquired_at: '2026-10-06T11:00:00Z',
    renewed_at: null,
    expires_at: '2026-10-06T13:12:00Z',
    ttl_seconds: 7200,
    seconds_remaining: 4320,
    ...overrides,
  };
}

function model(overrides: Partial<ModelInfo> = {}): ModelInfo {
  return {
    id: 1, name: 'm1', repo_id: 'org/m1', source: 'huggingface', quantization: 'FP16',
    params: '1B', disk_size_mb: 1, estimated_memory_mb: 1, local_path: '', status: 'loaded',
    created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z',
    ...overrides,
  };
}

const models = vi.hoisted(() => ({ list: [] as ModelInfo[] }));

vi.mock('@hooks/useModels', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/useModels')>();
  return {
    ...actual,
    useModels: () => ({
      models: models.list,
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
  };
});

describe('LeaseBadge', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it('renders holder, remaining time, reason and an aria-label', () => {
    render(<LeaseBadge lease={lease()} />);
    const badge = screen.getByTestId('lease-badge');
    expect(badge).toHaveTextContent('Leased by midataworks · expires in 1h 12m');
    expect(badge).toHaveTextContent('label run 7');
    expect(badge.getAttribute('aria-label')).toBe('Leased by midataworks, expires in 1h 12m');
    expect(badge.getAttribute('title')).toContain('label run 7');
    expect(badge.className).toContain('text-amber-400');
  });

  it('counts down every second and hides once the lease expires', () => {
    render(<LeaseBadge lease={lease({ expires_at: '2026-10-06T12:00:03Z' })} />);
    expect(screen.getByTestId('lease-badge')).toHaveTextContent('<1m');
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(screen.queryByTestId('lease-badge')).toBeNull();
  });

  it('formats remaining time', () => {
    expect(formatRemaining(72 * 60_000)).toBe('1h 12m');
    expect(formatRemaining(12 * 60_000 + 5_000)).toBe('12m');
    expect(formatRemaining(59_000)).toBe('<1m');
  });

  it('never renders a lease ID or any lease action', () => {
    const withId = { ...lease(), lease_id: 'SECRET-ID' } as LeaseSummary;
    const { container } = render(<LeaseBadge lease={withId} />);
    expect(container.innerHTML).not.toContain('SECRET-ID');
    expect(container.querySelector('button')).toBeNull();
  });
});

describe('Models page placement', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    vi.useRealTimers();
    useServerStore.getState().reset();
    models.list = [];
  });

  it('page: shows the lease beside the steering lock, and the lock keeps its own tooltip', async () => {
    models.list = [model({ locked: true, lease: lease() })];
    const { ModelsPage } = await import('@/pages/ModelsPage');
    render(<ModelsPage />);
    expect(screen.getAllByTestId('lease-badge')[0]).toHaveTextContent('Leased by midataworks');
    expect(screen.getByTitle('Locked for steering')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /lease|release|renew/i })).toBeNull();
  });

  it('page: an unleased model shows no badge', async () => {
    models.list = [model({ lease: null })];
    const { ModelsPage } = await import('@/pages/ModelsPage');
    render(<ModelsPage />);
    expect(screen.queryByTestId('lease-badge')).toBeNull();
  });
});

describe('modelsRefetchInterval', () => {
  it('poll: 10 s while any model is leased', () => {
    expect(modelsRefetchInterval([model({ lease: lease() })])).toBe(10_000);
  });
  it('keeps 2 s during downloads and loads', () => {
    expect(modelsRefetchInterval([model({ status: 'loading', lease: lease() })])).toBe(2000);
  });
  it('does not poll otherwise', () => {
    expect(modelsRefetchInterval([model()])).toBe(false);
    expect(modelsRefetchInterval(undefined)).toBe(false);
  });
});

describe('Loaded model card and details modal', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it('the loaded model card shows the lease under the name', async () => {
    const { LoadedModelCard } = await import('../LoadedModelCard');
    render(<LoadedModelCard model={model({ lease: lease() })} onUnload={vi.fn()} />);
    expect(screen.getByTestId('lease-badge')).toHaveTextContent('Leased by midataworks');
  });

  it('the details modal shows a Lease row with the reason', async () => {
    const { ModelDetailsModal } = await import('../ModelDetailsModal');
    render(<ModelDetailsModal model={model({ lease: lease() })} isOpen onClose={vi.fn()} />);
    expect(screen.getByText('Lease')).toBeInTheDocument();
    expect(screen.getByTestId('lease-badge')).toHaveTextContent('expires in 1h 12m');
    expect(screen.getAllByText('label run 7').length).toBeGreaterThan(0);
  });
});
