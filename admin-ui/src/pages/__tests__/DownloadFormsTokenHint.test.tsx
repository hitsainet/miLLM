/**
 * Feature 34: the PAGES pass the server's token status into the download forms.
 *
 * The forms' own hint is tested in HfTokenSettings.test.tsx; this pins the wiring that a form test
 * cannot see. MUTATION CONTROL: drop `tokenStatus={hfToken.status}` from ModelsPage (U8) or from
 * either SAEPage branch -> red.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useServerStore } from '@/stores/serverStore';

const api = vi.hoisted(() => ({ getHfToken: vi.fn() }));

vi.mock('@/services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/api')>();
  return { ...actual, settingsApi: { ...api } };
});

vi.mock('@hooks/useModels', () => ({
  useModels: () => ({
    models: [], isLoading: false, downloadModel: vi.fn(), isDownloading: false, load: vi.fn(),
    isLoadingModel: false, unloadModel: vi.fn(), isUnloading: false, delete: vi.fn(),
    isDeleting: false, previewModel: vi.fn(), isPreviewingModel: false, previewData: null,
    clearPreview: vi.fn(), lockModel: vi.fn(), unlockModel: vi.fn(), isLocking: false,
    isUnlockingModel: false,
  }),
}));

vi.mock('@hooks/useSAE', () => ({
  useSAE: () => ({
    saes: [], isLoading: false, downloadSAE: vi.fn(), isDownloading: false, attachSAE: vi.fn(),
    detachSAE: vi.fn(), isDetaching: false, deleteSAE: vi.fn(), cancelSAE: vi.fn(),
  }),
}));

vi.mock('@components/circuits', () => ({ AttachmentPanel: () => null }));

import { ModelsPage } from '../ModelsPage';
import { SAEPage } from '../SAEPage';

const SAVED = {
  configured: true, source: 'settings', masked: 'hf_…sav2', last_verified_at: null, user: null,
  stored: true, stored_error: null, last_check_error: null, encryption_available: true,
  env_configured: false,
};

afterEach(() => useServerStore.getState().reset());

describe('download forms on their pages', () => {
  it('the Models page tells the form a token is saved', async () => {
    api.getHfToken.mockResolvedValue(SAVED);
    render(<ModelsPage />);
    expect(await screen.findByText(/Leave empty to use the saved token \(hf_…sav2\)/)).toBeInTheDocument();
  });

  it('the SAE page tells the form, with no model loaded', async () => {
    api.getHfToken.mockResolvedValue(SAVED);
    render(<SAEPage />);
    expect(await screen.findByText(/Leave empty to use the saved token/)).toBeInTheDocument();
  });

  it('the SAE page tells the form, with a model loaded', async () => {
    api.getHfToken.mockResolvedValue(SAVED);
    useServerStore.setState({ loadedModel: { id: 1, name: 'm' } as never });
    render(<SAEPage />);
    expect(await screen.findByText(/Leave empty to use the saved token/)).toBeInTheDocument();
  });
});
