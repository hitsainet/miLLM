/**
 * Tests for ModelCard component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ModelCard } from './ModelCard';
import { Model, ModelStatus, QuantizationType, DownloadProgress } from '../../types';

const createMockModel = (overrides?: Partial<Model>): Model => ({
  id: 1,
  name: 'test-model',
  repoId: 'google/gemma-2-2b',
  source: 'huggingface',
  status: 'ready' as ModelStatus,
  quantization: 'fp16' as QuantizationType,
  createdAt: '2024-01-01T00:00:00Z',
  params: '2B',
  diskSizeMb: 4096,
  estimatedMemoryMb: 5000,
  ...overrides,
});

describe('ModelCard', () => {
  const mockOnLoad = vi.fn();
  const mockOnUnload = vi.fn();
  const mockOnDelete = vi.fn();
  const mockOnCancelDownload = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render model name and repo ID', () => {
    const model = createMockModel();
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText('test-model')).toBeInTheDocument();
    expect(screen.getByText('google/gemma-2-2b')).toBeInTheDocument();
  });

  it('should display model metadata', () => {
    const model = createMockModel();
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText('2B')).toBeInTheDocument();
    expect(screen.getByText('fp16')).toBeInTheDocument();
  });

  it('should show Load and Delete buttons for ready models', () => {
    const model = createMockModel({ status: 'ready' });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByRole('button', { name: /load/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
  });

  it('should show Unload button for loaded models', () => {
    const model = createMockModel({ status: 'loaded' });
    render(
      <ModelCard
        model={model}
        isLoaded={true}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByRole('button', { name: /unload/i })).toBeInTheDocument();
  });

  it('should show Cancel button for downloading models', () => {
    const model = createMockModel({ status: 'downloading' });
    const downloadProgress: DownloadProgress = {
      modelId: 1,
      progress: 50,
      downloadedBytes: 2048000000,
      totalBytes: 4096000000,
      speedBps: 10000000,
    };
    render(
      <ModelCard
        model={model}
        downloadProgress={downloadProgress}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
  });

  it('should show progress bar for downloading models', () => {
    const model = createMockModel({ status: 'downloading' });
    const downloadProgress: DownloadProgress = {
      modelId: 1,
      progress: 50,
      downloadedBytes: 2048000000,
      totalBytes: 4096000000,
    };
    render(
      <ModelCard
        model={model}
        downloadProgress={downloadProgress}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText(/downloading/i)).toBeInTheDocument();
  });

  it('should show loading progress for loading models', () => {
    const model = createMockModel({ status: 'loading' });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText(/loading to gpu/i)).toBeInTheDocument();
  });

  it('should show error message for error models', () => {
    const model = createMockModel({
      status: 'error',
      errorMessage: 'Download failed: Network error',
    });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText('Download failed: Network error')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
  });

  it('should call onLoad when Load button is clicked', async () => {
    const user = userEvent.setup();
    const model = createMockModel({ status: 'ready' });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    await user.click(screen.getByRole('button', { name: /load/i }));
    expect(mockOnLoad).toHaveBeenCalledTimes(1);
  });

  it('should call onUnload when Unload button is clicked', async () => {
    const user = userEvent.setup();
    const model = createMockModel({ status: 'loaded' });
    render(
      <ModelCard
        model={model}
        isLoaded={true}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    await user.click(screen.getByRole('button', { name: /unload/i }));
    expect(mockOnUnload).toHaveBeenCalledTimes(1);
  });

  it('should call onDelete when Delete button is clicked', async () => {
    const user = userEvent.setup();
    const model = createMockModel({ status: 'ready' });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    await user.click(screen.getByRole('button', { name: /delete/i }));
    expect(mockOnDelete).toHaveBeenCalledTimes(1);
  });

  it('should call onCancelDownload when Cancel button is clicked', async () => {
    const user = userEvent.setup();
    const model = createMockModel({ status: 'downloading' });
    const downloadProgress: DownloadProgress = {
      modelId: 1,
      progress: 50,
      downloadedBytes: 2048000000,
      totalBytes: 4096000000,
    };
    render(
      <ModelCard
        model={model}
        downloadProgress={downloadProgress}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    await user.click(screen.getByRole('button', { name: /cancel/i }));
    expect(mockOnCancelDownload).toHaveBeenCalledTimes(1);
  });

  it('should display formatted disk size', () => {
    const model = createMockModel({ diskSizeMb: 4096 });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText('4 GB')).toBeInTheDocument();
  });

  it('should display formatted memory estimate', () => {
    const model = createMockModel({ estimatedMemoryMb: 5000 });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    expect(screen.getByText(/~4\.9 GB/)).toBeInTheDocument();
  });

  it('should handle model without optional fields', () => {
    const model = createMockModel({
      params: undefined,
      diskSizeMb: undefined,
      estimatedMemoryMb: undefined,
      repoId: undefined,
    });
    render(
      <ModelCard
        model={model}
        isLoaded={false}
        onLoad={mockOnLoad}
        onUnload={mockOnUnload}
        onDelete={mockOnDelete}
        onCancelDownload={mockOnCancelDownload}
      />
    );

    // Should render without errors
    expect(screen.getByText('test-model')).toBeInTheDocument();
  });
});
