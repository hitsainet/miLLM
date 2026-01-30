/**
 * Integration tests for ModelsPage.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ModelsPage } from './ModelsPage';
import { useModelStore } from '../stores/modelStore';
import { modelService, socketService } from '../services';
import { Model, ModelStatus, QuantizationType } from '../types';

// Mock services
vi.mock('../services', () => ({
  modelService: {
    listModels: vi.fn(),
    downloadModel: vi.fn(),
    loadModel: vi.fn(),
    unloadModel: vi.fn(),
    deleteModel: vi.fn(),
    cancelDownload: vi.fn(),
  },
  socketService: {
    connect: vi.fn(),
    disconnect: vi.fn(),
    onDownloadProgress: vi.fn(() => vi.fn()),
    onDownloadComplete: vi.fn(() => vi.fn()),
    onDownloadError: vi.fn(() => vi.fn()),
    onLoadComplete: vi.fn(() => vi.fn()),
    onUnloadComplete: vi.fn(() => vi.fn()),
  },
}));

const mockModel: Model = {
  id: 1,
  name: 'gemma-2-2b',
  repoId: 'google/gemma-2-2b',
  source: 'huggingface',
  status: 'ready' as ModelStatus,
  quantization: 'Q4' as QuantizationType,
  createdAt: '2024-01-01T00:00:00Z',
  params: '2B',
  diskSizeMb: 1500,
  estimatedMemoryMb: 2000,
};

describe('ModelsPage', () => {
  beforeEach(() => {
    // Reset store state
    useModelStore.setState({
      models: [],
      loadedModelId: null,
      isLoading: false,
      downloadProgress: {},
      error: null,
    });
    vi.clearAllMocks();
    vi.mocked(modelService.listModels).mockResolvedValue([]);
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it('should render page title and description', async () => {
    render(<ModelsPage />);

    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(
      screen.getByText(/download and manage llm models for mechanistic interpretability/i)
    ).toBeInTheDocument();
  });

  it('should render download form', async () => {
    render(<ModelsPage />);

    expect(screen.getByText('Download Model')).toBeInTheDocument();
    expect(screen.getByLabelText(/huggingface repository/i)).toBeInTheDocument();
  });

  it('should fetch models on mount', async () => {
    vi.mocked(modelService.listModels).mockResolvedValue([mockModel]);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(modelService.listModels).toHaveBeenCalled();
    });
  });

  it('should display models after fetch', async () => {
    vi.mocked(modelService.listModels).mockResolvedValue([mockModel]);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText('gemma-2-2b')).toBeInTheDocument();
    });
  });

  it('should show loading spinner while fetching', () => {
    vi.mocked(modelService.listModels).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    useModelStore.setState({ isLoading: true });
    render(<ModelsPage />);

    expect(screen.getByText('Models')).toBeInTheDocument();
  });

  it('should connect to socket on mount', async () => {
    render(<ModelsPage />);

    await waitFor(() => {
      expect(socketService.connect).toHaveBeenCalled();
    });
  });

  it('should disconnect from socket on unmount', async () => {
    const { unmount } = render(<ModelsPage />);

    unmount();

    expect(socketService.disconnect).toHaveBeenCalled();
  });

  it('should show error banner when error exists', async () => {
    useModelStore.setState({ error: 'Something went wrong' });

    render(<ModelsPage />);

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('should clear error when close button is clicked', async () => {
    const user = userEvent.setup();
    useModelStore.setState({ error: 'Something went wrong' });

    render(<ModelsPage />);

    const errorBanner = screen.getByText('Something went wrong').closest('div');
    const closeButton = within(errorBanner!).getByRole('button');

    await user.click(closeButton);

    await waitFor(() => {
      expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    });
  });

  it('should start download when form is submitted', async () => {
    const user = userEvent.setup();
    const downloadingModel = { ...mockModel, status: 'downloading' as ModelStatus };
    vi.mocked(modelService.downloadModel).mockResolvedValue(downloadingModel);
    vi.mocked(modelService.listModels).mockResolvedValue([]);

    render(<ModelsPage />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'google/gemma-2-2b');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(modelService.downloadModel).toHaveBeenCalledWith({
        source: 'huggingface',
        repoId: 'google/gemma-2-2b',
        quantization: 'Q4',
      });
    });
  });

  it('should load model when Load button is clicked', async () => {
    const user = userEvent.setup();
    const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
    vi.mocked(modelService.listModels).mockResolvedValue([mockModel]);
    vi.mocked(modelService.loadModel).mockResolvedValue(loadedModel);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText('gemma-2-2b')).toBeInTheDocument();
    });

    const loadButton = screen.getByRole('button', { name: /load/i });
    await user.click(loadButton);

    await waitFor(() => {
      expect(modelService.loadModel).toHaveBeenCalledWith(1);
    });
  });

  it('should unload model when Unload button is clicked', async () => {
    const user = userEvent.setup();
    const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
    const readyModel = { ...mockModel, status: 'ready' as ModelStatus };
    vi.mocked(modelService.listModels).mockResolvedValue([loadedModel]);
    vi.mocked(modelService.unloadModel).mockResolvedValue(readyModel);

    useModelStore.setState({
      models: [loadedModel],
      loadedModelId: 1,
    });

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText('gemma-2-2b')).toBeInTheDocument();
    });

    const unloadButton = screen.getByRole('button', { name: /unload/i });
    await user.click(unloadButton);

    await waitFor(() => {
      expect(modelService.unloadModel).toHaveBeenCalledWith(1);
    });
  });

  it('should delete model when Delete button is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(modelService.listModels).mockResolvedValue([mockModel]);
    vi.mocked(modelService.deleteModel).mockResolvedValue(undefined);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText('gemma-2-2b')).toBeInTheDocument();
    });

    const deleteButton = screen.getByRole('button', { name: /delete/i });
    await user.click(deleteButton);

    await waitFor(() => {
      expect(modelService.deleteModel).toHaveBeenCalledWith(1);
    });
  });

  it('should show empty state when no models', async () => {
    vi.mocked(modelService.listModels).mockResolvedValue([]);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText(/no models yet/i)).toBeInTheDocument();
    });
  });

  it('should display multiple models', async () => {
    const model2: Model = {
      ...mockModel,
      id: 2,
      name: 'llama-2-7b',
      repoId: 'meta-llama/Llama-2-7b',
    };
    vi.mocked(modelService.listModels).mockResolvedValue([mockModel, model2]);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText('gemma-2-2b')).toBeInTheDocument();
      expect(screen.getByText('llama-2-7b')).toBeInTheDocument();
    });
  });

  it('should handle download error gracefully', async () => {
    const user = userEvent.setup();
    vi.mocked(modelService.downloadModel).mockRejectedValue(new Error('Network error'));
    vi.mocked(modelService.listModels).mockResolvedValue([]);

    render(<ModelsPage />);

    const input = screen.getByLabelText(/huggingface repository/i);
    await user.type(input, 'google/gemma-2-2b');

    const submitButton = screen.getByRole('button', { name: /download model/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/failed to start download/i)).toBeInTheDocument();
    });
  });
});
