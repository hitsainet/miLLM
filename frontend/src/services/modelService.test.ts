/**
 * Tests for Model API service.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { modelService } from './modelService';
import { api } from './api';
import { Model, ModelStatus, QuantizationType } from '../types';

// Mock the api module
vi.mock('./api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

const mockModel: Model = {
  id: 1,
  name: 'test-model',
  repoId: 'test/model',
  source: 'huggingface',
  status: 'ready' as ModelStatus,
  quantization: 'fp16' as QuantizationType,
  createdAt: '2024-01-01T00:00:00Z',
};

describe('ModelService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('listModels', () => {
    it('should return list of models', async () => {
      const mockModels = [mockModel];
      vi.mocked(api.get).mockResolvedValue({
        data: { success: true, data: mockModels },
      });

      const result = await modelService.listModels();

      expect(api.get).toHaveBeenCalledWith('/api/models');
      expect(result).toEqual(mockModels);
    });

    it('should return empty array when no data', async () => {
      vi.mocked(api.get).mockResolvedValue({
        data: { success: true, data: null },
      });

      const result = await modelService.listModels();

      expect(result).toEqual([]);
    });
  });

  describe('getModel', () => {
    it('should return model by ID', async () => {
      vi.mocked(api.get).mockResolvedValue({
        data: { success: true, data: mockModel },
      });

      const result = await modelService.getModel(1);

      expect(api.get).toHaveBeenCalledWith('/api/models/1');
      expect(result).toEqual(mockModel);
    });

    it('should throw error when model not found', async () => {
      vi.mocked(api.get).mockResolvedValue({
        data: { success: false, data: null },
      });

      await expect(modelService.getModel(999)).rejects.toThrow('Model not found');
    });
  });

  describe('downloadModel', () => {
    it('should start download with correct payload', async () => {
      const downloadingModel = { ...mockModel, status: 'downloading' as ModelStatus };
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: downloadingModel },
      });

      const result = await modelService.downloadModel({
        source: 'huggingface',
        repoId: 'test/model',
        quantization: 'fp16',
      });

      expect(api.post).toHaveBeenCalledWith('/api/models', {
        source: 'huggingface',
        repo_id: 'test/model',
        local_path: undefined,
        quantization: 'fp16',
        trust_remote_code: false,
        hf_token: undefined,
      });
      expect(result).toEqual(downloadingModel);
    });

    it('should include optional parameters', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: mockModel },
      });

      await modelService.downloadModel({
        source: 'huggingface',
        repoId: 'test/model',
        quantization: 'q4',
        trustRemoteCode: true,
        hfToken: 'test-token',
      });

      expect(api.post).toHaveBeenCalledWith('/api/models', {
        source: 'huggingface',
        repo_id: 'test/model',
        local_path: undefined,
        quantization: 'q4',
        trust_remote_code: true,
        hf_token: 'test-token',
      });
    });

    it('should throw error on failure', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: false, data: null },
      });

      await expect(
        modelService.downloadModel({
          source: 'huggingface',
          repoId: 'test/model',
          quantization: 'fp16',
        })
      ).rejects.toThrow('Failed to start download');
    });
  });

  describe('loadModel', () => {
    it('should load model by ID', async () => {
      const loadedModel = { ...mockModel, status: 'loaded' as ModelStatus };
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: loadedModel },
      });

      const result = await modelService.loadModel(1);

      expect(api.post).toHaveBeenCalledWith('/api/models/1/load');
      expect(result).toEqual(loadedModel);
    });

    it('should throw error on failure', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: false, data: null },
      });

      await expect(modelService.loadModel(1)).rejects.toThrow('Failed to load model');
    });
  });

  describe('unloadModel', () => {
    it('should unload model by ID', async () => {
      const readyModel = { ...mockModel, status: 'ready' as ModelStatus };
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: readyModel },
      });

      const result = await modelService.unloadModel(1);

      expect(api.post).toHaveBeenCalledWith('/api/models/1/unload');
      expect(result).toEqual(readyModel);
    });

    it('should throw error on failure', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: false, data: null },
      });

      await expect(modelService.unloadModel(1)).rejects.toThrow('Failed to unload model');
    });
  });

  describe('deleteModel', () => {
    it('should delete model by ID', async () => {
      vi.mocked(api.delete).mockResolvedValue({
        data: { success: true, data: null },
      });

      await modelService.deleteModel(1);

      expect(api.delete).toHaveBeenCalledWith('/api/models/1');
    });
  });

  describe('cancelDownload', () => {
    it('should cancel download by model ID', async () => {
      const errorModel = { ...mockModel, status: 'error' as ModelStatus };
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: errorModel },
      });

      const result = await modelService.cancelDownload(1);

      expect(api.post).toHaveBeenCalledWith('/api/models/1/cancel');
      expect(result).toEqual(errorModel);
    });

    it('should throw error on failure', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: false, data: null },
      });

      await expect(modelService.cancelDownload(1)).rejects.toThrow('Failed to cancel download');
    });
  });

  describe('previewModel', () => {
    it('should preview model from HuggingFace', async () => {
      const previewResponse = {
        name: 'test-model',
        repoId: 'test/model',
        params: '7B',
        isGated: false,
      };
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: previewResponse },
      });

      const result = await modelService.previewModel({
        repoId: 'test/model',
      });

      expect(api.post).toHaveBeenCalledWith('/api/models/preview', {
        repo_id: 'test/model',
        hf_token: undefined,
      });
      expect(result).toEqual(previewResponse);
    });

    it('should include HuggingFace token when provided', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: true, data: { name: 'gated-model', isGated: true } },
      });

      await modelService.previewModel({
        repoId: 'meta-llama/Llama-2-7b',
        hfToken: 'test-token',
      });

      expect(api.post).toHaveBeenCalledWith('/api/models/preview', {
        repo_id: 'meta-llama/Llama-2-7b',
        hf_token: 'test-token',
      });
    });

    it('should throw error on failure', async () => {
      vi.mocked(api.post).mockResolvedValue({
        data: { success: false, data: null },
      });

      await expect(
        modelService.previewModel({ repoId: 'invalid/model' })
      ).rejects.toThrow('Failed to preview model');
    });
  });
});
