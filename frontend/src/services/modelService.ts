/**
 * Model API service for miLLM frontend.
 */

import { api } from './api';
import {
  Model,
  ModelDownloadRequest,
  ModelPreviewRequest,
  ModelPreviewResponse,
  ApiResponse,
} from '../types';

class ModelService {
  private readonly basePath = '/api/models';

  /**
   * List all models.
   */
  async listModels(): Promise<Model[]> {
    const response = await api.get<ApiResponse<Model[]>>(this.basePath);
    return response.data.data || [];
  }

  /**
   * Get a single model by ID.
   */
  async getModel(modelId: number): Promise<Model> {
    const response = await api.get<ApiResponse<Model>>(`${this.basePath}/${modelId}`);
    if (!response.data.data) {
      throw new Error('Model not found');
    }
    return response.data.data;
  }

  /**
   * Start downloading a model.
   */
  async downloadModel(request: ModelDownloadRequest): Promise<Model> {
    const response = await api.post<ApiResponse<Model>>(this.basePath, {
      source: request.source,
      repo_id: request.repoId,
      local_path: request.localPath,
      quantization: request.quantization,
      trust_remote_code: request.trustRemoteCode || false,
      hf_token: request.hfToken,
    });
    if (!response.data.data) {
      throw new Error('Failed to start download');
    }
    return response.data.data;
  }

  /**
   * Load a model into GPU memory.
   */
  async loadModel(modelId: number): Promise<Model> {
    const response = await api.post<ApiResponse<Model>>(`${this.basePath}/${modelId}/load`);
    if (!response.data.data) {
      throw new Error('Failed to load model');
    }
    return response.data.data;
  }

  /**
   * Unload a model from GPU memory.
   */
  async unloadModel(modelId: number): Promise<Model> {
    const response = await api.post<ApiResponse<Model>>(`${this.basePath}/${modelId}/unload`);
    if (!response.data.data) {
      throw new Error('Failed to unload model');
    }
    return response.data.data;
  }

  /**
   * Delete a model.
   */
  async deleteModel(modelId: number): Promise<void> {
    await api.delete<ApiResponse<null>>(`${this.basePath}/${modelId}`);
  }

  /**
   * Cancel an in-progress download.
   */
  async cancelDownload(modelId: number): Promise<Model> {
    const response = await api.post<ApiResponse<Model>>(`${this.basePath}/${modelId}/cancel`);
    if (!response.data.data) {
      throw new Error('Failed to cancel download');
    }
    return response.data.data;
  }

  /**
   * Preview a model from HuggingFace without downloading.
   */
  async previewModel(request: ModelPreviewRequest): Promise<ModelPreviewResponse> {
    const response = await api.post<ApiResponse<ModelPreviewResponse>>(`${this.basePath}/preview`, {
      repo_id: request.repoId,
      hf_token: request.hfToken,
    });
    if (!response.data.data) {
      throw new Error('Failed to preview model');
    }
    return response.data.data;
  }
}

export const modelService = new ModelService();
