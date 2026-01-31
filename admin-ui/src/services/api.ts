/**
 * @fileoverview API client services for miLLM admin UI.
 *
 * This module provides typed API clients for all miLLM management endpoints:
 * - Model management (download, load, unload)
 * - SAE management (download, attach, detach)
 * - Feature steering control
 * - Monitoring configuration
 * - Profile management
 * - Server status
 *
 * All API calls use a consistent error handling pattern via the {@link ApiError} class.
 *
 * @module services/api
 */

import type {
  ApiResponse,
  ModelInfo,
  ModelDownloadRequest,
  ModelPreviewResponse,
  SAEInfo,
  DownloadSAERequest,
  AttachSAERequest,
  SteeringState,
  SetSteeringRequest,
  BatchSteeringRequest,
  MonitoringConfig,
  ConfigureMonitoringRequest,
  MonitoringHistory,
  Profile,
  CreateProfileRequest,
  UpdateProfileRequest,
  ProfileExport,
  ServerStatus,
} from '@/types';

/** Base URL for all API requests */
const API_BASE = '/api';

/**
 * Custom error class for API-related errors.
 *
 * Provides structured error information including error codes and optional details
 * for better error handling and user feedback.
 *
 * @extends Error
 *
 * @example
 * ```typescript
 * try {
 *   await modelApi.load(123);
 * } catch (error) {
 *   if (error instanceof ApiError) {
 *     console.error(`Error ${error.code}: ${error.message}`);
 *     if (error.details) {
 *       console.error('Details:', error.details);
 *     }
 *   }
 * }
 * ```
 */
export class ApiError extends Error {
  /** Error code from the API (e.g., 'MODEL_NOT_FOUND', 'INSUFFICIENT_MEMORY') */
  code: string;
  /** Additional error details from the API response */
  details?: Record<string, unknown>;

  /**
   * Creates a new ApiError instance.
   *
   * @param code - Error code identifier
   * @param message - Human-readable error message
   * @param details - Optional additional error details
   */
  constructor(code: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.details = details;
  }
}

/**
 * Makes an HTTP request to the API and handles the response.
 *
 * This is the core request function used by all API methods. It:
 * - Prepends the API base URL to endpoints
 * - Sets JSON content-type headers
 * - Parses the response and extracts data
 * - Throws {@link ApiError} for unsuccessful responses
 *
 * @template T - The expected type of the response data
 * @param endpoint - API endpoint path (e.g., '/models')
 * @param options - Fetch options (method, body, headers, etc.)
 * @returns Promise resolving to the response data
 * @throws {ApiError} When the API returns an error response
 *
 * @internal
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const response = await fetch(url, { ...options, headers });
  const data: ApiResponse<T> = await response.json();

  if (!data.success || data.error) {
    throw new ApiError(
      data.error?.code ?? 'UNKNOWN_ERROR',
      data.error?.message ?? 'An unknown error occurred',
      data.error?.details
    );
  }

  return data.data as T;
}

/**
 * Model management API client.
 *
 * Provides methods for downloading, loading, and managing LLM models
 * from HuggingFace. Supports quantization and memory estimation.
 *
 * @example
 * ```typescript
 * // List all downloaded models
 * const models = await modelApi.list();
 *
 * // Download a new model with quantization
 * const model = await modelApi.download({
 *   repo_id: 'google/gemma-2-2b',
 *   quantization: 'Q4',
 * });
 *
 * // Load a model into GPU memory
 * await modelApi.load(model.id);
 * ```
 */
export const modelApi = {
  /**
   * Lists all downloaded models.
   * @returns Promise resolving to array of model information
   */
  list: () => request<ModelInfo[]>('/models'),

  /**
   * Gets detailed information about a specific model.
   * @param id - Model ID
   * @returns Promise resolving to model information
   */
  get: (id: number) => request<ModelInfo>(`/models/${id}`),

  /**
   * Initiates download of a model from HuggingFace.
   * Progress updates are sent via WebSocket.
   * @param req - Download request with repo_id and quantization options
   * @returns Promise resolving to the created model record
   */
  download: (req: ModelDownloadRequest) =>
    request<ModelInfo>('/models', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Previews model information without downloading.
   * Fetches metadata like parameter count and memory requirements.
   * @param repo_id - HuggingFace repository ID (e.g., 'google/gemma-2-2b')
   * @returns Promise resolving to model preview information
   */
  preview: (repo_id: string) =>
    request<ModelPreviewResponse>('/models/preview', {
      method: 'POST',
      body: JSON.stringify({ repo_id }),
    }),

  /**
   * Loads a downloaded model into GPU memory.
   * Only one model can be loaded at a time.
   * @param id - Model ID to load
   * @returns Promise resolving to updated model information
   */
  load: (id: number) =>
    request<ModelInfo>(`/models/${id}/load`, {
      method: 'POST',
    }),

  /**
   * Unloads the currently loaded model from GPU memory.
   * @param id - Model ID to unload
   * @returns Promise resolving when unload is complete
   */
  unload: (id: number) =>
    request<void>(`/models/${id}/unload`, {
      method: 'POST',
    }),

  /**
   * Deletes a model from local storage.
   * Model must not be currently loaded.
   * @param id - Model ID to delete
   * @returns Promise resolving when deletion is complete
   */
  delete: (id: number) =>
    request<void>(`/models/${id}`, {
      method: 'DELETE',
    }),

  /**
   * Cancels an in-progress model download.
   * @param id - Model ID with active download
   * @returns Promise resolving when cancellation is processed
   */
  cancelDownload: (id: number) =>
    request<void>(`/models/${id}/cancel`, {
      method: 'POST',
    }),
};

/**
 * SAE (Sparse Autoencoder) management API client.
 *
 * Provides methods for downloading SAEs from HuggingFace and
 * attaching them to loaded models for feature steering.
 *
 * @example
 * ```typescript
 * // Download an SAE
 * const sae = await saeApi.download({
 *   repo_id: 'google/gemma-scope-2b-pt-res',
 *   model_id: 1,
 *   layer: 12,
 * });
 *
 * // Attach SAE to enable steering
 * await saeApi.attach({ sae_id: sae.id });
 * ```
 */
export const saeApi = {
  /**
   * Lists all downloaded SAEs.
   * @returns Promise resolving to array of SAE information
   */
  list: () => request<SAEInfo[]>('/saes'),

  /**
   * Gets detailed information about a specific SAE.
   * @param id - SAE ID
   * @returns Promise resolving to SAE information
   */
  get: (id: number) => request<SAEInfo>(`/saes/${id}`),

  /**
   * Initiates download of an SAE from HuggingFace.
   * SAE must be linked to a compatible downloaded model.
   * @param req - Download request with repo_id, model_id, and layer
   * @returns Promise resolving to the created SAE record
   */
  download: (req: DownloadSAERequest) =>
    request<SAEInfo>('/saes', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Attaches an SAE to the currently loaded model.
   * Enables feature steering and monitoring capabilities.
   * Only one SAE can be attached at a time.
   * @param req - Attach request with SAE ID
   * @returns Promise resolving to updated SAE information
   */
  attach: (req: AttachSAERequest) =>
    request<SAEInfo>('/saes/attach', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Detaches the currently attached SAE.
   * Disables feature steering until another SAE is attached.
   * @returns Promise resolving when detachment is complete
   */
  detach: () =>
    request<void>('/saes/detach', {
      method: 'POST',
    }),

  /**
   * Deletes an SAE from local storage.
   * SAE must not be currently attached.
   * @param id - SAE ID to delete
   * @returns Promise resolving when deletion is complete
   */
  delete: (id: number) =>
    request<void>(`/saes/${id}`, {
      method: 'DELETE',
    }),

  /**
   * Cancels an in-progress SAE download.
   * @param id - SAE ID with active download
   * @returns Promise resolving when cancellation is processed
   */
  cancelDownload: (id: number) =>
    request<void>(`/saes/${id}/cancel`, {
      method: 'POST',
    }),
};

/**
 * Feature steering API client.
 *
 * Provides methods for controlling SAE feature activation strengths
 * to influence model behavior during inference.
 *
 * @example
 * ```typescript
 * // Set a single feature strength
 * await steeringApi.set({ feature_index: 1234, strength: 5.0 });
 *
 * // Set multiple features at once
 * await steeringApi.batch({
 *   features: [
 *     { feature_index: 1234, strength: 5.0 },
 *     { feature_index: 892, strength: -2.0 },
 *   ],
 * });
 *
 * // Enable steering to apply changes
 * await steeringApi.enable();
 * ```
 */
export const steeringApi = {
  /**
   * Gets the current steering state including all active features.
   * @returns Promise resolving to current steering configuration
   */
  getState: () => request<SteeringState>('/steering'),

  /**
   * Sets the strength for a single feature.
   * Positive values amplify the feature, negative values suppress it.
   * @param req - Request with feature index and strength value
   * @returns Promise resolving to updated steering state
   */
  set: (req: SetSteeringRequest) =>
    request<SteeringState>('/steering', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Sets strengths for multiple features in a single request.
   * More efficient than multiple individual set calls.
   * @param req - Request with array of feature/strength pairs
   * @returns Promise resolving to updated steering state
   */
  batch: (req: BatchSteeringRequest) =>
    request<SteeringState>('/steering/batch', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Removes a feature from steering (resets to neutral).
   * @param featureIndex - Index of the feature to remove
   * @returns Promise resolving to updated steering state
   */
  remove: (featureIndex: number) =>
    request<SteeringState>(`/steering/${featureIndex}`, {
      method: 'DELETE',
    }),

  /**
   * Clears all steering values (resets all features to neutral).
   * @returns Promise resolving to updated steering state
   */
  clear: () =>
    request<SteeringState>('/steering/clear', {
      method: 'POST',
    }),

  /**
   * Enables steering to apply configured feature strengths.
   * Steering is applied to all subsequent inference requests.
   * @returns Promise resolving to updated steering state
   */
  enable: () =>
    request<SteeringState>('/steering/enable', {
      method: 'POST',
    }),

  /**
   * Disables steering without clearing configuration.
   * Feature strengths are preserved but not applied.
   * @returns Promise resolving to updated steering state
   */
  disable: () =>
    request<SteeringState>('/steering/disable', {
      method: 'POST',
    }),
};

/**
 * Feature monitoring API client.
 *
 * Provides methods for configuring and controlling real-time
 * observation of feature activations during inference.
 *
 * @example
 * ```typescript
 * // Configure which features to monitor
 * await monitoringApi.configure({
 *   feature_indices: [1234, 892, 2341],
 *   sample_rate: 1.0,
 * });
 *
 * // Enable monitoring
 * await monitoringApi.enable();
 *
 * // Get activation history
 * const history = await monitoringApi.getHistory(100);
 * ```
 */
export const monitoringApi = {
  /**
   * Gets the current monitoring configuration.
   * @returns Promise resolving to monitoring configuration
   */
  getConfig: () => request<MonitoringConfig>('/monitoring'),

  /**
   * Configures monitoring settings.
   * Specify which features to monitor and sampling rate.
   * @param req - Configuration with feature indices and options
   * @returns Promise resolving to updated monitoring configuration
   */
  configure: (req: ConfigureMonitoringRequest) =>
    request<MonitoringConfig>('/monitoring', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Enables monitoring to start capturing activations.
   * Activations are sent via WebSocket to connected clients.
   * @returns Promise resolving to updated monitoring configuration
   */
  enable: () =>
    request<MonitoringConfig>('/monitoring/enable', {
      method: 'POST',
    }),

  /**
   * Disables monitoring without clearing configuration.
   * @returns Promise resolving to updated monitoring configuration
   */
  disable: () =>
    request<MonitoringConfig>('/monitoring/disable', {
      method: 'POST',
    }),

  /**
   * Gets historical activation data.
   * @param limit - Optional maximum number of records to return
   * @returns Promise resolving to activation history
   */
  getHistory: (limit?: number) =>
    request<MonitoringHistory>(
      `/monitoring/history${limit ? `?limit=${limit}` : ''}`
    ),

  /**
   * Clears all stored activation history.
   * @returns Promise resolving when history is cleared
   */
  clearHistory: () =>
    request<void>('/monitoring/history', {
      method: 'DELETE',
    }),
};

/**
 * Profile management API client.
 *
 * Provides methods for saving, loading, and managing steering
 * configuration profiles. Supports import/export for miStudio compatibility.
 *
 * @example
 * ```typescript
 * // Create a new profile from current steering
 * const profile = await profileApi.create({
 *   name: 'yelling-demo',
 *   description: 'Makes model respond in caps',
 * });
 *
 * // Activate a profile (loads its steering config)
 * await profileApi.activate(profile.id);
 *
 * // Export for miStudio
 * const exported = await profileApi.export(profile.id);
 * ```
 */
export const profileApi = {
  /**
   * Lists all saved profiles.
   * @returns Promise resolving to array of profiles
   */
  list: () => request<Profile[]>('/profiles'),

  /**
   * Gets detailed information about a specific profile.
   * @param id - Profile ID
   * @returns Promise resolving to profile information
   */
  get: (id: number) => request<Profile>(`/profiles/${id}`),

  /**
   * Creates a new profile from current steering configuration.
   * @param req - Request with profile name and optional description
   * @returns Promise resolving to the created profile
   */
  create: (req: CreateProfileRequest) =>
    request<Profile>('/profiles', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Updates an existing profile.
   * @param id - Profile ID to update
   * @param req - Updated profile data
   * @returns Promise resolving to updated profile
   */
  update: (id: number, req: UpdateProfileRequest) =>
    request<Profile>(`/profiles/${id}`, {
      method: 'PUT',
      body: JSON.stringify(req),
    }),

  /**
   * Deletes a profile.
   * @param id - Profile ID to delete
   * @returns Promise resolving when deletion is complete
   */
  delete: (id: number) =>
    request<void>(`/profiles/${id}`, {
      method: 'DELETE',
    }),

  /**
   * Activates a profile, loading its steering configuration.
   * Replaces current steering settings with profile values.
   * @param id - Profile ID to activate
   * @returns Promise resolving to activated profile
   */
  activate: (id: number) =>
    request<Profile>(`/profiles/${id}/activate`, {
      method: 'POST',
    }),

  /**
   * Deactivates a profile without clearing steering.
   * @param id - Profile ID to deactivate
   * @returns Promise resolving to deactivated profile
   */
  deactivate: (id: number) =>
    request<Profile>(`/profiles/${id}/deactivate`, {
      method: 'POST',
    }),

  /**
   * Exports a profile in miStudio-compatible format.
   * @param id - Profile ID to export
   * @returns Promise resolving to exportable profile data
   */
  export: (id: number) => request<ProfileExport>(`/profiles/${id}/export`),

  /**
   * Imports a profile from miStudio export format.
   * @param data - Exported profile data to import
   * @returns Promise resolving to the imported profile
   */
  import: (data: ProfileExport) =>
    request<Profile>('/profiles/import', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

/**
 * Server status API client.
 *
 * Provides methods for checking server health and status,
 * including loaded model and SAE information.
 *
 * @example
 * ```typescript
 * // Check server health
 * const health = await serverApi.health();
 * console.log(`Server status: ${health.status}`);
 *
 * // Get detailed status
 * const status = await serverApi.getStatus();
 * console.log(`Loaded model: ${status.loaded_model?.name}`);
 * ```
 */
export const serverApi = {
  /**
   * Gets detailed server status including loaded model and SAE.
   * @returns Promise resolving to server status information
   */
  getStatus: () => request<ServerStatus>('/status'),

  /**
   * Performs a basic health check.
   * @returns Promise resolving to health status
   */
  health: () => request<{ status: string }>('/health'),
};

/**
 * Combined API client providing access to all miLLM management endpoints.
 *
 * This is the main entry point for API interactions. It aggregates
 * all individual API clients into a single namespace.
 *
 * @example
 * ```typescript
 * import api from '@/services/api';
 *
 * // Access different API areas
 * const models = await api.models.list();
 * const status = await api.server.getStatus();
 * await api.steering.enable();
 * ```
 */
export const api = {
  /** Model management operations */
  models: modelApi,
  /** SAE management operations */
  saes: saeApi,
  /** Feature steering operations */
  steering: steeringApi,
  /** Monitoring configuration operations */
  monitoring: monitoringApi,
  /** Profile management operations */
  profiles: profileApi,
  /** Server status operations */
  server: serverApi,
};

export default api;
