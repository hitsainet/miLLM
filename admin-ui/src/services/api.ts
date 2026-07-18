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
  SAEListResponse,
  DownloadSAERequest,
  PreviewSAERequest,
  PreviewSAEResponse,
  AttachSAERequest,
  SteeringState,
  SetSteeringRequest,
  BatchSteeringRequest,
  MonitoringConfig,
  ConfigureMonitoringRequest,
  MonitoringHistory,
  StatisticsResponse,
  TopFeaturesResponse,
  Profile,
  CreateProfileRequest,
  UpdateProfileRequest,
  ProfileExport,
  ServerStatus,
} from '@/types';
import type {
  ClusterDefinitionV1,
  ClusterImportItem,
  ClusterImportResult,
  ClusterListResponse,
  HubDefinitionRef,
  HubRepoInfo,
} from '@/types/clusters';


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

  // Check content type before attempting JSON parse — nginx error pages are HTML
  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('application/json')) {
    throw new ApiError(
      'SERVER_ERROR',
      `Server returned ${response.status} ${response.statusText || 'error'}`
    );
  }

  let data: ApiResponse<T>;
  try {
    data = await response.json();
  } catch {
    throw new ApiError('PARSE_ERROR', `Invalid response from server (status ${response.status})`);
  }

  // 422s come in two shapes: the management envelope ({success:false,
  // error:{...}} — e.g. the cluster activation gate, intensity-range
  // rejections) and FastAPI's request-validation {detail:[...]}. Check the
  // envelope FIRST — assuming the FastAPI shape used to reduce every server
  // gate message to a bare "Validation error" (review find).
  if (response.status === 422) {
    const envelope = data as unknown as {
      success?: boolean;
      error?: { code?: string; message?: string; details?: unknown };
    };
    if (envelope?.error?.message) {
      throw new ApiError(
        envelope.error.code ?? 'VALIDATION_ERROR',
        envelope.error.message,
        envelope.error.details as Record<string, unknown> | undefined
      );
    }
    const detail = (data as unknown as { detail: Array<{ msg: string }> | string })?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join('; ')
      : typeof detail === 'string'
        ? detail
        : 'Validation error';
    throw new ApiError('VALIDATION_ERROR', message);
  }

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
import type {
  SensingConfigResult,
  SensingEvent,
  SensingEventList,
  SensingStatus,
  SensingToggleResult,
} from '@/types/sensing';

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
  preview: (repo_id: string, hf_token?: string) =>
    request<ModelPreviewResponse>('/models/preview', {
      method: 'POST',
      body: JSON.stringify({ repo_id, ...(hf_token ? { hf_token } : {}) }),
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

  /**
   * Locks a loaded model for steering (prevents auto-unload).
   * @param id - Model ID to lock
   * @returns Promise resolving to updated model information
   */
  lock: (id: number) =>
    request<ModelInfo>(`/models/${id}/lock`, {
      method: 'POST',
    }),

  /**
   * Unlocks a model to allow auto-unload by inference requests.
   * @param id - Model ID to unlock
   * @returns Promise resolving to updated model information
   */
  unlock: (id: number) =>
    request<ModelInfo>(`/models/${id}/unlock`, {
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
   * Lists all downloaded SAEs with attachment status.
   * @returns Promise resolving to SAE list response with attachment info
   */
  list: async (): Promise<SAEInfo[]> => {
    const response = await request<SAEListResponse>('/saes');
    return response.saes;
  },

  /**
   * Lists all SAEs with full response including attachment status.
   * @returns Promise resolving to full SAE list response
   */
  listWithAttachment: () => request<SAEListResponse>('/saes'),

  /**
   * Gets detailed information about a specific SAE.
   * @param id - SAE ID
   * @returns Promise resolving to SAE information
   */
  get: (id: string) => request<SAEInfo>(`/saes/${id}`),

  /**
   * Initiates download of an SAE from HuggingFace.
   * @param req - Download request with repository_id and optional revision
   * @returns Promise resolving to the download response
   */
  download: (req: DownloadSAERequest) =>
    request<{ sae_id: string; status: string; message: string }>('/saes/download', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Previews an SAE repository to list available files without downloading.
   * @param req - Preview request with repository_id and optional revision
   * @returns Promise resolving to repository preview with file listing
   */
  preview: (req: PreviewSAERequest) =>
    request<PreviewSAEResponse>('/saes/preview', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /**
   * Attaches an SAE to the currently loaded model.
   * Enables feature steering and monitoring capabilities.
   * Only one SAE can be attached at a time.
   * @param req - Attach request with SAE ID and layer
   * @returns Promise resolving to attach response
   */
  attach: (req: AttachSAERequest) =>
    request<{ status: string; sae_id: string; layer: number; memory_usage_mb: number; warnings: string[] }>(
      `/saes/${req.sae_id}/attach`,
      {
        method: 'POST',
        body: JSON.stringify({ layer: req.layer }),
      }
    ),

  /**
   * Detaches the currently attached SAE.
   * Disables feature steering until another SAE is attached.
   * @param saeId - SAE ID to detach
   * @returns Promise resolving when detachment is complete
   */
  detach: (saeId?: string) =>
    request<{ status: string; sae_id: string; memory_freed_mb: number }>(
      `/saes/${saeId}/detach`,
      {
        method: 'POST',
      }
    ),

  /**
   * Deletes an SAE from local storage.
   * SAE must not be currently attached.
   * @param id - SAE ID to delete
   * @returns Promise resolving when deletion is complete
   */
  delete: (id: string) =>
    request<{ status: string; sae_id: string; freed_disk_mb: number }>(`/saes/${id}`, {
      method: 'DELETE',
    }),

  /**
   * Cancels an in-progress SAE download.
   * @param id - SAE ID with active download
   * @returns Promise resolving when cancellation is processed
   */
  cancelDownload: (id: string) =>
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
/**
 * Transforms backend steering response to frontend SteeringState format.
 * Backend returns: { enabled: bool, values: { feature_idx: strength } }
 * Frontend expects: { enabled: bool, sae_id: number|null, features: FeatureSteering[] }
 */
interface BackendSteeringResponse {
  enabled: boolean;
  values: Record<string, number>;
}

function transformSteeringResponse(response: BackendSteeringResponse): SteeringState {
  const features = Object.entries(response.values).map(([index, strength]) => ({
    index: parseInt(index, 10),
    strength,
  }));
  return {
    enabled: response.enabled,
    sae_id: null, // Backend doesn't return sae_id in steering response
    features,
  };
}

export const steeringApi = {
  /**
   * Gets the current steering state including all active features.
   * @returns Promise resolving to current steering configuration
   */
  getState: async (): Promise<SteeringState> => {
    const response = await request<BackendSteeringResponse>('/saes/steering');
    return transformSteeringResponse(response);
  },

  /**
   * Sets the strength for a single feature.
   * Positive values amplify the feature, negative values suppress it.
   * @param req - Request with feature index and strength value
   * @returns Promise resolving to updated steering state
   */
  set: async (req: SetSteeringRequest): Promise<SteeringState> => {
    // Transform frontend format { feature_index, strength }
    // to backend format { feature_idx, value }
    const response = await request<BackendSteeringResponse>('/saes/steering', {
      method: 'POST',
      body: JSON.stringify({
        feature_idx: req.feature_index,
        value: req.strength,
      }),
    });
    return transformSteeringResponse(response);
  },

  /**
   * Sets strengths for multiple features in a single request.
   * More efficient than multiple individual set calls.
   * @param req - Request with array of feature/strength pairs
   * @returns Promise resolving to updated steering state
   */
  batch: async (req: BatchSteeringRequest): Promise<SteeringState> => {
    // Transform frontend format { features: [{ index, strength }] }
    // to backend format { steering: { index: strength } }
    const steering: Record<number, number> = {};
    for (const feature of req.features) {
      steering[feature.index] = feature.strength;
    }
    const response = await request<BackendSteeringResponse>('/saes/steering/batch', {
      method: 'POST',
      body: JSON.stringify({ steering }),
    });
    return transformSteeringResponse(response);
  },

  /**
   * Removes a feature from steering (resets to neutral).
   * @param featureIndex - Index of the feature to remove
   * @returns Promise resolving to updated steering state
   */
  remove: async (featureIndex: number): Promise<SteeringState> => {
    const response = await request<BackendSteeringResponse>(`/saes/steering/${featureIndex}`, {
      method: 'DELETE',
    });
    return transformSteeringResponse(response);
  },

  /**
   * Clears all steering values (resets all features to neutral).
   * @returns Promise resolving to updated steering state
   */
  clear: async (): Promise<SteeringState> => {
    const response = await request<BackendSteeringResponse>('/saes/steering', {
      method: 'DELETE',
    });
    return transformSteeringResponse(response);
  },

  /**
   * Enables steering to apply configured feature strengths.
   * Steering is applied to all subsequent inference requests.
   * @returns Promise resolving to updated steering state
   */
  enable: async (): Promise<SteeringState> => {
    const response = await request<BackendSteeringResponse>('/saes/steering/enable', {
      method: 'POST',
    });
    return transformSteeringResponse(response);
  },

  /**
   * Disables steering without clearing configuration.
   * Feature strengths are preserved but not applied.
   * @returns Promise resolving to updated steering state
   */
  disable: async (): Promise<SteeringState> => {
    const response = await request<BackendSteeringResponse>('/saes/steering/disable', {
      method: 'POST',
    });
    return transformSteeringResponse(response);
  },
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
  /** GET /api/monitoring — current state (enabled, top_k, sae_attached, …) */
  getConfig: () => request<MonitoringConfig>('/monitoring'),

  /**
   * POST /api/monitoring/configure
   * Maps frontend ConfigureMonitoringRequest fields to the backend schema:
   *   top_k          → top_k
   *   features       → features   (backend field name; frontend may also pass feature_indices)
   *   history_size   → history_size
   *   enabled        → enabled
   */
  configure: (req: ConfigureMonitoringRequest) =>
    request<MonitoringConfig>('/monitoring/configure', {
      method: 'POST',
      body: JSON.stringify({
        enabled: req.enabled ?? true,
        features: req.features ?? null,
        history_size: req.history_size ?? 100,
        top_k: req.top_k ?? 10,
      }),
    }),

  /** POST /api/monitoring/enable with enabled:true */
  enable: () =>
    request<MonitoringConfig>('/monitoring/enable', {
      method: 'POST',
      body: JSON.stringify({ enabled: true }),
    }),

  /** POST /api/monitoring/enable with enabled:false */
  disable: () =>
    request<MonitoringConfig>('/monitoring/enable', {
      method: 'POST',
      body: JSON.stringify({ enabled: false }),
    }),

  /**
   * GET /api/monitoring/history
   * @param limit   Max records (default 50)
   * @param requestId  Filter to a specific inference request
   */
  getHistory: (limit?: number, requestId?: string) => {
    const params = new URLSearchParams();
    if (limit) params.set('limit', String(limit));
    if (requestId) params.set('request_id', requestId);
    const qs = params.toString();
    return request<MonitoringHistory>(`/monitoring/history${qs ? `?${qs}` : ''}`);
  },

  /** DELETE /api/monitoring/history */
  clearHistory: () =>
    request<{ cleared: number; message: string }>('/monitoring/history', {
      method: 'DELETE',
    }),

  /**
   * GET /api/monitoring/statistics
   * Per-feature running stats (mean, std, min, max, active_ratio, count).
   * @param featureIndices  Comma-separated indices to filter; omit for all.
   */
  getStatistics: (featureIndices?: number[]) => {
    const qs = featureIndices?.length
      ? `?features=${featureIndices.join(',')}`
      : '';
    return request<StatisticsResponse>(`/monitoring/statistics${qs}`);
  },

  /** DELETE /api/monitoring/statistics — reset all running stats */
  resetStatistics: () =>
    request<{ cleared: number; message: string }>('/monitoring/statistics', {
      method: 'DELETE',
    }),

  /**
   * POST /api/monitoring/statistics/top
   * Get top-K features ranked by a metric.
   * @param k       Number of features (1–100, default 10)
   * @param metric  mean | max | active_ratio | count (default mean)
   */
  getTopFeatures: (k = 10, metric: 'mean' | 'max' | 'active_ratio' | 'count' = 'mean') =>
    request<TopFeaturesResponse>('/monitoring/statistics/top', {
      method: 'POST',
      body: JSON.stringify({ k, metric }),
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
  get: (id: string) => request<Profile>(`/profiles/${id}`),

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
  update: (id: string, req: UpdateProfileRequest) =>
    request<Profile>(`/profiles/${id}`, {
      method: 'PUT',
      body: JSON.stringify(req),
    }),

  /**
   * Deletes a profile.
   * @param id - Profile ID to delete
   * @returns Promise resolving when deletion is complete
   */
  delete: (id: string) =>
    request<void>(`/profiles/${id}`, {
      method: 'DELETE',
    }),

  /**
   * Activates a profile, loading its steering configuration.
   * Replaces current steering settings with profile values.
   * @param id - Profile ID to activate
   * @returns Promise resolving to activated profile
   */
  activate: (id: string) =>
    request<Profile>(`/profiles/${id}/activate`, {
      method: 'POST',
    }),

  /**
   * Deactivates a profile without clearing steering.
   * @param id - Profile ID to deactivate
   * @returns Promise resolving to deactivated profile
   */
  deactivate: (id: string) =>
    request<Profile>(`/profiles/${id}/deactivate`, {
      method: 'POST',
    }),

  /**
   * Exports a profile in miStudio-compatible format.
   * @param id - Profile ID to export
   * @returns Promise resolving to exportable profile data
   */
  export: (id: string) => request<ProfileExport>(`/profiles/${id}/export`),

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

// ============================================================
// Cluster API (Feature 8)
// ============================================================

/**
 * API client for imported cluster definitions (Feature 8).
 * Import portable mistudio.cluster-definition/v1 documents, browse public
 * Hugging Face cluster packs (consume-only), activate clusters, and adjust
 * the lambda intensity dial.
 */
export const clusterApi = {
  /** Lists imported cluster profiles. */
  list: () => request<ClusterListResponse>('/clusters'),

  /**
   * Imports a definition or bundle (kind-keyed).
   * @param payload - Parsed definition/bundle JSON
   */
  import: (payload: unknown, opts?: { onConflict?: 'rename' | 'fail'; activate?: boolean }) => {
    const params = new URLSearchParams();
    if (opts?.onConflict) params.set('on_conflict', opts.onConflict);
    if (opts?.activate) params.set('activate', 'true');
    const qs = params.toString();
    return request<ClusterImportResult>(`/clusters/import${qs ? `?${qs}` : ''}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /** Searches public Hub cluster packs (anonymous, tag-filtered). */
  hubSearch: (opts?: { q?: string; baseModel?: string; limit?: number }) => {
    const params = new URLSearchParams();
    if (opts?.q) params.set('q', opts.q);
    if (opts?.baseModel) params.set('base_model', opts.baseModel);
    if (opts?.limit) params.set('limit', String(opts.limit));
    const qs = params.toString();
    return request<HubRepoInfo[]>(`/clusters/hub/search${qs ? `?${qs}` : ''}`);
  },

  /** Lists a repo's definitions (manifest preferred). */
  hubDefinitions: (repoId: string) =>
    request<HubDefinitionRef[]>(`/clusters/hub/${repoId}/definitions`),

  /** Imports one definition file from a Hub repo. */
  hubImport: (req: { repo_id: string; filename: string; revision?: string; activate?: boolean }) =>
    request<ClusterImportItem>('/clusters/hub/import', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  /** Activates a cluster (hard compatibility gate server-side). */
  activate: (id: string) =>
    request<{ profile_id: string; applied_steering: boolean; feature_count: number }>(
      `/clusters/${id}/activate`,
      { method: 'POST' }
    ),

  /** Deactivates a cluster. */
  deactivate: (id: string) =>
    request<{ profile_id: string }>(`/clusters/${id}/deactivate`, { method: 'POST' }),

  /** Deletes a cluster (server deactivates + clears steering first if active). */
  delete: (id: string) =>
    request<{ profile_id: string; was_active: boolean }>(`/clusters/${id}`, {
      method: 'DELETE',
    }),

  /** Sets a cluster's lambda intensity (re-applies when active). */
  setIntensity: (id: string, intensity: number, reapply = true) =>
    request<{ profile_id: string; intensity: number; reapplied: boolean }>(
      `/clusters/${id}/intensity`,
      { method: 'PUT', body: JSON.stringify({ intensity, reapply }) }
    ),

  /**
   * Exports the lossless original definition. Raw artifact — served without
   * the ApiResponse envelope, so this bypasses request().
   */
  export: async (id: string): Promise<ClusterDefinitionV1> => {
    const response = await fetch(`${API_BASE}/clusters/${id}/export`);
    if (!response.ok) {
      throw new ApiError('EXPORT_FAILED', `Export failed (${response.status})`);
    }
    return (await response.json()) as ClusterDefinitionV1;
  },
};

export const sensingApi = {
  /** Runtime status: armed cluster, threshold mode, overhead. */
  status: () => request<SensingStatus>('/sensing/status'),

  /** Newest-first events, optionally scoped to a cluster. */
  events: (opts?: { profileId?: string; limit?: number }) => {
    const params = new URLSearchParams();
    if (opts?.profileId) params.set('profile_id', opts.profileId);
    if (opts?.limit) params.set('limit', String(opts.limit));
    const qs = params.toString();
    return request<SensingEventList>(`/sensing/events${qs ? `?${qs}` : ''}`);
  },

  /** Event detail — the only path that carries context text. */
  eventDetail: (id: number) => request<SensingEvent>(`/sensing/events/${id}`),

  /** Clears events (all, or one cluster's). */
  clearEvents: (profileId?: string) => {
    const qs = profileId ? `?profile_id=${encodeURIComponent(profileId)}` : '';
    return request<{ deleted: number }>(`/sensing/events${qs}`, { method: 'DELETE' });
  },

  /** Persists the per-cluster toggle and live-arms/disarms when active. */
  setEnabled: (profileId: string, enabled: boolean) =>
    request<SensingToggleResult>(
      `/sensing/${profileId}/${enabled ? 'enable' : 'disable'}`,
      { method: 'POST' }
    ),

  /** Runtime overrides (min_k quorum); null clears back to the default. */
  setConfig: (profileId: string, minK: number | null) =>
    request<SensingConfigResult>(`/sensing/${profileId}/config`, {
      method: 'PUT',
      body: JSON.stringify({ min_k: minK }),
    }),
};

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
  /** Imported cluster operations (Feature 8) */
  clusters: clusterApi,
  sensing: sensingApi,
  /** Server status operations */
  server: serverApi,
};

export default api;
