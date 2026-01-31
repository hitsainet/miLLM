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

const API_BASE = '/api';

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

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

// Model API
export const modelApi = {
  list: () => request<ModelInfo[]>('/models'),

  get: (id: number) => request<ModelInfo>(`/models/${id}`),

  download: (req: ModelDownloadRequest) =>
    request<ModelInfo>('/models', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  preview: (repo_id: string) =>
    request<ModelPreviewResponse>('/models/preview', {
      method: 'POST',
      body: JSON.stringify({ repo_id }),
    }),

  load: (id: number) =>
    request<ModelInfo>(`/models/${id}/load`, {
      method: 'POST',
    }),

  unload: (id: number) =>
    request<void>(`/models/${id}/unload`, {
      method: 'POST',
    }),

  delete: (id: number) =>
    request<void>(`/models/${id}`, {
      method: 'DELETE',
    }),

  cancelDownload: (id: number) =>
    request<void>(`/models/${id}/cancel`, {
      method: 'POST',
    }),
};

// SAE API
export const saeApi = {
  list: () => request<SAEInfo[]>('/saes'),

  get: (id: number) => request<SAEInfo>(`/saes/${id}`),

  download: (req: DownloadSAERequest) =>
    request<SAEInfo>('/saes', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  attach: (req: AttachSAERequest) =>
    request<SAEInfo>('/saes/attach', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  detach: () =>
    request<void>('/saes/detach', {
      method: 'POST',
    }),

  delete: (id: number) =>
    request<void>(`/saes/${id}`, {
      method: 'DELETE',
    }),

  cancelDownload: (id: number) =>
    request<void>(`/saes/${id}/cancel`, {
      method: 'POST',
    }),
};

// Steering API
export const steeringApi = {
  getState: () => request<SteeringState>('/steering'),

  set: (req: SetSteeringRequest) =>
    request<SteeringState>('/steering', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  batch: (req: BatchSteeringRequest) =>
    request<SteeringState>('/steering/batch', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  remove: (featureIndex: number) =>
    request<SteeringState>(`/steering/${featureIndex}`, {
      method: 'DELETE',
    }),

  clear: () =>
    request<SteeringState>('/steering/clear', {
      method: 'POST',
    }),

  enable: () =>
    request<SteeringState>('/steering/enable', {
      method: 'POST',
    }),

  disable: () =>
    request<SteeringState>('/steering/disable', {
      method: 'POST',
    }),
};

// Monitoring API
export const monitoringApi = {
  getConfig: () => request<MonitoringConfig>('/monitoring'),

  configure: (req: ConfigureMonitoringRequest) =>
    request<MonitoringConfig>('/monitoring', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  enable: () =>
    request<MonitoringConfig>('/monitoring/enable', {
      method: 'POST',
    }),

  disable: () =>
    request<MonitoringConfig>('/monitoring/disable', {
      method: 'POST',
    }),

  getHistory: (limit?: number) =>
    request<MonitoringHistory>(
      `/monitoring/history${limit ? `?limit=${limit}` : ''}`
    ),

  clearHistory: () =>
    request<void>('/monitoring/history', {
      method: 'DELETE',
    }),
};

// Profile API
export const profileApi = {
  list: () => request<Profile[]>('/profiles'),

  get: (id: number) => request<Profile>(`/profiles/${id}`),

  create: (req: CreateProfileRequest) =>
    request<Profile>('/profiles', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  update: (id: number, req: UpdateProfileRequest) =>
    request<Profile>(`/profiles/${id}`, {
      method: 'PUT',
      body: JSON.stringify(req),
    }),

  delete: (id: number) =>
    request<void>(`/profiles/${id}`, {
      method: 'DELETE',
    }),

  activate: (id: number) =>
    request<Profile>(`/profiles/${id}/activate`, {
      method: 'POST',
    }),

  deactivate: (id: number) =>
    request<Profile>(`/profiles/${id}/deactivate`, {
      method: 'POST',
    }),

  export: (id: number) => request<ProfileExport>(`/profiles/${id}/export`),

  import: (data: ProfileExport) =>
    request<Profile>('/profiles/import', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

// Server API
export const serverApi = {
  getStatus: () => request<ServerStatus>('/status'),

  health: () => request<{ status: string }>('/health'),
};

// Export combined API object
export const api = {
  models: modelApi,
  saes: saeApi,
  steering: steeringApi,
  monitoring: monitoringApi,
  profiles: profileApi,
  server: serverApi,
};

export default api;
