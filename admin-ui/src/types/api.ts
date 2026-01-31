// API Response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: ErrorDetails | null;
}

export interface ErrorDetails {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// Model types
export type ModelStatus = 'ready' | 'loaded' | 'downloading' | 'loading' | 'error';
export type QuantizationType = 'Q4' | 'Q8' | 'FP16';
export type ModelSource = 'huggingface' | 'local';

export interface ModelInfo {
  id: number;
  name: string;
  repo_id: string;
  source: ModelSource;
  quantization: QuantizationType;
  params: string;
  memory_mb: number;
  local_path: string;
  status: ModelStatus;
  created_at: string;
  updated_at: string;
}

export interface LoadModelRequest {
  model_id: string;
  device?: 'auto' | 'cuda' | 'cpu';
  dtype?: string;
}

export interface ModelDownloadRequest {
  repo_id: string;
  quantization: QuantizationType;
  trust_remote_code?: boolean;
  hf_token?: string;
}

export interface ModelPreviewResponse {
  repo_id: string;
  name: string;
  params: string;
  estimated_memory_mb: Record<QuantizationType, number>;
  available_memory_mb: number;
  can_load: Record<QuantizationType, boolean>;
}

// SAE types
export type SAEStatus = 'ready' | 'attached' | 'downloading' | 'error';

export interface SAEInfo {
  id: number;
  name: string;
  repo_id: string;
  filename: string;
  layer: number;
  num_features: number;
  size_mb: number;
  local_path: string;
  linked_model_id: number | null;
  linked_model_name: string | null;
  status: SAEStatus;
  created_at: string;
  updated_at: string;
}

export interface DownloadSAERequest {
  repo_id: string;
  filename?: string;
  linked_model_id?: number;
  hf_token?: string;
}

export interface AttachSAERequest {
  sae_id: number;
}

// Steering types
export interface FeatureSteering {
  index: number;
  strength: number;
  label?: string;
}

export interface SteeringState {
  enabled: boolean;
  sae_id: number | null;
  features: FeatureSteering[];
}

export interface SetSteeringRequest {
  feature_index: number;
  strength: number;
}

export interface BatchSteeringRequest {
  features: FeatureSteering[];
}

// Monitoring types
export interface MonitoringConfig {
  enabled: boolean;
  sae_id: number | null;
  top_k: number;
  feature_indices: number[] | null;
}

export interface ActivationRecord {
  timestamp: string;
  request_id: string;
  activations: Record<number, number>;
}

export interface FeatureStatistics {
  feature_index: number;
  label?: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  count: number;
}

export interface MonitoringHistory {
  records: ActivationRecord[];
  statistics: FeatureStatistics[];
}

export interface ConfigureMonitoringRequest {
  top_k?: number;
  feature_indices?: number[];
}

// Profile types
export interface Profile {
  id: number;
  name: string;
  description?: string;
  model_repo_id: string;
  sae_repo_id: string;
  sae_layer: number;
  features: FeatureSteering[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CreateProfileRequest {
  name: string;
  description?: string;
  save_current?: boolean;
  features?: FeatureSteering[];
}

export interface UpdateProfileRequest {
  name?: string;
  description?: string;
  features?: FeatureSteering[];
}

export interface ProfileExport {
  version: string;
  name: string;
  description?: string;
  model: {
    repo_id: string;
    quantization: QuantizationType;
  };
  sae: {
    repo_id: string;
    layer: number;
  };
  features: FeatureSteering[];
  exported_at: string;
}

// Server status
export interface ServerStatus {
  status: 'running' | 'error';
  model_loaded: boolean;
  loaded_model: ModelInfo | null;
  sae_attached: boolean;
  attached_sae: SAEInfo | null;
  steering_enabled: boolean;
  monitoring_enabled: boolean;
  active_profile: Profile | null;
  gpu_memory_used_mb: number;
  gpu_memory_total_mb: number;
  gpu_utilization: number;
}

// WebSocket event types
export interface DownloadProgressEvent {
  model_id?: number;
  sae_id?: number;
  progress: number;
  downloaded_mb: number;
  total_mb: number;
  speed_mbps: number;
}

export interface LoadProgressEvent {
  model_id: number;
  stage: 'loading_weights' | 'loading_tokenizer' | 'moving_to_device' | 'complete';
  progress: number;
}

export interface ActivationEvent {
  timestamp: string;
  request_id: string;
  activations: Record<number, number>;
}

export interface SystemMetricsEvent {
  cpu_percent: number;
  ram_used_mb: number;
  ram_total_mb: number;
  gpu_utilization: number;
  gpu_memory_used_mb: number;
  gpu_memory_total_mb: number;
  gpu_temperature: number;
}
