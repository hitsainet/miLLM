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
  // Extended properties
  num_parameters?: number;
  memory_footprint?: number;
  device?: string;
  dtype?: string;
  architecture?: string;
  download_progress?: number;
}

export interface LoadModelRequest {
  model_id: string;
  device?: 'auto' | 'cuda' | 'cpu';
  dtype?: string;
}

export interface ModelDownloadRequest {
  source: ModelSource;
  repo_id?: string;
  local_path?: string;
  quantization: QuantizationType;
  device?: 'auto' | 'cuda' | 'cpu';
  trust_remote_code?: boolean;
  hf_token?: string;
}

export interface SizeEstimate {
  disk_mb: number;
  memory_mb: number;
}

export interface ModelPreviewResponse {
  name: string;
  params: string | null;
  architecture: string | null;
  requires_trust_remote_code: boolean;
  is_gated: boolean;
  estimated_sizes: Record<QuantizationType, SizeEstimate> | null;
}

// SAE types
export type SAEStatus = 'cached' | 'attached' | 'downloading' | 'error';

export interface SAEInfo {
  id: string;
  repository_id: string;
  revision: string;
  name: string;
  format: string;
  d_in: number;
  d_sae: number;
  trained_on: string | null;
  trained_layer: number | null;
  file_size_bytes: number | null;
  status: SAEStatus;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface AttachmentStatus {
  is_attached: boolean;
  sae_id: string | null;
  layer: number | null;
  memory_usage_mb: number | null;
  steering_enabled: boolean;
  monitoring_enabled: boolean;
}

export interface SAEListResponse {
  saes: SAEInfo[];
  total: number;
  attachment: AttachmentStatus;
}

export interface DownloadSAERequest {
  repository_id: string;
  revision?: string;
}

export interface PreviewSAERequest {
  repository_id: string;
  revision?: string;
  hf_token?: string;
}

export interface SAEFileInfo {
  path: string;
  size_bytes: number;
  layer: number | null;
  width: string | null;
}

export interface PreviewSAEResponse {
  repository_id: string;
  revision: string;
  model_id: string | null;
  files: SAEFileInfo[];
  total_files: number;
}

export interface AttachSAERequest {
  sae_id: string;
  layer: number;
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

export interface FeatureActivation {
  feature_index: number;
  activation: number;
  label?: string;
}

export interface ActivationRecord {
  timestamp: string;
  request_id: string;
  activations: FeatureActivation[];
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
  model_repo_id?: string;
  sae_repo_id?: string;
  sae_layer?: number;
  features: FeatureSteering[];
  is_active?: boolean;
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
  activations: FeatureActivation[];
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
