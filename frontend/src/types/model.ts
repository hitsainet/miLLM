/**
 * Model types for miLLM frontend.
 */

export type ModelStatus = 'downloading' | 'ready' | 'loading' | 'loaded' | 'error';

export type ModelSource = 'huggingface' | 'local';

export type QuantizationType = 'Q4' | 'Q8' | 'FP16';

export interface Model {
  id: number;
  name: string;
  source: ModelSource;
  repoId: string | null;
  localPath: string | null;
  params: string | null;
  architecture: string | null;
  quantization: QuantizationType;
  diskSizeMb: number | null;
  estimatedMemoryMb: number | null;
  cachePath: string;
  configJson: Record<string, unknown> | null;
  trustRemoteCode: boolean;
  status: ModelStatus;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
  loadedAt: string | null;
}

export interface ModelDownloadRequest {
  source: ModelSource;
  repoId?: string;
  localPath?: string;
  quantization: QuantizationType;
  trustRemoteCode?: boolean;
  hfToken?: string;
}

export interface ModelPreviewRequest {
  repoId: string;
  hfToken?: string;
}

export interface ModelPreviewResponse {
  name: string;
  repoId: string;
  params: string | null;
  architecture: string | null;
  isGated: boolean;
  requiresTrustRemoteCode: boolean;
}

export interface DownloadProgress {
  modelId: number;
  progress: number;
  downloadedBytes: number;
  totalBytes: number;
  speedBps?: number;
}

export interface LoadProgress {
  modelId: number;
  stage: string;
  progress: number;
}
