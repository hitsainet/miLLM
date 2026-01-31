import {
  Server,
  HardDrive,
  Cpu,
  Calendar,
  AlertTriangle,
  Play,
  Square,
  Trash2,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import { Modal, Button, Spinner, Badge } from '@components/common';
import type { ModelInfo, ModelPreviewResponse } from '@/types';

export interface ModelDetailsModalProps {
  /** The downloaded model to show details for */
  model?: ModelInfo | null;
  /** Preview data from HuggingFace (for models not yet downloaded) */
  previewData?: ModelPreviewResponse | null;
  /** Whether the modal is open */
  isOpen: boolean;
  /** Callback when modal is closed */
  onClose: () => void;
  /** Callback to load the model */
  onLoad?: (id: number) => void;
  /** Callback to unload the model */
  onUnload?: (id: number) => void;
  /** Callback to delete the model */
  onDelete?: (id: number) => void;
  /** Whether a model is currently being loaded */
  isLoadingModel?: boolean;
  /** Whether a model is currently being unloaded */
  isUnloading?: boolean;
  /** Whether a model is currently being deleted */
  isDeleting?: boolean;
  /** The currently loaded model (to check if another model is loaded) */
  loadedModel?: ModelInfo | null;
}

export function ModelDetailsModal({
  model,
  previewData,
  isOpen,
  onClose,
  onLoad,
  onUnload,
  onDelete,
  isLoadingModel,
  isUnloading,
  isDeleting,
  loadedModel,
}: ModelDetailsModalProps) {
  // Determine if we're showing a downloaded model or preview data
  const isPreview = !model && !!previewData;
  const title = model?.name || previewData?.name || 'Model Details';

  const getStatusBadge = (status: ModelInfo['status']) => {
    switch (status) {
      case 'loaded':
        return <Badge variant="success">Loaded</Badge>;
      case 'ready':
        return <Badge variant="primary">Ready</Badge>;
      case 'downloading':
        return <Badge variant="warning">Downloading</Badge>;
      case 'loading':
        return <Badge variant="warning">Loading</Badge>;
      case 'error':
        return <Badge variant="danger">Error</Badge>;
      default:
        return <Badge>{status}</Badge>;
    }
  };

  const hasValidId = model && model.id !== undefined && model.id !== null;
  const canLoad = hasValidId && (model.status === 'ready' || model.status === 'error');
  const canUnload = hasValidId && model?.status === 'loaded';
  const canDelete = hasValidId && model.status !== 'loaded' && model.status !== 'downloading';
  const anotherModelLoaded = loadedModel && model && loadedModel.id !== model.id;

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatSize = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb} MB`;
  };

  return (
    <Modal
      id="model-details"
      title={title}
      isOpen={isOpen}
      onClose={onClose}
      size="lg"
      footer={
        model && (
          <div className="flex items-center gap-3 w-full">
            {canDelete && (
              <Button
                variant="danger"
                size="sm"
                onClick={() => onDelete?.(model.id)}
                disabled={isDeleting}
              >
                {isDeleting ? <Spinner size="sm" /> : <Trash2 className="w-4 h-4" />}
                Delete
              </Button>
            )}
            <div className="flex-1" />
            {canUnload && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onUnload?.(model.id)}
                disabled={isUnloading}
              >
                {isUnloading ? <Spinner size="sm" /> : <Square className="w-4 h-4" />}
                Unload
              </Button>
            )}
            {canLoad && (
              <Button
                variant="primary"
                size="sm"
                onClick={() => onLoad?.(model.id)}
                disabled={isLoadingModel || !!anotherModelLoaded}
                title={anotherModelLoaded ? 'Unload current model first' : undefined}
              >
                {isLoadingModel ? <Spinner size="sm" /> : <Play className="w-4 h-4" />}
                {anotherModelLoaded ? 'Unload current first' : 'Load Model'}
              </Button>
            )}
          </div>
        )
      }
    >
      <div className="space-y-6">
        {/* Status & Repository */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-slate-400 mb-1">Repository</p>
            <div className="flex items-center gap-2">
              <code className="text-sm text-slate-200 bg-slate-800 px-2 py-1 rounded">
                {model?.repo_id || 'N/A'}
              </code>
              {model?.repo_id && (
                <a
                  href={`https://huggingface.co/${model.repo_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-400 hover:text-primary-300"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>
          {model && getStatusBadge(model.status)}
        </div>

        {/* Error Message */}
        {model?.status === 'error' && (
          <div className="flex items-start gap-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-300">Load Failed</p>
              <p className="text-sm text-red-400/80 mt-1">
                The model failed to load. You can try loading it again.
              </p>
            </div>
          </div>
        )}

        {/* Download Progress */}
        {model?.status === 'downloading' && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">Download Progress</span>
              <span className="text-primary-400 font-medium">
                {model.download_progress || 0}%
              </span>
            </div>
            <div className="w-full bg-slate-700/50 rounded-full h-2 overflow-hidden">
              <div
                className="bg-primary-500 h-full rounded-full transition-all duration-300"
                style={{ width: `${model.download_progress || 0}%` }}
              />
            </div>
          </div>
        )}

        {/* Model Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <Server className="w-3 h-3" />
              <span className="text-xs">Parameters</span>
            </div>
            <p className="text-lg font-semibold text-slate-200">
              {model?.params || previewData?.params || 'Unknown'}
            </p>
          </div>

          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <RefreshCw className="w-3 h-3" />
              <span className="text-xs">Quantization</span>
            </div>
            <p className="text-lg font-semibold text-slate-200">
              {model?.quantization || 'N/A'}
            </p>
          </div>

          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <HardDrive className="w-3 h-3" />
              <span className="text-xs">Disk Size</span>
            </div>
            <p className="text-lg font-semibold text-slate-200">
              {model?.memory_mb ? formatSize(model.memory_mb) : 'Unknown'}
            </p>
          </div>

          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <Cpu className="w-3 h-3" />
              <span className="text-xs">Architecture</span>
            </div>
            <p className="text-lg font-semibold text-slate-200">
              {model?.architecture || previewData?.architecture || 'Unknown'}
            </p>
          </div>
        </div>

        {/* Preview-specific: Gated & Trust Remote Code warnings */}
        {isPreview && (
          <>
            {previewData?.is_gated && (
              <div className="flex items-center gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0" />
                <p className="text-sm text-yellow-300">
                  This model is gated. You'll need a HuggingFace token with access to download it.
                </p>
              </div>
            )}

            {previewData?.requires_trust_remote_code && (
              <div className="flex items-center gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0" />
                <p className="text-sm text-yellow-300">
                  This model requires "Trust Remote Code" to be enabled. Make sure you trust the model source.
                </p>
              </div>
            )}

            {/* Estimated sizes by quantization */}
            {previewData?.estimated_sizes && (
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-3">
                  Estimated Requirements by Quantization
                </h4>
                <div className="grid grid-cols-3 gap-3">
                  {(['Q4', 'Q8', 'FP16'] as const).map((quant) => {
                    const size = previewData.estimated_sizes?.[quant];
                    if (!size) return null;
                    return (
                      <div
                        key={quant}
                        className="bg-slate-800/30 border border-slate-700/50 rounded-lg p-3"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-200">{quant}</span>
                          {quant === 'Q4' && (
                            <span className="text-xs bg-primary-500/20 text-primary-300 px-2 py-0.5 rounded">
                              Recommended
                            </span>
                          )}
                        </div>
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 text-xs text-slate-400">
                            <HardDrive className="w-3 h-3" />
                            <span>Disk: {formatSize(size.disk_mb)}</span>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-slate-400">
                            <Cpu className="w-3 h-3" />
                            <span>VRAM: {formatSize(size.memory_mb)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}

        {/* Timestamps for downloaded models */}
        {model && (
          <div className="flex items-center gap-6 text-xs text-slate-500 pt-2 border-t border-slate-700/50">
            <div className="flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              <span>Added: {formatDate(model.created_at)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              <span>Updated: {formatDate(model.updated_at)}</span>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}

export default ModelDetailsModal;
