import { useState } from 'react';
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
  TrendingUp,
  Heart,
  Download,
  Scale,
  Lock,
  Unlock,
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
  /** Callback to download from preview with selected quantization */
  onDownloadFromPreview?: (quantization: string, trustRemoteCode: boolean) => void;
  /** Callback to lock the model for steering */
  onLock?: (id: number) => void;
  /** Callback to unlock the model */
  onUnlock?: (id: number) => void;
  /** Whether a model is currently being loaded */
  isLoadingModel?: boolean;
  /** Whether a model is currently being unloaded */
  isUnloading?: boolean;
  /** Whether a model is currently being deleted */
  isDeleting?: boolean;
  /** Whether a model is currently being locked */
  isLocking?: boolean;
  /** Whether a model is currently being unlocked */
  isUnlockingModel?: boolean;
  /** The currently loaded model (to check if another model is loaded) */
  loadedModel?: ModelInfo | null;
}

const QUANTIZATION_OPTIONS = [
  { format: 'FP32', label: 'FP32 (Full Precision)' },
  { format: 'FP16', label: 'FP16 (Half Precision)' },
  { format: 'Q8', label: 'Q8 (8-bit)' },
  { format: 'Q4', label: 'Q4 (4-bit)', recommended: true },
  { format: 'Q2', label: 'Q2 (2-bit)' },
];

export function ModelDetailsModal({
  model,
  previewData,
  isOpen,
  onClose,
  onLoad,
  onUnload,
  onDelete,
  onDownloadFromPreview,
  onLock,
  onUnlock,
  isLoadingModel,
  isUnloading,
  isDeleting,
  isLocking,
  isUnlockingModel,
  loadedModel,
}: ModelDetailsModalProps) {
  const [selectedQuantization, setSelectedQuantization] = useState('Q4');
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);

  // Determine if we're showing a downloaded model or preview data
  const isPreview = !model && !!previewData;
  const title = model?.name || previewData?.name || 'Model Details';
  const repoId = model?.repo_id || (previewData ? `${previewData.name}` : null);

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

  const handleDownload = () => {
    if (onDownloadFromPreview) {
      onDownloadFromPreview(selectedQuantization, trustRemoteCode);
      onClose();
    }
  };

  // Build footer based on mode
  const footer = model ? (
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
      {canUnload && model.locked && (
        <Button
          variant="secondary"
          size="sm"
          onClick={() => onUnlock?.(model.id)}
          disabled={isUnlockingModel}
        >
          {isUnlockingModel ? <Spinner size="sm" /> : <Unlock className="w-4 h-4" />}
          Unlock
        </Button>
      )}
      {canUnload && !model.locked && (
        <Button
          variant="secondary"
          size="sm"
          onClick={() => onLock?.(model.id)}
          disabled={isLocking}
          title="Lock model to prevent auto-unload during steering"
        >
          {isLocking ? <Spinner size="sm" /> : <Lock className="w-4 h-4" />}
          Lock
        </Button>
      )}
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
          disabled={isLoadingModel}
        >
          {isLoadingModel ? <Spinner size="sm" /> : <Play className="w-4 h-4" />}
          {anotherModelLoaded ? 'Switch Model' : 'Load Model'}
        </Button>
      )}
    </div>
  ) : isPreview ? (
    <div className="flex items-center justify-between w-full">
      <button
        onClick={onClose}
        className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors text-slate-300 text-sm"
      >
        Cancel
      </button>
      {onDownloadFromPreview && (
        <button
          onClick={handleDownload}
          disabled={previewData?.requires_trust_remote_code && !trustRemoteCode}
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium text-sm flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          {`Download with ${selectedQuantization}`}
        </button>
      )}
    </div>
  ) : null;

  return (
    <Modal
      id="model-details"
      title={title}
      isOpen={isOpen}
      onClose={onClose}
      size="2xl"
      footer={footer}
    >
      <div className="space-y-6">
        {/* Status & Repository */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-slate-400 mb-1">Repository</p>
            <div className="flex items-center gap-2">
              <code className="text-sm text-slate-200 bg-slate-800 px-2 py-1 rounded">
                {model?.repo_id || repoId || 'N/A'}
              </code>
              {(model?.repo_id || repoId) && (
                <a
                  href={`https://huggingface.co/${model?.repo_id || repoId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-400 hover:text-primary-300"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {model?.locked && (
              <Badge variant="warning">
                <Lock className="w-3 h-3 mr-1" />
                Locked
              </Badge>
            )}
            {model && getStatusBadge(model.status)}
          </div>
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

        {/* Trust Remote Code Warning (preview) */}
        {isPreview && previewData?.requires_trust_remote_code && (
          <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium text-yellow-300">Trust Remote Code Required</p>
                <p className="text-sm text-yellow-200/80 mt-1">
                  This model requires executing custom code from the repository. Enable
                  "Trust Remote Code" below to proceed. Only download if you trust the source.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Gated Model Warning (preview) */}
        {isPreview && previewData?.is_gated && (
          <div className="flex items-center gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-300">
              This model is gated. You'll need a HuggingFace token with access to download it.
            </p>
          </div>
        )}

        {/* Downloads / Likes / Pipeline Stats Row (preview) */}
        {isPreview && previewData && (
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <TrendingUp className="w-4 h-4" />
                <span className="text-xs">Downloads</span>
              </div>
              <p className="text-xl font-semibold text-slate-100">
                {previewData.downloads.toLocaleString()}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <Heart className="w-4 h-4" />
                <span className="text-xs">Likes</span>
              </div>
              <p className="text-xl font-semibold text-slate-100">
                {previewData.likes.toLocaleString()}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4">
              <div className="text-xs text-slate-500 mb-1">Pipeline</div>
              <p className="text-sm font-medium text-slate-300">
                {previewData.pipeline_tag || 'N/A'}
              </p>
            </div>
          </div>
        )}

        {/* Model Stats Grid (downloaded models) */}
        {model && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <Server className="w-3 h-3" />
                <span className="text-xs">Parameters</span>
              </div>
              <p className="text-lg font-semibold text-slate-200">
                {model.params || 'Unknown'}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <RefreshCw className="w-3 h-3" />
                <span className="text-xs">Quantization</span>
              </div>
              <p className="text-lg font-semibold text-slate-200">
                {model.quantization || 'N/A'}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <HardDrive className="w-3 h-3" />
                <span className="text-xs">Disk Size</span>
              </div>
              <p className="text-lg font-semibold text-slate-200">
                {model.memory_mb ? formatSize(model.memory_mb) : 'Unknown'}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <Cpu className="w-3 h-3" />
                <span className="text-xs">Architecture</span>
              </div>
              <p className="text-lg font-semibold text-slate-200">
                {model.architecture || 'Unknown'}
              </p>
            </div>
          </div>
        )}

        {/* Tags (preview) */}
        {isPreview && previewData?.tags && previewData.tags.length > 0 && (
          <div>
            <div className="text-sm font-medium text-slate-300 mb-2">Tags</div>
            <div className="flex flex-wrap gap-2">
              {previewData.tags.slice(0, 10).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-primary-500/10 border border-primary-500/30 rounded text-xs text-primary-300"
                >
                  {tag}
                </span>
              ))}
              {previewData.tags.length > 10 && (
                <span className="px-2 py-1 text-xs text-slate-500">
                  +{previewData.tags.length - 10} more
                </span>
              )}
            </div>
          </div>
        )}

        {/* Architecture Info (preview) */}
        {isPreview && previewData?.architectures && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-5 h-5 text-primary-400" />
              <h3 className="text-base font-semibold text-slate-100">Architecture</h3>
            </div>
            <div className="bg-slate-800/30 border border-slate-700 rounded-lg p-4">
              <div className="space-y-2">
                <div>
                  <div className="text-xs text-slate-500">Model Type</div>
                  <div className="text-sm text-slate-300 font-mono">
                    {previewData.model_type || 'Unknown'}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500">Architectures</div>
                  <div className="text-sm text-slate-300 font-mono">
                    {previewData.architectures.join(', ')}
                  </div>
                </div>
                {previewData.params && (
                  <div>
                    <div className="text-xs text-slate-500">Parameters</div>
                    <div className="text-sm text-slate-300 font-mono">
                      {previewData.params}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* License & Language (preview) */}
        {isPreview && (previewData?.license || previewData?.language) && (
          <div className="grid grid-cols-2 gap-4">
            {previewData.license && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 text-slate-500 mb-1">
                  <Scale className="w-3.5 h-3.5" />
                  <span className="text-xs">License</span>
                </div>
                <div className="text-sm text-slate-300 font-medium">{previewData.license}</div>
              </div>
            )}
            {previewData.language && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="text-xs text-slate-500 mb-1">Language</div>
                <div className="text-sm text-slate-300 font-medium">
                  {Array.isArray(previewData.language)
                    ? previewData.language.join(', ')
                    : previewData.language}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Memory Requirements Table (preview) */}
        {isPreview && previewData?.estimated_sizes && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Download className="w-5 h-5 text-primary-400" />
              <h3 className="text-base font-semibold text-slate-100">Memory Requirements</h3>
            </div>
            <div className="bg-slate-800/30 border border-slate-700 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead className="bg-slate-800">
                  <tr>
                    <th className="text-left px-4 py-3 text-sm font-medium text-slate-300 w-12">
                      Select
                    </th>
                    <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">
                      Quantization
                    </th>
                    <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                      Est. Disk
                    </th>
                    <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                      Est. VRAM
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {QUANTIZATION_OPTIONS.map(({ format, label, recommended }) => {
                    const size = previewData.estimated_sizes?.[format];
                    if (!size) return null;
                    const isSelected = selectedQuantization === format;
                    return (
                      <tr
                        key={format}
                        onClick={() => setSelectedQuantization(format)}
                        className={`
                          border-t border-slate-700/50 cursor-pointer transition-colors
                          ${isSelected
                            ? 'bg-primary-500/10'
                            : 'hover:bg-slate-800/50'
                          }
                        `}
                      >
                        <td className="px-4 py-3">
                          <input
                            type="radio"
                            name="quantization"
                            value={format}
                            checked={isSelected}
                            onChange={() => setSelectedQuantization(format)}
                            className="w-4 h-4 text-primary-600 bg-slate-700 border-slate-600 focus:ring-primary-500 focus:ring-2"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-slate-100 text-sm">{label}</span>
                          {recommended && (
                            <span className="ml-2 text-xs bg-primary-500/20 text-primary-300 px-2 py-0.5 rounded">
                              Recommended
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span className="text-slate-300 text-sm font-mono">
                            {formatSize(size.disk_mb)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span className="text-slate-300 text-sm font-mono">
                            {formatSize(size.memory_mb)}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              * Memory estimates include ~20% overhead for inference. Actual requirements may vary.
            </p>
          </div>
        )}

        {/* Trust Remote Code Checkbox (preview footer area) */}
        {isPreview && previewData?.requires_trust_remote_code && (
          <div className="flex items-start gap-3 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <input
              type="checkbox"
              id="preview-trust-remote-code"
              checked={trustRemoteCode}
              onChange={(e) => setTrustRemoteCode(e.target.checked)}
              className="mt-1 w-4 h-4 rounded border-yellow-500/50 bg-slate-900 text-yellow-500 focus:ring-yellow-500 focus:ring-offset-slate-950"
            />
            <div className="flex-1">
              <label htmlFor="preview-trust-remote-code" className="block text-sm font-medium text-yellow-300 cursor-pointer">
                Trust Remote Code
              </label>
              <p className="mt-1 text-xs text-yellow-200/80">
                I understand this model requires executing custom code and I trust the source.
              </p>
            </div>
          </div>
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
