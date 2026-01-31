import { useState } from 'react';
import { Server, Square, HardDrive, Cpu, Layers, ExternalLink } from 'lucide-react';
import { Card, CardHeader, Button, Modal, Badge } from '@components/common';
import type { ModelInfo } from '@/types';

interface LoadedModelCardProps {
  model: ModelInfo;
  onUnload: () => void;
  isUnloading?: boolean;
}

export function LoadedModelCard({
  model,
  onUnload,
  isUnloading,
}: LoadedModelCardProps) {
  const [showConfirmModal, setShowConfirmModal] = useState(false);

  const handleUnload = () => {
    setShowConfirmModal(false);
    onUnload();
  };

  const formatBytes = (bytes: number): string => {
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <>
      <Card className="border-l-4 border-l-green-500">
        <CardHeader
          title="Loaded Model"
          subtitle="Currently active model for inference"
          icon={<Server className="w-5 h-5 text-green-400" />}
          action={
            <Button
              variant="danger"
              size="sm"
              onClick={() => setShowConfirmModal(true)}
              loading={isUnloading}
              leftIcon={<Square className="w-4 h-4" />}
            >
              Unload
            </Button>
          }
        />

        <div className="space-y-4">
          {/* Model Name */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-100">{model.name}</h3>
              <a
                href={`https://huggingface.co/${model.repo_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary-400 hover:text-primary-300 flex items-center gap-1"
              >
                {model.repo_id}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
            <Badge variant="success">Loaded</Badge>
          </div>

          {/* Model Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                <Layers className="w-3 h-3" />
                Parameters
              </div>
              <p className="text-slate-200 font-semibold">
                {model.num_parameters
                  ? (model.num_parameters / 1e9).toFixed(1) + 'B'
                  : 'Unknown'
                }
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                <HardDrive className="w-3 h-3" />
                Memory
              </div>
              <p className="text-slate-200 font-semibold">
                {model.memory_footprint
                  ? formatBytes(model.memory_footprint)
                  : 'Unknown'
                }
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                <Cpu className="w-3 h-3" />
                Device
              </div>
              <p className="text-slate-200 font-semibold uppercase">
                {model.device || 'Unknown'}
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                Data Type
              </div>
              <p className="text-slate-200 font-semibold">
                {model.dtype || 'Unknown'}
              </p>
            </div>
          </div>

          {/* Model Architecture */}
          {model.architecture && (
            <div className="bg-slate-800/30 rounded-lg p-3">
              <p className="text-xs text-slate-400 mb-1">Architecture</p>
              <p className="text-sm text-slate-300 font-mono">{model.architecture}</p>
            </div>
          )}
        </div>
      </Card>

      {/* Unload Confirmation Modal */}
      <Modal
        id="unload-model-confirm"
        title="Unload Model"
        isOpen={showConfirmModal}
        onClose={() => setShowConfirmModal(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setShowConfirmModal(false)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={handleUnload} loading={isUnloading}>
              Unload Model
            </Button>
          </>
        }
      >
        <p className="text-slate-300">
          Are you sure you want to unload <strong>{model.name}</strong>?
        </p>
        <p className="text-slate-400 text-sm mt-2">
          This will also detach any attached SAE and clear steering configuration.
        </p>
      </Modal>
    </>
  );
}
