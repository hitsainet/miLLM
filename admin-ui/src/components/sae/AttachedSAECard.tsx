import { useState } from 'react';
import { Layers, Unlink, ExternalLink, Hash, Box } from 'lucide-react';
import { Card, CardHeader, Button, Modal, Badge } from '@components/common';
import type { SAEInfo } from '@/types';

interface AttachedSAECardProps {
  sae: SAEInfo;
  onDetach: () => void;
  isDetaching?: boolean;
}

export function AttachedSAECard({
  sae,
  onDetach,
  isDetaching,
}: AttachedSAECardProps) {
  const [showConfirmModal, setShowConfirmModal] = useState(false);

  const handleDetach = () => {
    setShowConfirmModal(false);
    onDetach();
  };

  return (
    <>
      <Card className="border-l-4 border-l-purple-500">
        <CardHeader
          title="Attached SAE"
          subtitle="Currently active for feature steering"
          icon={<Layers className="w-5 h-5 text-purple-400" />}
          action={
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setShowConfirmModal(true)}
              loading={isDetaching}
              leftIcon={<Unlink className="w-4 h-4" />}
            >
              Detach
            </Button>
          }
        />

        <div className="space-y-4">
          {/* SAE Name */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-100">{sae.name}</h3>
              <a
                href={`https://huggingface.co/${sae.repo_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary-400 hover:text-primary-300 flex items-center gap-1"
              >
                {sae.repo_id}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
            <Badge variant="primary">Attached</Badge>
          </div>

          {/* SAE Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                <Layers className="w-3 h-3" />
                Layer
              </div>
              <p className="text-slate-200 font-semibold">{sae.layer}</p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                <Hash className="w-3 h-3" />
                Features
              </div>
              <p className="text-slate-200 font-semibold">
                {sae.num_features?.toLocaleString() || 'Unknown'}
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                <Box className="w-3 h-3" />
                d_model
              </div>
              <p className="text-slate-200 font-semibold">
                {sae.d_model?.toLocaleString() || 'Unknown'}
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                Status
              </div>
              <p className="text-slate-200 font-semibold capitalize">
                {sae.status}
              </p>
            </div>
          </div>

          {/* Additional Info */}
          {sae.filename && (
            <div className="bg-slate-800/30 rounded-lg p-3">
              <p className="text-xs text-slate-400 mb-1">Filename</p>
              <p className="text-sm text-slate-300 font-mono truncate">{sae.filename}</p>
            </div>
          )}
        </div>
      </Card>

      {/* Detach Confirmation Modal */}
      <Modal
        id="detach-sae-confirm"
        title="Detach SAE"
        isOpen={showConfirmModal}
        onClose={() => setShowConfirmModal(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setShowConfirmModal(false)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={handleDetach} loading={isDetaching}>
              Detach SAE
            </Button>
          </>
        }
      >
        <p className="text-slate-300">
          Are you sure you want to detach <strong>{sae.name}</strong>?
        </p>
        <p className="text-slate-400 text-sm mt-2">
          This will clear all current steering configuration and monitoring data.
        </p>
      </Modal>
    </>
  );
}
