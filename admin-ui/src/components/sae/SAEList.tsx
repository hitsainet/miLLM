import { Layers } from 'lucide-react';
import { Card, CardHeader, EmptyState } from '@components/common';
import { SAEListItem } from './SAEListItem';
import type { SAEInfo } from '@/types';

interface SAEListProps {
  saes: SAEInfo[];
  attachedSAEId?: string;
  onAttach: (sae: SAEInfo) => void;
  onDetach: () => void;
  onDelete: (id: string) => void;
  onCancel?: (id: string) => void;
  attachingId?: string;
  isDetaching?: boolean;
  deletingId?: string;
  cancellingId?: string;
  canAttach?: boolean;
}

export function SAEList({
  saes,
  attachedSAEId,
  onAttach,
  onDetach,
  onDelete,
  onCancel,
  attachingId,
  isDetaching,
  deletingId,
  cancellingId,
  canAttach = true,
}: SAEListProps) {
  if (saes.length === 0) {
    return (
      <Card>
        <CardHeader
          title="Downloaded SAEs"
          subtitle="Sparse Autoencoders available locally"
          icon={<Layers className="w-5 h-5 text-slate-400" />}
        />
        <EmptyState
          icon={<Layers className="w-8 h-8" />}
          title="No SAEs downloaded"
          description="Download an SAE from Hugging Face to enable feature steering"
        />
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title="Downloaded SAEs"
        subtitle={`${saes.length} SAE${saes.length !== 1 ? 's' : ''} available`}
        icon={<Layers className="w-5 h-5 text-slate-400" />}
      />
      <div className="space-y-3">
        {saes.map((sae) => (
          <SAEListItem
            key={sae.id}
            sae={sae}
            isAttached={sae.id === attachedSAEId}
            onAttach={() => onAttach(sae)}
            onDetach={onDetach}
            onDelete={() => onDelete(sae.id)}
            onCancel={onCancel ? () => onCancel(sae.id) : undefined}
            isAttaching={attachingId === sae.id}
            isDetaching={sae.id === attachedSAEId && isDetaching}
            isDeleting={deletingId === sae.id}
            isCancelling={cancellingId === sae.id}
            canAttach={canAttach && attachedSAEId === undefined}
          />
        ))}
      </div>
    </Card>
  );
}
