import { Layers } from 'lucide-react';
import { Card, CardHeader, EmptyState } from '@components/common';
import { SAEListItem } from './SAEListItem';
import type { SAEInfo } from '@/types';

interface SAEListProps {
  saes: SAEInfo[];
  attachedSAEId?: number;
  onAttach: (id: number) => void;
  onDetach: () => void;
  onDelete: (id: number) => void;
  attachingId?: number;
  isDetaching?: boolean;
  deletingId?: number;
  canAttach?: boolean;
}

export function SAEList({
  saes,
  attachedSAEId,
  onAttach,
  onDetach,
  onDelete,
  attachingId,
  isDetaching,
  deletingId,
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
            onAttach={() => onAttach(sae.id)}
            onDetach={onDetach}
            onDelete={() => onDelete(sae.id)}
            isAttaching={attachingId === sae.id}
            isDetaching={sae.id === attachedSAEId && isDetaching}
            isDeleting={deletingId === sae.id}
            canAttach={canAttach && attachedSAEId === undefined}
          />
        ))}
      </div>
    </Card>
  );
}
