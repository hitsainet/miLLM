import { Layers, Link, Unlink, Trash2, ExternalLink } from 'lucide-react';
import { Button, Badge, Spinner } from '@components/common';
import type { SAEInfo } from '@/types';

interface SAEListItemProps {
  sae: SAEInfo;
  isAttached: boolean;
  onAttach: () => void;
  onDetach: () => void;
  onDelete: () => void;
  isAttaching?: boolean;
  isDetaching?: boolean;
  isDeleting?: boolean;
  canAttach?: boolean;
}

export function SAEListItem({
  sae,
  isAttached,
  onAttach,
  onDetach,
  onDelete,
  isAttaching,
  isDetaching,
  isDeleting,
  canAttach = true,
}: SAEListItemProps) {
  const isLoading = isAttaching || isDetaching || isDeleting;

  return (
    <div
      className={`
        flex items-center justify-between p-4 rounded-lg border transition-colors
        ${isAttached
          ? 'bg-purple-500/5 border-purple-500/30'
          : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
        }
      `}
    >
      <div className="flex items-center gap-3">
        <div className={`
          p-2 rounded-lg
          ${isAttached
            ? 'bg-purple-500/10 text-purple-400'
            : 'bg-slate-700/50 text-slate-400'
          }
        `}>
          <Layers className="w-5 h-5" />
        </div>

        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-medium text-slate-200">{sae.name}</h4>
            {isAttached && (
              <Badge variant="primary" size="sm">Attached</Badge>
            )}
            {sae.status === 'downloading' && (
              <Badge variant="warning" size="sm">Downloading</Badge>
            )}
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
            <span>Layer {sae.layer}</span>
            <span>•</span>
            <span>{sae.num_features?.toLocaleString()} features</span>
            {sae.d_model && (
              <>
                <span>•</span>
                <span>d_model: {sae.d_model}</span>
              </>
            )}
          </div>
          <a
            href={`https://huggingface.co/${sae.repo_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary-400/70 hover:text-primary-400 flex items-center gap-1 mt-1"
          >
            {sae.repo_id}
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {sae.status === 'downloading' ? (
          <div className="flex items-center gap-2">
            <Spinner size="sm" />
            <span className="text-xs text-slate-400">
              {sae.download_progress ? `${sae.download_progress}%` : 'Downloading...'}
            </span>
          </div>
        ) : (
          <>
            {isAttached ? (
              <Button
                variant="secondary"
                size="sm"
                onClick={onDetach}
                loading={isDetaching}
                disabled={isLoading}
                leftIcon={<Unlink className="w-3 h-3" />}
              >
                Detach
              </Button>
            ) : (
              <Button
                variant="primary"
                size="sm"
                onClick={onAttach}
                loading={isAttaching}
                disabled={isLoading || !canAttach}
                leftIcon={<Link className="w-3 h-3" />}
              >
                Attach
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              loading={isDeleting}
              disabled={isLoading || isAttached}
              className="text-slate-400 hover:text-red-400"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
