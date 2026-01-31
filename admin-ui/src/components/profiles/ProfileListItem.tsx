import { FileText, Play, Square, Edit, Trash2, Hash } from 'lucide-react';
import { Button, Badge } from '@components/common';
import type { Profile } from '@/types';

interface ProfileListItemProps {
  profile: Profile;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
  onEdit: () => void;
  onDelete: () => void;
  isActivating?: boolean;
  isDeactivating?: boolean;
  isDeleting?: boolean;
  disabled?: boolean;
}

export function ProfileListItem({
  profile,
  isActive,
  onActivate,
  onDeactivate,
  onEdit,
  onDelete,
  isActivating,
  isDeactivating,
  isDeleting,
  disabled,
}: ProfileListItemProps) {
  const isLoading = isActivating || isDeactivating || isDeleting;
  const featureCount = profile.features?.length ?? 0;

  return (
    <div
      className={`
        p-4 rounded-lg border transition-colors
        ${isActive
          ? 'bg-green-500/5 border-green-500/30'
          : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
        }
      `}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className={`
            p-2 rounded-lg
            ${isActive
              ? 'bg-green-500/10 text-green-400'
              : 'bg-slate-700/50 text-slate-400'
            }
          `}>
            <FileText className="w-5 h-5" />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-medium text-slate-200">{profile.name}</h4>
              {isActive && (
                <Badge variant="success" size="sm">Active</Badge>
              )}
            </div>

            {profile.description && (
              <p className="text-xs text-slate-500 mt-1">{profile.description}</p>
            )}

            <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
              <span className="flex items-center gap-1">
                <Hash className="w-3 h-3" />
                {featureCount} feature{featureCount !== 1 ? 's' : ''}
              </span>
              {profile.created_at && (
                <span>
                  Created {new Date(profile.created_at).toLocaleDateString()}
                </span>
              )}
            </div>

            {/* Feature Preview */}
            {featureCount > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {(profile.features || []).slice(0, 5).map((feature) => (
                  <span
                    key={feature.index}
                    className={`
                      px-2 py-0.5 rounded text-xs font-mono
                      ${feature.strength > 0
                        ? 'bg-green-500/10 text-green-400'
                        : feature.strength < 0
                          ? 'bg-red-500/10 text-red-400'
                          : 'bg-slate-700/50 text-slate-400'
                      }
                    `}
                  >
                    #{feature.index}: {feature.strength > 0 ? '+' : ''}{feature.strength}
                  </span>
                ))}
                {featureCount > 5 && (
                  <span className="text-xs text-slate-500 self-center">
                    +{featureCount - 5} more
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isActive ? (
            <Button
              variant="secondary"
              size="sm"
              onClick={onDeactivate}
              loading={isDeactivating}
              disabled={isLoading || disabled}
              leftIcon={<Square className="w-3 h-3" />}
            >
              Deactivate
            </Button>
          ) : (
            <Button
              variant="primary"
              size="sm"
              onClick={onActivate}
              loading={isActivating}
              disabled={isLoading || disabled}
              leftIcon={<Play className="w-3 h-3" />}
            >
              Activate
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onEdit}
            disabled={isLoading || disabled}
            className="text-slate-400 hover:text-primary-400"
          >
            <Edit className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onDelete}
            loading={isDeleting}
            disabled={isLoading || disabled || isActive}
            className="text-slate-400 hover:text-red-400"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
