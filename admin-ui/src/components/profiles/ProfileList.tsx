import { FileText } from 'lucide-react';
import { Card, CardHeader, EmptyState } from '@components/common';
import { ProfileListItem } from './ProfileListItem';
import type { Profile } from '@/types';

interface ProfileListProps {
  profiles: Profile[];
  activeProfileId?: number;
  onActivate: (id: number) => void;
  onDeactivate: (id: number) => void;
  onEdit: (profile: Profile) => void;
  onDelete: (id: number) => void;
  activatingId?: number;
  deactivatingId?: number;
  deletingId?: number;
  disabled?: boolean;
}

export function ProfileList({
  profiles,
  activeProfileId,
  onActivate,
  onDeactivate,
  onEdit,
  onDelete,
  activatingId,
  deactivatingId,
  deletingId,
  disabled,
}: ProfileListProps) {
  if (profiles.length === 0) {
    return (
      <Card>
        <CardHeader
          title="Saved Profiles"
          subtitle="Your steering configurations"
          icon={<FileText className="w-5 h-5 text-slate-400" />}
        />
        <EmptyState
          icon={<FileText className="w-8 h-8" />}
          title="No profiles yet"
          description="Create a profile to save your steering configuration for later use"
        />
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title="Saved Profiles"
        subtitle={`${profiles.length} profile${profiles.length !== 1 ? 's' : ''}`}
        icon={<FileText className="w-5 h-5 text-slate-400" />}
      />
      <div className="space-y-3">
        {profiles.map((profile) => (
          <ProfileListItem
            key={profile.id}
            profile={profile}
            isActive={profile.id === activeProfileId}
            onActivate={() => onActivate(profile.id)}
            onDeactivate={() => onDeactivate(profile.id)}
            onEdit={() => onEdit(profile)}
            onDelete={() => onDelete(profile.id)}
            isActivating={activatingId === profile.id}
            isDeactivating={deactivatingId === profile.id}
            isDeleting={deletingId === profile.id}
            disabled={disabled}
          />
        ))}
      </div>
    </Card>
  );
}
