import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Plus, Info } from 'lucide-react';
import { useProfiles } from '@hooks/useProfiles';
import { useServerStore } from '@stores/serverStore';
import {
  ProfileList,
  ProfileForm,
  ImportExportButtons,
} from '@components/profiles';
import { Card, Button, Spinner } from '@components/common';
import type { Profile, CreateProfileRequest, UpdateProfileRequest, ProfileExport } from '@/types';

export function ProfilesPage() {
  const location = useLocation();
  // Use 'steering' directly instead of 'steeringState' getter for proper Zustand reactivity
  const { steering, activeProfile } = useServerStore();
  const {
    profiles,
    isLoading,
    createProfile,
    isCreating,
    updateProfile,
    isUpdating,
    deleteProfile,
    activateProfile,
    deactivateProfile,
    exportProfile,
    isExporting,
    importProfile,
    isImporting,
  } = useProfiles();

  const [showForm, setShowForm] = useState(false);
  const [editingProfile, setEditingProfile] = useState<Profile | undefined>();
  const [activatingId, setActivatingId] = useState<string | undefined>();
  const [deactivatingId, setDeactivatingId] = useState<string | undefined>();
  const [deletingId, setDeletingId] = useState<string | undefined>();

  // Check if we should auto-open the form (from steering page)
  const createFromCurrent = (location.state as { createFromCurrent?: boolean })?.createFromCurrent;
  useState(() => {
    if (createFromCurrent) {
      setShowForm(true);
    }
  });

  const currentFeatures = steering?.features || [];

  // Imported clusters are profiles under the hood but are managed on the
  // dedicated Clusters page (lambda dial, lossless export, compat gate) —
  // rendering them here as ordinary editable rows invited double-scaling and
  // lossy flat exports (review find).
  const manualProfiles = profiles.filter((p) => p.source_kind !== 'cluster');
  const clusterCount = profiles.length - manualProfiles.length;

  const handleCreate = async (data: CreateProfileRequest) => {
    await createProfile(data);
    setShowForm(false);
  };

  const handleUpdate = async (data: UpdateProfileRequest) => {
    if (editingProfile) {
      await updateProfile(editingProfile.id, data);
      setEditingProfile(undefined);
      setShowForm(false);
    }
  };

  const handleFormSubmit = async (data: CreateProfileRequest | UpdateProfileRequest) => {
    if (editingProfile) {
      await handleUpdate(data as UpdateProfileRequest);
    } else {
      await handleCreate(data as CreateProfileRequest);
    }
  };

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      await deleteProfile(id);
    } finally {
      setDeletingId(undefined);
    }
  };

  const handleActivate = async (id: string) => {
    setActivatingId(id);
    try {
      await activateProfile(id);
    } finally {
      setActivatingId(undefined);
    }
  };

  const handleDeactivate = async (id: string) => {
    setDeactivatingId(id);
    try {
      await deactivateProfile(id);
    } finally {
      setDeactivatingId(undefined);
    }
  };

  const handleEdit = (profile: Profile) => {
    setEditingProfile(profile);
    setShowForm(true);
  };

  const handleExport = async (profileId: string): Promise<ProfileExport | null> => {
    return exportProfile(profileId);
  };

  const handleImport = async (data: ProfileExport) => {
    await importProfile(data);
  };

  const handleCloseForm = () => {
    setShowForm(false);
    setEditingProfile(undefined);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Actions Bar */}
      <Card>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-slate-200">Profile Management</h3>
            <p className="text-xs text-slate-500 mt-1">
              Save and manage steering configurations
            </p>
          </div>
          <div className="flex items-center gap-3">
            <ImportExportButtons
              onExport={handleExport}
              onImport={handleImport}
              profiles={manualProfiles.map((p) => ({ id: p.id, name: p.name }))}
              isExporting={isExporting}
              isImporting={isImporting}
            />
            <Button
              variant="primary"
              size="sm"
              onClick={() => setShowForm(true)}
              leftIcon={<Plus className="w-4 h-4" />}
            >
              New Profile
            </Button>
          </div>
        </div>
      </Card>

      {/* Profile List */}
      {clusterCount > 0 && (
        <p className="text-xs text-slate-500">
          {clusterCount} imported cluster{clusterCount === 1 ? '' : 's'} managed on the{' '}
          <a href="/clusters" className="text-cyan-400 hover:underline">Clusters page</a>.
        </p>
      )}
      <ProfileList
        profiles={manualProfiles}
        activeProfileId={activeProfile?.id}
        onActivate={handleActivate}
        onDeactivate={handleDeactivate}
        onEdit={handleEdit}
        onDelete={handleDelete}
        activatingId={activatingId}
        deactivatingId={deactivatingId}
        deletingId={deletingId}
      />

      {/* Info Section */}
      <Card padding="sm">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="mb-2">
              <strong className="text-slate-300">Profiles:</strong> Save your steering configuration to quickly restore it later.
            </p>
            <p className="mb-2">
              <strong className="text-slate-300">Export/Import:</strong> Share profiles with others or move them between systems using JSON files.
            </p>
            <p>
              <strong className="text-slate-300">miStudio:</strong> Profiles use a compatible format for future miStudio integration.
            </p>
          </div>
        </div>
      </Card>

      {/* Profile Form Modal */}
      <ProfileForm
        isOpen={showForm}
        onClose={handleCloseForm}
        onSubmit={handleFormSubmit}
        isSubmitting={isCreating || isUpdating}
        profile={editingProfile}
        currentFeatures={currentFeatures}
      />
    </div>
  );
}

export default ProfilesPage;
