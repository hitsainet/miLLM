import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { profileApi } from '@/services/api';
import { useServerStore } from '@/stores/serverStore';
import { useToast } from './useToast';
import type { CreateProfileRequest, UpdateProfileRequest, ProfileExport } from '@/types';

export function useProfiles() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const { setProfiles, setActiveProfile } = useServerStore();

  const profilesQuery = useQuery({
    queryKey: ['profiles'],
    queryFn: async () => {
      const profiles = await profileApi.list();
      setProfiles(profiles);
      const active = profiles.find((p) => p.is_active);
      if (active) setActiveProfile(active);
      return profiles;
    },
  });

  const createMutation = useMutation({
    mutationFn: (req: CreateProfileRequest) => profileApi.create(req),
    onSuccess: (profile) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      toast.success(`Profile "${profile.name}" created`);
    },
    onError: (error: Error) => {
      toast.error(`Create failed: ${error.message}`);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: UpdateProfileRequest }) =>
      profileApi.update(id, data),
    onSuccess: (profile) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      toast.success(`Profile "${profile.name}" updated`);
    },
    onError: (error: Error) => {
      toast.error(`Update failed: ${error.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => profileApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      toast.success('Profile deleted');
    },
    onError: (error: Error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  const activateMutation = useMutation({
    mutationFn: (id: number) => profileApi.activate(id),
    onSuccess: (profile) => {
      setActiveProfile(profile);
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      queryClient.invalidateQueries({ queryKey: ['steering'] });
      toast.success(`Profile "${profile.name}" activated`);
    },
    onError: (error: Error) => {
      toast.error(`Activate failed: ${error.message}`);
    },
  });

  const deactivateMutation = useMutation({
    mutationFn: (id: number) => profileApi.deactivate(id),
    onSuccess: () => {
      setActiveProfile(null);
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      toast.info('Profile deactivated');
    },
    onError: (error: Error) => {
      toast.error(`Deactivate failed: ${error.message}`);
    },
  });

  const exportProfile = async (id: number): Promise<ProfileExport | null> => {
    try {
      const data = await profileApi.export(id);
      return data;
    } catch (error) {
      toast.error(`Export failed: ${(error as Error).message}`);
      return null;
    }
  };

  const importMutation = useMutation({
    mutationFn: (data: ProfileExport) => profileApi.import(data),
    onSuccess: (profile) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      toast.success(`Profile "${profile.name}" imported`);
    },
    onError: (error: Error) => {
      toast.error(`Import failed: ${error.message}`);
    },
  });

  return {
    profiles: profilesQuery.data ?? [],
    isLoading: profilesQuery.isLoading,
    error: profilesQuery.error?.message,
    refetch: profilesQuery.refetch,
    create: createMutation.mutate,
    createProfile: createMutation.mutateAsync,
    update: updateMutation.mutate,
    updateProfile: (id: number, data: UpdateProfileRequest) =>
      updateMutation.mutateAsync({ id, data }),
    delete: deleteMutation.mutate,
    deleteProfile: deleteMutation.mutateAsync,
    activate: activateMutation.mutate,
    activateProfile: activateMutation.mutateAsync,
    deactivate: deactivateMutation.mutate,
    deactivateProfile: deactivateMutation.mutateAsync,
    exportProfile,
    importProfile: importMutation.mutate,
    isCreating: createMutation.isPending,
    isUpdating: updateMutation.isPending,
    isDeleting: deleteMutation.isPending,
    isActivating: activateMutation.isPending,
    isDeactivating: deactivateMutation.isPending,
    isExporting: false, // Export is sync
    isImporting: importMutation.isPending,
  };
}

export default useProfiles;
