import { useState } from 'react';
import { Save } from 'lucide-react';
import { Modal, Button, Input } from '@components/common';
import type { Profile, CreateProfileRequest, UpdateProfileRequest, FeatureSteering } from '@/types';

interface ProfileFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: CreateProfileRequest | UpdateProfileRequest) => void;
  isSubmitting?: boolean;
  profile?: Profile;
  currentFeatures?: FeatureSteering[];
}

// Inner form component that resets when key changes
function ProfileFormContent({
  profile,
  currentFeatures,
  onSubmit,
  onClose,
  isSubmitting,
}: Omit<ProfileFormProps, 'isOpen'>) {
  const isEditing = !!profile;

  // Initial values computed from props (no useEffect needed)
  const [name, setName] = useState(profile?.name ?? '');
  const [description, setDescription] = useState(profile?.description ?? '');
  const [useCurrentFeatures, setUseCurrentFeatures] = useState(!isEditing);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!name.trim()) {
      newErrors.name = 'Profile name is required';
    } else if (name.length > 100) {
      newErrors.name = 'Profile name must be less than 100 characters';
    }

    if (description.length > 500) {
      newErrors.description = 'Description must be less than 500 characters';
    }

    if (!isEditing && useCurrentFeatures && (!currentFeatures || currentFeatures.length === 0)) {
      newErrors.features = 'No features configured to save';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) return;

    if (isEditing) {
      onSubmit({
        name: name.trim(),
        description: description.trim() || undefined,
      });
    } else {
      onSubmit({
        name: name.trim(),
        description: description.trim() || undefined,
        features: useCurrentFeatures ? currentFeatures : undefined,
      });
    }
  };

  const currentFeatureCount = currentFeatures?.length ?? 0;

  return (
    <>
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Profile Name"
          placeholder="e.g., Formal Writing"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setErrors({ ...errors, name: '' });
          }}
          error={errors.name}
          autoFocus
        />

        <Input
          label="Description (Optional)"
          placeholder="Brief description of this profile's purpose"
          value={description}
          onChange={(e) => {
            setDescription(e.target.value);
            setErrors({ ...errors, description: '' });
          }}
          error={errors.description}
        />

        {!isEditing && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useCurrentFeatures"
                checked={useCurrentFeatures}
                onChange={(e) => setUseCurrentFeatures(e.target.checked)}
                className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary-500 focus:ring-primary-500 focus:ring-offset-0"
              />
              <label htmlFor="useCurrentFeatures" className="text-sm text-slate-300">
                Save current steering configuration
              </label>
            </div>

            {useCurrentFeatures && (
              <div className="pl-6">
                {currentFeatureCount > 0 ? (
                  <p className="text-xs text-slate-500">
                    Will save {currentFeatureCount} feature{currentFeatureCount !== 1 ? 's' : ''}
                  </p>
                ) : (
                  <p className="text-xs text-yellow-400">
                    No features currently configured
                  </p>
                )}
              </div>
            )}

            {errors.features && (
              <p className="text-sm text-red-400 pl-6">{errors.features}</p>
            )}
          </div>
        )}
      </form>
      <div className="flex justify-end gap-3 mt-6">
        <Button variant="secondary" onClick={onClose}>
          Cancel
        </Button>
        <Button
          variant="primary"
          onClick={handleSubmit}
          loading={isSubmitting}
          leftIcon={<Save className="w-4 h-4" />}
        >
          {isEditing ? 'Save Changes' : 'Create Profile'}
        </Button>
      </div>
    </>
  );
}

export function ProfileForm({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting,
  profile,
  currentFeatures,
}: ProfileFormProps) {
  const isEditing = !!profile;

  // Key changes when profile changes or modal opens, causing form to reset
  const formKey = `${isOpen}-${profile?.id ?? 'new'}`;

  return (
    <Modal
      id="profile-form"
      title={isEditing ? 'Edit Profile' : 'Create Profile'}
      isOpen={isOpen}
      onClose={onClose}
    >
      {isOpen && (
        <ProfileFormContent
          key={formKey}
          profile={profile}
          currentFeatures={currentFeatures}
          onSubmit={onSubmit}
          onClose={onClose}
          isSubmitting={isSubmitting}
        />
      )}
    </Modal>
  );
}
