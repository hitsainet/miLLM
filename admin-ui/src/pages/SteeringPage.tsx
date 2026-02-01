import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, Sliders, Info } from 'lucide-react';
import { useSteering } from '@hooks/useSteering';
import { useSAE } from '@hooks/useSAE';
import { useModels } from '@hooks/useModels';
import { useServerStore } from '@stores/serverStore';
import {
  SteeringControls,
  FeatureInput,
  BatchAddForm,
  SteeringSlider,
} from '@components/steering';
import { Card, CardHeader, Spinner, EmptyState, Button, Modal } from '@components/common';

export function SteeringPage() {
  const navigate = useNavigate();
  // Use hooks to keep queries active and store populated
  // This ensures loadedModel and attachedSAE stay in sync with backend state
  useModels();
  useSAE();
  // Use 'steering' directly instead of 'steeringState' getter for proper Zustand reactivity
  const { loadedModel, attachedSAE, steering } = useServerStore();
  const {
    isLoading,
    setFeature,
    isSetting,
    batchSetFeatures,
    removeFeature,
    isRemoving,
    clearFeatures,
    isClearing,
    enableSteering,
    disableSteering,
    isEnabling,
    isDisabling,
  } = useSteering();

  const [saveProfileModal, setSaveProfileModal] = useState(false);

  const features = steering?.features || [];
  const featureCount = features.length;
  const isEnabled = steering?.enabled || false;

  const handleAddFeature = async (featureIndex: number, strength: number = 1.0) => {
    await setFeature({ index: featureIndex, strength });
  };

  const handleBatchAdd = async (featureList: Array<{ index: number; strength: number }>) => {
    await batchSetFeatures(featureList);
  };

  const handleStrengthChange = async (featureIndex: number, strength: number) => {
    await setFeature({ index: featureIndex, strength });
  };

  const handleRemove = async (featureIndex: number) => {
    await removeFeature(featureIndex);
  };

  const handleToggle = async () => {
    if (isEnabled) {
      await disableSteering();
    } else {
      await enableSteering();
    }
  };

  const handleClear = async () => {
    await clearFeatures();
  };

  const handleSaveProfile = () => {
    // Navigate to profiles page with state to create new profile
    navigate('/profiles', { state: { createFromCurrent: true } });
    setSaveProfileModal(false);
  };

  // No model loaded
  if (!loadedModel) {
    return (
      <Card className="border-yellow-500/30 bg-yellow-500/5">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-yellow-400">No Model Loaded</h3>
            <p className="text-sm text-slate-400 mt-1">
              You need to load a model and attach an SAE before you can configure steering.
            </p>
            <Button
              variant="primary"
              size="sm"
              className="mt-3"
              onClick={() => navigate('/models')}
            >
              Go to Models
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  // No SAE attached
  if (!attachedSAE) {
    return (
      <Card className="border-yellow-500/30 bg-yellow-500/5">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-yellow-400">No SAE Attached</h3>
            <p className="text-sm text-slate-400 mt-1">
              You need to attach an SAE to the loaded model before you can configure steering.
            </p>
            <Button
              variant="primary"
              size="sm"
              className="mt-3"
              onClick={() => navigate('/sae')}
            >
              Go to SAE
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Steering Controls */}
      <SteeringControls
        isEnabled={isEnabled}
        featureCount={featureCount}
        onToggle={handleToggle}
        onClear={handleClear}
        onSaveProfile={() => setSaveProfileModal(true)}
        isToggling={isEnabling || isDisabling}
        isClearing={isClearing}
      />

      {/* Add Features */}
      <Card>
        <CardHeader
          title="Add Features"
          subtitle="Add feature indices to configure steering"
          icon={<Sliders className="w-5 h-5 text-primary-400" />}
          action={
            <BatchAddForm
              onBatchAdd={handleBatchAdd}
              maxFeatureIndex={attachedSAE.d_sae}
            />
          }
        />
        <FeatureInput
          onAdd={handleAddFeature}
          maxFeatureIndex={attachedSAE.d_sae}
        />
      </Card>

      {/* Feature Sliders */}
      <Card>
        <CardHeader
          title="Active Features"
          subtitle={featureCount > 0 ? `${featureCount} feature${featureCount !== 1 ? 's' : ''} configured` : 'No features added yet'}
        />

        {featureCount > 0 ? (
          <div className="space-y-2">
            {[...features]
              .sort((a, b) => a.index - b.index)
              .map((feature) => (
                  <SteeringSlider
                    key={feature.index}
                    featureIndex={feature.index}
                    strength={feature.strength}
                    onStrengthChange={(newStrength) => handleStrengthChange(feature.index, newStrength)}
                    onRemove={() => handleRemove(feature.index)}
                    disabled={isSetting || isRemoving}
                  />
                ))}
          </div>
        ) : (
          <EmptyState
            icon={<Sliders className="w-8 h-8" />}
            title="No features configured"
            description="Add feature indices above to start steering model behavior"
          />
        )}
      </Card>

      {/* Info Section */}
      <Card padding="sm">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="mb-2">
              <strong className="text-slate-300">Positive values:</strong> Amplify the feature (e.g., +5 for more of that behavior)
            </p>
            <p className="mb-2">
              <strong className="text-slate-300">Negative values:</strong> Suppress the feature (e.g., -5 for less of that behavior)
            </p>
            <p>
              <strong className="text-slate-300">Tip:</strong> Use Neuronpedia to find feature indices and their meanings.
            </p>
          </div>
        </div>
      </Card>

      {/* Save Profile Modal */}
      <Modal
        id="save-profile"
        title="Save as Profile"
        isOpen={saveProfileModal}
        onClose={() => setSaveProfileModal(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setSaveProfileModal(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleSaveProfile}>
              Go to Profiles
            </Button>
          </>
        }
      >
        <p className="text-slate-300">
          You can save your current steering configuration as a profile from the Profiles page.
        </p>
        <p className="text-slate-400 text-sm mt-2">
          This will allow you to quickly restore these settings later.
        </p>
      </Modal>
    </div>
  );
}

export default SteeringPage;
