import { describe, it, expect, beforeEach } from 'vitest';
import { useServerStore } from '../serverStore';
import type {
  ModelInfo,
  SAEInfo,
  SteeringState,
  MonitoringConfig,
  Profile,
  FeatureSteering,
  ActivationRecord,
  FeatureStatistics,
  FeatureActivation,
} from '@/types';

// Helper factories for test data
const createMockModel = (overrides: Partial<ModelInfo> = {}): ModelInfo => ({
  id: 1,
  name: 'gemma-2-2b',
  repo_id: 'google/gemma-2-2b',
  source: 'huggingface',
  quantization: 'Q4',
  params: '2.5B',
  memory_mb: 1800,
  local_path: '/data/models/gemma-2-2b',
  status: 'ready',
  created_at: '2026-01-30T12:00:00Z',
  updated_at: '2026-01-30T12:00:00Z',
  ...overrides,
});

const createMockSAE = (overrides: Partial<SAEInfo> = {}): SAEInfo => ({
  id: 'sae-1',
  repository_id: 'google/gemma-scope-2b-pt-res',
  revision: 'main',
  name: 'gemma-scope-2b-L12',
  format: 'saelens',
  d_in: 2048,
  d_sae: 16384,
  trained_on: 'gemma-2-2b',
  trained_layer: 12,
  width: '8x',
  average_l0: 50.0,
  file_size_bytes: 268435456,
  status: 'cached',
  error_message: null,
  created_at: '2026-01-30T12:00:00Z',
  updated_at: '2026-01-30T12:00:00Z',
  ...overrides,
});

const createMockProfile = (overrides: Partial<Profile> = {}): Profile => ({
  id: 1,
  name: 'yelling-demo',
  description: 'Makes model yell',
  features: [{ index: 1234, strength: 5.0, label: 'Yelling/Capitalization' }],
  is_active: false,
  created_at: '2026-01-30T12:00:00Z',
  updated_at: '2026-01-30T12:00:00Z',
  ...overrides,
});

describe('serverStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useServerStore.getState().reset();
  });

  describe('initial state', () => {
    it('has null loaded model by default', () => {
      const { loadedModel } = useServerStore.getState();
      expect(loadedModel).toBeNull();
    });

    it('has empty models list by default', () => {
      const { models } = useServerStore.getState();
      expect(models).toEqual([]);
    });

    it('has null attached SAE by default', () => {
      const { attachedSAE } = useServerStore.getState();
      expect(attachedSAE).toBeNull();
    });

    it('has steering disabled with empty features by default', () => {
      const { steering } = useServerStore.getState();
      expect(steering).toEqual({
        enabled: false,
        sae_id: null,
        features: [],
      });
    });

    it('has monitoring disabled by default', () => {
      const { monitoring } = useServerStore.getState();
      expect(monitoring).toEqual({
        enabled: false,
        sae_id: null,
        top_k: 10,
        feature_indices: null,
      });
    });

    it('has disconnected connection status by default', () => {
      const { connectionStatus } = useServerStore.getState();
      expect(connectionStatus).toBe('disconnected');
    });

    it('has zero GPU metrics by default', () => {
      const state = useServerStore.getState();
      expect(state.gpuMemoryUsed).toBe(0);
      expect(state.gpuMemoryTotal).toBe(0);
      expect(state.gpuUtilization).toBe(0);
      expect(state.gpuTemperature).toBe(0);
    });

    it('has no profiles by default', () => {
      const { profiles, activeProfile } = useServerStore.getState();
      expect(profiles).toEqual([]);
      expect(activeProfile).toBeNull();
    });

    it('has all loading flags set to false by default', () => {
      const state = useServerStore.getState();
      expect(state.modelLoading).toBe(false);
      expect(state.saeLoading).toBe(false);
      expect(state.steeringLoading).toBe(false);
      expect(state.monitoringLoading).toBe(false);
      expect(state.profilesLoading).toBe(false);
    });
  });

  describe('model actions', () => {
    it('setLoadedModel sets model data', () => {
      const model = createMockModel({ status: 'loaded' });
      useServerStore.getState().setLoadedModel(model);

      expect(useServerStore.getState().loadedModel).toEqual(model);
    });

    it('setLoadedModel(null) clears the loaded model', () => {
      const model = createMockModel({ status: 'loaded' });
      useServerStore.getState().setLoadedModel(model);
      useServerStore.getState().setLoadedModel(null);

      expect(useServerStore.getState().loadedModel).toBeNull();
    });

    it('setModels sets the models list', () => {
      const models = [
        createMockModel({ id: 1, name: 'model-a' }),
        createMockModel({ id: 2, name: 'model-b' }),
      ];
      useServerStore.getState().setModels(models);

      expect(useServerStore.getState().models).toHaveLength(2);
      expect(useServerStore.getState().models[0].name).toBe('model-a');
      expect(useServerStore.getState().models[1].name).toBe('model-b');
    });

    it('addModel appends a model to the list', () => {
      const model1 = createMockModel({ id: 1 });
      const model2 = createMockModel({ id: 2, name: 'model-b' });
      useServerStore.getState().setModels([model1]);
      useServerStore.getState().addModel(model2);

      expect(useServerStore.getState().models).toHaveLength(2);
    });

    it('updateModel updates a specific model and loaded model if matching', () => {
      const model = createMockModel({ id: 1, status: 'loaded' });
      useServerStore.getState().setModels([model]);
      useServerStore.getState().setLoadedModel(model);

      useServerStore.getState().updateModel(1, { name: 'updated-name' });

      expect(useServerStore.getState().models[0].name).toBe('updated-name');
      expect(useServerStore.getState().loadedModel?.name).toBe('updated-name');
    });

    it('removeModel removes a model and clears loadedModel if matching', () => {
      const model = createMockModel({ id: 1 });
      useServerStore.getState().setModels([model]);
      useServerStore.getState().setLoadedModel(model);

      useServerStore.getState().removeModel(1);

      expect(useServerStore.getState().models).toHaveLength(0);
      expect(useServerStore.getState().loadedModel).toBeNull();
    });

    it('setModelLoading sets the loading flag', () => {
      useServerStore.getState().setModelLoading(true);
      expect(useServerStore.getState().modelLoading).toBe(true);

      useServerStore.getState().setModelLoading(false);
      expect(useServerStore.getState().modelLoading).toBe(false);
    });

    it('setDownloadProgress tracks download progress per model', () => {
      useServerStore.getState().setDownloadProgress(1, 50);
      useServerStore.getState().setDownloadProgress(2, 75);

      const { downloadProgress } = useServerStore.getState();
      expect(downloadProgress[1]).toBe(50);
      expect(downloadProgress[2]).toBe(75);
    });

    it('clearDownloadProgress removes progress for a specific model', () => {
      useServerStore.getState().setDownloadProgress(1, 50);
      useServerStore.getState().setDownloadProgress(2, 75);
      useServerStore.getState().clearDownloadProgress(1);

      const { downloadProgress } = useServerStore.getState();
      expect(downloadProgress[1]).toBeUndefined();
      expect(downloadProgress[2]).toBe(75);
    });
  });

  describe('SAE actions', () => {
    it('setAttachedSAE sets SAE data', () => {
      const sae = createMockSAE({ status: 'attached' });
      useServerStore.getState().setAttachedSAE(sae);

      expect(useServerStore.getState().attachedSAE).toEqual(sae);
    });

    it('setAttachedSAE(null) clears the attached SAE', () => {
      const sae = createMockSAE({ status: 'attached' });
      useServerStore.getState().setAttachedSAE(sae);
      useServerStore.getState().setAttachedSAE(null);

      expect(useServerStore.getState().attachedSAE).toBeNull();
    });

    it('setSAEs sets the SAEs list', () => {
      const saes = [
        createMockSAE({ id: 'sae-1' }),
        createMockSAE({ id: 'sae-2', name: 'sae-b' }),
      ];
      useServerStore.getState().setSAEs(saes);

      expect(useServerStore.getState().saes).toHaveLength(2);
    });

    it('addSAE appends an SAE to the list', () => {
      useServerStore.getState().setSAEs([createMockSAE({ id: 'sae-1' })]);
      useServerStore.getState().addSAE(createMockSAE({ id: 'sae-2' }));

      expect(useServerStore.getState().saes).toHaveLength(2);
    });

    it('updateSAE updates a specific SAE and attachedSAE if matching', () => {
      const sae = createMockSAE({ id: 'sae-1', status: 'attached' });
      useServerStore.getState().setSAEs([sae]);
      useServerStore.getState().setAttachedSAE(sae);

      useServerStore.getState().updateSAE('sae-1', { name: 'updated-sae' });

      expect(useServerStore.getState().saes[0].name).toBe('updated-sae');
      expect(useServerStore.getState().attachedSAE?.name).toBe('updated-sae');
    });

    it('removeSAE removes an SAE and clears attachedSAE if matching', () => {
      const sae = createMockSAE({ id: 'sae-1' });
      useServerStore.getState().setSAEs([sae]);
      useServerStore.getState().setAttachedSAE(sae);

      useServerStore.getState().removeSAE('sae-1');

      expect(useServerStore.getState().saes).toHaveLength(0);
      expect(useServerStore.getState().attachedSAE).toBeNull();
    });

    it('setSAEDownloadProgress and clearSAEDownloadProgress work correctly', () => {
      useServerStore.getState().setSAEDownloadProgress('sae-1', 60);
      expect(useServerStore.getState().saeDownloadProgress['sae-1']).toBe(60);

      useServerStore.getState().clearSAEDownloadProgress('sae-1');
      expect(useServerStore.getState().saeDownloadProgress['sae-1']).toBeUndefined();
    });
  });

  describe('steering actions', () => {
    it('setSteering updates the full steering state', () => {
      const steeringState: SteeringState = {
        enabled: true,
        sae_id: 1,
        features: [{ index: 1234, strength: 5.0 }],
      };
      useServerStore.getState().setSteering(steeringState);

      expect(useServerStore.getState().steering).toEqual(steeringState);
    });

    it('setSteeringEnabled toggles only the enabled flag', () => {
      useServerStore.getState().setSteering({
        enabled: false,
        sae_id: 1,
        features: [{ index: 100, strength: 2.0 }],
      });

      useServerStore.getState().setSteeringEnabled(true);

      const { steering } = useServerStore.getState();
      expect(steering.enabled).toBe(true);
      expect(steering.features).toHaveLength(1);
      expect(steering.sae_id).toBe(1);
    });

    it('setFeatureSteering adds a new feature or replaces existing by index', () => {
      useServerStore.getState().setFeatureSteering({ index: 100, strength: 2.0 });
      expect(useServerStore.getState().steering.features).toHaveLength(1);

      // Add another feature
      useServerStore.getState().setFeatureSteering({ index: 200, strength: -1.0 });
      expect(useServerStore.getState().steering.features).toHaveLength(2);

      // Replace existing feature by same index
      useServerStore.getState().setFeatureSteering({ index: 100, strength: 5.0 });
      const { features } = useServerStore.getState().steering;
      expect(features).toHaveLength(2);
      const feature100 = features.find((f) => f.index === 100);
      expect(feature100?.strength).toBe(5.0);
    });

    it('removeFeatureSteering removes a feature by index', () => {
      useServerStore.getState().setFeatureSteering({ index: 100, strength: 2.0 });
      useServerStore.getState().setFeatureSteering({ index: 200, strength: -1.0 });

      useServerStore.getState().removeFeatureSteering(100);

      const { features } = useServerStore.getState().steering;
      expect(features).toHaveLength(1);
      expect(features[0].index).toBe(200);
    });

    it('clearSteering removes all features but preserves other steering state', () => {
      useServerStore.getState().setSteering({
        enabled: true,
        sae_id: 1,
        features: [
          { index: 100, strength: 2.0 },
          { index: 200, strength: -1.0 },
        ],
      });

      useServerStore.getState().clearSteering();

      const { steering } = useServerStore.getState();
      expect(steering.features).toHaveLength(0);
      expect(steering.enabled).toBe(true);
      expect(steering.sae_id).toBe(1);
    });

    it('setSteeringLoading sets the loading flag', () => {
      useServerStore.getState().setSteeringLoading(true);
      expect(useServerStore.getState().steeringLoading).toBe(true);
    });
  });

  describe('monitoring actions', () => {
    it('setMonitoring updates the full monitoring config', () => {
      const config: MonitoringConfig = {
        enabled: true,
        sae_id: 1,
        top_k: 20,
        feature_indices: [100, 200, 300],
      };
      useServerStore.getState().setMonitoring(config);

      expect(useServerStore.getState().monitoring).toEqual(config);
    });

    it('setMonitoringEnabled toggles only the enabled flag', () => {
      useServerStore.getState().setMonitoring({
        enabled: false,
        sae_id: 1,
        top_k: 10,
        feature_indices: [100],
      });

      useServerStore.getState().setMonitoringEnabled(true);

      const { monitoring } = useServerStore.getState();
      expect(monitoring.enabled).toBe(true);
      expect(monitoring.top_k).toBe(10);
      expect(monitoring.feature_indices).toEqual([100]);
    });

    it('addActivationRecord appends a record and caps at 100', () => {
      const record: ActivationRecord = {
        timestamp: '2026-01-30T12:00:00Z',
        request_id: 'req-1',
        activations: [{ feature_index: 100, activation: 0.85 }],
      };

      useServerStore.getState().addActivationRecord(record);

      expect(useServerStore.getState().activationHistory).toHaveLength(1);
      expect(useServerStore.getState().activationHistory[0].request_id).toBe('req-1');
    });

    it('addActivationRecord keeps at most 100 records', () => {
      // Add 105 records
      for (let i = 0; i < 105; i++) {
        useServerStore.getState().addActivationRecord({
          timestamp: `2026-01-30T12:00:${String(i).padStart(2, '0')}Z`,
          request_id: `req-${i}`,
          activations: [],
        });
      }

      const { activationHistory } = useServerStore.getState();
      expect(activationHistory.length).toBeLessThanOrEqual(100);
      // The earliest records should have been dropped
      expect(activationHistory[0].request_id).toBe('req-5');
    });

    it('clearActivationHistory empties the history', () => {
      useServerStore.getState().addActivationRecord({
        timestamp: '2026-01-30T12:00:00Z',
        request_id: 'req-1',
        activations: [],
      });

      useServerStore.getState().clearActivationHistory();

      expect(useServerStore.getState().activationHistory).toHaveLength(0);
    });

    it('setFeatureStatistics sets statistics data', () => {
      const stats: FeatureStatistics[] = [
        { feature_index: 100, min: 0.1, max: 0.9, mean: 0.5, std: 0.2, count: 50 },
      ];
      useServerStore.getState().setFeatureStatistics(stats);

      expect(useServerStore.getState().featureStatistics).toEqual(stats);
    });

    it('setLatestActivations sets the latest activations', () => {
      const activations: FeatureActivation[] = [
        { feature_index: 100, activation: 0.85, label: 'Test' },
      ];
      useServerStore.getState().setLatestActivations(activations);

      expect(useServerStore.getState().latestActivations).toEqual(activations);
    });
  });

  describe('profile actions', () => {
    it('setProfiles sets the profiles list', () => {
      const profiles = [
        createMockProfile({ id: 1, name: 'profile-a' }),
        createMockProfile({ id: 2, name: 'profile-b' }),
      ];
      useServerStore.getState().setProfiles(profiles);

      expect(useServerStore.getState().profiles).toHaveLength(2);
    });

    it('addProfile appends a profile', () => {
      useServerStore.getState().setProfiles([createMockProfile({ id: 1 })]);
      useServerStore.getState().addProfile(createMockProfile({ id: 2, name: 'new-profile' }));

      expect(useServerStore.getState().profiles).toHaveLength(2);
    });

    it('updateProfile updates a specific profile and activeProfile if matching', () => {
      const profile = createMockProfile({ id: 1 });
      useServerStore.getState().setProfiles([profile]);
      useServerStore.getState().setActiveProfile(profile);

      useServerStore.getState().updateProfile(1, { name: 'updated-profile' });

      expect(useServerStore.getState().profiles[0].name).toBe('updated-profile');
      expect(useServerStore.getState().activeProfile?.name).toBe('updated-profile');
    });

    it('removeProfile removes a profile and clears activeProfile if matching', () => {
      const profile = createMockProfile({ id: 1 });
      useServerStore.getState().setProfiles([profile]);
      useServerStore.getState().setActiveProfile(profile);

      useServerStore.getState().removeProfile(1);

      expect(useServerStore.getState().profiles).toHaveLength(0);
      expect(useServerStore.getState().activeProfile).toBeNull();
    });

    it('setActiveProfile sets the active profile', () => {
      const profile = createMockProfile({ id: 1 });
      useServerStore.getState().setActiveProfile(profile);

      expect(useServerStore.getState().activeProfile).toEqual(profile);
    });
  });

  describe('connection actions', () => {
    it('setConnectionStatus updates connection state', () => {
      useServerStore.getState().setConnectionStatus('connected');
      expect(useServerStore.getState().connectionStatus).toBe('connected');

      useServerStore.getState().setConnectionStatus('error');
      expect(useServerStore.getState().connectionStatus).toBe('error');

      useServerStore.getState().setConnectionStatus('connecting');
      expect(useServerStore.getState().connectionStatus).toBe('connecting');

      useServerStore.getState().setConnectionStatus('disconnected');
      expect(useServerStore.getState().connectionStatus).toBe('disconnected');
    });

    it('setServerUrl updates the server URL', () => {
      useServerStore.getState().setServerUrl('http://localhost:9000');
      expect(useServerStore.getState().serverUrl).toBe('http://localhost:9000');
    });
  });

  describe('system metrics actions', () => {
    it('setSystemMetrics updates GPU metrics', () => {
      useServerStore.getState().setSystemMetrics({
        gpuMemoryUsed: 4096,
        gpuMemoryTotal: 24576,
        gpuUtilization: 45,
        gpuTemperature: 52,
      });

      const state = useServerStore.getState();
      expect(state.gpuMemoryUsed).toBe(4096);
      expect(state.gpuMemoryTotal).toBe(24576);
      expect(state.gpuUtilization).toBe(45);
      expect(state.gpuTemperature).toBe(52);
    });

    it('setSystemMetrics partially updates metrics preserving others', () => {
      useServerStore.getState().setSystemMetrics({
        gpuMemoryUsed: 4096,
        gpuMemoryTotal: 24576,
        gpuUtilization: 45,
        gpuTemperature: 52,
      });

      // Only update utilization
      useServerStore.getState().setSystemMetrics({ gpuUtilization: 80 });

      const state = useServerStore.getState();
      expect(state.gpuMemoryUsed).toBe(4096);
      expect(state.gpuMemoryTotal).toBe(24576);
      expect(state.gpuUtilization).toBe(80);
      expect(state.gpuTemperature).toBe(52);
    });
  });

  describe('reset', () => {
    it('resets all state to default values', () => {
      // Set up various state
      useServerStore.getState().setLoadedModel(createMockModel({ status: 'loaded' }));
      useServerStore.getState().setAttachedSAE(createMockSAE({ status: 'attached' }));
      useServerStore.getState().setSteering({
        enabled: true,
        sae_id: 1,
        features: [{ index: 100, strength: 5.0 }],
      });
      useServerStore.getState().setMonitoring({
        enabled: true,
        sae_id: 1,
        top_k: 20,
        feature_indices: [100],
      });
      useServerStore.getState().setConnectionStatus('connected');
      useServerStore.getState().setSystemMetrics({
        gpuMemoryUsed: 4096,
        gpuMemoryTotal: 24576,
        gpuUtilization: 45,
        gpuTemperature: 52,
      });
      useServerStore.getState().setActiveProfile(createMockProfile());

      // Reset
      useServerStore.getState().reset();

      const state = useServerStore.getState();
      expect(state.loadedModel).toBeNull();
      expect(state.attachedSAE).toBeNull();
      expect(state.steering.enabled).toBe(false);
      expect(state.steering.features).toEqual([]);
      expect(state.monitoring.enabled).toBe(false);
      expect(state.connectionStatus).toBe('disconnected');
      expect(state.gpuMemoryUsed).toBe(0);
      expect(state.gpuMemoryTotal).toBe(0);
      expect(state.gpuUtilization).toBe(0);
      expect(state.gpuTemperature).toBe(0);
      expect(state.activeProfile).toBeNull();
      expect(state.models).toEqual([]);
      expect(state.saes).toEqual([]);
      expect(state.profiles).toEqual([]);
    });
  });
});
