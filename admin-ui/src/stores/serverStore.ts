import { create } from 'zustand';
import type {
  ModelInfo,
  SAEInfo,
  SteeringState,
  MonitoringConfig,
  Profile,
  FeatureSteering,
  ActivationRecord,
  FeatureStatistics,
} from '@/types';
import type { ConnectionStatus } from '@/types/ui';

interface ServerState {
  // Connection
  connectionStatus: ConnectionStatus;

  // Model state
  models: ModelInfo[];
  loadedModel: ModelInfo | null;
  modelLoading: boolean;
  downloadProgress: Record<number, number>;

  // SAE state
  saes: SAEInfo[];
  attachedSAE: SAEInfo | null;
  saeLoading: boolean;
  saeDownloadProgress: Record<number, number>;

  // Steering state
  steering: SteeringState;
  steeringLoading: boolean;

  // Monitoring state
  monitoring: MonitoringConfig;
  activationHistory: ActivationRecord[];
  featureStatistics: FeatureStatistics[];
  monitoringLoading: boolean;

  // Profiles
  profiles: Profile[];
  activeProfile: Profile | null;
  profilesLoading: boolean;

  // System metrics
  gpuMemoryUsed: number;
  gpuMemoryTotal: number;
  gpuUtilization: number;
  gpuTemperature: number;
}

interface ServerActions {
  // Connection actions
  setConnectionStatus: (status: ConnectionStatus) => void;

  // Model actions
  setModels: (models: ModelInfo[]) => void;
  addModel: (model: ModelInfo) => void;
  updateModel: (id: number, updates: Partial<ModelInfo>) => void;
  removeModel: (id: number) => void;
  setLoadedModel: (model: ModelInfo | null) => void;
  setModelLoading: (loading: boolean) => void;
  setDownloadProgress: (modelId: number, progress: number) => void;
  clearDownloadProgress: (modelId: number) => void;

  // SAE actions
  setSAEs: (saes: SAEInfo[]) => void;
  addSAE: (sae: SAEInfo) => void;
  updateSAE: (id: number, updates: Partial<SAEInfo>) => void;
  removeSAE: (id: number) => void;
  setAttachedSAE: (sae: SAEInfo | null) => void;
  setSAELoading: (loading: boolean) => void;
  setSAEDownloadProgress: (saeId: number, progress: number) => void;
  clearSAEDownloadProgress: (saeId: number) => void;

  // Steering actions
  setSteering: (steering: SteeringState) => void;
  setSteeringEnabled: (enabled: boolean) => void;
  setFeatureSteering: (feature: FeatureSteering) => void;
  removeFeatureSteering: (index: number) => void;
  clearSteering: () => void;
  setSteeringLoading: (loading: boolean) => void;

  // Monitoring actions
  setMonitoring: (config: MonitoringConfig) => void;
  setMonitoringEnabled: (enabled: boolean) => void;
  addActivationRecord: (record: ActivationRecord) => void;
  setActivationHistory: (records: ActivationRecord[]) => void;
  clearActivationHistory: () => void;
  setFeatureStatistics: (stats: FeatureStatistics[]) => void;
  setMonitoringLoading: (loading: boolean) => void;

  // Profile actions
  setProfiles: (profiles: Profile[]) => void;
  addProfile: (profile: Profile) => void;
  updateProfile: (id: number, updates: Partial<Profile>) => void;
  removeProfile: (id: number) => void;
  setActiveProfile: (profile: Profile | null) => void;
  setProfilesLoading: (loading: boolean) => void;

  // System metrics actions
  setSystemMetrics: (metrics: {
    gpuMemoryUsed?: number;
    gpuMemoryTotal?: number;
    gpuUtilization?: number;
    gpuTemperature?: number;
  }) => void;

  // Reset
  reset: () => void;
}

const initialState: ServerState = {
  connectionStatus: 'disconnected',
  models: [],
  loadedModel: null,
  modelLoading: false,
  downloadProgress: {},
  saes: [],
  attachedSAE: null,
  saeLoading: false,
  saeDownloadProgress: {},
  steering: {
    enabled: false,
    sae_id: null,
    features: [],
  },
  steeringLoading: false,
  monitoring: {
    enabled: false,
    sae_id: null,
    top_k: 10,
    feature_indices: null,
  },
  activationHistory: [],
  featureStatistics: [],
  monitoringLoading: false,
  profiles: [],
  activeProfile: null,
  profilesLoading: false,
  gpuMemoryUsed: 0,
  gpuMemoryTotal: 0,
  gpuUtilization: 0,
  gpuTemperature: 0,
};

export const useServerStore = create<ServerState & ServerActions>((set) => ({
  ...initialState,

  // Connection actions
  setConnectionStatus: (status) => set({ connectionStatus: status }),

  // Model actions
  setModels: (models) => set({ models }),
  addModel: (model) => set((state) => ({ models: [...state.models, model] })),
  updateModel: (id, updates) =>
    set((state) => ({
      models: state.models.map((m) => (m.id === id ? { ...m, ...updates } : m)),
      loadedModel:
        state.loadedModel?.id === id
          ? { ...state.loadedModel, ...updates }
          : state.loadedModel,
    })),
  removeModel: (id) =>
    set((state) => ({
      models: state.models.filter((m) => m.id !== id),
      loadedModel: state.loadedModel?.id === id ? null : state.loadedModel,
    })),
  setLoadedModel: (model) => set({ loadedModel: model }),
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setDownloadProgress: (modelId, progress) =>
    set((state) => ({
      downloadProgress: { ...state.downloadProgress, [modelId]: progress },
    })),
  clearDownloadProgress: (modelId) =>
    set((state) => {
      const { [modelId]: _, ...rest } = state.downloadProgress;
      return { downloadProgress: rest };
    }),

  // SAE actions
  setSAEs: (saes) => set({ saes }),
  addSAE: (sae) => set((state) => ({ saes: [...state.saes, sae] })),
  updateSAE: (id, updates) =>
    set((state) => ({
      saes: state.saes.map((s) => (s.id === id ? { ...s, ...updates } : s)),
      attachedSAE:
        state.attachedSAE?.id === id
          ? { ...state.attachedSAE, ...updates }
          : state.attachedSAE,
    })),
  removeSAE: (id) =>
    set((state) => ({
      saes: state.saes.filter((s) => s.id !== id),
      attachedSAE: state.attachedSAE?.id === id ? null : state.attachedSAE,
    })),
  setAttachedSAE: (sae) => set({ attachedSAE: sae }),
  setSAELoading: (loading) => set({ saeLoading: loading }),
  setSAEDownloadProgress: (saeId, progress) =>
    set((state) => ({
      saeDownloadProgress: { ...state.saeDownloadProgress, [saeId]: progress },
    })),
  clearSAEDownloadProgress: (saeId) =>
    set((state) => {
      const { [saeId]: _, ...rest } = state.saeDownloadProgress;
      return { saeDownloadProgress: rest };
    }),

  // Steering actions
  setSteering: (steering) => set({ steering }),
  setSteeringEnabled: (enabled) =>
    set((state) => ({
      steering: { ...state.steering, enabled },
    })),
  setFeatureSteering: (feature) =>
    set((state) => {
      const features = state.steering.features.filter(
        (f) => f.index !== feature.index
      );
      return {
        steering: {
          ...state.steering,
          features: [...features, feature],
        },
      };
    }),
  removeFeatureSteering: (index) =>
    set((state) => ({
      steering: {
        ...state.steering,
        features: state.steering.features.filter((f) => f.index !== index),
      },
    })),
  clearSteering: () =>
    set((state) => ({
      steering: { ...state.steering, features: [] },
    })),
  setSteeringLoading: (loading) => set({ steeringLoading: loading }),

  // Monitoring actions
  setMonitoring: (config) => set({ monitoring: config }),
  setMonitoringEnabled: (enabled) =>
    set((state) => ({
      monitoring: { ...state.monitoring, enabled },
    })),
  addActivationRecord: (record) =>
    set((state) => ({
      activationHistory: [...state.activationHistory.slice(-99), record],
    })),
  setActivationHistory: (records) => set({ activationHistory: records }),
  clearActivationHistory: () => set({ activationHistory: [] }),
  setFeatureStatistics: (stats) => set({ featureStatistics: stats }),
  setMonitoringLoading: (loading) => set({ monitoringLoading: loading }),

  // Profile actions
  setProfiles: (profiles) => set({ profiles }),
  addProfile: (profile) =>
    set((state) => ({ profiles: [...state.profiles, profile] })),
  updateProfile: (id, updates) =>
    set((state) => ({
      profiles: state.profiles.map((p) =>
        p.id === id ? { ...p, ...updates } : p
      ),
      activeProfile:
        state.activeProfile?.id === id
          ? { ...state.activeProfile, ...updates }
          : state.activeProfile,
    })),
  removeProfile: (id) =>
    set((state) => ({
      profiles: state.profiles.filter((p) => p.id !== id),
      activeProfile: state.activeProfile?.id === id ? null : state.activeProfile,
    })),
  setActiveProfile: (profile) => set({ activeProfile: profile }),
  setProfilesLoading: (loading) => set({ profilesLoading: loading }),

  // System metrics actions
  setSystemMetrics: (metrics) =>
    set((state) => ({
      gpuMemoryUsed: metrics.gpuMemoryUsed ?? state.gpuMemoryUsed,
      gpuMemoryTotal: metrics.gpuMemoryTotal ?? state.gpuMemoryTotal,
      gpuUtilization: metrics.gpuUtilization ?? state.gpuUtilization,
      gpuTemperature: metrics.gpuTemperature ?? state.gpuTemperature,
    })),

  // Reset
  reset: () => set(initialState),
}));
