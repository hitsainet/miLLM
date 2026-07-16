/**
 * Cluster import types (Feature 8) — mirrors millm/api/schemas/cluster.py.
 */

export interface ClusterMember {
  feature_idx: number;
  label?: string | null;
  similarity?: number | null;
  activation_frequency?: number | null;
  max_activation?: number | null;
  strength: number;
  sign?: 1 | -1;
  pinned?: boolean;
}

export interface ClusterBudget {
  B?: number | null;
  B_dir?: number | null;
  G?: number | null;
  f_eff?: number | null;
  formula_id?: string | null;
  constants?: Record<string, number> | null;
  intensity?: number;
  intensity_range?: number[];
}

/** The portable artifact (frozen v1 contract). */
export interface ClusterDefinitionV1 {
  kind: 'mistudio.cluster-definition';
  schema_version: '1';
  name: string;
  narrative?: string | null;
  display_token?: string | null;
  model?: { hf_id?: string | null; mistudio_model_id?: string | null };
  sae?: {
    mistudio_sae_id?: string | null;
    layer?: number | null;
    hook_type?: string | null;
    n_features?: number | null;
    d_model?: number | null;
    source_hint?: string | null;
  };
  members: ClusterMember[];
  budget?: ClusterBudget | null;
  provenance?: Record<string, unknown>;
}

export interface ClusterSummary {
  id: string;
  name: string;
  description?: string | null;
  model_id?: string | null;
  sae_id?: string | null;
  layer?: number | null;
  is_active: boolean;
  intensity: number;
  sensing_enabled: boolean;
  member_count: number;
  display_token?: string | null;
  bound: boolean;
  warnings: string[];
  hub_ref?: { repo_id: string; revision: string; path: string } | null;
  created_at: string;
  updated_at: string;
}

export interface ClusterListResponse {
  clusters: ClusterSummary[];
  active_cluster_id: string | null;
}

export interface ClusterImportItem {
  name: string;
  status: 'imported' | 'imported_unbound' | 'blocked' | 'error';
  profile_id?: string | null;
  warnings: string[];
  error?: string | null;
}

export interface ClusterImportResult {
  results: ClusterImportItem[];
  imported: number;
  blocked: number;
  errors: number;
}

export interface HubRepoInfo {
  repo_id: string;
  likes: number;
  downloads: number;
  last_modified?: string | null;
  tags: string[];
}

export interface HubDefinitionRef {
  file: string;
  name?: string | null;
  member_count?: number | null;
  base_model?: string | null;
}
