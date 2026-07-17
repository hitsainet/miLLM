/**
 * Sensing types (Feature 11: cluster co-activation sensing).
 */

export interface SensingEvent {
  id: number;
  profile_id: string;
  request_id: string;
  phase: 'prefill' | 'decode';
  pos_start: number;
  pos_end: number;
  /** [feature_idx, peak_activation] pairs */
  fired_members: [number, number][];
  fired_count: number;
  score: number;
  ambient_fired_count: number | null;
  /** Present on detail fetches only — WS payloads exclude context */
  context_text?: string | null;
  context_token_ids?: number[] | null;
  /** Separately decoded segments — span = the fired position(s) */
  context_parts?: { before: string; span: string; after: string } | null;
  summary: string;
  truncated: boolean;
  created_at: string | null;
}

export interface SensingStatus {
  armed: boolean;
  profile_id: string | null;
  profile_name: string | null;
  member_count: number;
  /** Members with usable thresholds (only these can fire) */
  sensable_count?: number;
  min_k: number | null;
  threshold_mode: 'epsilon_max' | 'floor_only' | null;
  context_tokens: number | null;
  last_request_overhead_ms: number;
  overhead_warn_threshold_ms: number;
  events_recorded_since_start: number;
  ws_events_dropped: number;
  /** Persistent per-cluster intent (distinct from `armed`) */
  enabled_clusters: { id: string; name: string; is_active: boolean }[];
  retention: {
    max_events_per_cluster?: number;
    max_age_days?: number;
  };
}

export interface SensingEventList {
  events: SensingEvent[];
  total: number;
}

export interface SensingToggleResult {
  profile_id: string;
  sensing_enabled: boolean;
  armed: boolean;
}

export interface SensingConfigResult {
  profile_id: string;
  min_k: number | null;
  effective_min_k: number | null;
  armed: boolean;
}
