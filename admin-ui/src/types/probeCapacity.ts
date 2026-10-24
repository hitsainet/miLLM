/** Per-model probe limits and measured capacity (operator 2026-10-21). */

export interface CostLine {
  decode_own_ms: number;
  decode_fixed_ms: number;
  prefill_own_ms_per_1k: number;
  prefill_fixed_ms_per_1k: number;
  k?: number;
  layer?: number;
}

export interface CapacityMeasurement {
  id: number;
  repo_id: string;
  quantization: string;
  /** `cuda:GPU-…` by UUID; several joined by `+` for a model split across cards. */
  device: string;
  decode_ms_per_token: number | null;
  prefill_ms_per_1k: number | null;
  dense: CostLine | null;
  sparse: CostLine | null;
  method: string;
  source: string | null;
  measured_at: string;
}

export interface BudgetSetting {
  value: number;
  unit: 'ms' | 'pct';
}

export interface ConfiguredLimits {
  max_armed: number | null;
  decode_budget: BudgetSetting | null;
  prefill_budget: BudgetSetting | null;
  reason: string | null;
  updated_at?: string;
  /** Feature 39: the count limit in BEHAVIOURS, and who set what. */
  max_behaviours?: number | null;
  source?: 'recommended-approved' | 'operator' | null;
  set_by?: string | null;
  approved_at?: string | null;
}

export interface LimitsApproval {
  source: string;
  set_by: string | null;
  approved_at: string | null;
  measurement_id: number | null;
  newer_recommendation_available: boolean;
}

export interface EffectiveLimits {
  max_armed: number;
  decode_ms_per_pass: number;
  prefill_ms_per_1k: number;
  headroom: number;
  sources: Record<string, string>;
  /** Feature 39 (D13): behaviours (probe sets) armed at once. */
  max_behaviours?: number;
  /** The approval was measured under another model, precision or card; it stays in force. */
  stale?: boolean;
  approval?: LimitsApproval | null;
}

/** Feature 39: a model's explained recommendation (`GET …/probe-limits/recommendation`). */
export interface ProbeLimitRecommendation {
  model_id: number;
  values: { max_behaviours: number; decode_budget_ms: number; prefill_budget_ms: number } | null;
  shares?: { decode_pct: number | null; prefill_pct: number | null };
  reasons: string[];
  estimated?: boolean;
  achieved_sets?: number;
  target_sets?: number;
  measurement?: { id: number; measured_at: string; devices: string[] };
  recommendation_digest: string | null;
}

export interface ImpliedHalf {
  fits_budget: number | null;
  fits_with_headroom: number | null;
}

export interface ImpliedKind {
  decode: ImpliedHalf;
  prefill: ImpliedHalf;
  binding?: 'decode' | 'prefill';
  n?: number;
}

export interface ProbeCapacity {
  measured: CapacityMeasurement | null;
  configured: ConfiguredLimits;
  effective: EffectiveLimits;
  implied: { dense?: ImpliedKind; sparse?: ImpliedKind };
  armed: {
    count: number;
    projected: { decode: number; prefill: number };
    share: { count: number | null; decode: number; prefill: number };
  } | null;
}

export interface ProbeLimitsBody {
  max_armed?: number | null;
  max_behaviours?: number | null;
  decode_budget?: BudgetSetting | null;
  prefill_budget?: BudgetSetting | null;
  /** `recommended-approved` must carry the recommendation's digest; the server writes ITS values. */
  source?: 'recommended-approved' | 'operator';
  recommendation_digest?: string;
  set_by?: string;
  reason?: string | null;
}
