/**
 * Circuit types (Feature 13) — mirrors millm/api/schemas/circuit.py.
 *
 * NOTE on evidence language: `rung_language` / `rung_next_step` are ALWAYS
 * server-rendered from the evidence ladder. The UI renders them verbatim and
 * must never derive or re-phrase an evidence claim client-side — a rung below
 * 2 is not causally validated and must never be described as such.
 */

export type PerSAEVerdictKind = 'bind' | 'warn' | 'block' | 'unbound';

export interface PerSAEVerdict {
  layer: number;
  sae_id?: string | null;
  verdict: PerSAEVerdictKind;
  reason?: string | null;
}

export type ServingMode = 'full' | 'slice_fallback' | null;

export interface CircuitSummary {
  id: string;
  name: string;
  description?: string | null;
  /** 0 mined · 1 attribution-supported · 2 causally validated · 3 faithfulness-tested */
  rung: number;
  /** Server-rendered evidence phrase — render verbatim. */
  rung_language: string;
  rung_next_step: string;
  /** rung >= 2. Below this, activation requires an explicit acknowledgement. */
  validated: boolean;
  edge_count: number;
  layers: number[];
  serveable: boolean;
  is_active: boolean;
  serving_mode: ServingMode;
  intensity: number;
  per_sae_warnings: PerSAEVerdict[];
  created_at?: string;
  updated_at?: string;
}

export interface CircuitListResponse {
  circuits: CircuitSummary[];
  active_circuit_id: string | null;
  total: number;
}

export interface CircuitActivationResponse extends CircuitSummary {
  bound_layers: number[];
  slice_layer?: number | null;
  applied_per_layer?: Record<string, Record<string, number>> | null;
  hazards: Array<Record<string, unknown>>;
  warnings: string[];
  acknowledged_unvalidated: boolean;
}

/** The portable artifact (frozen v1 contract) — imported, never produced here. */
export interface CircuitDefinitionV1 {
  kind: 'mistudio.circuit-definition';
  schema_version: '1';
  name: string;
  narrative?: string | null;
  saes: Array<Record<string, unknown>>;
  members: Array<Record<string, unknown>>;
  edges?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}
