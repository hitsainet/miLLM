/**
 * Circuit edge sensing types (Feature 15) — mirrors
 * millm/api/schemas/circuit_sensing.py.
 *
 * NOTE on evidence language: `edge_rung_language` is ALWAYS the server-rendered
 * ladder phrase carried from the moment of observation. The UI renders it
 * verbatim and must never derive, map, or re-phrase an evidence claim — an edge
 * below rung 2 is not validated and must never be described as though it were.
 */

/** One end of an observed edge firing. */
export interface EdgeEndpoint {
  layer: number;
  feature_idx: number;
  /** Absolute token position within the request */
  pos: number;
  act: number;
}

/**
 * An edge that could not be watched, and why. Surfaced prominently so a user
 * never reads "no events" as "the edge never fired" — absence of observation
 * is not evidence of absence.
 */
export interface UnsensableEdgeInfo {
  edge_key: string;
  /** layer_not_attached | no_activation_threshold | endpoint_not_a_feature */
  reason: string;
  detail: string;
}

/** Runtime state, reconciled against the SAEs actually armed. */
export interface CircuitSensingStatus {
  armed: boolean;
  circuit_id: string | null;
  circuit_name: string | null;
  layers: number[];
  sensable_edges: number;
  unsensable_edges: UnsensableEdgeInfo[];
  max_token_lag: number;
  last_request_overhead_ms: number;
  events_recorded: number;
  ws_dropped: number;
  /** Persistent operator intent, distinct from runtime `armed`. */
  enabled_circuits: { id: string; name: string; is_active: boolean }[];
}

/** One observed up→down firing. */
export interface CircuitSensingEvent {
  id: number;
  circuit_id: string;
  request_id: string;
  phase: string;
  edge_key: string;
  up: EdgeEndpoint;
  down: EdgeEndpoint;
  token_lag: number;
  edge_rung: number;
  /** Server-rendered evidence phrase — render verbatim, never re-phrase. */
  edge_rung_language: string;
  edge_type?: string | null;
  ambient_fired_count?: number | null;
  summary: string;
  truncated: boolean;
  created_at?: string | null;
  /** Present on detail fetches only — WS payloads exclude context. */
  context_text?: string | null;
  context_token_ids?: number[] | null;
  /** Separately decoded segments — span = the fired position(s). */
  context_parts?: { before: string; span: string; after: string } | null;
}

export interface CircuitSensingEventList {
  total: number;
  events: CircuitSensingEvent[];
}

export interface CircuitSensingToggleResult {
  circuit_id: string;
  enabled: boolean;
  armed: boolean;
  unsensable_edges: UnsensableEdgeInfo[];
  message: string;
}
