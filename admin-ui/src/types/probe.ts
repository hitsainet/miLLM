/**
 * Probe monitor types (Feature 24).
 *
 * ⚠ THE RUNG'S WORDING COMES FROM THE SERVER. `rung_language` and `next_step` are fields, not
 * something this client composes from `rung`. miStudio owns the vocabulary and miLLM mirrors it
 * verbatim; a number→phrase map here would be a third copy, free to drift — and the thing most
 * likely to drift is a detector's language rising above its evidence.
 */

/** `all | prompt | response` — the CONTRACT's vocabulary, not miStudio's internal role scopes. */
export type ProbeScope = 'all' | 'prompt' | 'response';
export type ProbeBasis = 'residual' | 'sae_features';
export type ProbeRule =
  | 'mean'
  | 'max'
  | 'last'
  | 'softmax'
  | 'attention'
  | 'rolling_mean_max';

export interface ProbeParityReport {
  passed: boolean;
  tolerance: number;
  /** `null` when no vector could be compared at all. */
  max_abs_diff: number | null;
  vector_index: number | null;
  vectors: Array<{
    index: number;
    max_abs_diff: number | null;
    combined_diff: number | null;
    comparable: boolean;
    reason: string | null;
  }>;
  /** Informational only — never contributes to `passed`. See the parity engine. */
  tokenization_drift: Record<string, unknown>;
  error: string | null;
}

export interface Probe {
  id: string;
  name: string;
  hf_id: string;
  layer: number;
  rule: ProbeRule;
  scope: ProbeScope;
  basis: ProbeBasis;
  streamable: boolean;
  /** ⚠ `null` means NO threshold was placed — the probe ranks but does not decide. Not zero. */
  threshold: number | null;
  target_fpr: number | null;
  rung: number;
  /** Server-rendered. Never derived from `rung` here. */
  rung_language: string;
  next_step: string;
  armed: boolean;
  /** Why an armed probe is not scoring. Displayed whenever present. */
  paused_reason: string | null;
  parity: ProbeParityReport | null;
  created_at: string;
  /** Only on the detail route. */
  definition?: Record<string, unknown>;
}

export interface ProbeEvent {
  id: number;
  probe_id: string;
  /** The `/v1` completion id — the only link between a response and its verdict. */
  request_id: string | null;
  scored: boolean;
  /** Present whenever `scored` is false. A probe never goes silently quiet. */
  not_scored_reason: string | null;
  score: number | null;
  threshold: number | null;
  /** ⚠ `null` is not `false`: either unscored, or no threshold was placed. */
  verdict: boolean | null;
  /** The rung AS OF THIS OBSERVATION, not the probe's rung today. */
  rung: number | null;
  top_positions: number[] | null;
  n_scored_tokens: number | null;
  summary: string | null;
  created_at: string;
  /**
   * ⚠ ONLY EVER PRESENT ON THE DETAIL ROUTE. The list route and the socket omit it: it is the
   * decoded window around a firing position, i.e. user content.
   */
  context_text?: string | null;
  context_token_ids?: number[] | null;
}

export interface ProbeStatusEntry {
  id: string;
  name: string;
  layer: number;
  rule: ProbeRule;
  rung: number;
  rung_language: string;
  next_step: string;
  streamable: boolean;
  basis: ProbeBasis;
  paused_reason: string | null;
}

export interface ProbeStatus {
  armed: ProbeStatusEntry[];
  armed_count: number;
  max_armed: number;
  imported_count: number;
  paused_reasons: string[];
  /** `null` when no request has been scored yet — distinguishable from "field missing". */
  last_request_overhead_ms: number | null;
  /** Absolute per-request backstop (pathology only). The RATE below is what decides the warning. */
  overhead_warn_threshold_ms: number;
  /** Forward passes that did probe work: one prefill plus one per generated token.
   *  ⚠ OPTIONAL ON THE WIRE. A backend from before 2026-09-30 does not send these three, and
   *  during a rolling deploy this bundle can meet one. `!== null` is TRUE for `undefined`, so a
   *  strict check here crashed the whole status tile on `.toFixed`. */
  last_request_n_passes?: number;
  /** null when nothing has been scored — distinct from a rate of 0. */
  last_overhead_ms_per_pass?: number | null;
  overhead_warn_threshold_ms_per_pass?: number;
  events_recorded: number;
  socket_events_dropped: number;
  /** While true, continuous batching is off whenever anything is armed. */
  force_serial: boolean;
}

/** The rung at or above which arming needs no explicit acknowledgement. */
export const ARM_WITHOUT_ACK_MIN_RUNG = 2;

/** A Hub repo tagged `mistudio-probe-definition`. */
export interface ProbeHubRepo {
  repo_id: string;
  author?: string | null;
  downloads?: number | null;
  likes?: number | null;
  last_modified?: string | null;
  tags?: string[];
}

/** One `.probe.json` inside a Hub repo. */
export interface ProbeHubDefinition {
  filename: string;
  name?: string | null;
  hf_id?: string | null;
  layer?: number | null;
  rung?: number | null;
  size_bytes?: number | null;
}
