/**
 * `mi.probe.result/v1` (038 FR-38.2) and the `telemetry` block of `GET /api/health/detailed`
 * (FR-38.13), written by hand from `docs/schemas/probe-result-v1.json` and the health route's
 * model. The fixtures under `src/test/fixtures/probe-result/` are generated and validated against
 * the schema (`scripts/gen-probe-result-fixtures.mjs`), and `probeResultFixtures.test.ts` checks
 * they still type as `ProbeResultV1`.
 *
 * ⚠ OPEN VALUES. The published schema admits a value added to a list later in v1, so every listed
 * value is typed `Known | string`: a consumer reads an unknown one as "other", never crashes.
 */

type Open<T extends string> = T | (string & {});

export type ProbeOutcome = Open<'completed' | 'skipped' | 'failed' | 'interrupted'>;
export type ProbeResultStatus = Open<'final' | 'partial' | 'unavailable'>;
export type ProbePersistence = Open<'persisted' | 'not_persisted' | 'unknown'>;
export type DigestSource = Open<'producer' | 'computed' | 'unknown'>;

export interface ProbeResultV1 {
  schema: 'mi.probe.result/v1';
  kind: 'probe';
  result_id: string;
  event_time: string;
  event_time_source: Open<'finalized' | 'inserted'>;
  completed_at: string;
  origin: Open<'live' | 'batch' | 'stateless'>;
  producer: { name: string; version: string };
  migrated: boolean;
  request_id: string | null;
  trace_id: string | null;
  span_id: string | null;
  /** Recorded copies only. */
  source_event_id?: number | null;
  batch_id: string | null;
  batch_line: number | null;
  session_id: string | null;
  turn_index: number | null;
  behaviour: { key: string; version: number | null } | null;
  set: unknown | null;
  outcome: ProbeOutcome;
  /** A code from the published list; `null` when completed. */
  reason: string | null;
  /** Recorded copies only. */
  persistence?: ProbePersistence | null;
  window: Open<'all' | 'prompt' | 'response' | 'last_user'>;
  status: ProbeResultStatus;
  choice_index: number | null;
  input_index: number | null;
  probe_id: string;
  probe_name: string | null;
  /** 64 hex characters, or `unknown`. */
  detector_digest: string;
  detector_digest_source: DigestSource;
  identity_algorithm: string | null;
  definition_schema: string;
  target: string;
  score_semantics: string;
  layer: number | null;
  hook: string;
  basis: string;
  rule: string;
  served_model: string;
  model_revision: string;
  template_sha256: string;
  precision: string;
  engine: Open<'transformers' | 'llamacpp' | 'cbm'>;
  execution_mode: Open<'serial' | 'batched' | 'continuous_batching' | 'speculative'>;
  steering_ref: { kind: string; id: string | null };
  definition_scope: string;
  n_scored_tokens: number;
  scored: boolean;
  score: number | null;
  probability: null;
  calibration_ref: null;
  threshold: number | null;
  /** ⚠ `null` is not `false`: no threshold, or not completed. */
  verdict: boolean | null;
  fires: boolean | null;
  comparator: string;
  decision_digest: string;
  /** miStudio's revision of the decision; `null` when the producer stated none. */
  decision_revision: number | null;
  /** miLLM's own counter of the bar that judged this result. */
  threshold_revision: number;
  decision: { calibration_id: string | null; target_fpr: number | null; provisional: boolean };
  rung: number;
  rung_language: string;
  parity_status: string;
  model_load_id: string;
  evaluation_ms: number | null;
}

export type TelemetrySignalState = Open<'disabled' | 'healthy' | 'degraded' | 'unavailable'>;

export interface TelemetrySignalHealth {
  state: TelemetrySignalState;
  exported: number;
  failed: number;
  dropped: number;
  last_success_at: string | null;
  /** The error's CLASS (or `HTTP401`) — never its message. */
  last_error: { type: string; at: string | null } | null;
  queue: { used: number; capacity: number };
}

export interface TelemetryConfigView {
  endpoint: string | null;
  protocol: string;
  transport: string | null;
  headers_set: boolean;
  tls_ca_file_set: boolean;
  tls_client_cert_set: boolean;
  tls_client_key_set: boolean;
  service_name: string | null;
  service_instance_id: string | null;
  deployment_environment: string | null;
  sampler: { kind: string; ratio: number } | null;
  series_cap: number | null;
  queue_size: number | null;
  batch_size: number | null;
  export_timeout_s: number | null;
  retry_horizon_s: number | null;
  drain_deadline_s: number | null;
  metric_interval_s: number | null;
  stateless_export_permitted: boolean;
  result_log_filter: { origins: string[] } | null;
  sdk_disabled: boolean;
  sdk_version: string | null;
  semconv_schema_url: string | null;
}

/** `GET /api/health/detailed` → `.telemetry`. `error` (a class name) only when the snapshot
 *  itself failed; then the other fields are absent. */
export interface TelemetryHealth {
  enabled?: boolean | null;
  config?: TelemetryConfigView | null;
  signals?: Record<string, TelemetrySignalHealth> | null;
  series_overflow?: number | null;
  attributes_dropped?: number | null;
  deletion_note?: string | null;
  error?: string | null;
}
