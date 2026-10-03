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
  /** ⚠ THE PER-TOKEN TOLERANCE, WHICH DECIDES NOTHING. Pairs with `max_abs_diff` below, and both
   *  are informational — see `per_token_is_informational`. The badge rendered exactly this pair
   *  and printed "parity passed · max Δ 9.37e+0 of 0.05", a number 187x its stated limit beside
   *  the word passed. */
  tolerance: number;
  /** `null` when no vector could be compared at all. Informational: a single token can drift far
   *  while the aggregate a probe actually scores on does not. */
  max_abs_diff: number | null;
  /** ⚠ THE PAIR THAT DECIDES `passed`: the aggregate score difference against its own tolerance.
   *  Optional because a backend or a stored report from before these existed omits them. */
  score_tolerance?: number;
  max_gated_diff?: number | null;
  max_combined_diff?: number | null;
  per_token_is_informational?: boolean;
  at_risk_tokens?: number;
  scored_tokens?: number;
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
  /**
   * WHICH CUT OF THE BAR THE STORED ROW IS AT. 1 is the one the producer's training run placed.
   *
   * ⚠ Optional on the wire: a backend from before 2026-10-02 omits it. An event carrying a
   * DIFFERENT revision was judged against a bar that has since moved — which is the only way a
   * reader can tell two verdicts apart once the two cuts land on similar numbers.
   */
  threshold_revision?: number;
  rung: number;
  /** Server-rendered. Never derived from `rung` here. */
  rung_language: string;
  next_step: string;
  /**
   * miStudio's own sentence for what positive means (`"positive = high-stakes"`), carried
   * through from the definition. Optional on the wire: a backend predating 2026-10-01 omits it.
   */
  concept?: string | null;
  /**
   * `{raw_label: "positive" | "negative" | "excluded"}` from the training view, verbatim.
   * Sent RAW so the client names both sides; the server does not format it.
   */
  label_mapping?: Record<string, string> | null;
  /** The precision the probe was fitted at, as its definition STATES it. Absent/null = the
   *  definition predates the field (miStudio 2026-10-03) — not assumed to be float16. */
  load_dtype?: string | null;
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
  /**
   * Which slice of the request this verdict read. One probe reports several — the prompt says
   * something about the USER, the response about the MODEL — and without this the rows are
   * indistinguishable, since they share a probe and a request.
   *
   * ⚠ OPTIONAL ON THE WIRE. A backend from before 2026-09-30 does not send it, and during a
   * rolling deploy this bundle meets one. A strict check on an absent field has already taken
   * this page down once this week.
   */
  window?: string;
  /**
   * Either the window has no threshold of its own and is not the probe's scope, or the probe's
   * weights were never fitted on what it reads (`response`, unless the probe's scope is
   * `response`). A window's own bar retires only the first reason.
   */
  provisional?: boolean;
  /**
   * WHICH CUT OF THE BAR JUDGED THIS VERDICT. 1 is the bar the producer's training run placed.
   *
   * ⚠ A THRESHOLD CAN NOW MOVE UNDER AN ARMED PROBE, so two events are distinguishable by
   * `threshold` alone only while the two cuts landed on different numbers — and a reader who
   * watches `provisional` disappear between two events cannot otherwise tell whether a window
   * gained its own bar or never needed one. Optional on the wire, like `window`: a bundle meets
   * an older backend during a rolling deploy.
   */
  threshold_revision?: number;
  created_at: string;
  /**
   * ⚠ ONLY EVER PRESENT ON THE DETAIL ROUTE. The list route and the socket omit it: it is the
   * decoded window around a firing position, i.e. user content.
   */
  context_text?: string | null;
  context_token_ids?: number[] | null;
}

export interface ProbeStatusEntry {
  /**
   * Which windows this probe is actually reporting, from the LIVE registry — not what the
   * picker currently shows, which is an intention that resets on re-render.
   *
   * ⚠ Optional on the wire: a backend from before 2026-10-01 omits it. `null`/absent means no
   * live ArmedProbe, i.e. it is scoring nothing.
   */
  windows?: string[] | null;
  /**
   * THE BAR ACTUALLY IN FORCE, from the LIVE registry — and the row's, beside it.
   *
   * ⚠ THE ONLY SURFACE THAT CAN SEE A FAILED REGISTRY REFRESH. `GET /api/probes` serialises the
   * ROW, so if a threshold re-cut wrote the database and did not replace the in-memory
   * `ArmedProbe`, the tile would show the new number while every verdict kept using the old
   * one — invisible-but-visible. `threshold_disagreement` is the server's sentence for that, and
   * it is deliberately NOT folded into `paused_reason`: a probe judging against a previous bar
   * is still scoring, and conflating the two would dilute the field whose whole job is "a probe
   * never goes silently quiet".
   *
   * `threshold` is `null` when nothing is armed here. Not scoring is not a bar of zero.
   */
  threshold?: number | null;
  threshold_revision?: number | null;
  row_threshold?: number | null;
  row_threshold_revision?: number;
  /** Which bar the operator armed against, so consent can be read against the bar in force. */
  armed_threshold_revision?: number | null;
  threshold_disagreement?: string | null;
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
