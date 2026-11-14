/**
 * Probe sessions (Feature 36): one conversation, as the armed probes saw it.
 *
 * ⚠ NO TEXT ON THESE SHAPES. A session and its timeline say WHETHER a turn has recorded text
 * (`has_transcript`); the text itself comes only from `ProbeTranscriptTurn`, fetched on demand from
 * the one route that serves it.
 */

export interface ProbeBehaviourRef {
  key: string;
  name: string | null;
  version: number | string | null;
}

export interface SessionBehaviourSummary {
  n_fired: number;
  n_scored: number;
  max_score: number | null;
}

export interface ProbeSessionRow {
  id: string;
  client_session_id: string | null;
  source: 'header' | 'fingerprint' | 'single';
  predecessor_id: string | null;
  link_ambiguous: boolean;
  served_model: string | null;
  opened_at: string;
  last_seen_at: string;
  open: boolean;
  n_turns: number;
  n_transcribed: number;
  /** Keyed by the timeline column: the behaviour key, else the probe's own name. */
  behaviours: Record<string, SessionBehaviourSummary>;
}

export interface SessionVerdictCell {
  column: string;
  /** 039 S2: `rule` for a rule signal's entry (no score; `count` and `fires`). */
  kind?: 'probe' | 'rule';
  rule_result_id?: number;
  rule_set_id?: string;
  signal?: string;
  form?: string;
  fires?: boolean | null;
  count?: number | null;
  outcome?: string;
  reason?: string | null;
  event_id: number;
  probe_id: string;
  probe_name: string;
  window: string;
  scored: boolean;
  not_scored_reason: string | null;
  verdict: boolean | null;
  score: number | null;
  threshold: number | null;
  provisional: boolean;
}

export interface SessionTurn {
  turn_index: number;
  request_id: string | null;
  created_at: string | null;
  has_transcript: boolean;
  transcript_incomplete: boolean;
  verdicts: SessionVerdictCell[];
}

export interface SessionColumn {
  column: string;
  behaviour: ProbeBehaviourRef | null;
  probe_ids: string[];
}

export interface ProbeSessionDetail extends ProbeSessionRow {
  columns: SessionColumn[];
  turns: SessionTurn[];
}

/** ⚠ USER TEXT. Only from `GET /api/probes/sessions/{id}/transcript`. */
export interface ProbeTranscriptTurn {
  turn_index: number;
  request_id: string;
  user_text: string | null;
  reply_text: string | null;
  n_choices: number;
  incomplete: boolean;
  created_at: string | null;
}

export interface SessionListFilters {
  since?: string;
  servedModel?: string;
  behaviour?: string;
  firedOnly?: boolean;
  source?: string;
  limit?: number;
}

export interface SessionDeleteResult {
  turns: number;
  transcripts: number;
  events_deleted: number;
  events_kept: number;
}
