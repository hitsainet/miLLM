import type { ProbeResultV1 } from '@/types';

/**
 * The identity of one recorded probe result, in the event detail (038 FR-38.3 event detail,
 * FTDD §6.6, FTASKS 9.4): `result_id`, `outcome` / `reason`, `persistence`, the detector digest
 * (first 12 hex characters, the full value in `title`), both revisions, and a "migrated" marker
 * for a row recorded before 038.
 *
 * ⚠ ONLY NAMED FIELDS, and none of them can carry text: they are ids, codes and digests.
 * Every field is optional — the event may come from a backend that predates 038.
 */

export type ResultIdentityFields = Partial<
  Pick<
    ProbeResultV1,
    | 'result_id'
    | 'outcome'
    | 'reason'
    | 'persistence'
    | 'detector_digest'
    | 'detector_digest_source'
    | 'threshold_revision'
    | 'decision_revision'
    | 'migrated'
  >
>;

function digestText(digest: string | undefined): { short: string; full?: string } {
  if (!digest || digest === 'unknown') return { short: 'unknown' };
  return { short: digest.slice(0, 12), full: digest };
}

export function ProbeResultIdentity({ result }: { result: ResultIdentityFields }) {
  if (!result.result_id) return null;
  const digest = digestText(result.detector_digest);
  const outcome = result.reason ? `${result.outcome ?? 'unknown'} (${result.reason})` : result.outcome;
  return (
    <dl data-testid="result-identity" className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px]">
      <dt className="text-slate-500">Result</dt>
      <dd data-testid="result-id" className="text-slate-300 font-mono break-all">
        {result.result_id}
        {result.migrated && (
          <span
            data-testid="result-migrated"
            title="Recorded before 038; its identity was filled in by the migration"
            className="ml-1 px-1 rounded-sm bg-slate-700 text-slate-200"
          >
            migrated
          </span>
        )}
      </dd>
      <dt className="text-slate-500">Outcome</dt>
      <dd data-testid="result-outcome" className="text-slate-300">{outcome ?? 'unknown'}</dd>
      <dt className="text-slate-500">Persistence</dt>
      <dd data-testid="result-persistence" className="text-slate-300">
        {result.persistence ?? 'unknown'}
      </dd>
      <dt className="text-slate-500">Detector</dt>
      <dd data-testid="result-digest" className="text-slate-300 font-mono" title={digest.full}>
        {digest.short}
        {result.detector_digest_source && result.detector_digest_source !== 'unknown' && (
          <span className="text-slate-500"> ({result.detector_digest_source})</span>
        )}
      </dd>
      <dt className="text-slate-500">Bar revision</dt>
      <dd data-testid="result-threshold-revision" className="text-slate-300">
        {result.threshold_revision ?? '—'}
      </dd>
      <dt className="text-slate-500">Decision revision</dt>
      <dd data-testid="result-decision-revision" className="text-slate-300">
        {result.decision_revision ?? 'not stated'}
      </dd>
    </dl>
  );
}
