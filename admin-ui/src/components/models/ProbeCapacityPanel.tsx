/**
 * Probe capacity for one model (operator 2026-10-21): "I want to know the actual limits, what
 * drives them, and to be able to adjust on a per model basis accordingly."
 *
 * Four things, in the order an operator reads them: the MEASURED cost lines (own cost per probe +
 * the shared part per hooked layer, per kind), the limits those lines IMPLY under the budget in
 * force, the EFFECTIVE limits admission uses (and where each came from), and — for the resident
 * model — how close the armed set is to each. The editable fields set this model's own limits;
 * an empty field means the global default.
 */

import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { modelApi } from '@/services/api';
import type {
  BudgetSetting,
  CostLine,
  ProbeCapacity,
  ProbeLimitRecommendation,
} from '@/types/probeCapacity';

function line(kind: string, c: CostLine | null | undefined) {
  if (!c) return null;
  return (
    <li key={kind} data-testid={`cost-line-${kind}`}>
      {kind}: {c.decode_fixed_ms.toFixed(3)} + {c.decode_own_ms.toFixed(4)} × N ms per generated
      token; {c.prefill_fixed_ms_per_1k.toFixed(3)} + {c.prefill_own_ms_per_1k.toFixed(4)} × N ms
      per 1,000 prompt tokens
    </li>
  );
}

function pct(share: number | null | undefined): string {
  return share == null ? '—' : `${Math.round(share * 100)}%`;
}

function BudgetField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: BudgetSetting | null;
  onChange: (v: BudgetSetting | null) => void;
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="w-40">{label}</span>
      <input
        aria-label={`${label} value`}
        type="number"
        step="any"
        className="w-24 bg-slate-900 border border-slate-700 rounded-sm px-1"
        value={value?.value ?? ''}
        placeholder="global"
        onChange={(e) =>
          onChange(e.target.value === '' ? null : { value: Number(e.target.value), unit: value?.unit ?? 'ms' })
        }
      />
      <select
        aria-label={`${label} unit`}
        value={value?.unit ?? 'ms'}
        onChange={(e) => value && onChange({ ...value, unit: e.target.value as 'ms' | 'pct' })}
        className="bg-slate-900 border border-slate-700 rounded-sm"
      >
        <option value="ms">ms</option>
        <option value="pct">% of step time</option>
      </select>
    </label>
  );
}

/**
 * Feature 39 (D13): the explained recommendation and its one-click approval.
 *
 * ⚠ APPROVE SENDS THE RECOMMENDATION'S OWN VALUES AND DIGEST, NOTHING TYPED. The server recomputes
 * the recommendation and refuses `RECOMMENDATION_CHANGED` if the digest no longer matches, then
 * writes exactly its values — so what the operator read is what is written.
 */
function Recommendation({
  rec,
  onApprove,
  busy,
}: {
  rec: ProbeLimitRecommendation;
  onApprove: (rec: ProbeLimitRecommendation) => void;
  busy: boolean;
}) {
  if (!rec.values || !rec.recommendation_digest) {
    return <p className="text-xs text-slate-400" data-testid="probe-recommendation">{rec.reasons[0]}</p>;
  }
  return (
    <div className="text-xs stack-y-1" data-testid="probe-recommendation">
      <p className="text-slate-200">
        Recommended: {rec.values.max_behaviours} behaviour{rec.values.max_behaviours === 1 ? '' : 's'},{' '}
        {rec.values.decode_budget_ms.toFixed(2)} ms per generated token,{' '}
        {rec.values.prefill_budget_ms.toFixed(2)} ms per 1,000 prompt tokens
        {rec.estimated ? ' (estimated)' : ''}.
      </p>
      <ul className="list-disc ml-4 text-slate-400">
        {rec.reasons.map((r) => <li key={r}>{r}</li>)}
      </ul>
      <button type="button" data-testid="approve-recommendation" disabled={busy}
              className="rounded-sm bg-emerald-800 text-emerald-100 px-2 py-1"
              onClick={() => onApprove(rec)}>
        Approve
      </button>
    </div>
  );
}

export function ProbeCapacityPanel({ modelId }: { modelId: number }) {
  const queryClient = useQueryClient();
  const key = ['models', modelId, 'probe-capacity'];
  const query = useQuery({ queryKey: key, queryFn: () => modelApi.probeCapacity(modelId) });
  const recQuery = useQuery({
    queryKey: [...key, 'recommendation'],
    queryFn: () => modelApi.probeLimitRecommendation(modelId),
  });
  const [maxBehaviours, setMaxBehaviours] = useState<string>('');
  const [maxArmed, setMaxArmed] = useState<string>('');
  const [decode, setDecode] = useState<BudgetSetting | null>(null);
  const [prefill, setPrefill] = useState<BudgetSetting | null>(null);
  const data: ProbeCapacity | undefined = query.data;

  useEffect(() => {
    if (!data) return;
    setMaxBehaviours(
      data.configured.max_behaviours == null ? '' : String(data.configured.max_behaviours));
    setMaxArmed(data.configured.max_armed == null ? '' : String(data.configured.max_armed));
    setDecode(data.configured.decode_budget);
    setPrefill(data.configured.prefill_budget);
  }, [data]);

  const save = useMutation({
    mutationFn: () =>
      modelApi.setProbeLimits(modelId, {
        max_behaviours: maxBehaviours === '' ? null : Number(maxBehaviours),
        max_armed: maxArmed === '' ? null : Number(maxArmed),
        decode_budget: decode,
        prefill_budget: prefill,
        // An override, recorded as the operator's.
        source: 'operator',
        set_by: 'admin-ui',
      }),
    onSuccess: (fresh) => queryClient.setQueryData(key, fresh),
  });
  const approve = useMutation({
    mutationFn: (rec: ProbeLimitRecommendation) =>
      modelApi.setProbeLimits(modelId, {
        source: 'recommended-approved',
        recommendation_digest: rec.recommendation_digest as string,
        set_by: 'admin-ui',
      }),
    onSuccess: (fresh) => {
      queryClient.setQueryData(key, fresh);
      void queryClient.invalidateQueries({ queryKey: [...key, 'recommendation'] });
    },
  });
  const measure = useMutation({
    mutationFn: () => modelApi.measureProbeCapacity(modelId),
    onSuccess: (fresh) => queryClient.setQueryData(key, fresh),
  });

  if (query.isLoading) return <p className="text-xs text-slate-400">Loading probe capacity…</p>;
  if (!data) return null;
  const m = data.measured;
  const e = data.effective;
  return (
    <section data-testid="probe-capacity" className="stack-y-2 border-t border-slate-700/50 pt-3 text-sm">
      <h3 className="font-medium text-slate-200">Probe capacity</h3>
      {m ? (
        <div className="text-xs text-slate-400">
          <p>
            Measured on {m.device}: {m.decode_ms_per_token != null ? `${m.decode_ms_per_token} ms per generated token` : 'decode step time not measured'};{' '}
            {m.prefill_ms_per_1k != null ? `${m.prefill_ms_per_1k} ms per 1,000 prompt tokens` : 'prefill step time not measured'}.
          </p>
          <ul>{line('dense', m.dense)}{line('k-sparse', m.sparse)}</ul>
          <p className="italic">{m.method}</p>
        </div>
      ) : (
        <p className="text-xs text-slate-400">Not measured yet for this model.</p>
      )}
      <ul className="text-xs" data-testid="probe-implied">
        {Object.entries(data.implied).map(([kind, v]) =>
          v ? (
            <li key={kind}>
              {kind}: up to {v.decode.fits_with_headroom ?? '—'} by the per-token budget,{' '}
              {v.prefill.fits_with_headroom ?? '—'} by the prompt budget
              {v.binding ? ` — the ${v.binding} budget binds` : ''}
            </li>
          ) : null,
        )}
      </ul>
      {recQuery.data && (
        <Recommendation rec={recQuery.data} busy={approve.isPending}
                        onApprove={(rec) => approve.mutate(rec)} />
      )}
      {e.max_behaviours != null && (
        <p className="text-xs" data-testid="probe-behaviour-limit">
          Behaviours: {e.max_behaviours} ({e.sources.max_behaviours ?? 'global'}
          {e.approval?.set_by ? `, set by ${e.approval.set_by}` : ''})
          {e.stale && (
            <span data-testid="limits-stale" className="ml-2 px-1 rounded-sm bg-amber-900/60 text-amber-200"
                  title="Measured under another model, precision or card; still in force">
              stale
            </span>
          )}
          {e.approval?.newer_recommendation_available && (
            <span className="ml-2 text-slate-400">a newer recommendation is available</span>
          )}
        </p>
      )}
      <p className="text-xs" data-testid="probe-effective">
        In force: {e.max_armed} probes ({e.sources.max_armed}); {e.decode_ms_per_pass.toFixed(3)} ms per
        generated token ({e.sources.decode}); {e.prefill_ms_per_1k.toFixed(3)} ms per 1,000 prompt
        tokens ({e.sources.prefill}); headroom {e.headroom}.
      </p>
      {data.armed && (
        <p className="text-xs" data-testid="probe-armed-share">
          Armed now: {data.armed.count} — count {pct(data.armed.share.count)}, per-token budget{' '}
          {pct(data.armed.share.decode)}, prompt budget {pct(data.armed.share.prefill)}.
        </p>
      )}
      <div className="stack-y-1 text-xs">
        <label className="flex items-center gap-2">
          <span className="w-40">Most armed behaviours</span>
          <input
            aria-label="Most armed behaviours"
            type="number"
            min={1}
            max={64}
            className="w-24 bg-slate-900 border border-slate-700 rounded-sm px-1"
            value={maxBehaviours}
            placeholder="global"
            onChange={(e) => setMaxBehaviours(e.target.value)}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-40">Most armed probes</span>
          <input
            aria-label="Most armed probes"
            type="number"
            min={1}
            max={64}
            className="w-24 bg-slate-900 border border-slate-700 rounded-sm px-1"
            value={maxArmed}
            placeholder="global"
            onChange={(e) => setMaxArmed(e.target.value)}
          />
        </label>
        <BudgetField label="Per-token budget" value={decode} onChange={setDecode} />
        <BudgetField label="Prompt budget (per 1,000)" value={prefill} onChange={setPrefill} />
        <div className="flex gap-2">
          <button
            type="button"
            data-testid="save-probe-limits"
            className="rounded-sm bg-primary-700 px-2 py-1"
            onClick={() => save.mutate()}
          >
            Override
          </button>
          <button
            type="button"
            data-testid="measure-probe-capacity"
            className="rounded-sm bg-slate-700 px-2 py-1"
            title="Runs on the graphics card: the model must be loaded and no probe armed"
            onClick={() => measure.mutate()}
          >
            Measure on its card
          </button>
        </div>
        {(save.error || measure.error || approve.error) && (
          <p className="text-red-300">
            {String((save.error || measure.error || approve.error) as Error)}
          </p>
        )}
      </div>
    </section>
  );
}
