/**
 * The armed set's projected cost against its budget (Feature 37, FR-37.7).
 *
 * Two figures, the two halves of the budget: per generated token (each decode pass) and per 1,000
 * prompt tokens (prefill). Projections are measured at arm on each probe's own card. When an
 * operator lowers a budget nothing is disarmed; the meter says the set is over budget instead.
 */

import type { ProbeBudget, ProbeProjection } from '@/types/probe';

function pct(value: number, limit: number): number {
  return limit > 0 ? Math.min(100, (value / limit) * 100) : 0;
}

export function ProbeBudgetMeter({
  budget,
  projected,
}: {
  budget?: ProbeBudget;
  projected?: ProbeProjection;
}) {
  if (!budget || !projected) return null;
  const over = projected.over_budget;
  const rows = [
    {
      label: `projected ${projected.decode.toFixed(3)} of ${budget.decode_ms_per_pass} ms per generated token`,
      value: pct(projected.decode, budget.decode_ms_per_pass),
    },
    {
      label: `${projected.prefill.toFixed(3)} of ${budget.prefill_ms_per_1k} ms per 1,000 prompt tokens`,
      value: pct(projected.prefill, budget.prefill_ms_per_1k),
    },
  ];
  return (
    <div data-testid="probe-budget-meter" className="stack-y-1 text-xs">
      <p className={over ? 'text-amber-300' : 'text-slate-400'}>
        Probe cost budget{over ? ' — over budget: the armed set no longer fits; nothing was disarmed' : ''}
      </p>
      {rows.map((r) => (
        <div key={r.label}>
          <div
            role="meter"
            aria-label={r.label}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(r.value)}
            className="h-1.5 w-full rounded-sm bg-slate-800"
          >
            <div
              className={`h-1.5 rounded-sm ${over ? 'bg-amber-500' : 'bg-primary-500'}`}
              style={{ width: `${r.value}%` }}
            />
          </div>
          <span className="text-slate-300">{r.label}</span>
        </div>
      ))}
    </div>
  );
}
