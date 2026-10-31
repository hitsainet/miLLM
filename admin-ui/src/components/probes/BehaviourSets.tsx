/**
 * Behaviour sets (Feature 39): probes grouped by the behaviour their definitions name, armed and
 * disarmed one behaviour at a time, with a dry run first.
 *
 * ⚠ ALL OR NOTHING. Arming a behaviour runs every probe's own gates, then checks the whole set once
 * against the behaviour limit, the probe backstop and the cost budget; one refusal arms none, and
 * the refusal lists every probe with the gate that stopped it. The dry run runs the same gates,
 * arms nothing, and shows each probe's result and the set's projected cost against both halves.
 *
 * The list is paged by cursor and filtered by key prefix: nothing here lists behaviours from a
 * constant.
 */

import { useState } from 'react';
import { useBehaviourActions, useBehaviours } from '@/hooks/useBehaviours';
import type { BehaviourProbeOutcome, BehaviourSummary } from '@/types/probe';
import { ProbeBudgetMeter } from './ProbeBudgetMeter';

function OutcomeTable({ probes }: { probes: BehaviourProbeOutcome[] }) {
  return (
    <table className="text-xs w-full" data-testid="behaviour-outcomes">
      <thead>
        <tr className="text-slate-400 text-left">
          <th>Probe</th>
          <th>Gate</th>
          <th>Outcome</th>
          <th>Per generated token</th>
          <th>Per 1,000 prompt tokens</th>
        </tr>
      </thead>
      <tbody>
        {probes.map((p) => (
          <tr key={p.probe_id} data-testid={`outcome-${p.probe_id}`}
              className={p.outcome === 'refused' ? 'text-amber-300' : 'text-slate-200'}>
            <td className="font-mono">{p.probe_id}</td>
            <td>{p.gate}</td>
            <td>{p.outcome}{p.code ? ` (${p.code})` : ''}</td>
            <td>{p.cost?.decode != null ? `${p.cost.decode.toFixed(3)} ms` : '—'}</td>
            <td>{p.cost?.prefill != null ? `${p.cost.prefill.toFixed(3)} ms` : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BehaviourGroup({
  behaviour,
  busy,
  onArm,
  onDryRun,
  onDisarm,
}: {
  behaviour: BehaviourSummary;
  busy: boolean;
  onArm: () => void;
  onDryRun: () => void;
  onDisarm: () => void;
}) {
  return (
    <div className="border border-slate-700 rounded-card p-3 stack-y-1"
         data-testid={`behaviour-${behaviour.key}`}>
      <div className="flex items-center justify-between gap-2">
        <div>
          <span className="font-mono text-slate-100">{behaviour.key}</span>
          <span className="text-xs text-slate-400 ml-2">
            {behaviour.armed} of {behaviour.imported} armed
          </span>
          {behaviour.set && !behaviour.set.complete && (
            <span data-testid="partial-badge"
                  className="ml-2 text-xs px-1 rounded-full bg-amber-900/60 text-amber-200"
                  title={`missing: ${behaviour.set.missing_probe_ids.join(', ')}`}>
              {behaviour.set.text}
            </span>
          )}
          {behaviour.armed_before_unload && (
            <span data-testid="armed-before"
                  className="ml-2 text-xs px-1 rounded-full bg-slate-700 text-slate-200">
              armed before unload or restart
            </span>
          )}
        </div>
        <div className="flex gap-2 text-xs">
          <button type="button" disabled={busy} onClick={onDryRun}
                  className="px-2 py-1 rounded-control border border-slate-600">
            Dry run
          </button>
          <button type="button" disabled={busy} onClick={onArm}
                  className="px-2 py-1 rounded-control bg-emerald-800 text-emerald-100">
            {busy ? 'Arming…' : 'Arm'}
          </button>
          <button type="button" disabled={busy || behaviour.armed === 0} onClick={onDisarm}
                  className="px-2 py-1 rounded-control bg-slate-700">
            Disarm
          </button>
        </div>
      </div>
      {behaviour.versions.length > 1 && (
        <p className="text-xs text-amber-300">
          {behaviour.versions.length} versions imported — arm one at a time.
        </p>
      )}
    </div>
  );
}

export function BehaviourSets() {
  const [keyFilter, setKeyFilter] = useState('');
  const query = useBehaviours(keyFilter);
  const actions = useBehaviourActions();
  const behaviours = query.data?.pages.flatMap((p) => p.data) ?? [];
  const refusalProbes = (actions.refusal?.details?.probes ?? []) as BehaviourProbeOutcome[];

  return (
    <section className="stack-y-2" data-testid="behaviour-sets">
      <div className="flex items-center justify-between">
        <h2 className="text-slate-200 font-medium">Behaviours</h2>
        <input
          aria-label="Filter behaviours by key"
          placeholder="key starts with…"
          value={keyFilter}
          onChange={(e) => setKeyFilter(e.target.value)}
          className="text-sm bg-slate-900 border border-slate-700 rounded-control px-2 py-1"
        />
      </div>
      {behaviours.length === 0 ? (
        <p className="text-slate-500 text-sm">
          No probe names a behaviour yet. A probe exported from miStudio's behaviour catalogue
          carries its behaviour's key.
        </p>
      ) : (
        <div className="stack-y-2">
          {behaviours.map((b) => (
            <BehaviourGroup
              key={b.key}
              behaviour={b}
              busy={actions.busyKey === b.key}
              onArm={() => actions.arm(b.key)}
              onDryRun={() => actions.dryRun(b.key)}
              onDisarm={() => actions.disarm(b.key)}
            />
          ))}
        </div>
      )}
      {query.hasNextPage && (
        <button type="button" data-testid="behaviours-more"
                onClick={() => void query.fetchNextPage()}
                className="text-xs text-slate-300 underline">
          More behaviours
        </button>
      )}
      {actions.dryRunResult && (
        <div className="border border-slate-600 rounded-card p-3 stack-y-2" data-testid="dry-run">
          <p className="text-sm text-slate-200">
            Dry run of {actions.dryRunResult.behaviour.key}: nothing was armed.
            {actions.dryRunResult.would_refuse
              ? ` Arming would be refused: ${actions.dryRunResult.would_refuse.code}.`
              : ' Arming would succeed.'}
          </p>
          <ProbeBudgetMeter
            budget={actions.dryRunResult.budget}
            projected={{ ...actions.dryRunResult.projected,
                         over_budget: actions.dryRunResult.would_refuse?.code === 'PROBE_BUDGET' }}
          />
          <OutcomeTable probes={actions.dryRunResult.probes} />
        </div>
      )}
      {actions.refusal && (
        <div className="border border-amber-700 rounded-card p-3 stack-y-2" data-testid="set-refusal">
          <p className="text-sm text-amber-200">
            {actions.refusal.key} was not armed — {actions.refusal.code}: {actions.refusal.message}
            {' '}None of its probes was armed.
          </p>
          {refusalProbes.length > 0 && <OutcomeTable probes={refusalProbes} />}
        </div>
      )}
    </section>
  );
}
