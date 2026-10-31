/**
 * Feature 35 (ENH-003 §3.2): the Settings page's "Environment" section, in miForge's layout.
 *
 * The environment stays the source of truth. A value saved here is an OVERRIDE, stored in miLLM's
 * database; it replaces the environment's value until "Use the environment value" clears it.
 *
 * - **Changeable while miLLM runs** — the server's 30 runtime keys. Each row shows a plain name,
 *   what it does, WHEN a change takes effect, the environment's value next to the one in force,
 *   and the last change (who, when, why). Save sends the typed value and an optional reason; the
 *   server checks it against the setting's own rules and its refusal is shown on the row.
 * - **Applies at the next model load** and **Needs a restart** — read-only, with when they apply.
 * - **From the environment** — read-only. Secrets show only "set" or "not set": the server never
 *   sends their values.
 * - **Recent changes** — the last 50 saves and resets.
 *
 * The control for a row follows the `type` the server sends (number, choice, switch or text), so
 * this file holds no copy of the server's list. ENH-006 D78: the telemetry export settings are
 * runtime rows too, in the "Telemetry and logging" group, on the same cards.
 */

import { useState } from 'react';
import { History, RotateCcw, Save, SlidersHorizontal } from 'lucide-react';
import { Badge, Button, Card, CardHeader, Input, Select } from '@components/common';
import { ApiError, environmentApi } from '@/services/api';
import type { EnvironmentChange, EnvironmentRow } from '@/types/environment';
import type { EnvironmentSettingsState } from '@/hooks/useEnvironmentSettings';
import { TELEMETRY_NOTE, parseDraft, showValue } from './environmentValues';
import { groupSettings } from './environmentGroups';

function formatWhen(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function errorText(error: unknown): string {
  return error instanceof ApiError || error instanceof Error ? error.message : 'Request failed';
}

function rangeText(row: EnvironmentRow): string | null {
  if (row.type === 'choice' || row.type === 'text' || row.minimum === null || row.maximum === null)
    return null;
  return `${row.minimum} to ${row.maximum}`;
}

function RuntimeRow({ row, onChanged }: { row: EnvironmentRow; onChanged: () => Promise<void> }) {
  const [draft, setDraft] = useState<string | null>(null);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const shown = draft ?? (row.value === null ? '' : String(row.value));
  const overridden = row.source === 'override';

  const send = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setProblem(null);
    try {
      await action();
      setDraft(null);
      setReason('');
      await onChanged();
    } catch (e) {
      setProblem(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const save = () => {
    const parsed = parseDraft(row, shown);
    if ('error' in parsed) {
      setProblem(parsed.error);
      return;
    }
    void send(() => environmentApi.set(row.key, parsed.value, reason.trim() || undefined));
  };

  const control =
    row.type === 'choice' || row.type === 'bool' ? (
      <Select
        aria-label={row.label}
        options={(row.type === 'bool' ? ['true', 'false'] : (row.choices ?? [])).map((c) => ({
          value: c,
          label: c,
        }))}
        value={shown}
        onChange={(e) => setDraft(e.target.value)}
        data-testid={`env-input-${row.key}`}
      />
    ) : row.type === 'text' ? (
      <Input
        aria-label={row.label}
        type="text"
        spellCheck={false}
        autoComplete="off"
        value={shown}
        onChange={(e) => setDraft(e.target.value)}
        data-testid={`env-input-${row.key}`}
      />
    ) : (
      <Input
        aria-label={row.label}
        type="number"
        inputMode="decimal"
        step={row.type === 'int' ? 1 : 'any'}
        min={row.minimum ?? undefined}
        max={row.maximum ?? undefined}
        value={shown}
        onChange={(e) => setDraft(e.target.value)}
        data-testid={`env-input-${row.key}`}
      />
    );

  return (
    <li className="py-4 stack-y-2" data-testid={`env-row-${row.key}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm text-slate-200">{row.label}</p>
          <p className="text-xs text-slate-500 font-mono">{row.key}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="primary" size="sm">
            Takes effect: {row.effect}
          </Badge>
          <Badge variant={overridden ? 'warning' : 'neutral'} size="sm">
            {overridden ? 'override' : 'environment'}
          </Badge>
        </div>
      </div>
      {row.description && <p className="text-xs text-slate-400">{row.description}</p>}
      <p className="text-xs text-slate-400" data-testid={`env-values-${row.key}`}>
        In force: <span className="font-mono text-slate-200">{showValue(row.value)}</span>
        {' · '}Environment value: <span className="font-mono">{showValue(row.env_value)}</span>
        {rangeText(row) && <> · Allowed: {rangeText(row)}</>}
      </p>
      {row.updated_at && (
        <p className="text-xs text-slate-500" data-testid={`env-last-${row.key}`}>
          Last changed by {row.updated_by ?? 'unknown'}, {formatWhen(row.updated_at)}
          {row.reason ? ` — “${row.reason}”` : ''}
        </p>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-[minmax(0,10rem)_minmax(0,1fr)_auto] gap-2 items-start">
        {control}
        <Input
          aria-label={`Reason for changing ${row.label}`}
          placeholder="Reason (optional)"
          maxLength={500}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          data-testid={`env-reason-${row.key}`}
        />
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={save}
            disabled={busy}
            leftIcon={<Save className="w-4 h-4" />}
            data-testid={`env-save-${row.key}`}
          >
            Save
          </Button>
          {overridden && (
            <Button
              size="sm"
              variant="secondary"
              disabled={busy}
              onClick={() =>
                void send(() => environmentApi.reset(row.key, reason.trim() || undefined))
              }
              leftIcon={<RotateCcw className="w-4 h-4" />}
              data-testid={`env-reset-${row.key}`}
            >
              Use the environment value
            </Button>
          )}
        </div>
      </div>
      {problem && (
        <p className="text-xs text-red-400" role="alert" data-testid={`env-problem-${row.key}`}>
          {problem}
        </p>
      )}
    </li>
  );
}

/** A subsystem group's heading inside a section (2026-10-28). */
function GroupHeading({ title, count }: { title: string; count: number }) {
  return (
    <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400 mt-3 mb-1">
      {title} <span className="font-normal normal-case text-slate-500">({count})</span>
    </h4>
  );
}

function ReadOnlyList({ rows, testId }: { rows: EnvironmentRow[]; testId: string }) {
  return (
    <div data-testid={testId}>
      {groupSettings(rows).map((group) => (
        <div key={group.id} data-testid={`${testId}-group-${group.id}`}>
          <GroupHeading title={group.title} count={group.rows.length} />
          <dl className="grid grid-cols-[minmax(0,14rem)_minmax(0,1fr)] gap-x-4 gap-y-1 text-xs">
            {group.rows.map((row) => (
              <div key={row.key} className="contents">
                <dt
                  className="font-mono text-slate-400 wrap-break-word"
                  title={row.description ?? undefined}
                  data-testid="env-fixed-key"
                >
                  {row.key}
                </dt>
                <dd
                  className="font-mono text-slate-200 wrap-break-word"
                  data-testid={`env-fixed-${row.key}`}
                >
                  {row.secret ? (row.set ? 'set' : 'not set') : showValue(row.value)}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}

function ChangeList({ changes }: { changes: EnvironmentChange[] }) {
  if (changes.length === 0) {
    return <p className="text-xs text-slate-500">No changes yet.</p>;
  }
  return (
    <ul className="stack-y-1 text-xs" data-testid="env-history">
      {changes.map((c) => (
        <li key={c.id} className="text-slate-400">
          <span className="text-slate-500">{formatWhen(c.changed_at)}</span>{' '}
          <span className="font-mono text-slate-200">{c.key}</span>{' '}
          {c.action === 'set' ? 'set to' : 'back to the environment value'}{' '}
          <span className="font-mono text-slate-200">{showValue(c.new_value)}</span>
          {' '}(was {showValue(c.previous_value)}) by {c.actor}
          {c.reason ? ` — “${c.reason}”` : ''}
        </li>
      ))}
    </ul>
  );
}

export function EnvironmentSettings({ state }: { state: EnvironmentSettingsState }) {
  const { listing, history, error, reload } = state;
  const rows = listing?.settings ?? [];
  const byGroup = (group: EnvironmentRow['group']) => rows.filter((r) => r.group === group);
  const runtime = byGroup('runtime');

  return (
    <Card>
      <CardHeader
        title="Environment"
        subtitle="Where miLLM's configuration comes from, and the settings you can change while it runs"
        icon={<SlidersHorizontal className="w-5 h-5 text-slate-400" />}
      />
      {error && (
        <p className="text-sm text-red-400 mb-3" role="alert" data-testid="env-error">
          Could not read the environment settings: {error}
        </p>
      )}
      {listing && (
        <div className="stack-y-6" data-testid="env-section">
          <section>
            <h3 className="text-sm font-semibold text-slate-200">Changeable while miLLM runs</h3>
            <p className="text-xs text-slate-500" data-testid="env-runtime-count">
              {runtime.length} of {rows.length} settings. A saved value is stored in miLLM&apos;s
              database and replaces the environment&apos;s value until you choose &quot;Use the
              environment value&quot;. Each row says when a change takes effect.
            </p>
            <div data-testid="env-runtime">
              {groupSettings(runtime).map((group) => (
                <div key={group.id} data-testid={`env-runtime-group-${group.id}`}>
                  <GroupHeading title={group.title} count={group.rows.length} />
                  {group.id === 'telemetry' &&
                    group.rows.some((r) => r.key.startsWith('TELEMETRY_')) && (
                      <p className="text-xs text-slate-500" data-testid="env-telemetry-note">
                        {TELEMETRY_NOTE}
                      </p>
                    )}
                  <ul className="divide-y divide-slate-800" aria-label={`Runtime settings: ${group.title}`}>
                    {group.rows.map((row) => (
                      <RuntimeRow key={row.key} row={row} onChanged={reload} />
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
          <section>
            <h3 className="text-sm font-semibold text-slate-200">Applies at the next model load</h3>
            <p className="text-xs text-slate-500 mb-2">
              Read when a model loads. Change these in miLLM&apos;s environment; they take effect
              the next time a model is loaded, without a restart.
            </p>
            <ReadOnlyList rows={byGroup('model_load')} testId="env-model-load" />
          </section>
          <section>
            <h3 className="text-sm font-semibold text-slate-200">Needs a restart of millm-backend</h3>
            <p className="text-xs text-slate-500 mb-2">
              Read once at startup. A restart ends requests and batches that are running.
            </p>
            <ReadOnlyList rows={byGroup('restart')} testId="env-restart" />
          </section>
          <section>
            <h3 className="text-sm font-semibold text-slate-200">From the environment</h3>
            <p className="text-xs text-slate-500 mb-2">
              Set in miLLM&apos;s environment; change them there and restart. Secrets show only
              whether they are set.
            </p>
            <ReadOnlyList rows={byGroup('environment')} testId="env-environment" />
          </section>
          <section>
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <History className="w-4 h-4" /> Recent changes
            </h3>
            <ChangeList changes={history} />
          </section>
        </div>
      )}
    </Card>
  );
}
