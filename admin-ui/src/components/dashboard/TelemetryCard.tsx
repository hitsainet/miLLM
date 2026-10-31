import { useState } from 'react';
import { RadioTower } from 'lucide-react';
import { StatusCard, type StatusType } from './StatusCard';
import type { TelemetryHealthView } from '@hooks/useTelemetryHealth';
import type { TelemetryHealth, TelemetrySignalHealth } from '@/types';

/**
 * The Dashboard "Telemetry" card (038 FR-38.13 UI, FTDD §6.6, FTASKS 9.3).
 *
 * ⚠ READ-ONLY BY DESIGN. ENH-006 D78 (2026-10-28): export settings are changed on the Settings
 * page (Environment, "Telemetry and logging"), never here. The card has no input and no button
 * that changes anything, and its fixed line points to Settings.
 * Only the fields named below are rendered — never a header value or a credential (the server
 * sends them only as set / not set), never an error message (only the error's class).
 *
 * Each signal shows its state as TEXT and colour, so the state never depends on colour alone.
 */

const FIXED_LINE =
  'Change export settings on the Settings page, under Environment → Telemetry and logging.';
const SIGNALS = ['traces', 'metrics', 'logs'] as const;

const STATE_STYLE: Record<string, { status: StatusType; className: string }> = {
  disabled: { status: 'neutral', className: 'bg-slate-700 text-slate-200' },
  healthy: { status: 'success', className: 'bg-green-900 text-green-200' },
  degraded: { status: 'warning', className: 'bg-yellow-900 text-yellow-100' },
  unavailable: { status: 'error', className: 'bg-red-900 text-red-100' },
};
const OTHER = { status: 'warning' as StatusType, className: 'bg-yellow-900 text-yellow-100' };

function styleOf(state: string) {
  return STATE_STYLE[state] ?? OTHER;
}

/** The card's own status: the worst enabled signal, or "Off" when every signal is disabled. */
function overall(health: TelemetryHealth): { status: StatusType; text: string } {
  const states = SIGNALS.map((s) => health.signals?.[s]?.state ?? 'disabled');
  if (states.every((s) => s === 'disabled')) return { status: 'neutral', text: 'Off' };
  for (const bad of ['unavailable', 'degraded']) {
    if (states.includes(bad)) return { status: styleOf(bad).status, text: bad[0].toUpperCase() + bad.slice(1) };
  }
  return { status: 'success', text: 'Exporting' };
}

function SignalRow({ name, signal }: { name: string; signal: TelemetrySignalHealth | undefined }) {
  const state = signal?.state ?? 'disabled';
  const style = styleOf(state);
  return (
    <div data-testid={`telemetry-signal-${name}`} className="stack-y-0.5">
      <div className="flex justify-between gap-2">
        <dt className="text-slate-400 capitalize">{name}</dt>
        <dd>
          <span
            data-testid={`telemetry-state-${name}`}
            className={`px-1.5 py-0.5 rounded-full text-[11px] font-medium ${style.className}`}
          >
            {state}
          </span>
        </dd>
      </div>
      {signal && state !== 'disabled' && (
        <dd className="text-slate-500 text-[11px] text-right">
          {`queue ${signal.queue.used}/${signal.queue.capacity} · exported ${signal.exported} · ` +
            `failed ${signal.failed} · dropped ${signal.dropped}`}
          {signal.last_error && (
            <span data-testid={`telemetry-error-${name}`} className="block text-amber-300">
              {`last error ${signal.last_error.type}${signal.last_error.at ? ` at ${signal.last_error.at}` : ''}`}
            </span>
          )}
        </dd>
      )}
    </div>
  );
}

function ConfigRows({ health }: { health: TelemetryHealth }) {
  const c = health.config;
  if (!c) return null;
  const rows: Array<[string, string]> = [
    ['Endpoint', c.endpoint ?? 'none'],
    ['Transport', c.transport ?? 'none'],
    ['Receiver credentials', c.headers_set ? 'set' : 'not set'],
    ['Client key', c.tls_client_key_set ? 'set' : 'not set'],
    ['Service', `${c.service_name ?? 'millm'} (${c.service_instance_id ?? '—'})`],
    ['Sampler', c.sampler ? `${c.sampler.kind} ${c.sampler.ratio}` : '—'],
    ['Result log origins', c.result_log_filter ? c.result_log_filter.origins.join(', ') || 'none' : '—'],
    ['Stateless export', c.stateless_export_permitted ? 'permitted' : 'not permitted'],
    ['Series cap', String(c.series_cap ?? '—')],
    ['SDK', c.sdk_version ?? '—'],
  ];
  return (
    <dl data-testid="telemetry-config" className="stack-y-0.5 text-[11px]">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-2">
          <dt className="text-slate-500">{label}</dt>
          <dd className="text-slate-300 text-right break-all">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function TelemetryCard(view: TelemetryHealthView) {
  const [open, setOpen] = useState(false);
  const icon = <RadioTower className="w-5 h-5" />;

  let display: { status: StatusType; text: string; details?: string };
  let body = null;
  if (view.isLoading) {
    display = { status: 'neutral', text: 'Loading' };
  } else if (view.error) {
    display = { status: 'warning', text: 'Unknown', details: 'health could not be read' };
  } else if (!view.health || !view.health.available) {
    display = { status: 'neutral', text: 'Not available' };
  } else if (view.health.data.error) {
    display = { status: 'warning', text: 'Unknown', details: `snapshot failed (${view.health.data.error})` };
  } else {
    const health = view.health.data;
    display = overall(health);
    body = (
      <div className="stack-y-2 text-xs">
        <dl className="stack-y-1">
          {SIGNALS.map((s) => (
            <SignalRow key={s} name={s} signal={health.signals?.[s]} />
          ))}
        </dl>
        <p data-testid="telemetry-counters" className="text-[11px] text-slate-500">
          {`series overflow ${health.series_overflow ?? 0} · attributes dropped ${health.attributes_dropped ?? 0}`}
        </p>
        <button
          type="button"
          data-testid="telemetry-config-toggle"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
          className="text-[11px] text-primary-400 hover:underline"
        >
          {open ? 'Hide configuration' : 'Show configuration'}
        </button>
        {open && <ConfigRows health={health} />}
        {health.deletion_note && (
          <p className="text-[11px] text-slate-500">{health.deletion_note}</p>
        )}
      </div>
    );
  }

  return (
    <div role="group" aria-label={`Telemetry: ${display.text}`} data-testid="telemetry-card">
      <StatusCard
        title="Telemetry"
        icon={icon}
        status={display.status}
        statusText={display.text}
        details={display.details}
        isLoading={view.isLoading}
      >
        {body}
        <p data-testid="telemetry-fixed-line" className="mt-2 text-[11px] text-slate-500">
          {FIXED_LINE}
        </p>
      </StatusCard>
    </div>
  );
}
