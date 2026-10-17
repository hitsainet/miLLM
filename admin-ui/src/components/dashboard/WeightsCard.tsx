import { Database } from 'lucide-react';
import { StatusCard, type StatusType } from './StatusCard';
import type { WeightsStatusView } from '@hooks/useWeightsStatus';
import type { WeightsState, WeightsStatus } from '@/types';

/**
 * The Dashboard "Weights" card (Feature 31 FTID §6; Feature 33 status block).
 *
 * Read-only by design: no buttons, no token, and only the fields named below are rendered, so
 * nothing the server adds to the payload reaches the DOM by accident (FR-31.16.2 — never the
 * attachment ID).
 */

const STATE_DISPLAY: Record<WeightsState, { status: StatusType; text: string; details?: string }> = {
  clean: { status: 'neutral', text: 'Clean' },
  writing: { status: 'warning', text: 'Writing' },
  // A torn set refuses every request (WEIGHTS_TORN, 503). Say what that means for the operator.
  torn: { status: 'error', text: 'Torn', details: 'serving refused' },
};

const NOT_AVAILABLE = 'not available';

function describeAttachment(status: WeightsStatus): string {
  const a = status.attachment;
  if (!a) return 'none';
  const holder = `${a.label} (${a.role})`;
  return a.state === 'stale' ? `${holder}, stale` : holder;
}

/** `null` is "not recorded" — never shown as 0, which would claim nothing has been written. */
function describeSteps(steps: number | null): string {
  return steps === null ? 'not recorded' : String(steps);
}

function describeLastSave(status: WeightsStatus): string {
  const save = status.last_save;
  if (!save) return 'none';
  return `${save.checkpoint_sha256.slice(0, 12)} at ${save.at}`;
}

function describeAdapter(view: WeightsStatusView): string {
  if (view.adapterLoading) return '…';
  if (view.adapterError) return 'unknown';
  if (!view.adapter || !view.adapter.available) return NOT_AVAILABLE;
  const adapter = view.adapter.data;
  if (!adapter.applied) return 'none';
  const name = adapter.name ?? 'unnamed';
  return adapter.rank === null ? name : `${name} (rank ${adapter.rank})`;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-slate-300 text-right">{value}</dd>
    </div>
  );
}

export function WeightsCard(view: WeightsStatusView) {
  const icon = <Database className="w-5 h-5" />;
  const adapterText = describeAdapter(view);

  let display: { status: StatusType; text: string; details?: string };
  let rows: Array<[string, string]> = [];
  let ariaLabel: string;

  if (view.statusLoading) {
    display = { status: 'neutral', text: 'Loading' };
    ariaLabel = 'Weights: loading';
  } else if (view.statusError) {
    display = { status: 'warning', text: 'Unknown', details: 'status could not be read' };
    ariaLabel = 'Weights: status could not be read';
  } else if (!view.status || !view.status.available) {
    display = { status: 'neutral', text: 'Not available' };
    ariaLabel = `Weights: ${NOT_AVAILABLE}`;
  } else {
    const status = view.status.data;
    display = STATE_DISPLAY[status.weights_state];
    const attachment = describeAttachment(status);
    rows = [
      ['Attachment', attachment],
      ['Steps since load', describeSteps(status.steps_since_load)],
      ['Last save', describeLastSave(status)],
    ];
    const stateWords = display.details ? `${status.weights_state}, ${display.details}` : status.weights_state;
    ariaLabel = `Weights: ${stateWords}; holder: ${attachment}`;
  }
  rows.push(['Adapter', adapterText]);

  return (
    <div role="group" aria-label={ariaLabel} data-testid="weights-card">
      <StatusCard
        title="Weights"
        icon={icon}
        status={display.status}
        statusText={display.text}
        details={display.details}
        isLoading={view.statusLoading}
      >
        <dl className="space-y-1 text-xs">
          {rows.map(([label, value]) => (
            <Row key={label} label={label} value={value} />
          ))}
        </dl>
      </StatusCard>
    </div>
  );
}
