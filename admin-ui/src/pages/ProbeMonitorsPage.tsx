/**
 * Probe Monitors (Feature 24).
 *
 * ⚠ DISTINCT FROM "Feature Monitor" (`/monitoring`), which watches SAE feature activations. A
 * probe monitor is a trained linear detector imported from miStudio. The older page was renamed
 * for exactly this reason (D8).
 *
 * Two display rules run through everything here, and both exist because the alternative reads as
 * a confident claim that was never made:
 *
 * 1. **A probe's rung language comes from the server**, never from a number→phrase map here.
 * 2. **`null` is not `false`.** A probe with no threshold ranks without deciding; an unscored
 *    request said nothing at all. Neither is "did not fire".
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, Radar, Trash2, Upload } from 'lucide-react';
import type { ProbeAckDetails } from '@/hooks/useProbes';
import { ProbeHubBrowser } from '@/components/probes/ProbeHubBrowser';
import { useProbeEventDetail, useProbes } from '@/hooks/useProbes';
import type { Probe, ProbeEvent } from '@/types/probe';
import { ARM_WITHOUT_ACK_MIN_RUNG } from '@/types/probe';

function RungBadge({ rung, language }: { rung: number; language: string }) {
  const weak = rung < ARM_WITHOUT_ACK_MIN_RUNG;
  return (
    <span
      data-testid="rung-badge"
      title={language}
      className={`px-2 py-0.5 rounded text-xs font-medium ${
        weak ? 'bg-amber-900/40 text-amber-300' : 'bg-slate-700 text-slate-200'
      }`}
    >
      rung {rung} · {language}
    </span>
  );
}

function ParitySummary({ probe }: { probe: Probe }) {
  if (!probe.parity) return <span className="text-slate-500 text-xs">not checked</span>;
  const { passed, max_abs_diff, tolerance } = probe.parity;
  return (
    <span
      data-testid="parity"
      className={`text-xs ${passed ? 'text-emerald-400' : 'text-rose-400'}`}
    >
      parity {passed ? 'passed' : 'FAILED'}
      {max_abs_diff !== null && ` · max Δ ${max_abs_diff.toExponential(2)} of ${tolerance}`}
    </span>
  );
}

function ProbeRow({
  probe,
  onArm,
  onDisarm,
  onDelete,
  onCheckParity,
  busy,
}: {
  probe: Probe;
  onArm: (id: string) => void;
  onDisarm: (id: string) => void;
  onDelete: (id: string) => void;
  onCheckParity: (id: string) => void;
  busy: boolean;
}) {
  return (
    <div
      data-testid="probe-row"
      className="border border-slate-700 rounded p-3 flex flex-col gap-2 bg-slate-800/40"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-slate-100 font-medium truncate">{probe.name}</p>
          <p className="font-mono text-xs text-slate-400">
            {probe.hf_id} · L{probe.layer} · {probe.rule} ·{' '}
            {probe.basis === 'sae_features' ? 'SAE basis' : 'dense residual'} · scope{' '}
            {probe.scope}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {probe.armed ? (
            <span
              data-testid="armed-chip"
              className="px-2 py-0.5 rounded text-xs bg-emerald-900/50 text-emerald-300"
            >
              armed
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-400">idle</span>
          )}
          {probe.armed ? (
            <button
              onClick={() => onDisarm(probe.id)}
              className="text-xs px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              Disarm
            </button>
          ) : (
            <button
              onClick={() => onArm(probe.id)}
              disabled={busy}
              className="text-xs px-2 py-1 rounded bg-emerald-700 text-emerald-50 hover:bg-emerald-600 disabled:opacity-50"
            >
              {busy ? 'Arming…' : 'Arm'}
            </button>
          )}
          {/* Parity without arming — the case after a model reload. */}
          <button
            onClick={() => onCheckParity(probe.id)}
            disabled={busy}
            title="Re-score the definition's test vectors against the loaded model"
            className="text-xs px-2 py-1 rounded border border-slate-600 text-slate-400 hover:bg-slate-700 disabled:opacity-50"
          >
            Check parity
          </button>
          <button
            onClick={() => onDelete(probe.id)}
            aria-label={`Delete ${probe.name}`}
            className="text-slate-500 hover:text-rose-400"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <RungBadge rung={probe.rung} language={probe.rung_language} />
        <ParitySummary probe={probe} />
        {/* ⚠ A null threshold is not zero. The probe ranks but does not decide. */}
        <span className="text-xs text-slate-400 font-mono">
          {probe.threshold === null
            ? 'no threshold — ranks, does not decide'
            : `threshold ${probe.threshold.toFixed(4)}`}
        </span>
        {!probe.streamable && (
          <span className="text-xs text-slate-500" title="its rule is only defined once generation ends">
            not streamable
          </span>
        )}
      </div>

      {/* ⚠ An armed probe that is not scoring must SAY SO. Silence reads as "nothing detected". */}
      {probe.armed && probe.paused_reason && (
        <p data-testid="paused-reason" className="text-xs text-amber-300 flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" /> paused — {probe.paused_reason}
        </p>
      )}
      {probe.rung < ARM_WITHOUT_ACK_MIN_RUNG && (
        <p className="text-xs text-amber-300/80">Next: {probe.next_step}</p>
      )}
    </div>
  );
}

/**
 * The below-rung-2 acknowledgement.
 *
 * ⚠ **THIS IS A DECISION, NOT A WARNING.** The server refuses to arm a probe below rung 2 until
 * someone says explicitly that they are choosing to monitor live traffic with it, and stores that
 * separately from the acknowledgement inside the definition — the person who exported a weak probe
 * and the person arming it here are not necessarily the same, and only the second is choosing.
 *
 * So: the rung, its language and the next step all come from the SERVER's refusal details, the
 * confirm button is not the default focus, and the dialog cannot be dismissed by arming. A
 * one-click "Arm anyway" would make the gate decorative.
 */
function AcknowledgeDialog({
  details,
  onConfirm,
  onCancel,
  busy,
}: {
  details: ProbeAckDetails;
  onConfirm: (reason: string) => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const [reason, setReason] = useState('');
  return (
    <section
      data-testid="ack-dialog"
      role="dialog"
      aria-label="Acknowledge arming a probe below rung 2"
      className="border border-amber-700/60 bg-amber-950/30 rounded p-4 space-y-3"
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="w-5 h-5 text-amber-300 shrink-0" aria-hidden />
        <div className="space-y-1">
          <h2 className="text-amber-200 font-medium">
            This probe is rung {details.rung ?? '?'} — {details.rung_language ?? 'below rung 2'}
          </h2>
          <p className="text-sm text-amber-100/80">
            It has not been shown to detect on unseen tasks. Arming it means monitoring live traffic
            with a detector whose evidence does not yet reach that bar.
          </p>
          {details.next_step && (
            <p className="text-xs text-amber-200/70">
              To raise it: {details.next_step}
            </p>
          )}
        </div>
      </div>
      <label className="block text-xs text-amber-200/80">
        Why are you arming it? (recorded with the acknowledgement)
        <input
          id="probe-ack-reason"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="e.g. triage only, not acted on"
          className="mt-1 w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-slate-100"
        />
      </label>
      <div className="flex items-center gap-2">
        <button
          data-testid="ack-confirm"
          onClick={() => onConfirm(reason)}
          disabled={busy}
          className="text-sm px-3 py-1.5 rounded bg-amber-700 text-amber-50 hover:bg-amber-600 disabled:opacity-50"
        >
          {busy ? 'Arming…' : 'I understand — arm it'}
        </button>
        <button
          data-testid="ack-cancel"
          onClick={onCancel}
          className="text-sm px-3 py-1.5 rounded border border-slate-600 text-slate-300 hover:bg-slate-700"
        >
          Cancel
        </button>
      </div>
    </section>
  );
}


function EventRow({
  event,
  probeName,
  onOpen,
}: {
  event: ProbeEvent;
  probeName: string;
  onOpen: (id: number) => void;
}) {
  return (
    <button
      data-testid="event-row"
      onClick={() => onOpen(event.id)}
      className="w-full text-left border-b border-slate-800 py-2 px-2 hover:bg-slate-800/50"
    >
      <div className="flex items-center justify-between gap-3">
        {/* ⚠ WHICH PROBE SAID THIS. Two probes armed on one layer produce two verdicts per
            request — often one firing and one not — and the row used to show neither name nor
            time, so the pair was indistinguishable. */}
        <span data-testid="event-probe-name" className="text-xs text-slate-300 truncate">
          {probeName}
        </span>
        {event.scored ? (
          <span
            className={`text-xs font-mono ${
              event.verdict === true
                ? 'text-violet-300'
                : event.verdict === false
                  ? 'text-slate-400'
                  : 'text-slate-500'
            }`}
          >
            {event.score?.toFixed(4)}
            {/* ⚠ null verdict renders as neither fired nor not-fired. */}
            {event.verdict === null ? ' (no threshold)' : event.verdict ? ' · fires' : ''}
          </span>
        ) : (
          <span data-testid="not-scored" className="text-xs text-amber-300">
            not scored — {event.not_scored_reason}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 mt-0.5">
        <span className="text-[11px] text-slate-500">
          {event.created_at ? new Date(event.created_at).toLocaleTimeString() : ''}
        </span>
        <span className="text-[11px] text-slate-500">·</span>
        <span className="text-[11px] text-slate-500">
          click for the prompt window
        </span>
      </div>
    </button>
  );
}


/** Verdicts from ONE request, which is the unit an operator reasons about.
 *
 * ⚠ The request id is the only link between a response and its verdicts, and one prompt produces
 * one verdict PER ARMED PROBE. Listed flat, two verdicts for the same prompt look like two
 * unrelated observations — which is exactly how a firing and a non-firing verdict on one prompt
 * read before this. The prompt text itself stays out of the list by design; it is fetched per
 * event from the detail route, so the group header names the request and the rows link to it.
 */
function RequestGroup({
  requestId,
  events,
  probeNames,
  onOpen,
}: {
  requestId: string;
  events: ProbeEvent[];
  probeNames: Map<string, string>;
  onOpen: (id: number) => void;
}) {
  return (
    <div data-testid="request-group" className="border-b border-slate-800 last:border-b-0">
      <div className="flex items-baseline gap-2 px-2 pt-2">
        <span className="text-[11px] uppercase tracking-wide text-slate-500">prompt</span>
        <span
          data-testid="group-request-id"
          className="font-mono text-xs text-slate-400 truncate"
        >
          {requestId}
        </span>
        <span className="text-[11px] text-slate-500">
          {events.length === 1 ? '1 verdict' : `${events.length} verdicts`}
        </span>
      </div>
      {events.map((event) => (
        <EventRow
          key={event.id}
          event={event}
          probeName={probeNames.get(event.probe_id) ?? event.probe_id}
          onOpen={onOpen}
        />
      ))}
    </div>
  );
}


/** Group in arrival order, preserving it. A Map keeps insertion order, so the newest request
 *  stays first without sorting by a timestamp the socket payload once did not carry. */
export function groupByRequest(events: ProbeEvent[]): [string, ProbeEvent[]][] {
  const groups = new Map<string, ProbeEvent[]>();
  for (const event of events) {
    const key = event.request_id ?? '—';
    const existing = groups.get(key);
    if (existing) existing.push(event);
    else groups.set(key, [event]);
  }
  return [...groups.entries()];
}

export function ProbeMonitorsPage() {
  const {
    probes,
    status,
    events,
    importProbe,
    importing,
    arm,
    arming,
    armingId,
    checkingParityId,
    pendingAck,
    dismissAck,
    checkParity,
    disarm,
    remove,
    clearEvents,
  } = useProbes();
  const [openEventId, setOpenEventId] = useState<number | null>(null);
  // probe_id -> name, so a verdict row says WHICH probe produced it. Falls back to the id for a
  // probe that has since been deleted — an orphaned verdict is still evidence and must render.
  const probeNames = useMemo(
    () => new Map(probes.map((probe) => [probe.id, probe.name])),
    [probes],
  );
  const [showHub, setShowHub] = useState(false);
  const eventDetail = useProbeEventDetail(openEventId);

  const onFile = async (file: File) => {
    try {
      importProbe({ definition: JSON.parse(await file.text()) });
    } catch {
      // A malformed file is a user error, not a crash; the mutation's own errors are toasted.
    }
  };

  return (
    <div className="p-6 space-y-6" data-testid="probe-monitors-page">
      <header className="flex items-start gap-3">
        <Radar className="w-6 h-6 text-emerald-400" aria-hidden />
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Probe Monitors</h1>
          <p className="text-sm text-slate-400">
            Linear detectors trained in miStudio, scored on live traffic. They record and report;
            they do not stop or alter a generation.
          </p>
        </div>
      </header>

      {status && (
        <section
          data-testid="probe-status"
          className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm"
        >
          <div className="border border-slate-700 rounded p-3">
            <p className="text-slate-400 text-xs">Armed</p>
            <p className="text-slate-100 font-mono">
              {status.armed_count} / {status.max_armed}
            </p>
          </div>
          <div className="border border-slate-700 rounded p-3">
            <p className="text-slate-400 text-xs">Last request overhead</p>
            <p className="text-slate-100 font-mono">
              {/* null means nothing has been scored yet — distinct from 0 ms. */}
              {status.last_request_overhead_ms === null
                ? '—'
                : `${status.last_request_overhead_ms.toFixed(2)} ms`}
              <span className="text-slate-500 text-xs">
                {' '}
                / {status.overhead_warn_threshold_ms} ms
              </span>
            </p>
          </div>
          <div className="border border-slate-700 rounded p-3">
            <p className="text-slate-400 text-xs">Events recorded</p>
            <p className="text-slate-100 font-mono">{status.events_recorded}</p>
          </div>
          <div className="border border-slate-700 rounded p-3">
            <p className="text-slate-400 text-xs">Continuous batching</p>
            <p className="text-slate-100 font-mono">
              {status.force_serial && status.armed_count > 0 ? 'off (serial)' : 'on'}
            </p>
          </div>
        </section>
      )}

      {pendingAck && (
        <AcknowledgeDialog
          details={pendingAck.details}
          onConfirm={(reason) =>
            arm({ id: pendingAck.id, acknowledgeBelowRung2: true, reason })
          }
          onCancel={dismissAck}
          busy={arming}
        />
      )}

      {status && status.paused_reasons.length > 0 && (
        <p data-testid="paused-summary" className="text-sm text-amber-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          Armed but not scoring: {status.paused_reasons.join(', ')}
        </p>
      )}

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-slate-200 font-medium">Imported probes</h2>
          <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHub((open) => !open)}
            className="text-sm px-3 py-1.5 rounded border border-slate-600 text-slate-200 hover:bg-slate-700"
          >
            {showHub ? 'Hide Hub' : 'Browse Hub'}
          </button>
          <label className="text-sm px-3 py-1.5 rounded bg-slate-700 text-slate-100 hover:bg-slate-600 cursor-pointer flex items-center gap-2">
            <Upload className="w-4 h-4" />
            {importing ? 'Importing…' : 'Import definition'}
            <input
              type="file"
              accept="application/json,.json"
              className="hidden"
              data-testid="import-input"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void onFile(file);
              }}
            />
          </label>
          </div>
        </div>

        {showHub && (
          <ProbeHubBrowser
            onImported={() => setShowHub(false)}
            onError={() => undefined}
          />
        )}

        {probes.length === 0 ? (
          <p className="text-slate-500 text-sm">
            No probes imported. Export one from miStudio and import its
            <code className="mx-1 text-slate-400">.probe.json</code> here.
          </p>
        ) : (
          <div className="space-y-2">
            {probes.map((probe) => (
              <ProbeRow
                key={probe.id}
                probe={probe}
                onArm={(id) => arm({ id })}
                onDisarm={disarm}
                onDelete={remove}
                onCheckParity={checkParity}
                // ⚠ THIS ROW'S id, not the global pending flag.
                //
                // It was `busy={arming || checkingParity}` — one mutation flag handed to every
                // row — so arming ONE probe made all of them read "Arming…" and disabled every
                // button, then appear to fail together when the single real request errored.
                // A shared pending flag is not a per-row state.
                busy={armingId === probe.id || checkingParityId === probe.id}
              />
            ))}
          </div>
        )}
      </section>

      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-slate-200 font-medium">Recent verdicts</h2>
          <button
            onClick={() => clearEvents(undefined)}
            className="text-xs text-slate-400 hover:text-slate-200"
          >
            Clear
          </button>
        </div>
        {events.length === 0 ? (
          <p className="text-slate-500 text-sm">No verdicts yet.</p>
        ) : (
          <div className="border border-slate-800 rounded">
            {groupByRequest(events).map(([requestId, group]) => (
              <RequestGroup
                key={requestId}
                requestId={requestId}
                events={group}
                probeNames={probeNames}
                onOpen={setOpenEventId}
              />
            ))}
          </div>
        )}
      </section>

      {/* ⚠ The context window is fetched HERE, per event, on demand. The list and the socket
          carry none of it — it is the user's words. */}
      {openEventId !== null && (
        <section
          data-testid="event-detail"
          className="border border-slate-700 rounded p-3 space-y-2"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-slate-200 text-sm font-medium">Event {openEventId}</h3>
            <button
              onClick={() => setOpenEventId(null)}
              className="text-xs text-slate-400 hover:text-slate-200"
            >
              Close
            </button>
          </div>
          {eventDetail.isLoading ? (
            <p className="text-slate-500 text-sm">Loading…</p>
          ) : eventDetail.data ? (
            <>
              <p className="text-xs text-slate-400">{eventDetail.data.summary}</p>
              {eventDetail.data.context_text && (
                <pre className="text-xs text-slate-300 whitespace-pre-wrap bg-slate-900 p-2 rounded">
                  {eventDetail.data.context_text}
                </pre>
              )}
            </>
          ) : null}
        </section>
      )}
    </div>
  );
}
