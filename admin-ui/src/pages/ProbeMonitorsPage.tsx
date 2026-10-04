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

import { useCallback, useMemo, useState } from 'react';
import { AlertTriangle, Radar, Trash2, Upload } from 'lucide-react';
import type { ProbeAckDetails } from '@/hooks/useProbes';
import { ProbeHubBrowser } from '@/components/probes/ProbeHubBrowser';
import { useProbeEventDetail, useProbes } from '@/hooks/useProbes';
import type { Probe, ProbeEvent } from '@/types/probe';
import { groupByRequest } from '@/utils/probeEvents';
import { ARM_WITHOUT_ACK_MIN_RUNG } from '@/types/probe';
import { labelSeparation } from '@/utils/probeLabels';

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
  const { passed, max_abs_diff, max_gated_diff, score_tolerance } = probe.parity;
  // ⚠ SHOW THE PAIR THAT DECIDED IT. This read `max_abs_diff` against `tolerance` — both
  // PER-TOKEN, both flagged informational by the engine — and rendered
  // "parity passed · max Δ 9.37e+0 of 0.05": a figure 187x its stated limit sitting beside the
  // word "passed". A reader then either distrusts a correct pass or believes a drift that
  // decided nothing. This estate has already shipped a parity check that told a correct
  // consumer it was wrong, and it was believed the first time.
  //
  // `passed` comes from the aggregate score difference against `score_tolerance`. The per-token
  // maximum is still worth showing — one token drifting far is worth knowing — but only when it
  // is labelled as what it is.
  const decided =
    max_gated_diff != null && score_tolerance != null
      ? `${max_gated_diff.toFixed(3)} / ${score_tolerance.toFixed(3)}`
      : null;
  return (
    <span
      data-testid="parity"
      className={`text-xs ${passed ? 'text-emerald-400' : 'text-rose-400'}`}
    >
      parity {passed ? 'passed' : 'FAILED'}
      {decided !== null && ` · ${decided}`}
      {max_abs_diff !== null &&
        ` · per-token max Δ ${max_abs_diff.toPrecision(3)} (informational)`}
    </span>
  );
}

/** The windows a probe can be asked to report, in the order they read. */
const WINDOWS = ['last_user', 'prompt', 'response', 'all'] as const;

/** Why each window is worth having, said once, where the operator chooses. */
const WINDOW_HELP: Record<string, string> = {
  all: "the whole request — what this did before windows existed. A long reply drags the mean down, so the same conversation scores differently depending on how much the model said.",
  last_user: "the newest user message only — its header, text and end-of-turn. Earlier turns, a system prompt and retrieved documents are excluded, so a high-stakes message earlier in a chat does not keep firing on every later one. The question \"is THIS message high-stakes?\".",
  prompt: "everything before the model's reply — system text, every earlier turn and the newest message. Independent of how much the model then said, but a high-stakes earlier turn keeps firing on every later one.",
  response: "the model's own output. ⚠ UNTRAINED — miStudio's training corpus is prose wrapped as a single user turn, so these weights never saw a model reply. Its threshold may be cut from reply negatives, but the readout itself is unvalidated, so it is always reported as provisional.",
};

/** What a probe's contract scope covers, in words — the same sentence miStudio's tile uses. */
const SCOPE_WORDS: Record<string, string> = {
  all: 'every token (prompt and reply)',
  prompt: "the person's side of the request",
  response: "the model's reply",
};

const WINDOW_ORDER = ['last_user', 'prompt', 'response', 'all'] as const;

/** How a window is named where a person reads it; `last_user` is an identifier, not a label. */
const WINDOW_LABELS: Record<string, string> = { last_user: 'last user turn' };

/**
 * WHICH TOKENS: what the probe was fitted on, and its own bar over each window.
 *
 * ⚠ A miStudio run's probes are layers x rules, each fitted on one scope with a bar per window.
 * Without this a list of them read as "one per window", and the bar each window is judged against
 * was invisible. `response` is marked provisional unless the probe was fitted on replies — the
 * same rule the runtime applies (`window_weights_trained`).
 */
export function ProbeWindowBars({ probe }: { probe: Probe }) {
  const bars = probe.window_thresholds ?? {};
  const windows = WINDOW_ORDER.filter((w) => w in bars);
  return (
    <div className="text-xs text-slate-400 mt-0.5" data-testid="probe-windows">
      <span data-testid="probe-scope">
        fitted on <span className="text-slate-300">{SCOPE_WORDS[probe.scope] ?? probe.scope}</span>
      </span>
      <div className="flex flex-wrap items-center gap-1.5 mt-1">
        {windows.length === 0 ? (
          <span
            className="px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-300 text-[10px]"
            data-testid="window-single-bar"
            title="No per-window bar was placed, so every window is judged against the single threshold."
          >
            one bar for every window
          </span>
        ) : (
          windows.map((window) => {
            const untrained = window === 'response' && probe.scope !== 'response';
            return (
              <span
                key={window}
                data-testid={`window-${window}`}
                className={`px-1.5 py-0.5 rounded text-[10px] ${
                  untrained ? 'bg-amber-900/40 text-amber-200' : 'bg-slate-700/60 text-slate-200'
                }`}
                title={
                  untrained
                    ? "This bar was cut from reply negatives, but the probe's weights never saw a model reply — verdicts here are provisional: a ranking, not a measured rate."
                    : `This window's own bar, cut from calibration negatives read over the ${window} window.`
                }
              >
                {WINDOW_LABELS[window] ?? window} ≥ {bars[window].toFixed(2)}
                {untrained ? ' · provisional' : ''}
              </span>
            );
          })
        )}
        {probe.length_band_count ? (
          <span
            className="text-[10px] text-slate-500"
            data-testid="window-length-bands"
            title="The bar over the probe's own scope also varies with how many tokens were scored. Only that window uses the bands."
          >
            + {probe.length_band_count} length bands on {probe.scope}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function ProbeRow({
  probe,
  onArm,
  onDisarm,
  onDelete,
  onCheckParity,
  liveWindows,
  liveBar,
  busy,
}: {
  probe: Probe;
  onArm: (id: string, windows: string[]) => void;
  onDisarm: (id: string) => void;
  onDelete: (id: string) => void;
  onCheckParity: (id: string) => void;
  /** Windows this probe is LIVE on, from the registry. `undefined` when nothing reports it. */
  liveWindows?: string[] | null;
  /**
   * The bar actually in force and the sentence the server writes when it is not the row's.
   *
   * ⚠ THE ONLY PLACE A FAILED REGISTRY REFRESH BECOMES VISIBLE. A threshold can now be re-cut on
   * an ARMED probe, and the write order is row-then-registry: if the refresh does not land, this
   * tile's `probe.threshold` (the ROW) shows the new number while every verdict is still judged
   * against the old one. `status()` reads the registry and reports the disagreement rather than
   * reconciling it; this renders what it says.
   */
  liveBar?: { threshold: number | null; revision: number | null; disagreement: string | null };
  busy: boolean;
}) {
  // Defaults to every window. A probe armed without a thought still gets both halves, which is
  // the point — the capability is useless if it only works for someone who remembers it exists.
  const [windows, setWindows] = useState<string[]>([...WINDOWS]);
  const separation = labelSeparation(probe.label_mapping);
  // The full mapping in the tooltip, so the caption's formatting hides nothing — including an
  // `excluded` label, which is on neither side of the boundary but is still an operator decision.
  const mappingDetail = Object.entries(probe.label_mapping ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, side]) => `${label} → ${side}`)
    .join(', ');
  const toggle = (name: string) =>
    setWindows((current) =>
      current.includes(name) ? current.filter((w) => w !== name) : [...current, name]
    );
  return (
    <div
      data-testid="probe-row"
      className="border border-slate-700 rounded p-3 flex flex-col gap-2 bg-slate-800/40"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-slate-100 font-medium truncate">{probe.name}</p>
          <p className="font-mono text-xs text-slate-400" data-testid="probe-readout">
            {probe.hf_id} · L{probe.layer} · {probe.rule}
            {/* The rolling window is part of the rule: two `rolling_mean_max` probes at one layer
                differ ONLY here, and without it their tiles read identically (2026-10-04). */}
            {typeof probe.rule_params?.window === 'number' ? ` w=${probe.rule_params.window}` : ''} ·{' '}
            {probe.basis === 'sae_features' ? 'SAE basis' : 'dense residual'}
          </p>
          <ProbeWindowBars probe={probe} />
          {/* ⚠ WHAT IT WAS FITTED ON. The line above is the READ POINT; it says nothing about
              the concept, and two probes here differed only in their run. A probe is a boundary
              between two sets of labelled rows, so both sides are named — the same positive
              label fitted against a different negative one is a different detector. The strings
              are the corpus's own; `labelSeparation` returns null rather than print half. */}
          {separation ? (
            <p
              className="text-xs text-slate-400 truncate"
              data-testid="probe-labels"
              title={
                'The training view\'s label mapping, as miStudio recorded it: ' +
                mappingDetail +
                '. The probe is a boundary between these sets and nothing else — it says nothing about any label it was never shown.'
              }
            >
              trained on <span className="font-mono text-slate-300">{separation}</span>
            </p>
          ) : null}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {probe.armed ? (
            <span
              data-testid="armed-chip"
              className="px-2 py-0.5 rounded text-xs bg-emerald-900/50 text-emerald-300"
            >
              armed
              {/* ⚠ WHICH WINDOWS IT IS ACTUALLY SCORING, from the live registry. The picker
                  below shows an INTENTION that resets on re-render; this is state. Arming took
                  a window choice and nothing reported it back until 2026-10-01. */}
              {liveWindows && liveWindows.length > 0 && (
                <span data-testid="armed-windows" className="ml-1 font-mono text-[10px] opacity-80">
                  · {liveWindows.join(' ')}
                </span>
              )}
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
            <>
            <span data-testid="window-picker" className="flex items-center gap-1">
              {WINDOWS.map((name) => {
                const on = windows.includes(name);
                return (
                  <button
                    key={name}
                    type="button"
                    aria-pressed={on}
                    onClick={() => toggle(name)}
                    title={WINDOW_HELP[name]}
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                      on
                        ? 'border-emerald-600 bg-emerald-900/40 text-emerald-200'
                        : 'border-slate-700 text-slate-500'
                    }`}
                  >
                    {WINDOW_LABELS[name] ?? name}
                  </button>
                );
              })}
            </span>
            <button
              onClick={() => onArm(probe.id, windows)}
              disabled={busy || windows.length === 0}
              title={windows.length === 0 ? 'Choose at least one window to report' : undefined}
              className="text-xs px-2 py-1 rounded bg-emerald-700 text-emerald-50 hover:bg-emerald-600 disabled:opacity-50"
            >
              {busy ? 'Arming…' : 'Arm'}
            </button>
            </>
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
        <span className="text-xs text-slate-400 font-mono" data-testid="probe-threshold">
          {probe.threshold === null
            ? 'no threshold — ranks, does not decide'
            : `threshold ${probe.threshold.toFixed(4)}`}
          {/* The cut, shown once a bar has ever moved. Silent at revision 1, because a number
              nobody has re-cut needs no version beside it. */}
          <span className="ml-1 opacity-70" data-testid="probe-load-dtype">
            · fitted at {probe.load_dtype ?? 'unrecorded precision'}
          </span>
          {probe.threshold_revision !== undefined && probe.threshold_revision > 1 && (
            <span className="ml-1 opacity-70">· rev {probe.threshold_revision}</span>
          )}
        </span>
        {!probe.streamable && (
          <span className="text-xs text-slate-500" title="its rule is only defined once generation ends">
            not streamable
          </span>
        )}
      </div>

      {/* ⚠ THE BAR IN FORCE DISAGREES WITH THE STORED ONE. Separate from `paused_reason` on
          purpose: this probe IS scoring, just against a previous cut, and folding the two
          together would dilute the field whose whole job is "a probe never goes silently
          quiet". The sentence is the server's — it names both revisions. */}
      {liveBar?.disagreement && (
        <p
          data-testid="threshold-disagreement"
          className="text-xs text-amber-300 flex items-center gap-1"
        >
          <AlertTriangle className="w-3 h-3 shrink-0" /> {liveBar.disagreement}
        </p>
      )}

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


/** One event's decoded prompt window, fetched on demand and rendered in place.
 *
 * ⚠ A SEPARATE COMPONENT SO THE HOOK IS UNCONDITIONAL. `useProbeEventDetail` cannot be called
 * from a loop or behind an `if` in the parent, so each expanded row mounts its own fetcher. That
 * is also what makes several windows open at once: each holds its own query, keyed by its own id.
 *
 * The list and the socket carry no prompt text — this route is the only one that serves it, and
 * fetching per event on expand is the point of that design, not a workaround.
 */
function EventContext({ eventId }: { eventId: number }) {
  const detail = useProbeEventDetail(eventId);

  if (detail.isLoading) {
    return <p className="text-xs text-slate-500 px-2 pb-2">Loading the prompt window…</p>;
  }
  if (detail.isError) {
    return (
      <p className="text-xs text-amber-300 px-2 pb-2">
        Could not load this event&rsquo;s prompt window.
      </p>
    );
  }
  const data = detail.data;
  if (!data) return null;

  return (
    <div data-testid="event-context" className="px-2 pb-2 space-y-1">
      {data.summary && <p className="text-xs text-slate-400">{data.summary}</p>}
      {data.context_text ? (
        <pre
          data-testid="event-context-text"
          className="text-xs text-slate-300 whitespace-pre-wrap break-words bg-slate-900 p-2 rounded max-h-64 overflow-y-auto"
        >
          {data.context_text}
        </pre>
      ) : (
        /* ⚠ Says WHY it is empty. A blank panel here is indistinguishable from the defect where
           nothing captured the window at all — which is what was reported. Events recorded before
           the capture shipped have no text and never will, so the absence is stated. */
        <p data-testid="event-context-absent" className="text-xs text-slate-500">
          No prompt window was recorded for this event.
        </p>
      )}
    </div>
  );
}


function EventRow({
  event,
  probeName,
  currentRevisions,
  expanded,
  onToggle,
}: {
  event: ProbeEvent;
  probeName: string;
  /** probe_id -> the revision its ROW is at now. Absent for an orphaned verdict, which is why
   *  the chip is OMITTED rather than defaulted there: a marker about a bar nobody can look up
   *  would warn about evidence it cannot classify. */
  currentRevisions: Record<string, number>;
  expanded: boolean;
  onToggle: (id: number) => void;
}) {
  return (
    <div className="border-b border-slate-800 last:border-b-0">
    <button
      data-testid="event-row"
      aria-expanded={expanded}
      onClick={() => onToggle(event.id)}
      className="w-full text-left py-2 px-2 hover:bg-slate-800/50"
    >
      <div className="flex items-center justify-between gap-3">
        {/* ⚠ WHICH PROBE SAID THIS. Two probes armed on one layer produce two verdicts per
            request — often one firing and one not — and the row used to show neither name nor
            time, so the pair was indistinguishable. */}
        <span data-testid="event-probe-name" className="text-xs text-slate-300 truncate">
          {probeName}
          {/* ⚠ WHICH SLICE THIS READ. One probe now reports several windows, so the probe's
              name no longer identifies a row — two rows in one request group share it. */}
          {event.window && (
            <span
              data-testid="event-window"
              className="ml-2 px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono text-[10px]"
            >
              {event.window}
            </span>
          )}
          {/* ⚠ FIRED AGAINST A BAR NEVER CUT FOR THIS WINDOW. The operator chose alerts from
              uncalibrated windows over silence; this marker is the condition that choice was
              made under, so it is never hidden behind a hover or a detail view. */}
          {event.provisional && (
            <span
              data-testid="event-provisional"
              title="Provisional: either this window has no threshold of its own, or the probe's weights were never trained on what it reads (a model reply). A ranking, not a rate."
              className="ml-2 px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-300 text-[10px]"
            >
              provisional
            </span>
          )}
          {/* ⚠ WHICH CUT OF THE BAR JUDGED THIS, shown only when it is not the one in force.
              A threshold can now be re-cut on an armed probe, so a list can hold verdicts
              judged under different bars — and a reader watching `provisional` disappear
              between two rows cannot otherwise tell whether a window gained its own bar or
              never needed one. Rendered on the same terms as `provisional`: never behind a
              hover, with the prose in `title`. */}
          {event.threshold_revision !== undefined &&
            currentRevisions[event.probe_id] !== undefined &&
            event.threshold_revision !== currentRevisions[event.probe_id] && (
              <span
                data-testid="event-threshold-revision"
                title={
                  `This verdict was judged against revision ${event.threshold_revision} of the ` +
                  `probe's threshold. The bar has since moved to revision ` +
                  `${currentRevisions[event.probe_id]}, so its score is not comparable to a ` +
                  `newer verdict by the fires/silent outcome alone.`
                }
                className="ml-2 px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono text-[10px]"
              >
                rev {event.threshold_revision}
              </span>
            )}
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
            {/* ⚠ THE BAR IT WAS JUDGED AGAINST, BESIDE THE SCORE. Two events with the same score
                and opposite verdicts were previously indistinguishable on this row, and the
                number was already on the wire — a threshold that can now MOVE makes that gap
                a lie rather than merely an omission. */}
            {event.threshold !== null && event.threshold !== undefined
              ? ` / ${event.threshold.toFixed(4)}`
              : ''}
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
          {expanded ? 'hide the prompt window' : 'show the prompt window'}
        </span>
      </div>
    </button>
    {/* Beneath THIS row, so the window sits with the verdict it belongs to. Several rows can be
        open at once — each is its own independent toggle. */}
    {expanded && <EventContext eventId={event.id} />}
    </div>
  );
}


/** Verdicts from ONE request, which is the unit an operator reasons about.
 *
 * ⚠ The request id is the only link between a response and its verdicts, and one prompt produces
 * one verdict PER ARMED PROBE **PER WINDOW** — a probe armed on all three reports three, and they
 * share a name. The window chip on each row is what tells them apart. Listed flat, verdicts for
 * the same prompt look like
 * unrelated observations — which is exactly how a firing and a non-firing verdict on one prompt
 * read before this. The prompt text itself stays out of the list by design; it is fetched per
 * event from the detail route, so the group header names the request and the rows link to it.
 */
function RequestGroup({
  requestId,
  events,
  probeNames,
  currentRevisions,
  expandedIds,
  onToggle,
}: {
  requestId: string;
  events: ProbeEvent[];
  probeNames: Map<string, string>;
  currentRevisions: Record<string, number>;
  expandedIds: ReadonlySet<number>;
  onToggle: (id: number) => void;
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
          currentRevisions={currentRevisions}
          expanded={expandedIds.has(event.id)}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
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
  // ⚠ A SET, not a single id. One window per event, several open at once — a single
  // `openEventId` could only ever show one, and it showed it in a panel detached from the row
  // it described.
  const [expandedIds, setExpandedIds] = useState<ReadonlySet<number>>(() => new Set());
  const toggleEvent = useCallback((id: number) => {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  // probe_id -> name, so a verdict row says WHICH probe produced it. Falls back to the id for a
  // probe that has since been deleted — an orphaned verdict is still evidence and must render.
  const probeNames = useMemo(
    () => new Map(probes.map((probe) => [probe.id, probe.name])),
    [probes],
  );
  // probe_id -> the revision its row is at now, so a verdict judged against an EARLIER bar can
  // say so. Built from the same list as `probeNames`, and deliberately a plain object so an
  // unknown probe reads as `undefined` rather than as revision 1.
  const currentRevisions = useMemo(
    () =>
      Object.fromEntries(
        probes
          .filter((probe) => probe.threshold_revision !== undefined)
          .map((probe) => [probe.id, probe.threshold_revision as number]),
      ),
    [probes],
  );
  const [showHub, setShowHub] = useState(false);

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
              {/* ⚠ The TOTAL is shown against the pass count, never against a budget: it scales
                  with the answer's length, so pairing it with a threshold invited reading a long
                  reply as a fault. The RATE below is the figure that is actually judged. */}
              {(status.last_request_n_passes ?? 0) > 0 && (
                <span className="text-slate-500 text-xs">
                  {' '}
                  over {status.last_request_n_passes} pass
                  {status.last_request_n_passes === 1 ? '' : 'es'}
                </span>
              )}
            </p>
            {status.last_overhead_ms_per_pass != null && (   // != covers undefined too
              <p
                className={
                  status.overhead_warn_threshold_ms_per_pass != null &&
                  status.last_overhead_ms_per_pass > status.overhead_warn_threshold_ms_per_pass
                    ? 'text-amber-400 font-mono text-xs mt-1'
                    : 'text-slate-400 font-mono text-xs mt-1'
                }
              >
                {status.last_overhead_ms_per_pass.toFixed(3)} /
                {status.overhead_warn_threshold_ms_per_pass ?? '?'} ms per pass
              </p>
            )}
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
            arm({
              id: pendingAck.id,
              acknowledgeBelowRung2: true,
              reason,
              // The choice made before the refusal, not a silent reset to the default.
              windows: pendingAck.windows,
            })
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
                onArm={(id, windows) => arm({ id, windows })}
                liveWindows={status?.armed.find((a) => a.id === probe.id)?.windows}
                liveBar={(() => {
                  const entry = status?.armed.find((a) => a.id === probe.id);
                  return entry
                    ? {
                        threshold: entry.threshold ?? null,
                        revision: entry.threshold_revision ?? null,
                        disagreement: entry.threshold_disagreement ?? null,
                      }
                    : undefined;
                })()}
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
                currentRevisions={currentRevisions}
                expandedIds={expandedIds}
                onToggle={toggleEvent}
              />
            ))}
          </div>
        )}
      </section>

    </div>
  );
}
