/**
 * ContentionDialog (Feature 19) — the informed refusal.
 *
 * Shown when activation is refused because another circuit already holds one
 * of the layers. It is NOT a generic error dialog: the refusal carries the
 * measurement behind it, and showing that measurement is a binding condition
 * of retaining the override at all (BR-011 / contention model §6.2).
 *
 * Two shapes, and the difference is load-bearing:
 *
 *  - CONTENTION (`overridable: true`) — the layer is shared but the features
 *    are distinct. Composition is additive and a coherent thing to intend, so
 *    "Compose anyway" is offered, next to what it costs.
 *
 *  - COLLISION (`overridable: false`) — both circuits steer the SAME
 *    (layer, feature). One strength would silently overwrite the other and the
 *    served value would belong to neither author. There is no honest
 *    composition of that case, so NO compose action is rendered at all.
 */

import { AlertTriangle, Layers, X } from 'lucide-react';

import { Button } from '@components/common';

export interface ContentionDetails {
  contended_layers: number[];
  incumbent: { id: string | null; name: string | null };
  requested?: { id: string | null; name: string | null };
  override_param?: string;
  rung_header_suppressed_if_overridden?: boolean;
  overridable: boolean;
  colliding_keys: Array<{ layer: number; feature_idx: number; incumbent: string }>;
  measured_hazard: {
    source: string;
    one_layer_at_strength_5: string;
    two_layers_at_strength_5: string;
    note: string;
  };
}

interface ContentionDialogProps {
  details: ContentionDetails;
  message: string;
  onDeactivateIncumbent?: () => void;
  onComposeAnyway?: () => void;
  onCancel: () => void;
  isBusy?: boolean;
}

export function ContentionDialog({
  details,
  message,
  onDeactivateIncumbent,
  onComposeAnyway,
  onCancel,
  isBusy = false,
}: ContentionDialogProps) {
  const incumbentName = details.incumbent?.name ?? 'another active circuit';
  const hazard = details.measured_hazard;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      data-testid="contention-dialog"
    >
      <div className="w-full max-w-lg rounded-lg border border-slate-700 bg-slate-900 shadow-xl">
        <div className="flex items-start justify-between border-b border-slate-800 p-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-400" />
            <h2 className="text-sm font-semibold text-slate-100">
              {details.overridable
                ? 'These layers are already being steered'
                : 'These circuits steer the same features'}
            </h2>
          </div>
          <button
            onClick={onCancel}
            className="text-slate-400 hover:text-slate-200"
            aria-label="Close"
            data-testid="contention-cancel-x"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-3 p-4 text-sm text-slate-300">
          <p data-testid="contention-message">{message}</p>

          <div className="flex items-center gap-2 text-xs">
            <Layers className="h-3.5 w-3.5 text-slate-400" />
            <span data-testid="contended-layers">
              Layers {details.contended_layers.join(', ')} — held by{' '}
              <strong className="text-slate-100">{incumbentName}</strong>
            </span>
          </div>

          {!details.overridable && details.colliding_keys.length > 0 && (
            <div
              className="rounded border border-red-900/60 bg-red-950/30 p-3 text-xs"
              data-testid="colliding-keys"
            >
              <p className="mb-1 font-medium text-red-200">
                Both circuits steer these exact features:
              </p>
              <ul className="space-y-0.5 text-red-100/90">
                {details.colliding_keys.map((k) => (
                  <li key={`${k.layer}-${k.feature_idx}`}>
                    L{k.layer} · feature {k.feature_idx}
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-red-200/80">
                One strength would silently overwrite the other, so the served
                value would belong to neither author. This cannot be composed —
                edit one circuit&apos;s members.
              </p>
            </div>
          )}

          {/* The measurement travels WITH the refusal. An operator who
              overrides has been told what happened last time — including that
              it is one model and one fixture, stated as part of the data
              rather than as a footnote. */}
          {details.overridable && (
            <div
              className="rounded border border-amber-900/60 bg-amber-950/20 p-3 text-xs"
              data-testid="measured-hazard"
            >
              <p className="mb-1 font-medium text-amber-200">
                What was measured
              </p>
              <ul className="space-y-0.5 text-amber-100/90">
                <li>1 steered layer at strength 5: {hazard.one_layer_at_strength_5}</li>
                <li>
                  <strong>2 steered layers</strong> at strength 5:{' '}
                  {hazard.two_layers_at_strength_5}
                </li>
              </ul>
              <p className="mt-2 text-amber-200/70">
                {hazard.source} — {hazard.note}
              </p>
            </div>
          )}

          {details.overridable && details.rung_header_suppressed_if_overridden && (
            <p className="text-xs text-slate-400" data-testid="rung-suppression-note">
              While any layer is composed the circuit-rung header is omitted,
              because no single circuit&apos;s evidence describes the response.
            </p>
          )}
        </div>

        <div className="flex flex-wrap justify-end gap-2 border-t border-slate-800 p-4">
          <Button variant="secondary" size="sm" onClick={onCancel} disabled={isBusy}>
            Cancel
          </Button>
          {onDeactivateIncumbent && details.incumbent?.id && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onDeactivateIncumbent}
              disabled={isBusy}
              data-testid="deactivate-incumbent"
            >
              Deactivate &lsquo;{incumbentName}&rsquo;
            </Button>
          )}
          {/* NO compose action on a collision — offering one would invite an
              override that cannot be honest. */}
          {details.overridable && onComposeAnyway && (
            <Button
              variant="danger"
              size="sm"
              onClick={onComposeAnyway}
              disabled={isBusy}
              data-testid="compose-anyway"
            >
              Compose anyway
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
