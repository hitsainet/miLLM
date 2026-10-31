/**
 * CircuitActivateControl (Feature 13) — the activation gate.
 *
 * A circuit whose evidence rung is below 2 is NOT causally validated. Serving
 * one is allowed, but only deliberately: the user must tick an explicit
 * acknowledgement first, mirroring the server's UNVALIDATED_CIRCUIT refusal.
 * The rung phrase shown here is the server's `rung_language`, verbatim.
 */

import { useState } from 'react';
import { Play } from 'lucide-react';

import { Button } from '@components/common';
import type { CircuitSummary } from '@/types/circuits';

interface CircuitActivateControlProps {
  circuit: CircuitSummary;
  onActivate: (acknowledgeUnvalidated: boolean) => void;
  isActivating?: boolean;
}

export function CircuitActivateControl({
  circuit,
  onActivate,
  isActivating = false,
}: CircuitActivateControlProps) {
  const [acknowledged, setAcknowledged] = useState(false);

  // A validated circuit (rung >= 2) activates directly.
  if (circuit.validated) {
    return (
      <Button
        variant="primary"
        size="sm"
        onClick={() => onActivate(false)}
        disabled={isActivating}
        data-testid="activate-button"
      >
        <Play className="w-4 h-4 mr-1" />
        Activate
      </Button>
    );
  }

  return (
    <div className="flex items-center gap-2 flex-wrap justify-end">
      <label
        className="flex items-center gap-1.5 text-xs text-yellow-200/90 cursor-pointer"
        data-testid="unvalidated-ack-label"
      >
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(e) => setAcknowledged(e.target.checked)}
          data-testid="unvalidated-ack"
          className="accent-yellow-500"
        />
        This circuit is {circuit.rung_language} — steer anyway
      </label>
      <Button
        variant="primary"
        size="sm"
        onClick={() => onActivate(true)}
        disabled={!acknowledged || isActivating}
        data-testid="activate-button"
      >
        <Play className="w-4 h-4 mr-1" />
        Activate
      </Button>
    </div>
  );
}
