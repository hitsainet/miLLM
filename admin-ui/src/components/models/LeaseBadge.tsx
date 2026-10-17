import { useEffect, useState } from 'react';
import { KeyRound } from 'lucide-react';
import type { LeaseSummary } from '@/types';
import { formatRemaining } from './leaseTime';

/** Milliseconds until `expiresAt`, by this browser's clock. */
function remainingMs(expiresAt: string, now: number): number {
  return new Date(expiresAt).getTime() - now;
}

export interface LeaseBadgeProps {
  lease: LeaseSummary;
}

/**
 * Who has pinned this model, why, and for how long (Feature 29, FR-29.5.2).
 *
 * Amber, with a key icon and a text label, so it never reads as the yellow steering lock
 * ("Locked for steering") beside it: the two guards are different things. Display only — there
 * is no lease action anywhere in the Admin UI (T-86). The remaining time is recomputed every
 * second from `expires_at`, and the badge hides itself the moment the lease expires, without
 * waiting for the next poll (FR-29.5.4).
 */
export function LeaseBadge({ lease }: LeaseBadgeProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const ms = remainingMs(lease.expires_at, now);
  if (!(ms > 0)) return null;

  const remaining = formatRemaining(ms);
  const exact = new Date(lease.expires_at).toLocaleString();
  return (
    <span
      className="inline-flex items-center gap-1 text-xs text-amber-400"
      title={`Leased by ${lease.holder}: ${lease.reason}. Expires ${exact}.`}
      aria-label={`Leased by ${lease.holder}, expires in ${remaining}`}
      data-testid="lease-badge"
    >
      <KeyRound className="w-3.5 h-3.5" aria-hidden="true" />
      <span>
        Leased by {lease.holder} · expires in {remaining}
      </span>
      <span className="sr-only">Reason: {lease.reason}</span>
    </span>
  );
}
