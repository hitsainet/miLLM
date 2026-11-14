/**
 * The old-address notice (040 FR-040.31; TID §6), mounted once in `Layout` between the header and
 * the page. Drawn only when this page was loaded from one of the server's transition hosts: it
 * says the address is retiring (with the date once the server has one), names the new address the
 * server pairs with it, and says that preferences saved in this browser stay with the old address
 * (browser storage is per address; the keys themselves are unchanged, DR-25).
 *
 * Dismissal is kept in sessionStorage, so it comes back in a new session until the address goes.
 */
import { useState } from 'react';
import { X } from 'lucide-react';

import { useLegacyHost } from '@/hooks/useLegacyHost';

export const LEGACY_NOTICE_DISMISSED_KEY = 'miserver.legacyNotice.dismissed';

function readDismissed(): boolean {
  try {
    return window.sessionStorage.getItem(LEGACY_NOTICE_DISMISSED_KEY) === '1';
  } catch {
    return false;
  }
}

function writeDismissed(): void {
  try {
    window.sessionStorage.setItem(LEGACY_NOTICE_DISMISSED_KEY, '1');
  } catch {
    /* storage blocked: the notice still hides for this render */
  }
}

/** The same page on the new host: scheme, port, path and query kept. */
function sameLocationOn(newHost: string): string {
  const url = new URL(window.location.href);
  url.hostname = newHost;
  return url.toString();
}

export function LegacyAddressNotice({ hostname = window.location.hostname }: { hostname?: string }) {
  const { hosts, retireOn, newHostFor } = useLegacyHost();
  const [dismissed, setDismissed] = useState(readDismissed);

  if (dismissed || !hosts.includes(hostname)) return null;
  const newHost = newHostFor(hostname);

  return (
    <div
      role="status"
      data-testid="legacy-address-notice"
      className="mx-6 mt-4 flex items-start justify-between gap-3 rounded-control border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200"
    >
      <p>
        This address is retiring{retireOn ? ` on ${retireOn}` : ''}.{' '}
        {newHost && (
          <>
            Use{' '}
            <a href={sameLocationOn(newHost)} className="font-mono underline hover:text-amber-100">
              {newHost}
            </a>
            .{' '}
          </>
        )}
        Preferences saved in this browser do not carry over.
      </p>
      <button
        type="button"
        aria-label="Dismiss the old-address notice"
        className="shrink-0 text-amber-300 hover:text-amber-100"
        onClick={() => {
          writeDismissed();
          setDismissed(true);
        }}
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

export default LegacyAddressNotice;
