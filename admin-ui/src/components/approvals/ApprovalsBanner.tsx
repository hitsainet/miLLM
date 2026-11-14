/**
 * The banner on every page (ENH-008 D5, BUILD_SPEC §2.3), mounted once in `Layout` between the
 * header and the page. While any request waits it shows up to three cards (the soonest to expire
 * first), "and N more waiting — Open Approvals" beyond that, and a Collapse / Expand toggle kept in
 * sessionStorage (a NEW request expands it again). It draws nothing when nothing waits and the last
 * load succeeded; a failed load says so. It is not drawn on the Approvals page. With two or more
 * settings changes (or code rules) waiting it also offers "Approve all settings changes (N)" /
 * "Approve all code rules (N)" (ENH-006 D79, D81).
 */
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { useApprovalsStore } from '@/stores/approvalsStore';

import { ApprovalCard } from './ApprovalCard';
import { ApproveAllButtons } from './ApproveAllBatch';
import { WORDS } from './approvalWords';

export const BANNER_MAX_CARDS = 3;
/**
 * The banner floats (operator 2026-10-28): the window scrolls, and the header is sticky at top-0
 * (h-16, z-30), so the dock sticks just under it at top-16, above page content (z-20) and under
 * the header, the sidebar (z-40) and dialogs (z-50). Opaque, because the region's tint is not.
 */
export const BANNER_DOCK_CLASS = 'sticky top-16 z-20 bg-slate-950 px-6 pt-4';
/** The region caps its height and scrolls inside, so it never hides the whole page. */
const REGION_CAP = 'max-h-[40vh] overflow-y-auto';

function Dock({ children }: { children: ReactNode }) {
  return (
    <div className={BANNER_DOCK_CLASS} data-testid="approvals-banner-dock">
      {children}
    </div>
  );
}

const COLLAPSE_KEY = 'millm.approvals.banner.collapsed';

function readCollapsed(): boolean {
  try {
    return window.sessionStorage.getItem(COLLAPSE_KEY) === '1';
  } catch {
    return false;
  }
}

function writeCollapsed(value: boolean): void {
  try {
    if (value) window.sessionStorage.setItem(COLLAPSE_KEY, '1');
    else window.sessionStorage.removeItem(COLLAPSE_KEY);
  } catch {
    /* storage blocked: the toggle still works for this render */
  }
}

export function ApprovalsBanner({ onApprovalsPage }: { onApprovalsPage: boolean }) {
  const waiting = useApprovalsStore((s) => s.waiting);
  const error = useApprovalsStore((s) => s.error);
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const seen = useRef<Set<string> | null>(null);

  // A request not seen before expands the banner again.
  useEffect(() => {
    const ids = new Set(waiting.map((a) => a.approval_id));
    if (seen.current !== null && [...ids].some((id) => !seen.current!.has(id))) {
      setCollapsed(false);
      writeCollapsed(false);
    }
    seen.current = ids;
  }, [waiting]);

  if (onApprovalsPage) return null;
  if (waiting.length === 0) {
    if (!error) return null;
    return (
      <Dock>
        <div
          role="region"
          aria-label={WORDS.bannerRegion}
          data-testid="approvals-banner"
          className={`${REGION_CAP} rounded-card border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200`}
        >
          Could not read the approval queue: {error}
        </div>
      </Dock>
    );
  }

  const shown = waiting.slice(0, BANNER_MAX_CARDS);
  const more = waiting.length - shown.length;
  const toggle = () => {
    setCollapsed(!collapsed);
    writeCollapsed(!collapsed);
  };

  return (
    <Dock>
      <div
        role="region"
        aria-label={WORDS.bannerRegion}
        data-testid="approvals-banner"
        className={`${REGION_CAP} rounded-card border border-amber-500/40 bg-amber-500/10 p-3`}
      >
        <div className="flex flex-wrap items-center gap-3">
          <h2 aria-live="polite" className="text-sm font-semibold text-amber-200">
            {WORDS.bannerHeading(waiting.length)}
          </h2>
          <Link to="/approvals" className="text-sm text-amber-300 underline hover:text-amber-200">
            {WORDS.bannerLink}
          </Link>
          <button
            type="button"
            onClick={toggle}
            aria-expanded={!collapsed}
            className="ml-auto rounded-control px-2 py-0.5 text-xs text-amber-300 hover:bg-amber-500/20"
          >
            {collapsed ? 'Expand' : 'Collapse'}
          </button>
        </div>
        <div className="mt-2">
          <ApproveAllButtons waiting={waiting} where="banner" />
        </div>
        {!collapsed && (
          <div className="mt-3 stack-y-3">
            {shown.map((a) => (
              <ApprovalCard key={a.approval_id} approval={a} />
            ))}
            {more > 0 && (
              <p className="text-sm text-amber-200">
                <Link to="/approvals" className="underline hover:text-amber-100">
                  {WORDS.bannerMore(more)}
                </Link>
              </p>
            )}
          </div>
        )}
      </div>
    </Dock>
  );
}

export default ApprovalsBanner;
