// A link into the public miLLM manual (config/manual.ts). Opens in a new tab: the manual is a
// separate site, and leaving the admin page mid-task would lose the user's place.
//
// Origin: miDataworks frontend/src/components/help/ManualLink.tsx @ 32cf406 (ENH-004).
import { BookOpen } from 'lucide-react';

import { manualUrl } from '@/config/manual';

export function ManualLink({ page, label, testId }: { page: string; label: string; testId?: string }) {
  return (
    <a
      href={manualUrl(page)}
      target="_blank"
      rel="noopener noreferrer"
      data-testid={testId}
      className="inline-flex items-center gap-1 text-xs text-primary-400 hover:text-primary-300 hover:underline rounded-control focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-primary-500"
    >
      <BookOpen className="w-3.5 h-3.5" aria-hidden="true" />
      {label}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}
