// A "Help" link: opens the Help page at its topic (/help/<topic>, so the address can be bookmarked
// or shared). Beside it, "Manual" opens the topic's page in the public manual.
//
// Origin: miDataworks frontend/src/components/help/HelpLink.tsx @ 32cf406 (ENH-004). miDataworks
// remembers the topic in a store because its hash route is only #/help; here the topic is in the
// path, so the router carries it and no store is needed.
import { CircleHelp } from 'lucide-react';
import { Link } from 'react-router-dom';

import { helpPath, helpTopic } from '@/config/help';

import { ManualLink } from './ManualLink';

export function HelpLink({ topic, label = 'Help' }: { topic: string; label?: string }) {
  const manual = helpTopic(topic)?.manual;
  return (
    <span className="inline-flex items-center gap-3">
      <Link
        to={helpPath(topic)}
        data-testid={`help-link-${topic}`}
        className="inline-flex items-center gap-1 text-xs text-primary-400 hover:text-primary-300 hover:underline rounded-sm focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-primary-500"
      >
        <CircleHelp className="w-3.5 h-3.5" aria-hidden="true" />
        {label}
      </Link>
      {manual && <ManualLink page={manual} label="Manual" testId={`manual-link-${topic}`} />}
    </span>
  );
}
