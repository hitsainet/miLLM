// The Help page (ENH-004): a short, plain-language summary of each screen, with a link to the
// full page of the public manual. /help/<topic> opens it at one topic, so a topic can be
// bookmarked or linked from chat or the manual; the header's Help link on every screen and the
// in-page links on Steering and Probe Monitors land here.
//
// Origin: miDataworks frontend/src/components/help/HelpPanel.tsx @ 32cf406. The page title is the
// header's <h1> ("Help"), as on every other miLLM page.
import { useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';

import { ManualLink } from '@/components/help';
import { HELP_TOPICS, helpPath, helpTopic } from '@/config/help';

export function HelpPage() {
  const { topic: requested } = useParams<{ topic?: string }>();
  const topic = helpTopic(requested)?.id;

  useEffect(() => {
    if (topic) document.getElementById(`help-${topic}`)?.scrollIntoView?.({ block: 'start' });
  }, [topic]);

  return (
    <div className="stack-y-6" data-testid="help-page">
      <p className="text-sm text-slate-400">
        Short summaries of each screen. The <ManualLink page="" label="miLLM manual" testId="manual-home" /> explains
        every screen in full.
      </p>
      {requested && !topic && (
        <p role="status" className="text-sm text-amber-400">
          There is no Help topic called “{requested}”. Every topic is listed below.
        </p>
      )}
      <nav aria-label="Help topics" className="flex flex-wrap gap-2 text-xs">
        {HELP_TOPICS.map((t) => (
          <Link
            key={t.id}
            to={helpPath(t.id)}
            aria-current={t.id === topic ? 'true' : undefined}
            className={`rounded-full border px-3 py-1 transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-primary-500 ${
              t.id === topic
                ? 'border-primary-500/60 bg-primary-500/10 text-primary-300'
                : 'border-slate-700 text-slate-300 hover:bg-slate-800'
            }`}
          >
            {t.title}
          </Link>
        ))}
      </nav>
      <div className="stack-y-4 max-w-3xl">
        {HELP_TOPICS.map((t) => (
          <section
            key={t.id}
            id={`help-${t.id}`}
            aria-labelledby={`help-${t.id}-title`}
            data-current={t.id === topic ? 'true' : undefined}
            className={`rounded-card border p-5 bg-slate-800/50 scroll-mt-20 ${
              t.id === topic ? 'border-primary-500/60' : 'border-slate-700/50'
            }`}
          >
            <h2 id={`help-${t.id}-title`} className="font-semibold text-slate-100 mb-2">
              {t.title}
            </h2>
            {t.body.map((p, i) => (
              <p key={i} className="text-sm text-slate-300 mb-2">
                {p}
              </p>
            ))}
            {t.steps && (
              <ol className="list-decimal pl-5 text-sm text-slate-300 stack-y-1">
                {t.steps.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ol>
            )}
            <div className="mt-3">
              <ManualLink page={t.manual} label="Read more in the manual" testId={`help-manual-${t.id}`} />
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

export default HelpPage;
