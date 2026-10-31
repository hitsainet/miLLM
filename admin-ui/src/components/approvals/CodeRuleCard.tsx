/**
 * The code-rule card (ENH-008 D4) on an operator-code import: each code document's hash, its
 * functions, the runner contract, and the FULL source, as text. miLLM keeps no per-hash approval
 * or revoked state; the request's facts say so ("Revocation").
 */
import type { CodeModuleCard } from '@/types/approvals';

import { asText } from './approvalWords';

export function CodeRuleCard({ modules }: { modules: unknown[] }) {
  return (
    <section className="mt-3 rounded-card border border-slate-700/60 bg-slate-950/60 p-3" aria-label="Operator code">
      <h4 className="font-semibold text-slate-200">Operator code in this import</h4>
      {(modules as Partial<CodeModuleCard>[]).map((m, i) => (
        <div key={`${asText(m.code_sha256)}-${i}`} className="mt-2">
          <dl className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-[minmax(8rem,auto)_1fr]">
            <dt className="text-slate-500">code_sha256</dt>
            <dd className="break-all font-mono text-xs">{asText(m.code_sha256)}</dd>
            <dt className="text-slate-500">Functions</dt>
            <dd className="break-words">{Array.isArray(m.functions) ? m.functions.map(asText).join(', ') : asText(m.functions)}</dd>
            <dt className="text-slate-500">Runner contract</dt>
            <dd className="break-words">{asText(m.runner_contract)}</dd>
          </dl>
          <details open className="mt-2">
            <summary className="cursor-pointer text-amber-300">Source (read it before approving)</summary>
            <pre className="mt-1 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-control bg-slate-900 p-2 font-mono text-xs text-slate-200">
              {asText(m.source)}
            </pre>
          </details>
        </div>
      ))}
    </section>
  );
}

export default CodeRuleCard;
