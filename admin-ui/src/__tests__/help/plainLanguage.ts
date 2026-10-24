/**
 * The plain-language rules for Help text (ENH-004, D8: Help topics in v1).
 *
 * Origin: the `internal` rules are miForge's manual/publish-rules.json (`planning-tree`,
 * `requirement-id`, `decision-id`) @ 36c1e50, copied verbatim so the suite shares one list;
 * plainLanguage.test.ts compares them with miForge's file when that sibling is checked out. The
 * `millm-*` rules add the id shapes miLLM's own history uses (FR-25.15, Feature 24, D8, D-47,
 * F20, R3-20), which miForge's patterns do not match.
 *
 * Two checks:
 *   1. no internal id or planning word, which means nothing to someone using the page;
 *   2. every acronym is spelled out in parentheses at its FIRST use in a topic, either
 *      "sparse autoencoder (SAE)" or "SAE (sparse autoencoder)". The expansions are seeded from
 *      the manual's vocabulary (concepts/interpretability#vocabulary); an acronym not listed here
 *      fails until someone adds its expansion, so a new one cannot slip through unexplained.
 */

export interface PlainRule {
  id: string;
  pattern: string;
  flags: string;
  why: string;
  /** A string the rule must catch: the guard's own negative control. */
  sample: string;
}

export const INTERNAL_RULES: PlainRule[] = [
  // ── verbatim from miForge manual/publish-rules.json (category "internal") ──
  {
    id: 'planning-tree',
    pattern: '0xcc|\\bAGENT\\.md\\b|\\bCLAUDE\\.md\\b|\\bGEMINI\\.md\\b',
    flags: '',
    why: 'Internal planning documents and agent memory, which the mirror strips.',
    sample: 'see 0xcc/tasks for the plan',
  },
  {
    id: 'requirement-id',
    pattern: '\\bR-0\\d\\.\\d|\\bBRD-|\\b(?:PPRD|PADR|FPRD|FTDD|FTID|FTASKS)\\b|\\bADR-\\d|\\bMIS-E2E|\\bFR-0\\d\\d\\.',
    flags: '',
    why: 'Internal requirement and review identifiers mean nothing to a reader of the manual.',
    sample: 'as FR-009.1 requires',
  },
  {
    id: 'decision-id',
    pattern: "\\(D\\d{1,2}\\)|\\bfeature 0\\d\\d\\b|\\bF\\d{1,2}'s\\b",
    flags: '',
    why: 'Internal decision and feature numbers.',
    sample: 'the name was chosen (D7)',
  },
  // ── miLLM's own id shapes ──
  {
    id: 'millm-requirement-id',
    pattern: '\\bFR-\\d+(?:\\.\\d+)*|\\bENH-\\d+|\\bT-\\d+\\b',
    flags: '',
    why: "miLLM's requirement and enhancement numbers (FR-25.15, ENH-004).",
    sample: 'float32 scoring, FR-25.15',
  },
  {
    id: 'millm-feature-number',
    pattern: '\\bFeature \\d{1,3}\\b|\\bF\\d{1,2}\\b',
    flags: '',
    why: 'Feature numbers from the planning documents (Feature 24, F20).',
    sample: 'until Feature 24 added probes',
  },
  {
    id: 'millm-decision-or-review-id',
    pattern: '\\bD-?\\d{1,2}\\b|\\bR\\d-\\d+\\b',
    flags: '',
    why: 'Decision and review-finding numbers (D8, D-47, R3-20).',
    sample: 'renamed by D8',
  },
];

/** Acronym -> its expansion, as the manual's vocabulary words it. */
export const EXPANSIONS: Record<string, string> = {
  SAE: 'sparse autoencoder',
  GPU: 'graphics processing unit',
  API: 'application programming interface',
  LLM: 'large language model',
  MCP: 'Model Context Protocol',
};

/** Capitalised words that are names, not acronyms a reader needs spelled out. None today:
 *  product names (miLLM, miStudio, OpenAI, Open WebUI) do not match the acronym pattern. */
export const NOT_ACRONYMS = new Set<string>([]);

export interface TopicText {
  id: string;
  title: string;
  body: string[];
  steps?: string[];
}

function escape(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Every plain-language problem in one topic, as readable messages. */
export function plainLanguageProblems(topic: TopicText): string[] {
  const parts = [topic.title, ...topic.body, ...(topic.steps ?? [])];
  const problems: string[] = [];

  for (const part of parts) {
    for (const rule of INTERNAL_RULES) {
      const hit = part.match(new RegExp(rule.pattern, rule.flags));
      if (hit) problems.push(`${topic.id}: internal id "${hit[0]}" (${rule.id}: ${rule.why})`);
    }
  }

  const text = parts.join('\n');
  const seen = new Set<string>();
  for (const m of text.matchAll(/\b([A-Z]{2,})(s?)\b/g)) {
    const acronym = m[1];
    if (seen.has(acronym) || NOT_ACRONYMS.has(acronym)) continue;
    seen.add(acronym);
    const expansion = EXPANSIONS[acronym];
    if (!expansion) {
      problems.push(`${topic.id}: "${acronym}" has no expansion in EXPANSIONS; spell it out or add one`);
      continue;
    }
    const at = m.index ?? 0;
    const word = m[0];
    const before = text.slice(0, at);
    const after = text.slice(at + word.length);
    const exp = escape(expansion);
    // "sparse autoencoder (SAE)" / "Sparse autoencoders (SAEs)"
    const spelledBefore =
      before.endsWith('(') && after.startsWith(')') && new RegExp(`${exp}s?\\s*\\($`, 'i').test(before);
    // "SAE (sparse autoencoder)"
    const spelledAfter = new RegExp(`^\\s*\\(${exp}s?\\)`, 'i').test(after);
    if (!spelledBefore && !spelledAfter) {
      problems.push(`${topic.id}: first use of "${acronym}" is not spelled out ("${expansion} (${acronym})")`);
    }
  }
  return problems;
}
