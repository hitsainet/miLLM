/**
 * Corners are two named sizes (operator decision 2026-10-31): `rounded-card` (--radius-card) and
 * `rounded-control` (--radius-control), plus `rounded-full` for pills, dots and bars and
 * `rounded-none` where a corner is deliberately square. Any other corner class (rounded-sm,
 * rounded-lg, rounded-xl, bare `rounded`, an arbitrary `rounded-[10px]`, a class assembled as
 * `rounded-${size}`) would take a size the two tokens do not control, so the look could no longer
 * be changed by editing two numbers in src/index.css.
 *
 * This reads every source file from disk (tests excluded) and scans its text, so a class in a
 * className string, a template literal, an array, a cva/clsx call or an @apply line is caught alike.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = path.resolve(__dirname, '../..');
const SRC = path.join(ROOT, 'src');

const SIDES = '(?:t|r|b|l|s|e|tl|tr|br|bl|ss|se|ee|es)';
const ALLOWED = new RegExp(`^rounded(?:-${SIDES})?-(?:card|control|full|none)!?$`);
const ALLOWED_SET = 'rounded[-side]-(card|control|full|none)';

/** Every `rounded…` token: from `rounded` to the next character that cannot be part of a class. */
const TOKEN = /(?<![\w-])!?rounded[^\s'"`;,)}<>]*/g;

interface Offence {
  line: number;
  token: string;
}

/** The corner classes in `text` outside the allowed set, and any raw radius declaration. */
function cornerOffences(text: string, fileName: string): Offence[] {
  const out: Offence[] = [];
  text.split('\n').forEach((line, i) => {
    for (const m of line.matchAll(TOKEN)) {
      const token = m[0].replace(/^!/, '');
      if (!ALLOWED.test(token)) out.push({ line: i + 1, token });
    }
    // A radius written past the utilities: an inline style, or a CSS declaration that is neither
    // a token nor the deliberate `revert` of the browser's own field and file-button corners.
    if (/\bborderRadius\b/.test(line)) out.push({ line: i + 1, token: 'borderRadius' });
    if (fileName.endsWith('.css')) {
      const decl = /border(?:-[a-z]+)*-radius\s*:\s*([^;]+)/.exec(line);
      if (decl && !/^(revert|var\(--radius-(card|control)\))\s*$/.test((decl[1] ?? '').trim()))
        out.push({ line: i + 1, token: decl[0] });
    }
  });
  return out;
}

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) {
      if (name === '__tests__' || name === 'test' || name === 'node_modules') continue;
      sourceFiles(p, out);
    } else if (/\.(tsx?|jsx?|css)$/.test(name) && !/\.(test|spec)\.[jt]sx?$/.test(name)) {
      out.push(p);
    }
  }
  return out;
}

describe('corner classes', () => {
  it('the scanner catches classes in strings, template literals, arrays and @apply', () => {
    const planted = [
      'className="p-2 rounded-lg"',
      'className={`p-2 ${on ? "rounded-sm" : ""}`}',
      "const c = ['border', 'rounded-xl'];",
      'className={`rounded-${size}`}',
      'className="rounded"',
      'className="rounded-[10px]"',
      'className="md:rounded-t-lg hover:rounded-md"',
      '@apply bg-slate-900 rounded-2xl;',
      'style={{ borderRadius: 4 }}',
    ].join('\n');
    expect(cornerOffences(planted, 'x.tsx').map((o) => o.token)).toEqual([
      'rounded-lg',
      'rounded-sm',
      'rounded-xl',
      'rounded-${size',
      'rounded',
      'rounded-[10px]',
      'rounded-t-lg',
      'rounded-md',
      'rounded-2xl',
      'borderRadius',
    ]);
    expect(cornerOffences('.x { border-top-left-radius: 4px; }', 'x.css')).toHaveLength(1);
  });

  it('the scanner lets the allowed set through', () => {
    const allowed =
      'rounded-card rounded-control rounded-full rounded-none rounded-t-card md:rounded-none ' +
      'file:rounded-control rounded-bl-control !rounded-card rounded-card! ' +
      '--radius-card: 12px; border-radius: revert; border-radius: var(--radius-card);';
    expect(cornerOffences(allowed, 'x.css')).toEqual([]);
  });

  it(`uses only ${ALLOWED_SET} in src/ (tests excluded)`, () => {
    const files = sourceFiles(SRC);
    // The walk reaches the app: the shell, a page and the stylesheet.
    const rel = files.map((f) => path.relative(ROOT, f));
    expect(rel).toEqual(
      expect.arrayContaining([
        'src/index.css',
        'src/components/common/Card.tsx',
        'src/pages/ModelsPage.tsx',
      ]),
    );
    const offences = files.flatMap((f) =>
      cornerOffences(readFileSync(f, 'utf8'), f).map(
        (o) => `${path.relative(ROOT, f)}:${o.line}  ${o.token}`,
      ),
    );
    expect(
      offences,
      `Corner classes outside ${ALLOWED_SET}. Use rounded-card for cards, panels, dialogs, ` +
        'callouts and list rows; rounded-control for inputs, buttons, menu items, tabs and small ' +
        'inline boxes; rounded-full for pills, dots and bars (src/index.css @theme).',
    ).toEqual([]);
    // And the allowed classes are what the app actually uses.
    const text = files.map((f) => readFileSync(f, 'utf8')).join('\n');
    expect(text).toContain('rounded-card');
    expect(text).toContain('rounded-control');
  });
});
