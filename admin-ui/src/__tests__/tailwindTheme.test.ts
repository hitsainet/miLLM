/**
 * Tailwind CSS 4 keeps the admin UI looking as it did on Tailwind CSS 3.4 (src/index.css).
 *
 * This compiles the real src/index.css through the plugins postcss.config.js names and pins the
 * theme and every v3 behaviour the file restores, so deleting or changing one of those lines turns
 * a test red. The before/after screenshot and computed-style comparison of every page is the wider
 * check; these are the pieces a test can hold.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import postcss from 'postcss';
import { beforeAll, describe, expect, it } from 'vitest';
import { SHELL } from '../config/tokens';

const ROOT = path.resolve(__dirname, '../..');
const SRC = path.join(ROOT, 'src');

// Utilities the pins below need, generated whether or not a component uses them today.
const PROBES = [
  'stack-y-2',
  'text-sm',
  'text-xs',
  'hover:bg-slate-700',
  'dark:bg-slate-900',
  'bg-primary-400',
  'bg-purple-500',
  'bg-purple-light',
  'bg-linear-to-r/srgb',
  'font-sans',
  'font-mono',
  'animate-slide-in',
  'animate-fade-in',
  'animate-pulse-slow',
  'text-mono',
  // The two corner tokens and a side variant.
  'rounded-card',
  'rounded-control',
  'rounded-t-card',
  // The raw scale, which must generate nothing, and the static pair that stays.
  'rounded-xs',
  'rounded-sm',
  'rounded-md',
  'rounded-lg',
  'rounded-xl',
  'rounded-2xl',
  'rounded-t-lg',
  'rounded',
  'rounded-full',
  'rounded-none',
  // A family v4 added (v3 had none): with the v3 palette in place it generates nothing.
  'bg-mauve-500',
];

let css = '';
let theme: Record<string, string> = {};

beforeAll(async () => {
  const input = `@import './index.css';\n@source inline(${JSON.stringify(PROBES.join(' '))});\n`;
  // The plugins postcss.config.js names, loaded as Vite loads them, so the app's own PostCSS
  // wiring is what compiles here (an empty or v3-style config leaves the utilities ungenerated).
  const config = (await import(path.join(ROOT, 'postcss.config.js'))) as {
    default: { plugins: Record<string, object> };
  };
  const plugins = await Promise.all(
    Object.entries(config.default.plugins).map(async ([name, opts]) => {
      const mod = (await import(name)) as { default: (o: object) => postcss.AcceptedPlugin };
      return mod.default({ ...opts, base: ROOT, optimize: false });
    }),
  );
  const result = await postcss(plugins).process(input, {
    from: path.join(SRC, '__tailwindTheme.test.css'),
  });
  css = result.css;
  // The last declaration of each theme variable wins, as in the browser.
  theme = {};
  for (const m of css.matchAll(/(--[\w-]+):\s*([^;]+);/g)) theme[m[1] ?? ''] = (m[2] ?? '').trim();
}, 60_000);

/** The CSS rule bodies whose selector contains `selector`, joined. */
function rulesFor(selector: string): string {
  const out: string[] = [];
  let i = css.indexOf(selector);
  while (i !== -1) {
    const open = css.indexOf('{', i);
    const close = css.indexOf('}', open);
    out.push(css.slice(open + 1, close));
    i = css.indexOf(selector, close);
  }
  return out.join('\n');
}

describe('corner tokens (operator decision 2026-10-31)', () => {
  const px = (n: number) => `${n}px`;

  it('defines --radius-card and --radius-control as the numbers SHELL names', () => {
    expect(theme['--radius-card']).toBe(px(SHELL.cardRadiusPx));
    expect(theme['--radius-control']).toBe(px(SHELL.controlRadiusPx));
  });

  it('rounded-card / rounded-control / rounded-t-card read the tokens', () => {
    expect(rulesFor('.rounded-card {')).toContain('border-radius: var(--radius-card)');
    expect(rulesFor('.rounded-control {')).toContain('border-radius: var(--radius-control)');
    const top = rulesFor('.rounded-t-card {');
    expect(top).toContain('border-top-left-radius: var(--radius-card)');
    expect(top).toContain('border-top-right-radius: var(--radius-card)');
  });

  it("Tailwind's raw corner scale generates no CSS; rounded-full and rounded-none still do", () => {
    expect(Object.keys(theme).filter((k) => k.startsWith('--radius-')).sort()).toEqual([
      '--radius-card',
      '--radius-control',
    ]);
    for (const raw of ['xs', 'sm', 'md', 'lg', 'xl', '2xl', 't-lg'])
      expect(css, `.rounded-${raw} generated`).not.toMatch(new RegExp(`\\.rounded-${raw}\\s*\\{`));
    expect(css).not.toMatch(/\.rounded\s*\{/);
    expect(rulesFor('.rounded-full {')).toContain('border-radius');
    expect(rulesFor('.rounded-none {')).toContain('border-radius: 0');
  });

  it('the component classes take the tokens', () => {
    expect(rulesFor('.card {')).toContain('var(--radius-card)');
    expect(rulesFor('.input-base {')).toContain('var(--radius-control)');
    expect(rulesFor('.btn-danger {')).toContain('var(--radius-control)');
  });
});

describe('theme values carried from Tailwind CSS 3', () => {
  it('keeps the v3 hex palette, not v4 OKLCH', () => {
    expect(theme['--color-slate-900']).toBe('#0f172a');
    expect(theme['--color-slate-950']).toBe('#020617');
    expect(theme['--color-red-500']).toBe('#ef4444');
    expect(
      Object.entries(theme).filter(([k, v]) => k.startsWith('--color-') && v.startsWith('oklch')),
    ).toEqual([]);
    expect(css).not.toContain('.bg-mauve-500');
  });

  it('keeps the primary scale, the status colours and purple with its v3 shades', () => {
    expect(theme['--color-primary-400']).toBe('#22d3ee');
    expect(theme['--color-primary-600']).toBe('#0891b2');
    expect(theme['--color-success']).toBe('#4ade80');
    expect(theme['--color-danger-light']).toBe('rgba(239, 68, 68, 0.15)');
    expect(theme['--color-warning-border']).toBe('rgba(251, 191, 36, 0.3)');
    expect(theme['--color-purple']).toBe('#c084fc');
    expect(theme['--color-purple-light']).toBe('rgba(168, 85, 247, 0.15)');
    expect(theme['--color-purple-500']).toBe('#a855f7');
    expect(rulesFor('.bg-purple-light')).toContain('var(--color-purple-light)');
  });

  it('keeps the v3 absolute line heights for text sizes', () => {
    expect(theme['--text-sm--line-height']).toBe('1.25rem');
    expect(theme['--text-xs--line-height']).toBe('1rem');
    expect(rulesFor('.text-sm')).toContain('var(--text-sm--line-height)');
  });

  it('keeps the fonts and the three animations', () => {
    expect(theme['--font-sans']).toMatch(/^Inter, -apple-system/);
    expect(theme['--font-mono']).toMatch(/^JetBrains Mono, monospace/);
    expect(theme['--animate-slide-in']).toBe('slideIn 0.2s ease-out');
    expect(theme['--animate-fade-in']).toBe('fadeIn 0.2s ease-out');
    expect(theme['--animate-pulse-slow']).toBe('pulse 2s infinite');
    expect(css).toContain('@keyframes slideIn');
    expect(css).toContain('@keyframes fadeIn');
    expect(rulesFor('.text-mono')).toContain("font-family: 'JetBrains Mono', monospace");
  });

  it('interpolates gradients in sRGB, as v3 did', () => {
    expect(rulesFor('.bg-linear-to-r\\/srgb')).toContain('to right in srgb');
    expect(rulesFor('.btn-primary')).toContain('in srgb');
  });
});

describe('variants and Preflight carried from Tailwind CSS 3', () => {
  it('dark: follows the dark class with the v3 selector', () => {
    expect(css).toMatch(
      /\.dark\\:bg-slate-900:is\(\.dark \*\)\s*\{\s*background-color: var\(--color-slate-900\)/,
    );
    expect(css).not.toContain('prefers-color-scheme: dark');
  });

  it('hover: applies on every device, as in v3', () => {
    expect(css).toContain('hover\\:bg-slate-700');
    expect(css).not.toMatch(/@media \(hover: hover\)/);
  });

  it('restores the v3 default border colour, placeholder colour and button cursor', () => {
    expect(css).toMatch(
      /::file-selector-button\s*\{\s*border-color: var\(--color-gray-200, currentcolor\)/,
    );
    expect(theme['--color-gray-200']).toBe('#e5e7eb');
    expect(rulesFor('textarea::placeholder')).toContain('color: var(--color-gray-400)');
    expect(rulesFor("[role='button']:not(:disabled)")).toContain('cursor: pointer');
  });

  it("restores the browser's own field backgrounds and file button", () => {
    const fields = rulesFor('optgroup,');
    expect(fields).toContain('background-color: revert');
    expect(fields).toContain('border-radius: revert');
    const fileButton = rulesFor('::file-selector-button {');
    for (const declaration of [
      'padding: revert',
      'border: revert',
      'background-color: revert',
      '-webkit-appearance: button',
    ])
      expect(fileButton).toContain(declaration);
  });

  it('stack-y-* gives every child after the first a top margin (v3 space-y-*)', () => {
    expect(css).toMatch(
      /\.stack-y-2 > :not\(\[hidden\]\) ~ :not\(\[hidden\]\)\s*\{\s*margin-top: calc\(var\(--spacing\) \* 2\);\s*margin-bottom: 0;/,
    );
  });

  it('keeps the component classes in the components layer, under the utilities', () => {
    const layers = css.indexOf('@layer components');
    expect(layers).toBeGreaterThan(-1);
    expect(css.indexOf('.btn-primary')).toBeGreaterThan(layers);
    expect(rulesFor('.btn-danger {')).toContain('var(--color-danger-light)');
  });
});

describe('source', () => {
  const files: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const p = path.join(dir, name);
      if (statSync(p).isDirectory()) walk(p);
      else if (
        /\.(tsx?|css)$/.test(name) &&
        !p.endsWith('tailwindTheme.test.ts') &&
        !p.endsWith('index.css')
      )
        files.push(p);
    }
  };
  walk(SRC);

  it('uses stack-y-* rather than the v4 space-y-*/space-x-* (different selector)', () => {
    expect(files.length).toBeGreaterThan(50);
    expect(files.some((f) => readFileSync(f, 'utf8').includes('stack-y-'))).toBe(true);
    const offenders = files.filter((f) =>
      /(^|[\s'"`:])-?space-[xy]-/m.test(readFileSync(f, 'utf8')),
    );
    expect(offenders.map((f) => path.relative(ROOT, f))).toEqual([]);
  });

  it('gives every gradient the /srgb modifier', () => {
    const offenders = files.filter((f) =>
      /bg-linear-to-[a-z]+(?!\/srgb)(?![\w-])/.test(readFileSync(f, 'utf8')),
    );
    expect(offenders.map((f) => path.relative(ROOT, f))).toEqual([]);
  });
});
