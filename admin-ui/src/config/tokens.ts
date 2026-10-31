// Shell measurements that a test compares to the compiled theme (src/index.css `@theme`).
//
// Corners are two named sizes in every miSuite app (operator decision 2026-10-31): a card radius
// for cards, panels, dialogs, callouts and list rows, and a control radius for inputs, buttons,
// menu items, tabs and small inline boxes. They are plain px numbers so a test and a document can
// quote them. To change the look, edit `--radius-card` / `--radius-control` in src/index.css and
// these two numbers together; src/__tests__/tailwindTheme.test.ts fails while they disagree.
// Origin: miForge frontend/src/config/tokens.ts (SHELL.cardRadiusPx / controlRadiusPx).

export const SHELL = {
  /** `rounded-card` (and `rounded-t-card` etc.). */
  cardRadiusPx: 12,
  /** `rounded-control`: inputs, buttons, menu items, tabs, code spans. */
  controlRadiusPx: 8,
} as const;
