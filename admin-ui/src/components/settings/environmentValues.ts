/** Feature 35: how the Environment section reads and parses values (kept apart from the component
 * so the component file exports components only). */

import type { EnvironmentRow } from '@/types/environment';

/** A value as the operator reads it. `null` is "not set", never an empty cell. */
export function showValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'not set';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/** The typed draft as the value `PUT` sends, or why it cannot be one. */
export function parseDraft(row: EnvironmentRow, draft: string): { value: unknown } | { error: string } {
  const text = draft.trim();
  if (row.type === 'choice') return text ? { value: text } : { error: 'Choose a value' };
  if (row.type === 'bool') {
    if (text === 'true' || text === 'false') return { value: text === 'true' };
    return { error: 'Choose true or false' };
  }
  // ENH-006 D75: text as typed (trimmed); the server checks it and refuses with the rule.
  if (row.type === 'text') return { value: text };
  const number = Number(text);
  if (text === '' || !Number.isFinite(number)) return { error: 'Enter a number' };
  if (row.type === 'int' && !Number.isInteger(number)) return { error: 'Enter a whole number' };
  return { value: number };
}


/** ENH-006 D78: what the Telemetry group's cards change, and what stays in the environment. */
export const TELEMETRY_NOTE =
  'Saving an export setting drains and restarts the export pipeline; nothing leaves until a ' +
  'switch is on and a receiver endpoint is set. Receiver credentials (TELEMETRY_HEADERS) and ' +
  'certificate files stay in the environment. A change an AI agent asks for waits on the ' +
  'Approvals page.';
