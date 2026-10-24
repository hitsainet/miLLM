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
  const number = Number(text);
  if (text === '' || !Number.isFinite(number)) return { error: 'Enter a number' };
  if (row.type === 'int' && !Number.isInteger(number)) return { error: 'Enter a whole number' };
  return { value: number };
}

