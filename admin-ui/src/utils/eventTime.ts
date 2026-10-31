/**
 * A recorded instant for a result tile: the DATE and the time, in the viewer's locale, with the
 * exact instant (ISO 8601, UTC) for a tooltip and a `<time dateTime>`.
 *
 * ⚠ A tile showed only `toLocaleTimeString()`, so results recorded on different days read as
 * the same moment (operator report, 2026-10-28). And a missing or unparseable value must render
 * NOTHING — `new Date('garbage').toLocaleString()` is the string "Invalid Date".
 */
export interface EventTime {
  /** Localised date and time, e.g. "28 Oct 2026, 14:03:07". */
  label: string;
  /** The exact instant, ISO 8601 in UTC with milliseconds. */
  iso: string;
}

export function eventTime(value: string | number | null | undefined, locale?: string): EventTime | null {
  if (value === null || value === undefined || value === '') return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return {
    label: date.toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'medium' }),
    iso: date.toISOString(),
  };
}

/**
 * ENH-008 D8b (shared by the suite's four apps): an approvals surface shows a moment in the
 * BROWSER'S time zone with its abbreviation, and the exact instant (ISO 8601, UTC) as the tooltip.
 *
 * ⚠ The build spec named `{dateStyle: 'medium', timeStyle: 'short', timeZoneName: 'short'}`, but
 * ECMA-402 refuses `timeZoneName` beside `dateStyle`/`timeStyle` (a TypeError, "Invalid option").
 * The explicit fields below are what that pair expands to, plus the zone — the same as miForge's
 * `formatLocal` (`components/common/time.ts`). A missing or unparseable value is "unknown", never
 * "Invalid Date", and has no tooltip.
 */
export interface LocalTime {
  readonly text: string;
  /** The exact instant, ISO 8601 in UTC; '' when the value could not be read. */
  readonly title: string;
}

export const UNKNOWN_TIME = 'unknown';

function parse(value: string | null | undefined): Date | null {
  if (value === null || value === undefined || value === '') return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatLocal(iso: string | null | undefined, timeZone?: string): LocalTime {
  const date = parse(iso);
  if (date === null) return { text: UNKNOWN_TIME, title: '' };
  const format = new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
    ...(timeZone === undefined ? {} : { timeZone }),
  });
  return { text: format.format(date), title: date.toISOString() };
}

/** "expires in 23 h", "expires in 40 min", "expires in less than 1 min"; "expired" once past;
 *  "never expires" for no value. */
export function formatExpiresIn(iso: string | null | undefined, nowMs: number = Date.now()): string {
  if (iso === null || iso === undefined || iso === '') return 'never expires';
  const date = parse(iso);
  if (date === null) return `expires ${UNKNOWN_TIME}`;
  const seconds = Math.round((date.getTime() - nowMs) / 1000);
  if (seconds <= 0) return 'expired';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 1) return 'expires in less than 1 min';
  if (minutes < 60) return `expires in ${minutes} min`;
  return `expires in ${Math.round(minutes / 60)} h`;
}
