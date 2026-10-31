/**
 * ENH-008 D8b / A5-L13: the approvals time format. ⚠ The spec's literal option set
 * (`dateStyle` + `timeStyle` + `timeZoneName`) THROWS a TypeError in every engine; this pins that
 * the format used does not, names the zone, keeps the exact instant as an ISO tooltip, and says
 * "unknown" — never "Invalid Date" — for a bad value.
 */
import { describe, expect, it } from 'vitest';

import { formatExpiresIn, formatLocal, UNKNOWN_TIME } from '@/utils/eventTime';

describe('formatLocal', () => {
  it('never throws, and names the zone', () => {
    expect(() => formatLocal('2026-10-28T18:00:00Z')).not.toThrow();
    expect(formatLocal('2026-10-28T18:00:00Z', 'America/New_York').text).toMatch(/Oct 28, 2026.*2:00\s?PM.*(EDT|GMT-4)/);
    expect(formatLocal('2026-10-28T18:00:00Z', 'UTC').text).toMatch(/6:00\s?PM.*UTC/);
  });

  it('the spec literal really is refused, which is why the fields are explicit', () => {
    expect(() => new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium', timeStyle: 'short', timeZoneName: 'short',
    } as Intl.DateTimeFormatOptions)).toThrow(TypeError);
  });

  it('keeps the exact instant as the title, in UTC', () => {
    expect(formatLocal('2026-10-28T14:00:00-04:00').title).toBe('2026-10-28T18:00:00.000Z');
  });

  it.each([null, undefined, '', 'garbage'])('says "unknown" for %s, with no title', (value) => {
    expect(formatLocal(value as string | null)).toEqual({ text: UNKNOWN_TIME, title: '' });
  });
});

describe('formatExpiresIn', () => {
  const now = Date.parse('2026-10-28T18:00:00Z');
  it.each([
    ['2026-10-29T17:00:00Z', 'expires in 23 h'],
    ['2026-10-28T18:40:00Z', 'expires in 40 min'],
    ['2026-10-28T18:00:30Z', 'expires in less than 1 min'],
    ['2026-10-28T17:00:00Z', 'expired'],
    [null, 'never expires'],
    ['garbage', 'expires unknown'],
  ])('%s -> %s', (iso, words) => {
    expect(formatExpiresIn(iso as string | null, now)).toBe(words);
  });
});
