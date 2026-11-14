import { describe, expect, it } from 'vitest';
import { eventTime } from '@/utils/eventTime';

describe('eventTime', () => {
  it('gives the date and the time in the requested locale, and the exact instant', () => {
    const when = eventTime('2026-10-28T13:05:09.250Z', 'en-GB');
    expect(when).not.toBeNull();
    expect(when!.iso).toBe('2026-10-28T13:05:09.250Z');
    expect(when!.label).toMatch(/2026/);
    expect(when!.label).toMatch(/Oct/);
    expect(when!.label).toMatch(/\d{2}:\d{2}:\d{2}/);
  });

  it('localises: two locales write the same instant differently', () => {
    const a = eventTime('2026-10-28T13:05:09Z', 'en-US')!.label;
    const b = eventTime('2026-10-28T13:05:09Z', 'de-DE')!.label;
    expect(a).not.toEqual(b);
  });

  it.each([null, undefined, '', 'garbage', '2026-13-45T99:99:99Z'])('is null for %j', (value) => {
    expect(eventTime(value as string | null | undefined)).toBeNull();
  });

  it('keeps an offset instant exact in UTC', () => {
    expect(eventTime('2026-10-28T09:00:00-04:00')!.iso).toBe('2026-10-28T13:00:00.000Z');
  });
});
