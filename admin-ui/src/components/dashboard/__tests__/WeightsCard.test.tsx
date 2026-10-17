/**
 * The Dashboard "Weights" card (Feature 31 FTASKS 8.3, Feature 33 status block).
 *
 * The tests worth keeping are about what the card must NOT say: a torn set has to read as
 * "serving refused" in red, an unrecorded step count must never read 0, and a server that does
 * not serve the routes yet must read "not available" rather than an error.
 *
 * MUTATION CONTROLS (each turns this file red):
 * - drop `details: 'serving refused'` from the torn mapping;
 * - render `steps ?? 0` instead of "not recorded";
 * - spread the payload instead of naming fields → the attachment-id test fails.
 */

import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';

import { WeightsCard } from '../WeightsCard';
import type { WeightsStatusView } from '@hooks/useWeightsStatus';
import type { AdapterStatus, WeightsStatus } from '@/types';

const NO_ADAPTER: AdapterStatus = {
  applied: false,
  name: null,
  base_name: null,
  manifest_sha256: null,
  manifest_base_name: null,
  manifest_base_sha256: null,
  weights_sha256: null,
  rank: null,
  target_count: null,
  loaded_at: null,
  base_steps_since_load: null,
};

const STATUS: WeightsStatus = {
  attached: false,
  attachment: null,
  barrier: 'serving',
  quiesced_at: null,
  held_requests: null,
  weights_state: 'clean',
  resident_model: { model_id: 3, name: 'llama-3.1-8b', manifest_hash: 'abc' },
  steps_since_load: 0,
  last_save: null,
  commit: null,
  commit_overran: false,
  last_hold_timeout_at: null,
  last_overrun: null,
  adapter: NO_ADAPTER,
};

function view(
  status: Partial<WeightsStatus> | 'absent' = {},
  adapter: Partial<AdapterStatus> | 'absent' = {},
  extra: Partial<WeightsStatusView> = {},
): WeightsStatusView {
  return {
    status: status === 'absent' ? { available: false } : { available: true, data: { ...STATUS, ...status } },
    adapter: adapter === 'absent' ? { available: false } : { available: true, data: { ...NO_ADAPTER, ...adapter } },
    statusLoading: false,
    adapterLoading: false,
    statusError: null,
    adapterError: null,
    ...extra,
  };
}

function card() {
  return screen.getByTestId('weights-card');
}

/** The StatusCard's left border carries the colour. */
function border() {
  return card().querySelector('.border-l-4') as HTMLElement;
}

function row(label: string) {
  const dt = within(card()).getByText(label);
  return dt.nextElementSibling?.textContent;
}

describe('WeightsCard', () => {
  it('clean reads "Clean", neutral', () => {
    render(<WeightsCard {...view()} />);
    expect(within(card()).getByText('Clean')).toBeInTheDocument();
    expect(border().className).toContain('border-l-slate-500');
    expect(within(card()).queryByText('serving refused')).toBeNull();
  });

  it('writing reads "Writing", amber', () => {
    render(<WeightsCard {...view({ weights_state: 'writing' })} />);
    expect(within(card()).getByText('Writing')).toBeInTheDocument();
    expect(border().className).toContain('border-l-yellow-500');
    expect(within(card()).getByText('Writing').className).toContain('text-yellow-400');
  });

  it('torn reads "serving refused", red', () => {
    render(<WeightsCard {...view({ weights_state: 'torn' })} />);
    expect(within(card()).getByText('Torn')).toBeInTheDocument();
    expect(within(card()).getByText('serving refused')).toBeInTheDocument();
    expect(border().className).toContain('border-l-red-500');
    expect(within(card()).getByText('Torn').className).toContain('text-red-400');
    expect(card()).toHaveAttribute('aria-label', expect.stringContaining('serving refused'));
  });

  it('attachment: none, label and role, stale', () => {
    const { unmount } = render(<WeightsCard {...view()} />);
    expect(row('Attachment')).toBe('none');
    unmount();

    const attachment = {
      state: 'active' as const,
      role: 'trainer',
      label: 'miforge',
      attached_at: '2026-10-14T12:00:00+00:00',
      last_heartbeat_at: null,
      stale_since: null,
    };
    const second = render(<WeightsCard {...view({ attached: true, attachment })} />);
    expect(row('Attachment')).toBe('miforge (trainer)');
    expect(card()).toHaveAttribute('aria-label', 'Weights: clean; holder: miforge (trainer)');
    second.unmount();

    render(
      <WeightsCard
        {...view({ attached: true, attachment: { ...attachment, state: 'stale', stale_since: '2026-10-14T12:05:00+00:00' } })}
      />,
    );
    expect(row('Attachment')).toBe('miforge (trainer), stale');
  });

  it('steps since load: null reads "not recorded", never 0', () => {
    const { unmount } = render(<WeightsCard {...view({ steps_since_load: null })} />);
    expect(row('Steps since load')).toBe('not recorded');
    expect(within(card()).queryByText('0')).toBeNull();
    unmount();

    render(<WeightsCard {...view({ steps_since_load: 7 })} />);
    expect(row('Steps since load')).toBe('7');
  });

  it('last save: none, or the checkpoint hash prefix and time', () => {
    const { unmount } = render(<WeightsCard {...view()} />);
    expect(row('Last save')).toBe('none');
    unmount();

    render(
      <WeightsCard
        {...view({
          last_save: { checkpoint_sha256: '0123456789abcdef0123', path: '/data/checkpoints/x', at: '2026-10-14T12:00:00+00:00' },
        })}
      />,
    );
    expect(row('Last save')).toBe('0123456789ab at 2026-10-14T12:00:00+00:00');
  });

  it('adapter: none, or name and rank', () => {
    const { unmount } = render(<WeightsCard {...view()} />);
    expect(row('Adapter')).toBe('none');
    unmount();

    render(<WeightsCard {...view({}, { applied: true, name: 'support-tone', rank: 16, target_count: 64 })} />);
    expect(row('Adapter')).toBe('support-tone (rank 16)');
  });

  it('a route that is not served reads "not available", not an error', () => {
    render(<WeightsCard {...view('absent', 'absent')} />);
    expect(within(card()).getByText('Not available')).toBeInTheDocument();
    expect(border().className).toContain('border-l-slate-500');
    expect(border().className).not.toContain('border-l-red-500');
    expect(row('Adapter')).toBe('not available');
    expect(card()).toHaveAttribute('aria-label', 'Weights: not available');
  });

  it('the adapter can be absent while the status is served', () => {
    render(<WeightsCard {...view({ weights_state: 'writing' }, 'absent')} />);
    expect(within(card()).getByText('Writing')).toBeInTheDocument();
    expect(row('Adapter')).toBe('not available');
  });

  it('a failed read (not a 404) says it could not be read', () => {
    render(<WeightsCard {...view({}, {}, { status: undefined, statusError: new Error('boom') })} />);
    expect(within(card()).getByText('Unknown')).toBeInTheDocument();
    expect(within(card()).getByText('status could not be read')).toBeInTheDocument();
  });

  it('renders no button and no attachment ID', () => {
    const smuggled = {
      ...STATUS,
      attached: true,
      attachment: {
        state: 'active' as const,
        role: 'trainer',
        label: 'miforge',
        attached_at: '2026-10-14T12:00:00+00:00',
        last_heartbeat_at: null,
        stale_since: null,
        attachment_id: 'att_SECRET_123',
      },
      attachment_id: 'att_SECRET_123',
    } as WeightsStatus;
    render(<WeightsCard {...view()} status={{ available: true, data: smuggled }} />);
    expect(within(card()).queryByRole('button')).toBeNull();
    expect(card().outerHTML).not.toContain('att_SECRET_123');
  });
});
