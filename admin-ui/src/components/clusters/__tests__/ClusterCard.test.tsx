/**
 * Tests for ClusterCard (Feature 8): badges, unbound gating, warnings,
 * intensity dial commit semantics, narrative toggle.
 */

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ClusterCard } from '../ClusterCard';
import type { ClusterSummary } from '@/types/clusters';

function makeCluster(overrides: Partial<ClusterSummary> = {}): ClusterSummary {
  return {
    id: 'prof_c1',
    name: 'fear cluster',
    description: 'Steers toward fear.',
    is_active: false,
    intensity: 1.0,
    sensing_enabled: false,
    member_count: 3,
    display_token: 'fear',
    bound: true,
    warnings: [],
    created_at: '2026-07-16T00:00:00Z',
    updated_at: '2026-07-16T00:00:00Z',
    ...overrides,
  };
}

const noop = () => {};

function renderCard(cluster: ClusterSummary, handlers: Partial<Record<string, () => void>> = {}) {
  const onSetIntensity = vi.fn();
  render(
    <ClusterCard
      cluster={cluster}
      onActivate={handlers.onActivate ?? noop}
      onDeactivate={handlers.onDeactivate ?? noop}
      onSetIntensity={onSetIntensity}
      onExport={handlers.onExport ?? noop}
    />
  );
  return { onSetIntensity };
}

describe('ClusterCard', () => {
  it('renders identity, token badge, and member count', () => {
    renderCard(makeCluster());
    expect(screen.getByText('fear cluster')).toBeInTheDocument();
    expect(screen.getByText('fear')).toBeInTheDocument();
    expect(screen.getByText(/3 members/)).toBeInTheDocument();
  });

  it('disables Activate for unbound clusters with a helpful title', () => {
    renderCard(makeCluster({ bound: false }));
    const btn = screen.getByRole('button', { name: /Activate/ });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('title', expect.stringContaining('Unbound'));
    expect(screen.getByText('unbound')).toBeInTheDocument();
  });

  it('shows Deactivate for the active cluster', () => {
    renderCard(makeCluster({ is_active: true }));
    expect(screen.getByRole('button', { name: /Deactivate/ })).toBeInTheDocument();
    expect(screen.getByText('active')).toBeInTheDocument();
  });

  it('surfaces import warnings', () => {
    renderCard(makeCluster({ warnings: ['Layer mismatch: definition L12, attached L6'] }));
    expect(screen.getByText(/Layer mismatch/)).toBeInTheDocument();
  });

  it('commits the intensity only on release (not every drag tick)', () => {
    const { onSetIntensity } = renderCard(makeCluster());
    const slider = screen.getByLabelText('Intensity for fear cluster');
    fireEvent.change(slider, { target: { value: '1.5' } });
    expect(onSetIntensity).not.toHaveBeenCalled();
    fireEvent.mouseUp(slider);
    expect(onSetIntensity).toHaveBeenCalledWith(1.5);
  });

  it('does not commit when released at the original value', () => {
    const { onSetIntensity } = renderCard(makeCluster({ intensity: 1.0 }));
    const slider = screen.getByLabelText('Intensity for fear cluster');
    fireEvent.mouseUp(slider);
    expect(onSetIntensity).not.toHaveBeenCalled();
  });

  it('toggles the narrative', () => {
    renderCard(makeCluster());
    expect(screen.queryByText('Steers toward fear.')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Narrative'));
    expect(screen.getByText('Steers toward fear.')).toBeInTheDocument();
  });

  it('shows hub provenance badge and repo ref', () => {
    renderCard(makeCluster({
      hub_ref: { repo_id: 'org/pack', revision: 'main', path: 'fear.cluster.json' },
    }));
    expect(screen.getByText('hub')).toBeInTheDocument();
    expect(screen.getByText(/org\/pack@main/)).toBeInTheDocument();
  });
});
