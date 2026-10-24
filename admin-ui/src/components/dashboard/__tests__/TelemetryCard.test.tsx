/**
 * The Dashboard "Telemetry" card (038 FR-38.13 UI, FTASKS 9.3).
 *
 * Each state reads as TEXT (not colour alone); the configuration is collapsed by default; the
 * fixed line is always there; and the card has NO edit control at all — export is configured in
 * the deployment, never here.
 *
 * MUTATION CONTROLS: render the configuration open by default; add an input; drop the fixed line;
 * render a signal's state by colour only.
 */

import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';

import { TelemetryCard } from '../TelemetryCard';
import type { TelemetryHealthView } from '@hooks/useTelemetryHealth';
import type { TelemetryHealth, TelemetrySignalHealth } from '@/types';

const FIXED = 'Configuration comes from deployment settings; nothing here can change it.';

function signal(state: string, over: Partial<TelemetrySignalHealth> = {}): TelemetrySignalHealth {
  return {
    state, exported: 0, failed: 0, dropped: 0, last_success_at: null, last_error: null,
    queue: { used: 0, capacity: state === 'disabled' ? 0 : 8192 }, ...over,
  };
}

function health(states: [string, string, string], over: Partial<TelemetryHealth> = {}): TelemetryHealth {
  return {
    enabled: states.some((s) => s !== 'disabled'),
    config: {
      endpoint: 'https://collector.example:4318', protocol: 'http/protobuf', transport: 'tls',
      headers_set: true, tls_ca_file_set: false, tls_client_cert_set: false,
      tls_client_key_set: false, service_name: 'millm', service_instance_id: 'a1b2',
      deployment_environment: null, sampler: { kind: 'parentbased_traceidratio', ratio: 1 },
      series_cap: 2000, queue_size: 8192, batch_size: 512, export_timeout_s: 10,
      retry_horizon_s: 60, drain_deadline_s: 5, metric_interval_s: 15,
      stateless_export_permitted: false, result_log_filter: { origins: ['live', 'batch'] },
      sdk_disabled: false, sdk_version: '1.45.1',
      semconv_schema_url: 'https://opentelemetry.io/schemas/1.44.0',
    },
    signals: { traces: signal(states[0]), metrics: signal(states[1]), logs: signal(states[2]) },
    series_overflow: 0,
    attributes_dropped: 0,
    deletion_note: 'Deleting probes or events in miLLM does not erase copies already exported.',
    ...over,
  };
}

function view(data: TelemetryHealth | 'absent', extra: Partial<TelemetryHealthView> = {}): TelemetryHealthView {
  return {
    health: data === 'absent' ? { available: false } : { available: true, data },
    isLoading: false,
    error: null,
    ...extra,
  };
}

describe('TelemetryCard', () => {
  it('off by default: every signal reads "disabled" and the card says Off', () => {
    render(<TelemetryCard {...view(health(['disabled', 'disabled', 'disabled']))} />);
    const card = screen.getByTestId('telemetry-card');
    expect(within(card).getByText('Off')).toBeInTheDocument();
    for (const s of ['traces', 'metrics', 'logs']) {
      expect(screen.getByTestId(`telemetry-state-${s}`)).toHaveTextContent('disabled');
    }
    expect(screen.getByTestId('telemetry-fixed-line')).toHaveTextContent(FIXED);
  });

  it.each([
    ['healthy', 'Exporting'],
    ['degraded', 'Degraded'],
    ['unavailable', 'Unavailable'],
  ])('a %s signal reads as text, and the card says %s', (state, cardText) => {
    render(<TelemetryCard {...view(health(['healthy', state, 'healthy']))} />);
    expect(screen.getByTestId('telemetry-state-metrics')).toHaveTextContent(state);
    expect(within(screen.getByTestId('telemetry-card')).getAllByText(cardText).length).toBeGreaterThan(0);
  });

  it('shows the last error CLASS and time, the queue and the counters', () => {
    const data = health(['healthy', 'healthy', 'unavailable']);
    data.signals!.logs = signal('unavailable', {
      exported: 3, failed: 2, dropped: 1, queue: { used: 5, capacity: 8192 },
      last_error: { type: 'HTTP401', at: '2026-10-23T12:00:00.000000Z' },
    });
    data.series_overflow = 7;
    data.attributes_dropped = 4;
    render(<TelemetryCard {...view(data)} />);
    const logs = screen.getByTestId('telemetry-signal-logs');
    expect(logs).toHaveTextContent('queue 5/8192');
    expect(logs).toHaveTextContent('exported 3');
    expect(logs).toHaveTextContent('failed 2');
    expect(logs).toHaveTextContent('dropped 1');
    expect(screen.getByTestId('telemetry-error-logs')).toHaveTextContent(
      'last error HTTP401 at 2026-10-23T12:00:00.000000Z');
    expect(screen.getByTestId('telemetry-counters')).toHaveTextContent('series overflow 7');
    expect(screen.getByTestId('telemetry-counters')).toHaveTextContent('attributes dropped 4');
  });

  it('keeps the configuration collapsed until asked, and shows credentials only as set', () => {
    render(<TelemetryCard {...view(health(['healthy', 'healthy', 'healthy']))} />);
    expect(screen.queryByTestId('telemetry-config')).toBeNull();
    const toggle = screen.getByTestId('telemetry-config-toggle');
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);
    const config = screen.getByTestId('telemetry-config');
    expect(config).toHaveTextContent('https://collector.example:4318');
    expect(config).toHaveTextContent('Receiver credentialsset');
    expect(config).toHaveTextContent('Result log originslive, batch');
  });

  it('has no edit control: no input, select or textarea, and one button (the toggle)', () => {
    const { container } = render(<TelemetryCard {...view(health(['healthy', 'degraded', 'disabled']))} />);
    fireEvent.click(screen.getByTestId('telemetry-config-toggle'));
    expect(container.querySelectorAll('input, select, textarea, [contenteditable="true"]')).toHaveLength(0);
    const buttons = within(container).getAllByRole('button');
    expect(buttons).toHaveLength(1);
    expect(buttons[0]).toHaveAttribute('data-testid', 'telemetry-config-toggle');
  });

  it('a server without the block says "Not available", a failed read "Unknown"', () => {
    const { rerender } = render(<TelemetryCard {...view('absent')} />);
    expect(screen.getByText('Not available')).toBeInTheDocument();
    rerender(<TelemetryCard {...view('absent', { health: undefined, error: new Error('x') })} />);
    expect(screen.getByText('Unknown')).toBeInTheDocument();
    rerender(<TelemetryCard {...view({ error: 'RuntimeError' })} />);
    expect(screen.getByText('snapshot failed (RuntimeError)')).toBeInTheDocument();
    expect(screen.getByTestId('telemetry-fixed-line')).toHaveTextContent(FIXED);
  });
});
