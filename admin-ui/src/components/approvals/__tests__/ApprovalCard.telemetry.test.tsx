/**
 * ENH-006 D76: an AI agent's change to telemetry settings waits as `telemetry_write`. Its card
 * shows, for each setting, the value now and the value after approval, exactly as the server
 * wrote them (the endpoint arrives redacted), plus when it takes effect and the agent's reason.
 * The shape is the one `runtime_settings.describe_save` returns (tests/unit/telemetry/
 * test_runtime_toggle.py pins the server side).
 */
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { approval } from '@/test/approvalsFake';

import { ApprovalCard } from '../ApprovalCard';
import { approveLabel } from '../approvalWords';

const telemetry = approval({
  action: 'telemetry_write',
  action_label: 'Change telemetry settings in miLLM',
  requested_by: 'agent:planner',
  summary: {
    'Export result logs (TELEMETRY_LOGS_ENABLED)': 'now off; after approval on',
    'Receiver endpoint (TELEMETRY_ENDPOINT)':
      'now https://deploy.example:4318; after approval http://collector.local:4318',
    'Takes effect': 'next export; drains and restarts the export pipeline',
    Reason: 'turn on the pilot',
  },
  payload: {
    body: {
      values: { TELEMETRY_LOGS_ENABLED: true, TELEMETRY_ENDPOINT: 'http://collector.local:4318' },
      reset: [],
      reason: 'turn on the pilot',
    },
  },
});

describe('a telemetry_write card', () => {
  it('shows each setting now and after approval, the effect and the reason', () => {
    render(<MemoryRouter><ApprovalCard approval={telemetry} /></MemoryRouter>);
    const card = screen.getByTestId(`approval-card-${telemetry.approval_id}`);
    expect(within(card).getByRole('heading', { level: 3 })).toHaveTextContent(
      'Change telemetry settings in miLLM');
    expect(within(card).getByText('asked by agent planner')).toBeInTheDocument();
    expect(within(card).getByText('Export result logs (TELEMETRY_LOGS_ENABLED)')).toBeInTheDocument();
    expect(within(card).getByText('now off; after approval on')).toBeInTheDocument();
    expect(within(card).getByText(
      'now https://deploy.example:4318; after approval http://collector.local:4318')).toBeInTheDocument();
    expect(within(card).getByText('next export; drains and restarts the export pipeline'))
      .toBeInTheDocument();
    expect(within(card).getByText('turn on the pilot')).toBeInTheDocument();
    expect(within(card).getByRole('button', { name: approveLabel(telemetry.action_label) }))
      .toBeInTheDocument();
  });
});
