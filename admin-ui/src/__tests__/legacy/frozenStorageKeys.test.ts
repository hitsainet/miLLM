/**
 * 040 DR-25 exercise (FR-040.7): the admin UI keeps its browser storage keys under the old brand.
 * Renaming either would silently drop every operator's saved preferences and banner state, so the
 * keys are frozen (suite register DR-25) and this test drives the real store and the real banner.
 */
import { act, createElement } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApprovalsBanner } from '@/components/approvals/ApprovalsBanner';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { useUIStore } from '@/stores/uiStore';
import { approval, installApprovalsFake, resetApprovalsStore } from '@/test/approvalsFake';

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  resetApprovalsStore(useApprovalsStore);
});

describe('frozen browser storage keys (DR-25)', () => {
  it('keeps the frozen browser storage keys', async () => {
    // tests/setup.ts replaces localStorage with vi.fn() stubs (getItem answers undefined), and the
    // persisted store captured that object when it was created: assert the WRITE it made.
    const setItem = vi.mocked(window.localStorage.setItem);
    setItem.mockClear();
    act(() => useUIStore.getState().setTheme('light'));
    const writes = setItem.mock.calls.filter(([key]) => key === 'millm-ui-preferences');
    expect(writes.length).toBeGreaterThan(0);
    expect(JSON.parse(writes[writes.length - 1][1] as string).state.theme).toBe('light');

    installApprovalsFake([approval()]);
    await act(() => useApprovalsStore.getState().refresh());
    render(createElement(MemoryRouter, null, createElement(ApprovalsBanner, { onApprovalsPage: false })));
    fireEvent.click(screen.getByRole('button', { name: /collapse/i }));
    expect(window.sessionStorage.getItem('millm.approvals.banner.collapsed')).toBe('1');
  });
});
