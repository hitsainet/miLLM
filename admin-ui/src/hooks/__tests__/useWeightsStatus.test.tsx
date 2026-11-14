/**
 * `useWeightsStatus` through the real API client: the two URLs, the 5 s poll, and a 404 (route
 * not deployed) resolving to `{ available: false }` instead of an error.
 *
 * MUTATION CONTROLS: drop the `status === 404` branch in `readOptional` -> the 404 tests fail;
 * drop `response.status` from the envelope throw in `request()` -> the FastAPI 404 test fails.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement, type ReactNode } from 'react';

import { useWeightsStatus, WEIGHTS_POLL_MS } from '../useWeightsStatus';

type Reply = { status: number; body: unknown; contentType?: string };

function json(status: number, body: unknown): Reply {
  return { status, body };
}

const OK_STATUS = {
  attached: false, attachment: null, barrier: 'serving', weights_state: 'torn',
  resident_model: null, steps_since_load: null, last_save: null, commit_overran: false,
  last_hold_timeout_at: null, last_overrun: null, quiesced_at: null, held_requests: null,
  commit: null, adapter: null,
};
const OK_ADAPTER = {
  applied: true, name: 'support-tone', base_name: 'llama', manifest_sha256: 'cd',
  manifest_base_name: 'llama', manifest_base_sha256: null, weights_sha256: 'ab', rank: 8,
  target_count: 4, loaded_at: '2026-10-14T12:00:00+00:00', base_steps_since_load: null,
};

function stubFetch(routes: Record<string, Reply>) {
  const fetchMock = vi.fn(async (url: string) => {
    const reply = routes[url];
    if (!reply) throw new Error(`unexpected fetch ${url}`);
    const body = typeof reply.body === 'string' ? reply.body : JSON.stringify(reply.body);
    return new Response(body, {
      status: reply.status,
      headers: { 'content-type': reply.contentType ?? 'application/json' },
    });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children);
  return { client, ...renderHook(() => useWeightsStatus(), { wrapper }) };
}

describe('useWeightsStatus', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('reads both endpoints and unwraps the envelope', async () => {
    const fetchMock = stubFetch({
      '/api/weights/status': json(200, { success: true, data: OK_STATUS, error: null }),
      '/api/weights/adapter': json(200, { success: true, data: OK_ADAPTER, error: null }),
    });
    const { result } = setup();
    await waitFor(() => expect(result.current.adapter).toBeDefined());
    await waitFor(() => expect(result.current.status).toBeDefined());

    expect(result.current.status).toEqual({ available: true, data: OK_STATUS });
    expect(result.current.adapter).toEqual({ available: true, data: OK_ADAPTER });
    const urls = fetchMock.mock.calls.map((c) => c[0]).sort();
    expect(urls).toEqual(['/api/weights/adapter', '/api/weights/status']);
  });

  it('a FastAPI 404 (route not served) is "not available", not an error', async () => {
    stubFetch({
      '/api/weights/status': json(404, { detail: 'Not Found' }),
      '/api/weights/adapter': json(404, { detail: 'Not Found' }),
    });
    const { result } = setup();
    await waitFor(() => expect(result.current.status).toBeDefined());
    await waitFor(() => expect(result.current.adapter).toBeDefined());

    expect(result.current.status).toEqual({ available: false });
    expect(result.current.adapter).toEqual({ available: false });
    expect(result.current.statusError).toBeNull();
    expect(result.current.adapterError).toBeNull();
  });

  it('an HTML 404 from the proxy is "not available" too', async () => {
    stubFetch({
      '/api/weights/status': { status: 404, body: '<html>404</html>', contentType: 'text/html' },
      '/api/weights/adapter': json(200, { success: true, data: OK_ADAPTER, error: null }),
    });
    const { result } = setup();
    await waitFor(() => expect(result.current.status).toBeDefined());
    expect(result.current.status).toEqual({ available: false });
  });

  it('any other failure stays an error', async () => {
    stubFetch({
      '/api/weights/status': json(500, { success: false, data: null, error: { code: 'INTERNAL', message: 'x' } }),
      '/api/weights/adapter': json(200, { success: true, data: OK_ADAPTER, error: null }),
    });
    const { result } = setup();
    await waitFor(() => expect(result.current.statusError).not.toBeNull());
    expect(result.current.status).toBeUndefined();
  });

  it('polls both every 5 s', async () => {
    stubFetch({
      '/api/weights/status': json(200, { success: true, data: OK_STATUS, error: null }),
      '/api/weights/adapter': json(200, { success: true, data: OK_ADAPTER, error: null }),
    });
    const { client, result } = setup();
    await waitFor(() => expect(result.current.status).toBeDefined());
    expect(WEIGHTS_POLL_MS).toBe(5000);
    for (const key of [['weights', 'status'], ['weights', 'adapter']]) {
      const query = client.getQueryCache().find({ queryKey: key });
      expect((query?.options as { refetchInterval?: unknown }).refetchInterval).toBe(5000);
    }
  });
});
