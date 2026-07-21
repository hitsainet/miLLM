/**
 * Circuit API client (Feature 13).
 *
 * Mirrors /api/circuits. Evidence phrasing (`rung_language`) always comes from
 * the server — this client never derives it.
 */

import { request } from './api';
import type {
  CircuitActivationResponse,
  CircuitDefinitionV1,
  CircuitListResponse,
  CircuitSummary,
  LayerClaim,
} from '@/types/circuits';

export const circuitApi = {
  /** Lists imported circuits with rung, layers and serveability. */
  list: (opts?: {
    minRung?: number;
    serveable?: boolean;
    limit?: number;
    offset?: number;
  }) => {
    const params = new URLSearchParams();
    if (opts?.minRung != null) params.set('min_rung', String(opts.minRung));
    if (opts?.serveable != null) params.set('serveable', String(opts.serveable));
    if (opts?.limit != null) params.set('limit', String(opts.limit));
    if (opts?.offset != null) params.set('offset', String(opts.offset));
    const qs = params.toString();
    return request<CircuitListResponse>(`/circuits${qs ? `?${qs}` : ''}`);
  },

  /** The currently serving circuit, or null. */
  active: () => request<CircuitSummary | null>('/circuits/active'),

  /** Imports a `mistudio.circuit-definition/v1` document. */
  import: (payload: unknown, opts?: { onConflict?: 'rename' | 'fail' }) => {
    const params = new URLSearchParams();
    if (opts?.onConflict) params.set('on_conflict', opts.onConflict);
    const qs = params.toString();
    return request<CircuitSummary>(`/circuits/import${qs ? `?${qs}` : ''}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Activates (serves) a circuit.
   * @param acknowledgeUnvalidated - required when rung < 2; the server refuses
   *   with UNVALIDATED_CIRCUIT otherwise.
   */
  activate: (
    circuitId: string,
    acknowledgeUnvalidated = false,
    allowLayerOverlap = false,
  ) => {
    const params = new URLSearchParams();
    if (acknowledgeUnvalidated) params.set('acknowledge_unvalidated', 'true');
    // F19: composing onto a layer another circuit holds. Defaults false — the
    // server refuses by default because two steered layers were measured to
    // destroy generation.
    if (allowLayerOverlap) params.set('allow_layer_overlap', 'true');
    const qs = params.toString() ? `?${params.toString()}` : '';
    return request<CircuitActivationResponse>(
      `/circuits/${circuitId}/activate${qs}`,
      { method: 'POST' },
    );
  },

  /** Live layer claims: who holds which layer, and what is composed (F19). */
  claims: () => request<LayerClaim[]>('/circuits/claims'),

  /** Stops serving a circuit and clears its steering. */
  deactivate: (circuitId: string) =>
    request<CircuitSummary>(`/circuits/${circuitId}/deactivate`, {
      method: 'POST',
    }),

  /** Sets the active circuit's global lambda (scales every layer together). */
  setActiveIntensity: (intensity: number, reapply = true) =>
    request<CircuitSummary>('/circuits/active/intensity', {
      method: 'PUT',
      body: JSON.stringify({ intensity, reapply }),
    }),

  /** Deletes an imported circuit. */
  remove: (circuitId: string) =>
    request<{ circuit_id: string; deleted: boolean }>(`/circuits/${circuitId}`, {
      method: 'DELETE',
    }),

  /** Raw lossless re-export (the response IS the portable artifact). */
  export: (circuitId: string) =>
    request<CircuitDefinitionV1>(`/circuits/${circuitId}/export`),
};
