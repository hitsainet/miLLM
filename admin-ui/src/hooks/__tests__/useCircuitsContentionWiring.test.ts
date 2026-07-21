/**
 * F19 R1-06 — the contention refusal reaches the UI.
 *
 * Both contention components were written, exported and unit-tested with NO
 * CONSUMER ANYWHERE: rendered by no page, and no code read
 * `CIRCUIT_LAYER_CONTENTION` or `composed_layers`. The whole task-5.0
 * deliverable was dead code, and BR-011 §6.2's binding condition — every
 * override is surfaced in the UI — was unmet in production while Vitest stayed
 * green, because the component tests render the component directly.
 *
 * These assert the WIRING exists at each link of the chain. They are
 * deliberately structural: a behavioural test of the mutation's error path
 * fights React Query's rejection propagation for no added signal, and the
 * defect being guarded against is "nothing calls this", which is exactly what
 * a wiring assertion catches.
 *
 * The BEHAVIOUR of the dialog itself (no compose action on a collision, the
 * measurement rendered, the caveat kept) is covered by
 * ContentionDialog.test.tsx against the real component.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const SRC = resolve(__dirname, '../..');
const read = (p: string) => readFileSync(resolve(SRC, p), 'utf8');

describe('F19 contention wiring', () => {
  it('the hook recognises CIRCUIT_LAYER_CONTENTION rather than toasting it', () => {
    const hook = read('hooks/useCircuits.ts');
    expect(hook).toContain("error.code === 'CIRCUIT_LAYER_CONTENTION'");
    // It must route to the handler, not just log: a toast discards the
    // incumbent, the measurement and both resolutions.
    expect(hook).toContain('onContentionRef.current?.(');
    const handlerIndex = hook.indexOf("error.code === 'CIRCUIT_LAYER_CONTENTION'");
    const toastIndex = hook.indexOf('toast.error(`Activation failed');
    expect(handlerIndex).toBeLessThan(toastIndex);
  });

  it('the page RENDERS the dialog and the claims strip', () => {
    // The finding: both components existed with zero consumers.
    const page = read('pages/CircuitsPage.tsx');
    expect(page).toContain('<ContentionDialog');
    expect(page).toContain('<ClaimsStrip');
    expect(page).toContain('circuitApi.claims()');
  });

  it('the page passes the override back through activate', () => {
    const page = read('pages/CircuitsPage.tsx');
    expect(page).toContain('allowLayerOverlap: true');
  });

  it('compose is offered only when the server says it is overridable', () => {
    // Mirrors the dialog's own rule at the call site, so the UI cannot offer
    // an override the server would refuse anyway.
    const page = read('pages/CircuitsPage.tsx');
    expect(page).toContain('contention.details.overridable');
  });

  it('the service forwards allow_layer_overlap to the API', () => {
    const service = read('services/circuits.ts');
    expect(service).toContain("params.set('allow_layer_overlap', 'true')");
    // Default must remain false — composition is refused by default because it
    // is measured to destroy generation.
    expect(service).toContain('allowLayerOverlap = false');
  });

  it('the service exposes the claims endpoint', () => {
    expect(read('services/circuits.ts')).toContain("request<LayerClaim[]>('/circuits/claims')");
  });
});
