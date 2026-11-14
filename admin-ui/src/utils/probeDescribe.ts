/**
 * A probe in plain words, for the tile header (operator, 2026-10-29: "The probes need more
 * information about what they are looking for specifically … the information on the tiles … is
 * not very human friendly").
 *
 * Every sentence here is built from a field the server sent. Nothing is inferred: a missing field
 * drops its sentence (or says it was not recorded) rather than being filled with a guess. The
 * rung's wording is NOT composed here; it stays the server's (`rung_language`).
 */

import type { Probe } from '@/types/probe';
import { trainingLabels } from './probeLabels';

/** The tile's title: the behaviour's name, else the probe's own (machine) name. */
export function probeTitle(probe: Probe): string {
  const name = probe.behaviour?.name;
  return typeof name === 'string' && name.trim() ? name.trim() : probe.name;
}

const quote = (labels: string[]) => labels.map((l) => `“${l}”`).join(' or ');

/**
 * "Fires when the text contains: “present”; stays quiet for: “absent”." from `label_mapping`,
 * or `null` when the mapping does not name the positive side. The labels are the corpus's own
 * strings, unchanged.
 */
export function detectSentence(mapping: Record<string, string> | null | undefined): string | null {
  const { positive, negative } = trainingLabels(mapping);
  if (positive.length === 0) return null;
  const fires = `Fires when the text contains: ${quote(positive)}`;
  return negative.length > 0 ? `${fires}; stays quiet for: ${quote(negative)}.` : `${fires}.`;
}

/**
 * miStudio's own one-line statement, when it says MORE than the detect sentence already does.
 * `"positive = present"` adds nothing beside a mapping whose positive side is `present`; a
 * concept that names something else (or a probe with no mapping) is shown.
 */
export function informativeConcept(probe: Probe): string | null {
  const concept = probe.concept?.trim();
  if (!concept) return null;
  const { positive } = trainingLabels(probe.label_mapping);
  const restated = positive.length > 0 && /^positive\s*=\s*/i.test(concept)
    && concept.replace(/^positive\s*=\s*/i, '').trim().toLowerCase()
      === positive.join(' or ').toLowerCase();
  return restated ? null : concept;
}

/** How each aggregation rule turns per-token scores into one score, in a short phrase. */
export function rulePhrase(rule: string, params?: Record<string, unknown>): string {
  switch (rule) {
    case 'max':
      return 'the single strongest token';
    case 'mean':
      return 'the average over all tokens';
    case 'attention':
      return 'a learned weighting of tokens';
    case 'rolling_mean_max': {
      const window = params?.window;
      return typeof window === 'number'
        ? `the strongest stretch of ${window} tokens`
        : 'the strongest stretch of tokens';
    }
    case 'last':
      return 'the last token only';
    case 'softmax':
      return 'a soft maximum that leans toward the strongest tokens';
    default:
      return `the “${rule}” rule`;
  }
}

/** The model's layer count, when anything the client holds states it. Never guessed. */
export function layerCount(probe: Probe): number | null {
  if (typeof probe.n_layers === 'number' && probe.n_layers > 0) return probe.n_layers;
  const model = (probe.definition as { model?: { n_layers?: unknown } } | undefined)?.model;
  const fromDefinition = model?.n_layers;
  return typeof fromDefinition === 'number' && fromDefinition > 0 ? fromDefinition : null;
}

/** `meta-llama/Llama-3.1-8B-Instruct` → `Llama-3.1-8B-Instruct`. */
export function shortModelName(hfId: string): string {
  const tail = hfId.split('/').pop();
  return tail && tail.trim() ? tail : hfId;
}

/** "Reads layer 4 of 16 of Llama-3.1-8B-Instruct; combines tokens using … (attention)." */
export function readsSentence(probe: Probe): string {
  const total = layerCount(probe);
  const layer = total !== null ? `layer ${probe.layer} of ${total}` : `layer ${probe.layer}`;
  const basis = probe.basis === 'sae_features' ? " through a sparse autoencoder's features" : '';
  return (
    `Reads ${layer} of ${shortModelName(probe.hf_id)}${basis}; ` +
    `combines tokens using ${rulePhrase(probe.rule, probe.rule_params)} (${probe.rule}).`
  );
}

/** `0.01` → "about 1 false alarm in 100"; `null` when no rate was recorded. */
export function falseAlarmWords(targetFpr: number | null | undefined): string | null {
  if (typeof targetFpr !== 'number' || !(targetFpr > 0) || targetFpr >= 1) return null;
  const inN = Math.round(1 / targetFpr);
  return `about 1 false alarm in ${inN.toLocaleString('en-US')}`;
}

/** "Alarm when the score passes 10.13 (set for about 1 false alarm in 100)." */
export function alarmSentence(probe: Probe): string {
  if (probe.threshold === null) {
    return 'No alarm bar is set: it ranks text by score but never raises an alarm.';
  }
  const rate = falseAlarmWords(probe.target_fpr);
  return (
    `Alarm when the score passes ${probe.threshold.toFixed(2)}` +
    (rate ? ` (set for ${rate}).` : ' (the false-alarm rate it was set for was not recorded).')
  );
}
