import type { GpuMetrics, GpuSelection } from '@/types';

const MIB = 1024 ** 2;

/**
 * The selection meaning "split the model across every visible card". The
 * backend honours it or refuses it with the figures, like a named card; it is
 * how an operator force-splits a model one card could hold.
 */
export const ALL_GPUS = 'all';

/** Free memory on a card, from the live metrics. */
export function gpuFreeMb(gpu: GpuMetrics): number {
  return Math.max(gpu.memory_total_mb - gpu.memory_used_mb, 0);
}

/**
 * The value a card is selected by: its UUID when the backend reported one,
 * which survives a card being added in front of it; its index otherwise.
 */
export function gpuSelectionValue(gpu: GpuMetrics): GpuSelection {
  return gpu.uuid ? gpu.uuid : String(gpu.index);
}

/**
 * "Auto" first, then "All GPUs (split)" when there is more than one card, then
 * one option per card with its name and free memory. On one card a split is
 * that card, so the option is not offered.
 */
export function gpuOptions(gpus: GpuMetrics[]): { value: string; label: string }[] {
  return [
    { value: 'auto', label: 'Auto (most free memory)' },
    ...(gpus.length > 1 ? [{ value: ALL_GPUS, label: 'All GPUs (split)' }] : []),
    ...gpus.map((gpu) => ({
      value: gpuSelectionValue(gpu),
      label: `GPU ${gpu.index} · ${gpu.name} · ${(gpuFreeMb(gpu) / 1024).toFixed(1)} GB free`,
    })),
  ];
}

/**
 * The selection as it applies to the cards that exist now.
 *
 * A remembered card that is no longer reported reads as 'auto', so the
 * selector never shows a value that is not among its options — and so does a
 * remembered split once only one card is left. With no card list at all (the
 * socket has not delivered one yet) the selection is kept: the backend
 * validates it, and dropping it would load on a card nobody chose.
 */
export function resolveGpuSelection(selection: GpuSelection, gpus: GpuMetrics[]): GpuSelection {
  if (!selection || selection === 'auto') return 'auto';
  if (!gpus.length) return selection;
  if (selection === ALL_GPUS) return gpus.length > 1 ? ALL_GPUS : 'auto';
  const known = gpus.some(
    (gpu) => gpu.uuid === selection || String(gpu.index) === selection,
  );
  return known ? selection : 'auto';
}

/** Card totals in bytes, for fit hints. */
export function gpuTotalsBytes(gpus: GpuMetrics[]): number[] {
  return gpus.map((gpu) => gpu.memory_total_mb * MIB).filter((bytes) => bytes > 0);
}
