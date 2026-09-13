import { HardDrive, Layers } from 'lucide-react';

import type { GGUFQuantInfo } from '../../types/api';

/**
 * Choose ONE quantization from a GGUF repository.
 *
 * A GGUF repo holds many mutually exclusive quantizations of the same model —
 * twenty-four is ordinary — and you want exactly one. Two things follow, and
 * both are easy to get wrong:
 *
 *   * A row is a QUANTIZATION, not a file. On large models a quant is split
 *     across numbered parts, and downloading one part gives a directory that
 *     looks populated and a model that will not load. Rows therefore show a
 *     file count and select every part together.
 *   * Sizes here are MEASURED, from HuggingFace's file metadata — not the
 *     bytes-per-parameter estimate shown for safetensors models, which means
 *     nothing for a mixed-precision quantization.
 */

export interface GGUFQuantPickerProps {
  quants: GGUFQuantInfo[];
  selectedLabel: string | null;
  onSelect: (label: string) => void;
  /**
   * Total VRAM of each card on the node, live from the metrics socket. Empty
   * or absent means unknown, and the fit column then says nothing.
   */
  gpuTotalsBytes?: number[] | null;
}

/** Binary units, because that is what `nvidia-smi` and the Hub both report in. */
export function formatBytes(bytes: number): string {
  if (bytes <= 0) return '—';
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

/**
 * Weights are not the whole story — KV cache and runtime overhead land on the
 * same card — so this is a HINT, hedged in the UI copy, never a gate. A quant
 * that reads "tight" may still run at a short context.
 *
 * 'fits' means on ONE card, which the backend then uses alone; 'fits-across'
 * means only the cards together hold it, which llama.cpp does by splitting
 * layers between them. The totals are the node's live cards — this was a
 * constant 24 GB, which was wrong the day a second card went in.
 */
export function fitsInVram(totalSizeBytes: number, gpuTotalsBytes?: number[] | null) {
  const cards = (gpuTotalsBytes ?? []).filter((bytes) => bytes > 0);
  if (!cards.length) return null;
  const withOverhead = totalSizeBytes * 1.15;
  const largest = Math.max(...cards);
  const combined = cards.reduce((sum, bytes) => sum + bytes, 0);
  if (withOverhead <= largest * 0.9) return 'fits' as const;
  if (cards.length > 1 && withOverhead <= combined * 0.9) return 'fits-across' as const;
  if (totalSizeBytes <= (cards.length > 1 ? combined : largest)) return 'tight' as const;
  return 'too-large' as const;
}

/** The fit column's heading, naming the cards it was judged against. */
function fitHeading(gpuTotalsBytes?: number[] | null): string {
  const cards = (gpuTotalsBytes ?? []).filter((bytes) => bytes > 0);
  if (!cards.length) return 'GPU fit';
  const sizes = cards.map((bytes) => `${Math.round(bytes / 1024 ** 3)}`);
  return cards.length === 1 ? `On a ${sizes[0]} GB card` : `On ${cards.length} cards (${sizes.join(' + ')} GB)`;
}

export function GGUFQuantPicker({
  quants,
  selectedLabel,
  onSelect,
  gpuTotalsBytes,
}: GGUFQuantPickerProps) {
  if (!quants.length) return null;

  return (
    <div data-testid="gguf-quant-picker">
      <div className="flex items-center gap-2 mb-3">
        <Layers className="w-5 h-5 text-primary-400" />
        <h3 className="text-base font-semibold text-slate-100">
          GGUF quantizations
        </h3>
        <span className="text-xs text-slate-500">
          {quants.length} available — choose one
        </span>
      </div>

      <div className="bg-slate-800/30 border border-slate-700 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-800">
            <tr>
              <th className="text-left px-4 py-3 text-sm font-medium text-slate-300 w-12">
                Select
              </th>
              <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">
                Quantization
              </th>
              <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                Size
              </th>
              <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                {fitHeading(gpuTotalsBytes)}
              </th>
            </tr>
          </thead>
          <tbody>
            {quants.map((q) => {
              const isSelected = selectedLabel === q.label;
              const fit = fitsInVram(q.total_size_bytes, gpuTotalsBytes);
              return (
                <tr
                  key={q.label}
                  onClick={() => onSelect(q.label)}
                  data-testid={`gguf-quant-${q.label}`}
                  className={`border-t border-slate-700 cursor-pointer transition-colors ${
                    isSelected ? 'bg-primary-500/10' : 'hover:bg-slate-800/50'
                  }`}
                >
                  <td className="px-4 py-3">
                    <input
                      type="radio"
                      name="gguf-quant"
                      checked={isSelected}
                      onChange={() => onSelect(q.label)}
                      aria-label={`Select ${q.label}`}
                      className="w-4 h-4 accent-primary-500"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm text-slate-100">{q.label}</span>
                      {/* STATED, not implied. A split quant downloads several
                          files; a reader who assumes one file and sees a
                          multi-file directory has no way to tell whether the
                          download finished. */}
                      {q.is_split && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300"
                          title={`Split across ${q.files.length} files, all downloaded together`}
                        >
                          {q.files.length} parts
                        </span>
                      )}
                      {!q.quant_parsed && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400"
                          title="The filename carries no recognised quantization token; this is the filename itself."
                        >
                          unrecognised name
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-sm text-slate-200 tabular-nums">
                      {formatBytes(q.total_size_bytes)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    {fit === null ? (
                      <span className="text-xs text-slate-500">—</span>
                    ) : fit === 'fits' ? (
                      <span className="text-xs text-emerald-400">fits on one card</span>
                    ) : fit === 'fits-across' ? (
                      <span className="text-xs text-sky-400">fits across cards</span>
                    ) : fit === 'tight' ? (
                      <span className="text-xs text-amber-400">tight</span>
                    ) : (
                      <span className="text-xs text-red-400">too large</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-500 mt-2 flex items-start gap-1.5">
        <HardDrive className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span>
          Sizes are measured from the repository, not estimated. The fit column
          accounts for weights plus about 15% overhead — KV cache grows with
          context, so a “tight” quantization may still run at a shorter one.
        </span>
      </p>
    </div>
  );
}
