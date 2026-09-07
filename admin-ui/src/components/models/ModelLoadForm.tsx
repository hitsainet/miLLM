import { useEffect, useRef, useState } from 'react';
import { Download, Play, HelpCircle } from 'lucide-react';
import { Card, CardHeader, Button, Input, Select } from '@components/common';
import type { GGUFQuantInfo } from '@/types';

export interface ModelLoadFormData {
  repo_id: string;
  quantization: 'FP32' | 'FP16' | 'Q8' | 'Q4' | 'Q2';
  device: 'auto' | 'cuda' | 'cpu';
  trust_remote_code: boolean;
  hf_token?: string;
  /** Set only when a GGUF quantization was chosen. */
  gguf_files?: string[];
  gguf_label?: string;
}

interface ModelLoadFormProps {
  onSubmit: (data: ModelLoadFormData) => void;
  onPreview?: (repo_id: string, hf_token?: string) => void;
  isLoading?: boolean;
  isPreviewLoading?: boolean;
  /** Quantizations from the most recent preview, if that repo was GGUF. */
  ggufQuants?: GGUFQuantInfo[] | null;
  /** WHICH repo those quantizations describe. See `ggufForCurrentRepo`. */
  previewedRepoId?: string | null;
  /**
   * Called once the typed repo id has stopped changing and looks like a repo.
   *
   * Separate from `onPreview`, which opens the details modal — this only fills
   * the quantization dropdown, so it must not surface anything.
   */
  onRepoIdSettled?: (repo_id: string, hf_token?: string) => void;
}

/** How long the repo id must stay unchanged before it is looked up. */
const REPO_SETTLE_MS = 600;

/** The shape a HuggingFace repo id takes; also what the form validates. */
function looksLikeRepoId(value: string): boolean {
  return /^[\w-]+\/[\w.-]+$/.test(value.trim());
}

/**
 * Runtime quantization levels, for an ordinary safetensors model.
 *
 * FIXED ON PURPOSE. These are bitsandbytes levels applied at load time, so they
 * are a property of the runtime and identical for every such repo — there is
 * nothing to look up. A GGUF repo is the opposite: the quantizations are files
 * that were baked ahead of time and differ per repo, so the list below is
 * replaced by the repo's own.
 */
const quantizationOptions = [
  { value: 'Q4', label: 'Q4 - 4-bit (Recommended)' },
  { value: 'Q8', label: 'Q8 - 8-bit' },
  { value: 'FP16', label: 'FP16 - Half Precision' },
  { value: 'FP32', label: 'FP32 - Full Precision' },
  { value: 'Q2', label: 'Q2 - 2-bit' },
];

function formatGB(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

const deviceOptions = [
  { value: 'auto', label: 'Auto' },
  { value: 'cuda', label: 'CUDA (GPU)' },
  { value: 'cpu', label: 'CPU' },
];

export function ModelLoadForm({
  onSubmit,
  onPreview,
  isLoading,
  isPreviewLoading,
  ggufQuants,
  previewedRepoId,
  onRepoIdSettled,
}: ModelLoadFormProps) {
  const [formData, setFormData] = useState<ModelLoadFormData>({
    repo_id: '',
    quantization: 'Q4',
    device: 'auto',
    trust_remote_code: false,
    hf_token: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [ggufLabel, setGgufLabel] = useState<string>('');

  /**
   * The previewed quantizations, but ONLY while they still describe the repo in
   * the box.
   *
   * A preview is true of one repository. Editing the repo id after previewing
   * leaves the old repo's quantizations on screen, and choosing one would send
   * file paths that do not exist in the new repo — a stale verdict presented as
   * a current one. Comparing against the previewed id makes the list disappear
   * the moment it stops being true.
   */
  // Look the repo up once typing settles, so the dropdown describes the repo
  // in the box without requiring a manual Preview first. Debounced because this
  // is a network call on a text field, and skipped until the id is well-formed.
  const settleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastLookedUp = useRef<string>('');
  useEffect(() => {
    const repo = formData.repo_id.trim();
    if (!onRepoIdSettled || !looksLikeRepoId(repo) || repo === lastLookedUp.current) {
      return;
    }
    if (settleTimer.current) clearTimeout(settleTimer.current);
    settleTimer.current = setTimeout(() => {
      lastLookedUp.current = repo;
      onRepoIdSettled(repo, formData.hf_token || undefined);
    }, REPO_SETTLE_MS);
    return () => {
      if (settleTimer.current) clearTimeout(settleTimer.current);
    };
  }, [formData.repo_id, formData.hf_token, onRepoIdSettled]);

  // A chosen quantization belongs to the repo it came from. Clearing it when
  // the box changes stops a stale label riding along to a different repo.
  useEffect(() => {
    setGgufLabel('');
  }, [formData.repo_id]);

  const ggufForCurrentRepo =
    previewedRepoId && previewedRepoId.trim() === formData.repo_id.trim()
      ? ggufQuants ?? null
      : null;
  const isGguf = !!ggufForCurrentRepo?.length;
  const selectedQuant = ggufForCurrentRepo?.find((q) => q.label === ggufLabel) ?? null;

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.repo_id.trim()) {
      newErrors.repo_id = 'Repository ID is required';
    } else if (!formData.repo_id.includes('/')) {
      newErrors.repo_id = 'Invalid format. Use: owner/model-name';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit({
        ...formData,
        hf_token: formData.hf_token || undefined,
        // EVERY file of the chosen quantization. The coarse `quantization`
        // above is ignored by the backend when a label is present — it derives
        // the bucket from the label so the two cannot disagree.
        ...(selectedQuant
          ? {
              gguf_files: selectedQuant.files.map((f) => f.path),
              gguf_label: selectedQuant.label,
            }
          : {}),
      });
    }
  };

  const handlePreview = () => {
    if (formData.repo_id.trim() && formData.repo_id.includes('/')) {
      onPreview?.(formData.repo_id, formData.hf_token || undefined);
    }
  };

  return (
    <Card>
      <CardHeader
        title="Load Model"
        subtitle="Download and load a model from Hugging Face"
        icon={<Download className="w-5 h-5 text-primary-400" />}
      />
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Hugging Face Repository ID"
          placeholder="e.g., google/gemma-2-2b"
          value={formData.repo_id}
          onChange={(e) => setFormData({ ...formData, repo_id: e.target.value })}
          error={errors.repo_id}
          helperText="Enter the model repository in format: owner/model-name"
        />

        <div className="grid grid-cols-2 gap-4">
          {isGguf ? (
            <Select
              label="Quantization"
              value={ggufLabel}
              onChange={(e) => setGgufLabel(e.target.value)}
              options={[
                { value: '', label: `Choose one of ${ggufForCurrentRepo!.length}…` },
                ...ggufForCurrentRepo!.map((q) => ({
                  value: q.label,
                  label: `${q.label} — ${formatGB(q.total_size_bytes)}${
                    q.is_split ? ` (${q.files.length} parts)` : ''
                  }`,
                })),
              ]}
              helper="From this repository, with measured sizes"
            />
          ) : (
            <Select
              label="Quantization"
              value={formData.quantization}
              onChange={(e) => setFormData({ ...formData, quantization: e.target.value as ModelLoadFormData['quantization'] })}
              options={quantizationOptions}
            />
          )}
          <Select
            label="Device"
            value={formData.device}
            onChange={(e) => setFormData({ ...formData, device: e.target.value as ModelLoadFormData['device'] })}
            options={deviceOptions}
          />
        </div>

        {/* An API token, NOT a password — see SAEDownloadForm for the same
            treatment. `autoComplete="off"` keeps browsers from offering to
            save it as a site credential. */}
        <Input
          label="Hugging Face Token"
          name="hf-token"
          type="password"
          autoComplete="off"
          data-lpignore="true"
          data-1p-ignore="true"
          data-form-type="other"
          placeholder="hf_xxxx... (optional)"
          value={formData.hf_token}
          onChange={(e) => setFormData({ ...formData, hf_token: e.target.value })}
          helperText="Required for gated models like Llama"
        />

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="trust_remote_code"
            checked={formData.trust_remote_code}
            onChange={(e) => setFormData({ ...formData, trust_remote_code: e.target.checked })}
            className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary-500 focus:ring-primary-500 focus:ring-offset-0"
          />
          <label htmlFor="trust_remote_code" className="text-sm text-slate-300 flex items-center gap-1">
            Trust remote code
            <span className="text-yellow-500 text-xs">(Required for some models)</span>
          </label>
        </div>

        <div className="flex gap-3 pt-2">
          {onPreview && (
            <Button
              type="button"
              variant="secondary"
              onClick={handlePreview}
              loading={isPreviewLoading}
              disabled={!formData.repo_id.includes('/')}
              leftIcon={<HelpCircle className="w-4 h-4" />}
            >
              Preview
            </Button>
          )}
          <Button
            type="submit"
            variant="primary"
            loading={isLoading}
            leftIcon={<Play className="w-4 h-4" />}
            className="flex-1"
          >
            Download & Load Model
          </Button>
        </div>
      </form>
    </Card>
  );
}
