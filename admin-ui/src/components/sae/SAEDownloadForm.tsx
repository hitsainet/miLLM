import { useState } from 'react';
import { Download, Layers, Search, File, Check } from 'lucide-react';
import { Card, CardHeader, Button, Input, Spinner } from '@components/common';
import { saeApi } from '@/services/api';
import type { DownloadSAERequest, PreviewSAEResponse, SAEFileInfo } from '@/types';

interface SAEDownloadFormProps {
  onSubmit: (data: DownloadSAERequest) => void;
  isLoading?: boolean;
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

// Natural sort comparator for file paths
// Sorts "l0_13" before "l0_105" by comparing numbers by magnitude
function naturalSortCompare(a: string, b: string): number {
  const regex = /(\d+)|(\D+)/g;
  const aParts = a.match(regex) || [];
  const bParts = b.match(regex) || [];

  for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
    const aPart = aParts[i] || '';
    const bPart = bParts[i] || '';

    // If both parts are numeric, compare as numbers
    const aNum = parseInt(aPart, 10);
    const bNum = parseInt(bPart, 10);

    if (!isNaN(aNum) && !isNaN(bNum)) {
      if (aNum !== bNum) return aNum - bNum;
    } else {
      // Compare as strings
      if (aPart !== bPart) return aPart.localeCompare(bPart);
    }
  }
  return 0;
}

export function SAEDownloadForm({
  onSubmit,
  isLoading,
}: SAEDownloadFormProps) {
  const [repositoryId, setRepositoryId] = useState('');
  const [revision, setRevision] = useState('main');
  const [hfToken, setHfToken] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Preview state
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [previewData, setPreviewData] = useState<PreviewSAEResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<SAEFileInfo | null>(null);

  const validateRepository = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!repositoryId.trim()) {
      newErrors.repositoryId = 'Repository ID is required';
    } else if (!repositoryId.includes('/')) {
      newErrors.repositoryId = 'Invalid format. Use: owner/repo-name';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handlePreview = async () => {
    if (!validateRepository()) return;

    setIsPreviewing(true);
    setPreviewError(null);
    setPreviewData(null);
    setSelectedFile(null);

    try {
      const data = await saeApi.preview({
        repository_id: repositoryId.trim(),
        revision: revision.trim() || 'main',
        hf_token: hfToken.trim() || undefined,
      });
      setPreviewData(data);
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : 'Failed to preview repository');
    } finally {
      setIsPreviewing(false);
    }
  };

  const handleDownload = () => {
    if (!validateRepository()) return;

    onSubmit({
      repository_id: repositoryId.trim(),
      revision: revision.trim() || undefined,
    });
  };

  // Group files by layer for better organization
  const groupedFiles = previewData?.files.reduce((acc, file) => {
    const layer = file.layer ?? -1;
    if (!acc[layer]) {
      acc[layer] = [];
    }
    acc[layer].push(file);
    return acc;
  }, {} as Record<number, SAEFileInfo[]>) || {};

  const sortedLayers = Object.keys(groupedFiles)
    .map(Number)
    .sort((a, b) => a - b);

  return (
    <Card>
      <CardHeader
        title="Download SAE from HuggingFace"
        subtitle="Preview repository contents before downloading"
        icon={<Download className="w-5 h-5 text-primary-400" />}
      />

      <div className="space-y-4">
        {/* Repository Input */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2">
            <Input
              label="HuggingFace Repository"
              placeholder="e.g., google/gemma-scope-2b-pt-res"
              value={repositoryId}
              onChange={(e) => {
                setRepositoryId(e.target.value);
                setPreviewData(null);
                setSelectedFile(null);
              }}
              error={errors.repositoryId}
            />
          </div>
          <div>
            <Input
              label="Revision"
              placeholder="main"
              value={revision}
              onChange={(e) => {
                setRevision(e.target.value);
                setPreviewData(null);
                setSelectedFile(null);
              }}
              helperText="Branch, tag, or commit"
            />
          </div>
        </div>

        {/* Access Token */}
        <Input
          label="Access Token"
          placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
          type="password"
          value={hfToken}
          onChange={(e) => setHfToken(e.target.value)}
          helperText="Optional - required for gated repositories"
        />

        {/* Preview Button */}
        <Button
          type="button"
          variant="secondary"
          onClick={handlePreview}
          loading={isPreviewing}
          leftIcon={<Search className="w-4 h-4" />}
          className="w-full"
        >
          Preview Repository
        </Button>

        {/* Preview Error */}
        {previewError && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
            <p className="text-sm text-red-400">{previewError}</p>
          </div>
        )}

        {/* Preview Results */}
        {previewData && (
          <div className="space-y-4">
            {/* Repository Info */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">
                Found <span className="text-slate-200 font-medium">{previewData.total_files}</span> SAE files
              </span>
              {previewData.model_id && (
                <span className="text-slate-500">
                  Model: <span className="text-primary-400">{previewData.model_id}</span>
                </span>
              )}
            </div>

            {/* File List */}
            <div className="max-h-80 overflow-y-auto border border-slate-700/50 rounded-lg">
              {sortedLayers.map((layer) => (
                <div key={layer}>
                  {/* Layer Header */}
                  {layer >= 0 && (
                    <div className="sticky top-0 px-3 py-2 bg-slate-800/90 border-b border-slate-700/50 backdrop-blur-sm">
                      <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                        Layer {layer}
                      </span>
                    </div>
                  )}

                  {/* Files in Layer - sorted naturally by path */}
                  {[...groupedFiles[layer]].sort((a, b) => naturalSortCompare(a.path, b.path)).map((file) => (
                    <div
                      key={file.path}
                      onClick={() => setSelectedFile(selectedFile?.path === file.path ? null : file)}
                      className={`
                        flex items-center justify-between px-3 py-2 cursor-pointer
                        transition-colors border-b border-slate-700/30
                        ${selectedFile?.path === file.path
                          ? 'bg-primary-500/10 border-primary-500/30'
                          : 'hover:bg-slate-800/50'
                        }
                      `}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className={`
                          w-5 h-5 rounded flex items-center justify-center flex-shrink-0
                          ${selectedFile?.path === file.path
                            ? 'bg-primary-500 text-white'
                            : 'bg-slate-700/50 text-slate-500'
                          }
                        `}>
                          {selectedFile?.path === file.path ? (
                            <Check className="w-3 h-3" />
                          ) : (
                            <File className="w-3 h-3" />
                          )}
                        </div>
                        <span className="text-sm text-slate-300 truncate font-mono">
                          {file.path}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 flex-shrink-0 ml-4">
                        {file.width && (
                          <span className="text-xs text-slate-500 font-mono">
                            {file.width}
                          </span>
                        )}
                        <span className="text-xs text-slate-500 min-w-[60px] text-right">
                          {formatSize(file.size_bytes)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ))}

              {previewData.total_files === 0 && (
                <div className="p-6 text-center text-slate-500">
                  No SAE files found in this repository
                </div>
              )}
            </div>

            {/* Selected File Info */}
            {selectedFile && (
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                <p className="text-sm text-slate-400">
                  Selected: <span className="text-slate-200 font-mono">{selectedFile.path}</span>
                </p>
                {selectedFile.layer !== null && (
                  <p className="text-xs text-slate-500 mt-1">
                    Layer {selectedFile.layer}
                    {selectedFile.width && ` • Width ${selectedFile.width}`}
                    {selectedFile.size_bytes > 0 && ` • ${formatSize(selectedFile.size_bytes)}`}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Download Button */}
        <Button
          type="button"
          variant="primary"
          onClick={handleDownload}
          loading={isLoading}
          disabled={!repositoryId.trim()}
          leftIcon={<Layers className="w-4 h-4" />}
          className="w-full"
        >
          {previewData ? 'Download SAE Repository' : 'Download SAE'}
        </Button>

        {/* Help Text */}
        <div className="text-xs text-slate-500 bg-slate-800/30 rounded-lg p-3">
          <p className="font-medium text-slate-400 mb-1">Note:</p>
          <p>Downloads the entire SAE repository. Layer selection is done when attaching the SAE to a model.</p>
        </div>
      </div>
    </Card>
  );
}
