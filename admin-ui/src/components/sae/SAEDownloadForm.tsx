import { useState, useMemo } from 'react';
import { Download, Layers, Search, FolderOpen, Check } from 'lucide-react';
import { Card, CardHeader, Button, Input } from '@components/common';
import { saeApi } from '@/services/api';
import type { DownloadSAERequest, PreviewSAEResponse, SAEFileInfo } from '@/types';

/** An SAE directory group containing all files for one SAE */
interface SAEGroup {
  dirPath: string;
  files: SAEFileInfo[];
  totalSizeBytes: number;
  layer: number | null;
  width: string | null;
}

/** Group individual SAE files into directory-level SAE groups */
function groupFilesIntoSAEs(files: SAEFileInfo[]): SAEGroup[] {
  const groups = new Map<string, SAEGroup>();
  for (const file of files) {
    const lastSlash = file.path.lastIndexOf('/');
    const dirPath = lastSlash > 0 ? file.path.substring(0, lastSlash) : '.';
    if (!groups.has(dirPath)) {
      groups.set(dirPath, {
        dirPath, files: [], totalSizeBytes: 0,
        layer: file.layer, width: file.width,
      });
    }
    const group = groups.get(dirPath)!;
    group.files.push(file);
    group.totalSizeBytes += file.size_bytes || 0;
    if (group.layer === null && file.layer !== null) group.layer = file.layer;
    if (!group.width && file.width) group.width = file.width;
  }
  return Array.from(groups.values());
}

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
  const [selectedDirs, setSelectedDirs] = useState<Set<string>>(new Set());

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
    setSelectedDirs(new Set());

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

  // Compute SAE groups from preview files
  const saeGroups = useMemo(() => {
    if (!previewData) return [];
    return groupFilesIntoSAEs(previewData.files);
  }, [previewData]);

  const toggleDirSelection = (dirPath: string) => {
    setSelectedDirs(prev => {
      const next = new Set(prev);
      if (next.has(dirPath)) {
        next.delete(dirPath);
      } else {
        next.add(dirPath);
      }
      return next;
    });
  };

  const handleDownload = () => {
    if (!validateRepository()) return;

    if (selectedDirs.size > 0) {
      // Download each selected SAE directory
      const selected = saeGroups.filter(g => selectedDirs.has(g.dirPath));
      selected.forEach((group) => {
        onSubmit({
          repository_id: repositoryId.trim(),
          revision: revision.trim() || undefined,
          file_path: group.files[0].path, // backend extracts parent dir
        });
      });
    } else {
      // Fallback: download entire repository
      onSubmit({
        repository_id: repositoryId.trim(),
        revision: revision.trim() || undefined,
      });
    }
  };

  // Group SAE directories by layer for display
  const groupsByLayer = useMemo(() => {
    const map: Record<number, SAEGroup[]> = {};
    saeGroups.forEach((group) => {
      const layer = group.layer ?? -1;
      if (!map[layer]) map[layer] = [];
      map[layer].push(group);
    });
    return map;
  }, [saeGroups]);

  const sortedLayers = Object.keys(groupsByLayer)
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
                setSelectedDirs(new Set());
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
                setSelectedDirs(new Set());
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
                Found <span className="text-slate-200 font-medium">{saeGroups.length}</span> SAE{saeGroups.length !== 1 ? 's' : ''}
              </span>
              {previewData.model_id && (
                <span className="text-slate-500">
                  Model: <span className="text-primary-400">{previewData.model_id}</span>
                </span>
              )}
            </div>

            {/* SAE Directory List */}
            <div className="max-h-80 overflow-y-auto border border-slate-700/50 rounded-lg">
              {sortedLayers.map((layer) => {
                const layerGroups = groupsByLayer[layer];
                const allSelected = layerGroups.every(g => selectedDirs.has(g.dirPath));
                return (
                  <div key={layer}>
                    {/* Layer Header */}
                    {layer >= 0 && (
                      <div className="sticky top-0 px-3 py-2 bg-slate-800/90 border-b border-slate-700/50 backdrop-blur-sm flex items-center justify-between">
                        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                          Layer {layer}
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (allSelected) {
                              setSelectedDirs(prev => {
                                const next = new Set(prev);
                                layerGroups.forEach(g => next.delete(g.dirPath));
                                return next;
                              });
                            } else {
                              setSelectedDirs(prev => {
                                const next = new Set(prev);
                                layerGroups.forEach(g => next.add(g.dirPath));
                                return next;
                              });
                            }
                          }}
                          className="text-xs text-primary-400/70 hover:text-primary-400 transition-colors"
                        >
                          {allSelected ? 'Deselect all' : 'Select all'}
                        </button>
                      </div>
                    )}

                    {/* SAE directories in this layer */}
                    {[...layerGroups].sort((a, b) => naturalSortCompare(a.dirPath, b.dirPath)).map((group) => {
                      const isSelected = selectedDirs.has(group.dirPath);
                      return (
                        <div
                          key={group.dirPath}
                          onClick={() => toggleDirSelection(group.dirPath)}
                          className={`
                            flex items-center justify-between px-3 py-2 cursor-pointer
                            transition-colors border-b border-slate-700/30
                            ${isSelected
                              ? 'bg-primary-500/10 border-primary-500/30'
                              : 'hover:bg-slate-800/50'
                            }
                          `}
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <div className={`
                              w-5 h-5 rounded flex items-center justify-center flex-shrink-0
                              ${isSelected
                                ? 'bg-primary-500 text-white'
                                : 'bg-slate-700/50 text-slate-500'
                              }
                            `}>
                              {isSelected ? (
                                <Check className="w-3 h-3" />
                              ) : (
                                <FolderOpen className="w-3 h-3" />
                              )}
                            </div>
                            <span className="text-sm text-slate-300 truncate font-mono">
                              {group.dirPath}
                            </span>
                          </div>
                          <div className="flex items-center gap-3 flex-shrink-0 ml-4">
                            {group.width && (
                              <span className="text-xs text-slate-500 font-mono">
                                {group.width}
                              </span>
                            )}
                            <span className="text-xs text-slate-500">
                              {group.files.length} file{group.files.length !== 1 ? 's' : ''}
                            </span>
                            <span className="text-xs text-slate-500 min-w-[60px] text-right">
                              {formatSize(group.totalSizeBytes)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })}

              {saeGroups.length === 0 && (
                <div className="p-6 text-center text-slate-500">
                  No SAE files found in this repository
                </div>
              )}
            </div>

            {/* Selected SAEs Info */}
            {selectedDirs.size > 0 && (
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-slate-400">
                    Selected: <span className="text-slate-200 font-medium">{selectedDirs.size} SAE{selectedDirs.size !== 1 ? 's' : ''}</span>
                  </p>
                  <button
                    type="button"
                    onClick={() => setSelectedDirs(new Set())}
                    className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    Clear all
                  </button>
                </div>
                <div className="space-y-1 max-h-24 overflow-y-auto">
                  {saeGroups.filter(g => selectedDirs.has(g.dirPath)).map((group) => (
                    <div key={group.dirPath} className="text-xs text-slate-500 font-mono truncate">
                      {group.dirPath}
                      {group.layer !== null && ` (L${group.layer})`}
                      {group.width && ` • ${group.width}`}
                    </div>
                  ))}
                </div>
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
          disabled={!repositoryId.trim() || (previewData !== null && selectedDirs.size === 0)}
          leftIcon={<Layers className="w-4 h-4" />}
          className="w-full"
        >
          {selectedDirs.size > 0
            ? `Download ${selectedDirs.size} Selected SAE${selectedDirs.size !== 1 ? 's' : ''}`
            : (previewData ? 'Select SAEs to Download' : 'Download SAE Repository')}
        </Button>

        {/* Help Text */}
        <div className="text-xs text-slate-500 bg-slate-800/30 rounded-lg p-3">
          <p className="font-medium text-slate-400 mb-1">Note:</p>
          <p>{previewData
            ? 'Click to select SAEs to download. Each SAE directory (config + weights) is downloaded as a unit.'
            : 'Downloads the entire SAE repository. Layer selection is done when attaching the SAE to a model.'
          }</p>
        </div>
      </div>
    </Card>
  );
}
