/**
 * ModelCard component for displaying a single model.
 */

import React from 'react';
import { Model, DownloadProgress } from '../../types';
import { Card, CardContent, Badge, Button, ProgressBar } from '../common';
import { formatBytes, formatMemory } from '../../utils/format';

interface ModelCardProps {
  model: Model;
  downloadProgress?: DownloadProgress;
  isLoaded: boolean;
  onLoad: () => void;
  onUnload: () => void;
  onDelete: () => void;
  onCancelDownload: () => void;
}

export function ModelCard({
  model,
  downloadProgress,
  isLoaded,
  onLoad,
  onUnload,
  onDelete,
  onCancelDownload,
}: ModelCardProps) {
  const isDownloading = model.status === 'downloading';
  const isLoading = model.status === 'loading';
  const isReady = model.status === 'ready';
  const isError = model.status === 'error';

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent>
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
            {model.repoId && (
              <p className="text-sm text-gray-500">{model.repoId}</p>
            )}
          </div>
          <Badge status={model.status} />
        </div>

        {/* Model Info */}
        <div className="grid grid-cols-2 gap-2 text-sm mb-4">
          {model.params && (
            <div>
              <span className="text-gray-500">Params:</span>{' '}
              <span className="font-medium">{model.params}</span>
            </div>
          )}
          <div>
            <span className="text-gray-500">Quant:</span>{' '}
            <span className="font-medium">{model.quantization}</span>
          </div>
          {model.diskSizeMb && (
            <div>
              <span className="text-gray-500">Disk:</span>{' '}
              <span className="font-medium">{formatMemory(model.diskSizeMb)}</span>
            </div>
          )}
          {model.estimatedMemoryMb && (
            <div>
              <span className="text-gray-500">VRAM:</span>{' '}
              <span className="font-medium">~{formatMemory(model.estimatedMemoryMb)}</span>
            </div>
          )}
        </div>

        {/* Download Progress */}
        {isDownloading && downloadProgress && (
          <div className="mb-4">
            <ProgressBar
              progress={downloadProgress.progress}
              label="Downloading"
              size="md"
            />
            <p className="text-xs text-gray-500 mt-1">
              {formatBytes(downloadProgress.downloadedBytes)} / {formatBytes(downloadProgress.totalBytes)}
              {downloadProgress.speedBps && ` - ${formatBytes(downloadProgress.speedBps)}/s`}
            </p>
          </div>
        )}

        {/* Loading Progress */}
        {isLoading && (
          <div className="mb-4">
            <ProgressBar
              progress={50}
              label="Loading to GPU"
              size="md"
              color="yellow"
            />
          </div>
        )}

        {/* Error Message */}
        {isError && model.errorMessage && (
          <div className="mb-4 p-2 bg-red-50 rounded text-sm text-red-700">
            {model.errorMessage}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          {isDownloading && (
            <Button variant="danger" size="sm" onClick={onCancelDownload}>
              Cancel
            </Button>
          )}
          {isReady && (
            <>
              <Button variant="primary" size="sm" onClick={onLoad}>
                Load
              </Button>
              <Button variant="danger" size="sm" onClick={onDelete}>
                Delete
              </Button>
            </>
          )}
          {isLoaded && (
            <Button variant="secondary" size="sm" onClick={onUnload}>
              Unload
            </Button>
          )}
          {isError && (
            <Button variant="danger" size="sm" onClick={onDelete}>
              Delete
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
