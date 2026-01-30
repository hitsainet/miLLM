/**
 * MemoryEstimate component for displaying memory requirements.
 */

import React from 'react';
import { formatMemory } from '../../utils/format';

interface MemoryEstimateProps {
  estimatedMemoryMb: number;
  availableMemoryMb?: number;
  className?: string;
}

export function MemoryEstimate({
  estimatedMemoryMb,
  availableMemoryMb,
  className = '',
}: MemoryEstimateProps) {
  const hasEnoughMemory = availableMemoryMb === undefined || availableMemoryMb >= estimatedMemoryMb;
  const memoryPercentage = availableMemoryMb
    ? Math.min(100, (estimatedMemoryMb / availableMemoryMb) * 100)
    : 0;

  return (
    <div className={`rounded-lg border p-4 ${className}`}>
      <h4 className="text-sm font-medium text-gray-700 mb-3">Memory Requirements</h4>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Estimated VRAM:</span>
          <span className="font-medium text-gray-900">
            ~{formatMemory(estimatedMemoryMb)}
          </span>
        </div>

        {availableMemoryMb !== undefined && (
          <>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Available VRAM:</span>
              <span className="font-medium text-gray-900">
                {formatMemory(availableMemoryMb)}
              </span>
            </div>

            {/* Memory usage bar */}
            <div className="mt-3">
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    hasEnoughMemory ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${memoryPercentage}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {memoryPercentage.toFixed(0)}% of available memory
              </p>
            </div>

            {/* Warning message */}
            {!hasEnoughMemory && (
              <div className="mt-2 p-2 bg-red-50 rounded text-sm text-red-700">
                <strong>Warning:</strong> Insufficient VRAM. Model requires{' '}
                {formatMemory(estimatedMemoryMb - availableMemoryMb)} more memory.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
