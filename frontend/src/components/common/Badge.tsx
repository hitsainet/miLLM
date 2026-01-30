/**
 * Reusable Badge component for status display.
 */

import React from 'react';
import { ModelStatus } from '../../types';

interface BadgeProps {
  status: ModelStatus;
  className?: string;
}

const statusConfig: Record<ModelStatus, { label: string; classes: string }> = {
  downloading: {
    label: 'Downloading',
    classes: 'bg-blue-100 text-blue-800',
  },
  ready: {
    label: 'Ready',
    classes: 'bg-gray-100 text-gray-800',
  },
  loading: {
    label: 'Loading',
    classes: 'bg-yellow-100 text-yellow-800',
  },
  loaded: {
    label: 'Loaded',
    classes: 'bg-green-100 text-green-800',
  },
  error: {
    label: 'Error',
    classes: 'bg-red-100 text-red-800',
  },
};

export function Badge({ status, className = '' }: BadgeProps) {
  const config = statusConfig[status];

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.classes} ${className}`}
    >
      {status === 'downloading' && (
        <svg
          className="animate-spin -ml-0.5 mr-1.5 h-3 w-3"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
      {status === 'loading' && (
        <svg
          className="animate-pulse -ml-0.5 mr-1.5 h-3 w-3"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <circle cx="10" cy="10" r="10" />
        </svg>
      )}
      {status === 'loaded' && (
        <svg
          className="-ml-0.5 mr-1.5 h-3 w-3"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
      )}
      {config.label}
    </span>
  );
}
