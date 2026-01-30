/**
 * API response types for miLLM frontend.
 */

export interface ErrorDetails {
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: ErrorDetails | null;
}

export interface ApiError extends Error {
  code: string;
  details: Record<string, unknown>;
}

export function isApiError(error: unknown): error is ApiError {
  return (
    error instanceof Error &&
    'code' in error &&
    typeof (error as ApiError).code === 'string'
  );
}

export function createApiError(errorDetails: ErrorDetails): ApiError {
  const error = new Error(errorDetails.message) as ApiError;
  error.code = errorDetails.code;
  error.details = errorDetails.details;
  return error;
}
