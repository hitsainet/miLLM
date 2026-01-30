/**
 * Axios API client configuration.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import { ApiResponse, ErrorDetails, createApiError } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor to handle API errors
api.interceptors.response.use(
  (response) => {
    // Check if response follows ApiResponse format
    const data = response.data as ApiResponse<unknown>;
    if (data && typeof data.success === 'boolean' && !data.success && data.error) {
      throw createApiError(data.error);
    }
    return response;
  },
  (error: AxiosError<ApiResponse<unknown>>) => {
    // Handle network errors
    if (!error.response) {
      throw createApiError({
        code: 'NETWORK_ERROR',
        message: 'Network error. Please check your connection.',
        details: {},
      });
    }

    // Handle API errors
    const data = error.response.data;
    if (data?.error) {
      throw createApiError(data.error);
    }

    // Handle generic HTTP errors
    throw createApiError({
      code: `HTTP_${error.response.status}`,
      message: error.message || 'An error occurred',
      details: {},
    });
  }
);

export function getApiUrl(): string {
  return API_BASE_URL;
}
