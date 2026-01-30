/**
 * Socket.IO client service for miLLM frontend.
 */

import { io, Socket } from 'socket.io-client';
import { getApiUrl } from './api';
import { DownloadProgress, LoadProgress } from '../types';

type DownloadProgressHandler = (data: DownloadProgress) => void;
type DownloadCompleteHandler = (data: { modelId: number; localPath: string }) => void;
type DownloadErrorHandler = (data: { modelId: number; error: { code: string; message: string } }) => void;
type LoadProgressHandler = (data: LoadProgress) => void;
type LoadCompleteHandler = (data: { modelId: number; memoryUsedMb: number }) => void;
type LoadErrorHandler = (data: { modelId: number; error: { code: string; message: string } }) => void;
type UnloadCompleteHandler = (data: { modelId: number }) => void;

class SocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  // Event handlers
  private downloadProgressHandlers: DownloadProgressHandler[] = [];
  private downloadCompleteHandlers: DownloadCompleteHandler[] = [];
  private downloadErrorHandlers: DownloadErrorHandler[] = [];
  private loadProgressHandlers: LoadProgressHandler[] = [];
  private loadCompleteHandlers: LoadCompleteHandler[] = [];
  private loadErrorHandlers: LoadErrorHandler[] = [];
  private unloadCompleteHandlers: UnloadCompleteHandler[] = [];

  connect(): void {
    if (this.socket?.connected) {
      return;
    }

    this.socket = io(getApiUrl(), {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 1000,
    });

    this.socket.on('connect', () => {
      console.log('Socket.IO connected');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket.IO disconnected:', reason);
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error);
      this.reconnectAttempts++;
    });

    // Model download events
    this.socket.on('model:download:progress', (data: DownloadProgress) => {
      this.downloadProgressHandlers.forEach((handler) => handler(data));
    });

    this.socket.on('model:download:complete', (data: { modelId: number; localPath: string }) => {
      this.downloadCompleteHandlers.forEach((handler) => handler(data));
    });

    this.socket.on('model:download:error', (data: { modelId: number; error: { code: string; message: string } }) => {
      this.downloadErrorHandlers.forEach((handler) => handler(data));
    });

    // Model load events
    this.socket.on('model:load:progress', (data: LoadProgress) => {
      this.loadProgressHandlers.forEach((handler) => handler(data));
    });

    this.socket.on('model:load:complete', (data: { modelId: number; memoryUsedMb: number }) => {
      this.loadCompleteHandlers.forEach((handler) => handler(data));
    });

    this.socket.on('model:load:error', (data: { modelId: number; error: { code: string; message: string } }) => {
      this.loadErrorHandlers.forEach((handler) => handler(data));
    });

    // Model unload events
    this.socket.on('model:unload:complete', (data: { modelId: number }) => {
      this.unloadCompleteHandlers.forEach((handler) => handler(data));
    });
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = null;
  }

  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  // Subscribe methods
  onDownloadProgress(handler: DownloadProgressHandler): () => void {
    this.downloadProgressHandlers.push(handler);
    return () => {
      this.downloadProgressHandlers = this.downloadProgressHandlers.filter((h) => h !== handler);
    };
  }

  onDownloadComplete(handler: DownloadCompleteHandler): () => void {
    this.downloadCompleteHandlers.push(handler);
    return () => {
      this.downloadCompleteHandlers = this.downloadCompleteHandlers.filter((h) => h !== handler);
    };
  }

  onDownloadError(handler: DownloadErrorHandler): () => void {
    this.downloadErrorHandlers.push(handler);
    return () => {
      this.downloadErrorHandlers = this.downloadErrorHandlers.filter((h) => h !== handler);
    };
  }

  onLoadProgress(handler: LoadProgressHandler): () => void {
    this.loadProgressHandlers.push(handler);
    return () => {
      this.loadProgressHandlers = this.loadProgressHandlers.filter((h) => h !== handler);
    };
  }

  onLoadComplete(handler: LoadCompleteHandler): () => void {
    this.loadCompleteHandlers.push(handler);
    return () => {
      this.loadCompleteHandlers = this.loadCompleteHandlers.filter((h) => h !== handler);
    };
  }

  onLoadError(handler: LoadErrorHandler): () => void {
    this.loadErrorHandlers.push(handler);
    return () => {
      this.loadErrorHandlers = this.loadErrorHandlers.filter((h) => h !== handler);
    };
  }

  onUnloadComplete(handler: UnloadCompleteHandler): () => void {
    this.unloadCompleteHandlers.push(handler);
    return () => {
      this.unloadCompleteHandlers = this.unloadCompleteHandlers.filter((h) => h !== handler);
    };
  }
}

export const socketService = new SocketService();
