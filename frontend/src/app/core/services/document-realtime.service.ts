import { Injectable, NgZone, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { DocumentItem, DocumentStatus } from '../models/api.models';

export interface DocumentListSocketMessage {
  type: 'snapshot';
  total: number;
  documents: DocumentItem[];
}

export interface DocumentDetailSocketMessage {
  type: 'document';
  document: DocumentItem | null;
  deleted?: boolean;
}

function resolveWsBaseUrl(): string {
  if (environment.wsBaseUrl) {
    return environment.wsBaseUrl;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api`;
}

@Injectable({ providedIn: 'root' })
export class DocumentRealtimeService {
  private readonly zone = inject(NgZone);

  watchList(options?: {
    search?: string;
    status?: DocumentStatus | '';
    contentType?: string;
  }): Observable<DocumentListSocketMessage> {
    const params = new URLSearchParams();
    if (options?.search) params.set('search', options.search);
    if (options?.status) params.set('status', options.status);
    if (options?.contentType) params.set('contentType', options.contentType);
    const query = params.toString();
    const url = `${resolveWsBaseUrl()}/ws/documents${query ? `?${query}` : ''}`;
    return this.connect<DocumentListSocketMessage>(url);
  }

  watchDocument(documentId: string): Observable<DocumentDetailSocketMessage> {
    const url = `${resolveWsBaseUrl()}/ws/documents?documentId=${encodeURIComponent(documentId)}`;
    return this.connect<DocumentDetailSocketMessage>(url);
  }

  private connect<T>(url: string): Observable<T> {
    return new Observable<T>((subscriber) => {
      let socket: WebSocket | null = null;
      let closedByUser = false;
      let retryMs = 1000;
      let retryTimer: ReturnType<typeof setTimeout> | undefined;

      const open = () => {
        if (closedByUser || subscriber.closed) return;
        socket = new WebSocket(url);

        socket.onopen = () => {
          retryMs = 1000;
        };

        socket.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as T;
            this.zone.run(() => subscriber.next(payload));
            retryMs = 1000;
          } catch (err) {
            this.zone.run(() => subscriber.error(err));
          }
        };

        socket.onerror = () => {
          // onclose handles retry
        };

        socket.onclose = () => {
          if (closedByUser || subscriber.closed) return;
          retryTimer = setTimeout(() => {
            retryMs = Math.min(retryMs * 2, 10000);
            open();
          }, retryMs);
        };
      };

      // Keep WebSocket outside Angular zone; push updates back in via zone.run.
      this.zone.runOutsideAngular(() => open());

      return () => {
        closedByUser = true;
        if (retryTimer) clearTimeout(retryTimer);
        socket?.close();
      };
    });
  }
}
