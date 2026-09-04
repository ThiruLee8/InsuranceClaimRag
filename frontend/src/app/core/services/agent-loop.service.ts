import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  AgentLoopMeta,
  AgentLoopMode,
  AgentLoopRun,
  AgentLoopStreamEvent,
  AgentMemoryList,
  AgentRaceResult,
  ApiResponse,
} from '../models/api.models';

export interface AgentLoopRunOptions {
  mode: AgentLoopMode;
  task: string;
  sessionId?: string | null;
  model?: string | null;
  maxSteps: number;
  maxLlmCalls: number;
  maxSeconds: number;
  persistMemory: boolean;
}

@Injectable({ providedIn: 'root' })
export class AgentLoopService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/agent-loop`;

  getMeta(): Observable<AgentLoopMeta> {
    return this.http
      .get<ApiResponse<AgentLoopMeta>>(`${this.baseUrl}/meta`)
      .pipe(map((res) => res.data!));
  }

  listMemory(sessionId?: string | null): Observable<AgentMemoryList> {
    let params = new HttpParams();
    if (sessionId) params = params.set('sessionId', sessionId);
    return this.http
      .get<ApiResponse<AgentMemoryList>>(`${this.baseUrl}/memory`, { params })
      .pipe(map((res) => res.data!));
  }

  clearMemory(sessionId?: string | null): Observable<{ removed: number; sessionId?: string | null }> {
    let params = new HttpParams();
    if (sessionId) params = params.set('sessionId', sessionId);
    return this.http
      .delete<ApiResponse<{ removed: number; sessionId?: string | null }>>(`${this.baseUrl}/memory`, {
        params,
      })
      .pipe(map((res) => res.data!));
  }

  run(options: AgentLoopRunOptions): Observable<AgentLoopRun | AgentRaceResult> {
    return this.http
      .post<ApiResponse<AgentLoopRun | AgentRaceResult>>(`${this.baseUrl}/run`, options)
      .pipe(map((res) => res.data!));
  }

  runStream(options: AgentLoopRunOptions): Observable<AgentLoopStreamEvent> {
    return new Observable<AgentLoopStreamEvent>((subscriber) => {
      const controller = new AbortController();

      (async () => {
        try {
          const response = await fetch(`${this.baseUrl}/run/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            body: JSON.stringify(options),
            signal: controller.signal,
          });

          if (!response.ok || !response.body) {
            const text = await response.text().catch(() => '');
            throw new Error(text || `Agent stream failed (${response.status})`);
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop() || '';

            for (const part of parts) {
              const dataLine = part
                .split('\n')
                .map((line) => line.trimEnd())
                .find((line) => line.startsWith('data:'));
              if (!dataLine) continue;
              const raw = dataLine.replace(/^data:\s?/, '').trim();
              if (!raw || raw === '[DONE]') continue;
              const event = JSON.parse(raw) as AgentLoopStreamEvent;
              subscriber.next(event);
              if (event.type === 'error') {
                subscriber.error(new Error(event.message));
                return;
              }
            }
          }

          subscriber.complete();
        } catch (err) {
          if ((err as Error).name === 'AbortError') {
            subscriber.complete();
            return;
          }
          subscriber.error(err);
        }
      })();

      return () => controller.abort();
    });
  }
}
