import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  AgentsResponse,
  ApiResponse,
  ChatResponse,
  RagSource,
  RetrieveResponse,
} from '../models/api.models';

export type ChatStreamEvent =
  | { type: 'meta'; conversationId: string }
  | {
      type: 'status';
      stage: string;
      agentId?: string;
      model?: string;
      sourceCount?: number;
      originalQuestion?: string;
      searchQuery?: string;
      searchMode?: string;
    }
  | { type: 'sources'; sources: RagSource[] }
  | { type: 'token'; content: string }
  | {
      type: 'done';
      conversationId: string;
      messageId: string;
      answer: string;
      sources: RagSource[];
      agentId?: string | null;
      model?: string | null;
      originalQuestion?: string | null;
      searchQuery?: string | null;
      searchMode?: string | null;
    }
  | { type: 'error'; message: string; errorCode?: string };

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/chat`;
  private readonly agentsUrl = `${environment.apiBaseUrl}/agents`;

  listAgents(): Observable<AgentsResponse> {
    return this.http
      .get<ApiResponse<AgentsResponse>>(this.agentsUrl)
      .pipe(map((res) => res.data!));
  }

  ask(
    question: string,
    conversationId?: string | null,
    options?: { agentId?: string | null; model?: string | null; debug?: boolean }
  ): Observable<ChatResponse> {
    return this.http
      .post<ApiResponse<ChatResponse>>(this.baseUrl, {
        conversationId: conversationId || null,
        question,
        agentId: options?.agentId || null,
        model: options?.model || null,
        debug: !!options?.debug,
      })
      .pipe(map((res) => res.data!));
  }

  retrieve(
    question: string,
    options?: {
      topK?: number;
      searchMode?: string | null;
      rewrite?: boolean;
      debug?: boolean;
    }
  ): Observable<RetrieveResponse> {
    return this.http
      .post<ApiResponse<RetrieveResponse>>(`${this.baseUrl}/retrieve`, {
        question,
        topK: options?.topK ?? null,
        searchMode: options?.searchMode || null,
        rewrite: options?.rewrite ?? true,
        debug: options?.debug ?? true,
      })
      .pipe(map((res) => res.data!));
  }

  askStream(
    question: string,
    conversationId?: string | null,
    options?: {
      agentId?: string | null;
      model?: string | null;
      regenerateMessageId?: string | null;
      reuseLastUserMessage?: boolean;
      debug?: boolean;
    }
  ): Observable<ChatStreamEvent> {
    const body = {
      conversationId: conversationId || null,
      question,
      agentId: options?.agentId || null,
      model: options?.model || null,
      regenerateMessageId: options?.regenerateMessageId || null,
      reuseLastUserMessage: !!options?.reuseLastUserMessage,
      debug: !!options?.debug,
    };

    return new Observable<ChatStreamEvent>((subscriber) => {
      const controller = new AbortController();

      (async () => {
        try {
          const response = await fetch(`${this.baseUrl}/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            body: JSON.stringify(body),
            signal: controller.signal,
          });

          if (!response.ok || !response.body) {
            const text = await response.text().catch(() => '');
            throw new Error(text || `Chat stream failed (${response.status})`);
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
              const event = JSON.parse(raw) as ChatStreamEvent;
              subscriber.next(event);
              if (event.type === 'error') {
                subscriber.error(new Error(event.message));
                return;
              }
              if (event.type === 'done') {
                subscriber.complete();
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

  regenerate(
    question: string,
    conversationId: string,
    regenerateMessageId: string,
    options?: { agentId?: string | null; model?: string | null; debug?: boolean }
  ): Observable<ChatResponse> {
    return this.http
      .post<ApiResponse<ChatResponse>>(this.baseUrl, {
        conversationId,
        question,
        regenerateMessageId,
        agentId: options?.agentId || null,
        model: options?.model || null,
        debug: !!options?.debug,
      })
      .pipe(map((res) => res.data!));
  }
}
