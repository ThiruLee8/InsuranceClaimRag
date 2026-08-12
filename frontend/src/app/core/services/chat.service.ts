import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AgentsResponse, ApiResponse, ChatResponse } from '../models/api.models';

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
    options?: { agentId?: string | null; model?: string | null }
  ): Observable<ChatResponse> {
    return this.http
      .post<ApiResponse<ChatResponse>>(this.baseUrl, {
        conversationId: conversationId || null,
        question,
        agentId: options?.agentId || null,
        model: options?.model || null,
      })
      .pipe(map((res) => res.data!));
  }

  regenerate(
    question: string,
    conversationId: string,
    regenerateMessageId: string,
    options?: { agentId?: string | null; model?: string | null }
  ): Observable<ChatResponse> {
    return this.http
      .post<ApiResponse<ChatResponse>>(this.baseUrl, {
        conversationId,
        question,
        regenerateMessageId,
        agentId: options?.agentId || null,
        model: options?.model || null,
      })
      .pipe(map((res) => res.data!));
  }
}
