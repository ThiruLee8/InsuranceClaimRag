import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  ApiResponse,
  ConversationItem,
  MessageItem,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ConversationService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/conversations`;

  list(): Observable<ConversationItem[]> {
    return this.http
      .get<ApiResponse<ConversationItem[]>>(this.baseUrl)
      .pipe(map((res) => res.data ?? []));
  }

  create(title = 'New Conversation'): Observable<ConversationItem> {
    return this.http
      .post<ApiResponse<ConversationItem>>(this.baseUrl, { title })
      .pipe(map((res) => res.data!));
  }

  get(id: string): Observable<ConversationItem> {
    return this.http
      .get<ApiResponse<ConversationItem>>(`${this.baseUrl}/${id}`)
      .pipe(map((res) => res.data!));
  }

  rename(id: string, title: string): Observable<ConversationItem> {
    return this.http
      .put<ApiResponse<ConversationItem>>(`${this.baseUrl}/${id}`, { title })
      .pipe(map((res) => res.data!));
  }

  delete(id: string): Observable<void> {
    return this.http
      .delete<ApiResponse<null>>(`${this.baseUrl}/${id}`)
      .pipe(map(() => void 0));
  }

  getMessages(id: string): Observable<MessageItem[]> {
    return this.http
      .get<ApiResponse<MessageItem[]>>(`${this.baseUrl}/${id}/messages`)
      .pipe(map((res) => res.data ?? []));
  }

  clearMessages(id: string): Observable<void> {
    return this.http
      .delete<ApiResponse<null>>(`${this.baseUrl}/${id}/messages`)
      .pipe(map(() => void 0));
  }
}
