import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  ApiResponse,
  DocumentItem,
  DocumentListResponse,
  DocumentStatus,
  DocumentStatusResponse,
  VectorStoreActionResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class DocumentService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/documents`;

  list(options?: {
    search?: string;
    status?: DocumentStatus | '';
    contentType?: string;
  }): Observable<DocumentListResponse> {
    let params = new HttpParams();
    if (options?.search) params = params.set('search', options.search);
    if (options?.status) params = params.set('status', options.status);
    if (options?.contentType) params = params.set('contentType', options.contentType);
    return this.http
      .get<ApiResponse<DocumentListResponse>>(this.baseUrl, { params })
      .pipe(map((res) => res.data!));
  }

  get(id: string): Observable<DocumentItem> {
    return this.http
      .get<ApiResponse<DocumentItem>>(`${this.baseUrl}/${id}`)
      .pipe(map((res) => res.data!));
  }

  getStatus(id: string): Observable<DocumentStatusResponse> {
    return this.http
      .get<ApiResponse<DocumentStatusResponse>>(`${this.baseUrl}/${id}/status`)
      .pipe(map((res) => res.data!));
  }

  upload(
    file: File,
    options?: { chunkSize?: number; chunkOverlap?: number }
  ): Observable<DocumentItem> {
    const form = new FormData();
    form.append('file', file, file.name);
    if (options?.chunkSize != null) {
      form.append('chunkSize', String(options.chunkSize));
    }
    if (options?.chunkOverlap != null) {
      form.append('chunkOverlap', String(options.chunkOverlap));
    }
    return this.http
      .post<ApiResponse<DocumentItem>>(`${this.baseUrl}/upload`, form)
      .pipe(map((res) => res.data!));
  }

  delete(id: string): Observable<void> {
    return this.http
      .delete<ApiResponse<null>>(`${this.baseUrl}/${id}`)
      .pipe(map(() => void 0));
  }

  reprocess(id: string): Observable<DocumentItem> {
    return this.http
      .post<ApiResponse<DocumentItem>>(`${this.baseUrl}/${id}/reprocess`, {})
      .pipe(map((res) => res.data!));
  }

  clearVectorStore(): Observable<VectorStoreActionResponse> {
    return this.http
      .post<ApiResponse<VectorStoreActionResponse>>(`${this.baseUrl}/vector-store/clear`, {})
      .pipe(map((res) => res.data!));
  }

  reprocessAll(): Observable<VectorStoreActionResponse> {
    return this.http
      .post<ApiResponse<VectorStoreActionResponse>>(`${this.baseUrl}/reprocess-all`, {})
      .pipe(map((res) => res.data!));
  }

  clearAndReprocessAll(): Observable<VectorStoreActionResponse> {
    return this.http
      .post<ApiResponse<VectorStoreActionResponse>>(
        `${this.baseUrl}/vector-store/clear-and-reprocess`,
        {}
      )
      .pipe(map((res) => res.data!));
  }

  downloadUrl(id: string): string {
    return `${this.baseUrl}/${id}/download`;
  }
}
