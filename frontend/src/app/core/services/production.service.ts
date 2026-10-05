import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ApiResponse } from '../models/api.models';

export interface ProductionSpan {
  spanId?: string;
  name: string;
  elapsedMs: number;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  costUsd: number;
  model?: string | null;
  status?: string;
  attributes: Record<string, unknown>;
}

export interface ProductionRequest {
  id: string;
  question: string;
  answer: string;
  model: string | null;
  route: string;
  elapsedMs: number;
  costUsd: number;
  billedTokens: number;
  spans: ProductionSpan[];
}

export interface SideSummary {
  mode: string;
  requests: number;
  meanCostUsd: number;
  costPerThousandUsd: number;
  meanElapsedMs: number;
  meanBilledTokens: number;
}

export interface ProductionReport {
  label: string;
  method: { note: string };
  baseline: SideSummary;
  improved: SideSummary;
  improvedItems: ProductionRequest[];
  savings: { costPct: number; latencyPct: number; billedTokenPct: number };
  fallback: {
    limit: number;
    second: { id: string; model: string; route: string; reason: string; fallbackFrom: string };
  };
  tenX: {
    breaksFirst: string;
    generateShareOfTime: number;
    oneWorkerRequestsPerMinute: number;
    atTenX: string;
    plan: string[];
  };
  failure: {
    id: string;
    sourceTraceId: string;
    question: string;
    badAnswer: string;
    fixedAnswer: string;
    note: string;
  };
  fineTuning: string;
  observability: {
    openTelemetry: string;
    phoenix: string;
    langsmith: string;
  };
}

export interface SupportHit {
  requestId: string;
  timestamp: string;
  question: string;
  answer: string;
  model: string;
  matched: string[];
  score: number;
}

export interface SupportResult {
  complaint: string;
  when: string | null;
  restrictedToDay: boolean;
  scanned: number;
  hits: SupportHit[];
}

@Injectable({ providedIn: 'root' })
export class ProductionService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/production`;

  getReport(): Observable<ProductionReport> {
    return this.http
      .get<ApiResponse<ProductionReport>>(`${this.baseUrl}/report`)
      .pipe(map((res) => res.data!));
  }

  search(complaint: string): Observable<SupportResult> {
    return this.http
      .post<ApiResponse<SupportResult>>(`${this.baseUrl}/support`, { complaint })
      .pipe(map((res) => res.data!));
  }
}
