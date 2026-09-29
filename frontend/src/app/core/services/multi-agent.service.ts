import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ApiResponse } from '../models/api.models';

export interface TeamSideSummary {
  quality: number | null;
  elapsedMs: number;
  estimatedTokens: number;
  costUnits: number;
  llmCalls?: number;
  workerElapsedMs?: number;
  answer?: string;
}

export interface TeamVerdict {
  ship: 'single' | 'team';
  winner: string;
  decidedBy: string;
  rule: string;
  rationale: string;
  comparison: Record<string, { single: number | null; team: number | null; delta: number | null }>;
}

export interface TeamCaseResult {
  id: string;
  label: string;
  task: string;
  why: string;
  parallel: boolean;
  workers: string[];
  planReason: string;
  contextResendTokens: number;
  single: TeamSideSummary;
  team: TeamSideSummary;
  verdict: TeamVerdict;
}

export interface TeamSuite {
  label: string;
  createdAt: string;
  method: { model: string; llmDelaySeconds: number; note: string };
  cases: TeamCaseResult[];
  summary: {
    cases: number;
    single: TeamSideSummary;
    team: TeamSideSummary;
    contextResendTokensMean: number;
    verdict: TeamVerdict;
    worthIt: { when: string; whenNot: string; parallelFasterOn: string[] };
  };
}

export interface TeamCase {
  id: string;
  label: string;
  task: string;
  why: string;
  must_contain_any: string[][];
}

export interface AgentCard {
  id: string;
  name: string;
  description: string;
  skills: string[];
  url: string;
  version: string;
}

export interface TeamMeta {
  cards: AgentCard[];
  lifecycle: string[];
  mcpVsA2a: string;
  frameworks: string;
  decisionRule: string;
  cases: TeamCase[];
  suite: TeamSuite | null;
}

export interface SpecialistTask {
  id: string;
  specialistId: string;
  specialistName: string;
  state: string;
  history: string[];
  instruction: string;
  artifact: string;
  contextResendTokens: number;
  elapsedMs: number;
  searchQuery: string;
}

export interface TeamRace {
  task: string;
  caseId?: string | null;
  single: { answer: string; summary: TeamSideSummary; metrics: { llmCalls: number; elapsedMs: number } };
  team: {
    answer: string;
    summary: TeamSideSummary;
    plan: { workers: string[]; parallel: boolean; reason: string };
    tasks: SpecialistTask[];
    metrics: {
      llmCalls: number;
      contextResendTokens: number;
      workerElapsedMs: number;
      elapsedMs: number;
      estimatedTokens: number;
      costUnits: number;
    };
  };
  verdict: TeamVerdict;
  plannerCallTokens: number;
  plannerNote: string;
}

@Injectable({ providedIn: 'root' })
export class MultiAgentService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/multi-agent`;

  getMeta(): Observable<TeamMeta> {
    return this.http
      .get<ApiResponse<TeamMeta>>(`${this.baseUrl}/meta`)
      .pipe(map((res) => res.data!));
  }

  race(task: string, caseId: string | null): Observable<TeamRace> {
    return this.http
      .post<ApiResponse<TeamRace>>(`${this.baseUrl}/race`, {
        task,
        caseId,
        maxSteps: 8,
        maxLlmCalls: 10,
        maxSeconds: 120,
      })
      .pipe(map((res) => res.data!));
  }
}
