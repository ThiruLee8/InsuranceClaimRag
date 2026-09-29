import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TeamComponent } from './team.component';
import { NotificationService } from '../../core/services/notification.service';

describe('TeamComponent', () => {
  let fixture: ComponentFixture<TeamComponent>;
  let http: HttpTestingController;

  const meta = {
    success: true,
    data: {
      cards: [
        {
          id: 'claims_manager',
          name: 'Claims Manager',
          description: 'Splits the work',
          skills: ['route'],
          url: '/api/multi-agent/race',
          version: '0.1',
        },
      ],
      lifecycle: ['submitted', 'working', 'completed', 'failed', 'canceled'],
      mcpVsA2a: 'MCP tools. A2A agents.',
      frameworks: 'CrewAI hides the hand-off.',
      decisionRule: 'Quality, then cost, then speed.',
      cases: [
        {
          id: 'simple-cause',
          label: 'One fact',
          task: 'What was the cause of the loss?',
          why: 'One lookup.',
          must_contain_any: [['burst']],
        },
      ],
      suite: {
        label: 'scripted',
        createdAt: '2026-09-28T00:00:00Z',
        method: { model: 'extractive-stand-in', llmDelaySeconds: 0.05, note: 'Same four questions.' },
        cases: [],
        summary: {
          cases: 4,
          single: { quality: 1, elapsedMs: 200, estimatedTokens: 1000, costUnits: 2 },
          team: { quality: 1, elapsedMs: 140, estimatedTokens: 1800, costUnits: 3.2 },
          contextResendTokensMean: 40,
          verdict: {
            ship: 'single',
            winner: 'single',
            decidedBy: 'cost',
            rule: 'Quality, then cost, then speed.',
            rationale: 'Quality is tied, so the lower token bill decides it.',
            comparison: {},
          },
          worthIt: {
            when: 'Independent slices.',
            whenNot: 'One lookup.',
            parallelFasterOn: ['parallel-brief'],
          },
        },
      },
    },
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TeamComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        {
          provide: NotificationService,
          useValue: { success: () => undefined, error: () => undefined, handleHttpError: () => undefined },
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(TeamComponent);
    http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    http.expectOne('/api/multi-agent/meta').flush(meta);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.verify();
  });

  it('shows the four suite numbers and the shipped side', () => {
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('Quality');
    expect(text).toContain('Speed');
    expect(text).toContain('Tokens');
    expect(text).toContain('Cost');
    expect(text).toContain('Single agent');
    expect(text).toContain('Quality is tied');
  });

  it('keeps the selected case id when that question is raced', () => {
    const cmp = fixture.componentInstance;
    expect(cmp.selectedCaseId).toBe('simple-cause');
    cmp.raceTask();
    const req = http.expectOne('/api/multi-agent/race');
    expect(req.request.body.caseId).toBe('simple-cause');
    expect(req.request.body.task).toContain('cause of the loss');
    req.flush({
      success: true,
      data: {
        task: cmp.task,
        caseId: 'simple-cause',
        single: {
          answer: 'burst supply line',
          summary: { quality: 1, elapsedMs: 10, estimatedTokens: 20, costUnits: 1 },
          metrics: { llmCalls: 2, elapsedMs: 10 },
        },
        team: {
          answer: 'burst supply line',
          summary: { quality: 1, elapsedMs: 12, estimatedTokens: 40, costUnits: 2 },
          plan: { workers: ['loss_investigator'], parallel: false, reason: 'Only loss.' },
          tasks: [
            {
              id: 't1',
              specialistId: 'loss_investigator',
              specialistName: 'Loss Investigator',
              state: 'completed',
              history: ['submitted', 'working', 'completed'],
              instruction: 'cause',
              artifact: 'burst',
              contextResendTokens: 8,
              elapsedMs: 4,
              searchQuery: 'cause',
            },
          ],
          metrics: {
            llmCalls: 2,
            contextResendTokens: 16,
            workerElapsedMs: 4,
            elapsedMs: 12,
            estimatedTokens: 40,
            costUnits: 2,
          },
        },
        verdict: {
          ship: 'single',
          winner: 'single',
          decidedBy: 'cost',
          rule: 'Quality, then cost.',
          rationale: 'Ship the single agent.',
          comparison: {},
        },
        plannerCallTokens: 30,
        plannerNote: 'Not included.',
      },
    });
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('submitted → working → completed');
  });
});
