import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { ProductionComponent } from './production.component';

describe('ProductionComponent', () => {
  let fixture: ComponentFixture<ProductionComponent>;
  let http: HttpTestingController;

  const report = {
    success: true,
    data: {
      label: 'scripted',
      method: { note: 'Same scripted answers.' },
      baseline: {
        mode: 'baseline',
        requests: 8,
        meanCostUsd: 0.000165,
        costPerThousandUsd: 0.1654,
        meanElapsedMs: 186,
        meanBilledTokens: 184,
      },
      improved: {
        mode: 'improved',
        requests: 8,
        meanCostUsd: 0.000041,
        costPerThousandUsd: 0.0414,
        meanElapsedMs: 125,
        meanBilledTokens: 80,
      },
      improvedItems: [
        {
          id: 'deductible',
          question: 'What deductible applies?',
          answer: 'USD 1,000',
          model: 'llama3.2:1b',
          route: 'cheap',
          elapsedMs: 88,
          costUsd: 0.000018,
          billedTokens: 158,
          spans: [
            {
              spanId: 's1',
              name: 'retrieve',
              elapsedMs: 45,
              inputTokens: 6,
              outputTokens: 0,
              cachedInputTokens: 0,
              costUsd: 0.0000006,
              model: null,
              attributes: {},
            },
            {
              spanId: 's2',
              name: 'generate',
              elapsedMs: 42,
              inputTokens: 140,
              outputTokens: 12,
              cachedInputTokens: 0,
              costUsd: 0.000017,
              model: 'llama3.2:1b',
              attributes: {},
            },
          ],
        },
      ],
      savings: { costPct: 75, latencyPct: 32.8, billedTokenPct: 56.6 },
      fallback: {
        limit: 1,
        second: {
          id: 'investigator',
          model: 'llama3.2',
          route: 'fallback',
          reason: 'rate_limit on llama3.2:1b; fell back to llama3.2',
          fallbackFrom: 'llama3.2:1b',
        },
      },
      tenX: {
        breaksFirst: 'generate',
        generateShareOfTime: 0.706,
        oneWorkerRequestsPerMinute: 480,
        atTenX: 'Ten times that rate queues on generate.',
        plan: ['Add a second model process when the generate queue grows.'],
      },
      failure: {
        id: 'hallucinated-investigator',
        sourceTraceId: 'seed-009-gen-halluc',
        question: 'Who investigated claim CLM-2024-00847?',
        badAnswer: 'The claim was investigated by field surveyor M. Chen.',
        fixedAnswer: 'The claim was investigated by field adjuster A. Ramirez.',
        note: 'Context names Ramirez.',
      },
      fineTuning: 'Fine-tuning is the last resort.',
      observability: {
        openTelemetry: 'Spans.',
        phoenix: 'Phoenix shows the spans.',
        langsmith: 'LangSmith is a hosted view.',
      },
    },
  };

  const support = {
    success: true,
    data: {
      complaint: 'yesterday it gave the wrong investigator name',
      when: '2026-10-04',
      restrictedToDay: true,
      scanned: 1009,
      hits: [
        {
          requestId: 'req-bad-investigator',
          timestamp: '2026-10-04T16:00:00+00:00',
          question: 'Who investigated claim CLM-2024-00847?',
          answer: 'The claim was investigated by field surveyor M. Chen.',
          model: 'llama3.2',
          matched: ['investigator'],
          score: 1,
        },
      ],
    },
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProductionComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(ProductionComponent);
    http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    http.expectOne('/api/production/report').flush(report);
    http.expectOne('/api/production/support').flush(support);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.verify();
  });

  it('shows per-step cost and the saving', () => {
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('75% lower');
    expect(text).toContain('generate');
    expect(text).toContain('retrieve');
    expect(text).toContain('M. Chen');
    expect(text).toContain('A. Ramirez');
    expect(text).toContain('req-bad-investigator');
  });

  it('searches again when the complaint changes', () => {
    const cmp = fixture.componentInstance;
    cmp.complaint = 'what deductible applies yesterday';
    cmp.runSearch();
    const req = http.expectOne('/api/production/support');
    expect(req.request.body.complaint).toContain('deductible');
    req.flush(support);
  });
});
