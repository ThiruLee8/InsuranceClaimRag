import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, ElementRef, OnInit, ViewChild, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import {
  AgentLoopMeta,
  AgentLoopMode,
  AgentLoopRun,
  AgentMemoryItem,
  AgentRaceResult,
  AgentSampleTask,
  AgentStepItem,
} from '../../core/models/api.models';
import { AgentLoopService } from '../../core/services/agent-loop.service';
import { NotificationService } from '../../core/services/notification.service';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';

const SESSION_KEY = 'claimintel.agentSessionId';

@Component({
  selector: 'app-agent',
  standalone: true,
  imports: [
    DatePipe,
    DecimalPipe,
    RouterLink,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatProgressBarModule,
    MarkdownPipe,
  ],
  templateUrl: './agent.component.html',
  styleUrl: './agent.component.scss',
})
export class AgentComponent implements OnInit {
  @ViewChild('stepPane') stepPane?: ElementRef<HTMLDivElement>;

  private readonly agentService = inject(AgentLoopService);
  private readonly notifications = inject(NotificationService);
  private readonly fb = inject(FormBuilder);

  meta: AgentLoopMeta | null = null;
  mode: AgentLoopMode = 'race';
  running = false;
  activeRunMode: string | null = null;
  sessionId = AgentComponent.readOrCreateSession();
  memories: AgentMemoryItem[] = [];

  agentSteps: AgentStepItem[] = [];
  workflowSteps: AgentStepItem[] = [];
  agentResult: AgentLoopRun | null = null;
  workflowResult: AgentLoopRun | null = null;
  race: AgentRaceResult | null = null;
  errorMessage: string | null = null;

  form = this.fb.group({
    task: ['', [Validators.required, Validators.maxLength(4000)]],
    maxSteps: [8, [Validators.required, Validators.min(1), Validators.max(20)]],
    maxLlmCalls: [10, [Validators.required, Validators.min(1), Validators.max(30)]],
    maxSeconds: [90, [Validators.required, Validators.min(0), Validators.max(300)]],
    persistMemory: [true],
  });

  readonly modes: { id: AgentLoopMode; label: string; hint: string }[] = [
    { id: 'agent', label: 'Agent loop', hint: 'Plan → act → observe until done' },
    { id: 'workflow', label: 'Fixed workflow', hint: 'Known checklist, one synthesis' },
    { id: 'race', label: 'Race both', hint: 'Same task, real numbers' },
  ];

  ngOnInit(): void {
    this.agentService.getMeta().subscribe({
      next: (meta) => {
        this.meta = meta;
        this.form.patchValue({
          maxSteps: meta.defaultMaxSteps,
          maxLlmCalls: meta.defaultMaxLlmCalls,
          maxSeconds: meta.defaultMaxSeconds,
        });
        const first = meta.sampleTasks[0];
        if (first && !this.form.value.task) {
          this.form.patchValue({ task: first.task });
        }
      },
    });
    this.refreshMemory();
  }

  get sampleTasks(): AgentSampleTask[] {
    return this.meta?.sampleTasks ?? [];
  }

  get answer(): string {
    if (this.mode === 'workflow') return this.workflowResult?.answer || '';
    if (this.mode === 'agent') return this.agentResult?.answer || '';
    return this.agentResult?.answer || this.workflowResult?.answer || '';
  }

  useSample(sample: AgentSampleTask): void {
    this.form.patchValue({ task: sample.task });
    if (sample.id === 'checklist') this.mode = 'race';
    if (sample.id === 'investigate') this.mode = 'agent';
    if (sample.id === 'simple') this.mode = 'race';
    if (sample.id === 'memory') this.mode = 'agent';
  }

  setMode(mode: AgentLoopMode): void {
    if (this.running) return;
    this.mode = mode;
  }

  run(): void {
    if (this.form.invalid || this.running) return;
    const value = this.form.getRawValue();
    const task = (value.task || '').trim();
    if (!task) return;

    this.running = true;
    this.errorMessage = null;
    this.agentSteps = [];
    this.workflowSteps = [];
    this.agentResult = null;
    this.workflowResult = null;
    this.race = null;
    this.activeRunMode = this.mode === 'race' ? 'agent' : this.mode;

    this.agentService
      .runStream({
        mode: this.mode,
        task,
        sessionId: this.sessionId,
        maxSteps: Number(value.maxSteps),
        maxLlmCalls: Number(value.maxLlmCalls),
        maxSeconds: Number(value.maxSeconds),
        persistMemory: !!value.persistMemory,
      })
      .subscribe({
        next: (event) => {
          if (event.type === 'run_start') {
            this.activeRunMode = event.mode;
            return;
          }
          if (event.type === 'step') {
            this.upsertStep(event.mode, event.step);
            this.scrollSteps();
            return;
          }
          if (event.type === 'done') {
            if (event.mode === 'workflow') this.workflowResult = event.result;
            else this.agentResult = event.result;
            return;
          }
          if (event.type === 'race') {
            this.race = event.result;
            this.agentResult = event.result.agent;
            this.workflowResult = event.result.workflow;
          }
        },
        error: (err) => {
          this.running = false;
          this.activeRunMode = null;
          this.errorMessage =
            err instanceof Error && err.message ? err.message : 'Agent run failed.';
          this.notifications.error('Agent run failed');
        },
        complete: () => {
          this.running = false;
          this.activeRunMode = null;
          this.refreshMemory();
        },
      });
  }

  clearMemory(): void {
    this.agentService.clearMemory(this.sessionId).subscribe({
      next: (res) => {
        this.memories = [];
        this.notifications.success(`Cleared ${res.removed} memor${res.removed === 1 ? 'y' : 'ies'}`);
      },
    });
  }

  newSession(): void {
    this.sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, this.sessionId);
    this.memories = [];
    this.notifications.success('New session — memory starts empty');
  }

  actionInputText(value: unknown): string {
    if (value == null || value === '') return '{}';
    if (typeof value === 'string') return value;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

  stopLabel(reason: string | undefined): string {
    switch (reason) {
      case 'finished':
        return 'Finished';
      case 'max_steps':
        return 'Stopped · max steps';
      case 'max_llm_calls':
        return 'Stopped · LLM budget';
      case 'max_seconds':
        return 'Stopped · time budget';
      case 'repeated_action':
        return 'Stopped · repeated action';
      case 'parse_failures':
        return 'Stopped · invalid model output';
      default:
        return reason || '—';
    }
  }

  private upsertStep(mode: string, step: AgentStepItem): void {
    const list = mode === 'workflow' ? this.workflowSteps : this.agentSteps;
    const idx = list.findIndex((s) => s.index === step.index);
    if (idx >= 0) list[idx] = step;
    else list.push(step);
    if (mode === 'workflow') this.workflowSteps = [...list];
    else this.agentSteps = [...list];
  }

  private refreshMemory(): void {
    this.agentService.listMemory(this.sessionId).subscribe({
      next: (res) => {
        this.memories = res.items;
      },
    });
  }

  private scrollSteps(): void {
    queueMicrotask(() => {
      const el = this.stepPane?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  private static readOrCreateSession(): string {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
    return id;
  }
}
