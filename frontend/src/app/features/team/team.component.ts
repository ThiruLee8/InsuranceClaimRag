import { DecimalPipe } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import {
  AgentCard,
  MultiAgentService,
  TeamCase,
  TeamMeta,
  TeamRace,
  TeamSideSummary,
  TeamSuite,
} from '../../core/services/multi-agent.service';
import { NotificationService } from '../../core/services/notification.service';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';

@Component({
  selector: 'app-team',
  standalone: true,
  imports: [DecimalPipe, RouterLink, MatIconModule, MatProgressBarModule, MarkdownPipe],
  templateUrl: './team.component.html',
  styleUrl: './team.component.scss',
})
export class TeamComponent implements OnInit {
  private readonly api = inject(MultiAgentService);
  private readonly notifications = inject(NotificationService);

  meta: TeamMeta | null = null;
  task = '';
  selectedCaseId: string | null = null;
  running = false;
  race: TeamRace | null = null;
  errorMessage: string | null = null;

  ngOnInit(): void {
    this.api.getMeta().subscribe({
      next: (meta) => {
        this.meta = meta;
        const first = meta.cases[0];
        if (first && !this.task) this.selectCase(first);
      },
      error: () => {
        this.errorMessage = 'Could not load the team race. Is the API running?';
      },
    });
  }

  get suite(): TeamSuite | null {
    return this.meta?.suite ?? null;
  }

  get cases(): TeamCase[] {
    return this.meta?.cases ?? [];
  }

  get cards(): AgentCard[] {
    return this.meta?.cards ?? [];
  }

  selectCase(item: TeamCase): void {
    if (this.running) return;
    this.selectedCaseId = item.id;
    this.task = item.task;
  }

  onTaskInput(event: Event): void {
    this.task = (event.target as HTMLTextAreaElement).value;
    const match = this.cases.find((item) => item.task === this.task);
    this.selectedCaseId = match?.id ?? null;
  }

  selectedWhy(): string {
    return this.cases.find((item) => item.id === this.selectedCaseId)?.why ?? '';
  }

  raceTask(): void {
    const task = this.task.trim();
    if (!task || this.running) return;
    this.running = true;
    this.errorMessage = null;
    this.race = null;
    this.api.race(task, this.selectedCaseId).subscribe({
      next: (race) => {
        this.race = race;
        this.running = false;
      },
      error: () => {
        this.running = false;
        this.errorMessage = 'The race failed. Ollama and the document index need to be up for a live run.';
        this.notifications.error('Race failed');
      },
    });
  }

  shipLabel(ship: string | undefined): string {
    return ship === 'team' ? 'Specialist team' : 'Single agent';
  }

  pct(value: number | null | undefined): string {
    if (value == null || Number.isNaN(value)) return 'n/a';
    return `${Math.round(value * 100)}%`;
  }

  history(states: string[]): string {
    return states.join(' → ');
  }

  side(summary: TeamSideSummary | undefined, key: keyof TeamSideSummary): number | null {
    const value = summary?.[key];
    return typeof value === 'number' ? value : null;
  }
}
