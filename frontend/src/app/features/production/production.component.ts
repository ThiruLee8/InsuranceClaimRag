import { DecimalPipe, PercentPipe } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import {
  ProductionReport,
  ProductionRequest,
  ProductionService,
  SupportResult,
} from '../../core/services/production.service';

@Component({
  selector: 'app-production',
  standalone: true,
  imports: [DecimalPipe, PercentPipe, MatIconModule],
  templateUrl: './production.component.html',
  styleUrl: './production.component.scss',
})
export class ProductionComponent implements OnInit {
  private readonly api = inject(ProductionService);

  report: ProductionReport | null = null;
  selectedId: string | null = null;
  complaint = 'yesterday it gave the wrong investigator name';
  support: SupportResult | null = null;
  searching = false;
  errorMessage: string | null = null;

  ngOnInit(): void {
    this.api.getReport().subscribe({
      next: (report) => {
        this.report = report;
        const withGenerate = report.improvedItems.find((item) =>
          item.spans.some((span) => span.name === 'generate'),
        );
        this.selectedId = withGenerate?.id ?? report.improvedItems[0]?.id ?? null;
      },
      error: () => {
        this.errorMessage = 'Could not load the production report. Is the API running?';
      },
    });
    this.runSearch();
  }

  get selected(): ProductionRequest | null {
    return this.report?.improvedItems.find((item) => item.id === this.selectedId) ?? null;
  }

  select(id: string): void {
    this.selectedId = id;
  }

  onComplaint(event: Event): void {
    this.complaint = (event.target as HTMLTextAreaElement).value;
  }

  runSearch(fromBox?: string): void {
    const complaint = (fromBox ?? this.complaint).trim();
    this.complaint = complaint;
    if (!complaint || this.searching) return;
    this.searching = true;
    this.api.search(complaint).subscribe({
      next: (result) => {
        this.support = result;
        this.searching = false;
      },
      error: () => {
        this.searching = false;
        this.support = null;
      },
    });
  }

  usd(value: number | null | undefined): string {
    if (value == null) return 'n/a';
    return `$${value.toFixed(6)}`;
  }

  perThousand(value: number | null | undefined): string {
    if (value == null) return 'n/a';
    return `$${value.toFixed(4)}`;
  }

  routeLabel(route: string): string {
    if (route === 'cache') return 'Semantic cache';
    if (route === 'cheap') return 'Cheap model';
    if (route === 'fallback') return 'Fallback';
    return 'Capable model';
  }
}
