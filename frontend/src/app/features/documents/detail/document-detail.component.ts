import { DatePipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Subscription } from 'rxjs';
import { DocumentItem } from '../../../core/models/api.models';
import { DocumentRealtimeService } from '../../../core/services/document-realtime.service';
import { DocumentService } from '../../../core/services/document.service';
import { NotificationService } from '../../../core/services/notification.service';
import { FileSizePipe } from '../../../shared/pipes/file-size.pipe';

@Component({
  selector: 'app-document-detail',
  standalone: true,
  imports: [
    DatePipe,
    RouterLink,
    MatButtonModule,
    MatCardModule,
    MatIconModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    FileSizePipe,
  ],
  templateUrl: './document-detail.component.html',
  styleUrl: './document-detail.component.scss',
})
export class DocumentDetailComponent implements OnInit, OnDestroy {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly documentService = inject(DocumentService);
  private readonly realtime = inject(DocumentRealtimeService);
  private readonly notifications = inject(NotificationService);
  private socketSub?: Subscription;

  loading = true;
  document: DocumentItem | null = null;

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) return;
    this.watch(id);
  }

  ngOnDestroy(): void {
    this.socketSub?.unsubscribe();
  }

  private watch(id: string): void {
    this.socketSub?.unsubscribe();
    this.loading = true;

    this.documentService.get(id).subscribe({
      next: (doc) => {
        this.document = doc;
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      },
    });

    this.socketSub = this.realtime.watchDocument(id).subscribe({
      next: (msg) => {
        if (msg.deleted || !msg.document) {
          this.notifications.error('Document was deleted');
          this.router.navigate(['/documents']);
          return;
        }
        this.document = msg.document;
        this.loading = false;
      },
    });
  }

  download(): void {
    if (!this.document) return;
    window.open(this.documentService.downloadUrl(this.document.id), '_blank');
  }

  reprocess(): void {
    if (!this.document) return;
    const id = this.document.id;
    const name = this.document.originalFileName;
    if (!confirm(`Reprocess only "${name}"?\n\nOther documents will not be queued.`)) {
      return;
    }
    this.documentService.reprocess(id).subscribe({
      next: () => {
        this.notifications.success(`Queued "${name}" for reprocessing`);
        this.watch(id);
      },
    });
  }

  delete(): void {
    if (!this.document) return;
    if (!confirm(`Delete ${this.document.originalFileName}?`)) return;
    this.documentService.delete(this.document.id).subscribe({
      next: () => {
        this.notifications.success('Document deleted');
        this.router.navigate(['/documents']);
      },
    });
  }
}
