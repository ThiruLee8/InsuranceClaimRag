import { DatePipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatTableModule } from '@angular/material/table';
import { MatChipsModule } from '@angular/material/chips';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { Subscription } from 'rxjs';
import { DocumentItem, DocumentStatus } from '../../core/models/api.models';
import { DocumentRealtimeService } from '../../core/services/document-realtime.service';
import { DocumentService } from '../../core/services/document.service';
import { NotificationService } from '../../core/services/notification.service';
import { FileSizePipe } from '../../shared/pipes/file-size.pipe';
import { DocumentUploadComponent } from './upload/document-upload.component';

@Component({
  selector: 'app-documents',
  standalone: true,
  imports: [
    DatePipe,
    RouterLink,
    ReactiveFormsModule,
    MatButtonModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatSelectModule,
    MatTableModule,
    MatChipsModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    FileSizePipe,
    DocumentUploadComponent,
  ],
  templateUrl: './documents.component.html',
  styleUrl: './documents.component.scss',
})
export class DocumentsComponent implements OnInit, OnDestroy {
  private readonly documentService = inject(DocumentService);
  private readonly realtime = inject(DocumentRealtimeService);
  private readonly notifications = inject(NotificationService);
  private readonly fb = inject(FormBuilder);
  private socketSub?: Subscription;

  readonly displayedColumns = [
    'fileName',
    'type',
    'size',
    'status',
    'progress',
    'uploadedAt',
    'processedAt',
    'actions',
  ];
  readonly statuses: Array<DocumentStatus | ''> = [
    '',
    'Uploaded',
    'Queued',
    'Processing',
    'ExtractingText',
    'Chunking',
    'GeneratingEmbeddings',
    'Indexing',
    'Completed',
    'Failed',
  ];

  loading = true;
  adminBusy = false;
  documents: DocumentItem[] = [];
  showUpload = false;

  filters = this.fb.group({
    search: [''],
    status: ['' as DocumentStatus | ''],
    contentType: [''],
  });

  ngOnInit(): void {
    this.load();
  }

  ngOnDestroy(): void {
    this.socketSub?.unsubscribe();
  }

  private connectRealtime(): void {
    this.socketSub?.unsubscribe();
    this.socketSub = this.realtime
      .watchList({
        search: this.filters.value.search || undefined,
        status: this.filters.value.status || undefined,
        contentType: this.filters.value.contentType || undefined,
      })
      .subscribe({
        next: (msg) => {
          this.documents = msg.documents;
          this.loading = false;
        },
        error: () => {
          this.notifications.error('Realtime document updates disconnected');
        },
      });
  }

  load(): void {
    this.loading = true;
    this.documentService
      .list({
        search: this.filters.value.search || undefined,
        status: this.filters.value.status || undefined,
        contentType: this.filters.value.contentType || undefined,
      })
      .subscribe({
        next: (res) => {
          this.documents = res.items;
          this.loading = false;
          this.connectRealtime();
        },
        error: () => {
          this.loading = false;
          // Still try websocket so progress can appear if list HTTP failed transiently.
          this.connectRealtime();
        },
      });
  }

  onUploaded(): void {
    this.showUpload = false;
    this.load();
  }

  delete(doc: DocumentItem): void {
    if (
      !confirm(
        `Delete ${doc.originalFileName}?\n\nThis removes the file from blob storage and soft-deletes the database record.`
      )
    ) {
      return;
    }
    this.documentService.delete(doc.id).subscribe({
      next: () => {
        this.notifications.success('Document soft-deleted (blob removed)');
      },
    });
  }

  reprocess(doc: DocumentItem): void {
    this.documentService.process(doc.id).subscribe({
      next: () => {
        this.notifications.success('Document queued for processing');
      },
    });
  }

  clearVectorStore(): void {
    if (!confirm('Clear the entire vector database? Indexed chunks will be removed.')) return;
    this.adminBusy = true;
    this.documentService.clearVectorStore().subscribe({
      next: () => {
        this.adminBusy = false;
        this.notifications.success('Vector store cleared');
      },
      error: () => {
        this.adminBusy = false;
      },
    });
  }

  reprocessAll(): void {
    if (!confirm('Reprocess all documents from blob storage into the vector database?')) return;
    this.adminBusy = true;
    this.documentService.processAll(true).subscribe({
      next: (res) => {
        this.adminBusy = false;
        this.notifications.success(
          `Queued ${res.documentsQueued} of ${res.totalDocumentsFound} document(s)`
        );
      },
      error: () => {
        this.adminBusy = false;
      },
    });
  }

  clearAndReprocessAll(): void {
    if (
      !confirm(
        'Clear the vector database and reprocess all documents from blob storage?\n\nThis rebuilds the entire index.'
      )
    ) {
      return;
    }
    this.adminBusy = true;
    this.documentService.clearAndReprocessAll().subscribe({
      next: (res) => {
        this.adminBusy = false;
        this.notifications.success(
          `Vector store cleared. Queued ${res.documentsQueued} document(s)`
        );
      },
      error: () => {
        this.adminBusy = false;
      },
    });
  }

  statusClass(status: DocumentStatus): string {
    return `status status--${status.toLowerCase()}`;
  }

  isInFlight(status: DocumentStatus): boolean {
    return ![
      'Uploaded',
      'Completed',
      'Processed',
      'Failed',
      'Deleted',
    ].includes(status);
  }
}
