import { DatePipe } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Subscription, interval, startWith, switchMap } from 'rxjs';
import { DocumentItem } from '../../../core/models/api.models';
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
  private readonly notifications = inject(NotificationService);
  private pollSub?: Subscription;

  loading = true;
  document: DocumentItem | null = null;

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) return;
    this.pollSub = interval(3000)
      .pipe(
        startWith(0),
        switchMap(() => this.documentService.get(id))
      )
      .subscribe({
        next: (doc) => {
          this.document = doc;
          this.loading = false;
          if (doc.status === 'Processed' || doc.status === 'Failed') {
            this.pollSub?.unsubscribe();
          }
        },
        error: () => {
          this.loading = false;
        },
      });
  }

  ngOnDestroy(): void {
    this.pollSub?.unsubscribe();
  }

  download(): void {
    if (!this.document) return;
    window.open(this.documentService.downloadUrl(this.document.id), '_blank');
  }

  reprocess(): void {
    if (!this.document) return;
    const id = this.document.id;
    this.documentService.reprocess(id).subscribe({
      next: (doc) => {
        this.document = doc;
        this.notifications.success('Reprocessing started');
        this.pollSub?.unsubscribe();
        this.pollSub = interval(3000)
          .pipe(
            startWith(0),
            switchMap(() => this.documentService.get(id))
          )
          .subscribe({
            next: (updated) => {
              this.document = updated;
              if (updated.status === 'Processed' || updated.status === 'Failed') {
                this.pollSub?.unsubscribe();
              }
            },
          });
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
