import { Component, EventEmitter, Output, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { DocumentService } from '../../../core/services/document.service';
import { NotificationService } from '../../../core/services/notification.service';

@Component({
  selector: 'app-document-upload',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatFormFieldModule,
    MatInputModule,
  ],
  templateUrl: './document-upload.component.html',
  styleUrl: './document-upload.component.scss',
})
export class DocumentUploadComponent {
  private readonly documentService = inject(DocumentService);
  private readonly notifications = inject(NotificationService);
  private readonly fb = inject(FormBuilder);

  @Output() uploaded = new EventEmitter<void>();

  readonly maxSizeMb = 100;
  readonly allowed = ['.pdf', '.docx', '.txt'];

  dragging = false;
  uploading = false;
  progressLabel = '';
  selectedFile: File | null = null;
  error = '';

  chunkForm = this.fb.group({
    chunkSize: [800, [Validators.required, Validators.min(100), Validators.max(8000)]],
    chunkOverlap: [150, [Validators.required, Validators.min(0)]],
  });

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging = false;
    const file = event.dataTransfer?.files?.[0];
    if (file) this.setFile(file);
  }

  onBrowse(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.setFile(file);
    input.value = '';
  }

  setFile(file: File): void {
    this.error = '';
    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!this.allowed.includes(ext)) {
      this.error = `Unsupported file type. Allowed: ${this.allowed.join(', ')}`;
      this.selectedFile = null;
      return;
    }
    if (file.size > this.maxSizeMb * 1024 * 1024) {
      this.error = `File exceeds ${this.maxSizeMb} MB limit.`;
      this.selectedFile = null;
      return;
    }
    this.selectedFile = file;
  }

  upload(): void {
    if (!this.selectedFile || this.chunkForm.invalid) {
      this.error = this.chunkForm.invalid
        ? 'Chunk size must be 100–8000 and overlap must be less than chunk size.'
        : '';
      return;
    }
    const chunkSize = Number(this.chunkForm.value.chunkSize);
    const chunkOverlap = Number(this.chunkForm.value.chunkOverlap);
    if (chunkOverlap >= chunkSize) {
      this.error = 'Chunk overlap must be less than chunk size.';
      return;
    }

    this.uploading = true;
    this.error = '';
    this.progressLabel = 'Uploading to storage...';
    this.documentService
      .upload(this.selectedFile, { chunkSize, chunkOverlap })
      .subscribe({
        next: () => {
          this.uploading = false;
          this.progressLabel = '';
          this.selectedFile = null;
          this.notifications.success('Upload complete. Processing started.');
          this.uploaded.emit();
        },
        error: () => {
          this.uploading = false;
          this.progressLabel = '';
        },
      });
  }
}
