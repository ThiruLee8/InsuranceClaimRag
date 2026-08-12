import { DatePipe } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTableModule } from '@angular/material/table';
import { ConversationItem } from '../../core/models/api.models';
import { ConversationService } from '../../core/services/conversation.service';
import { NotificationService } from '../../core/services/notification.service';

@Component({
  selector: 'app-conversations',
  standalone: true,
  imports: [
    DatePipe,
    RouterLink,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTableModule,
  ],
  templateUrl: './conversations.component.html',
  styleUrl: './conversations.component.scss',
})
export class ConversationsComponent implements OnInit {
  private readonly conversationService = inject(ConversationService);
  private readonly notifications = inject(NotificationService);

  loading = true;
  conversations: ConversationItem[] = [];
  displayedColumns = ['title', 'messages', 'updatedAt', 'actions'];

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.conversationService.list().subscribe({
      next: (items) => {
        this.conversations = items;
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      },
    });
  }

  rename(conv: ConversationItem): void {
    const title = prompt('Rename conversation', conv.title);
    if (!title?.trim()) return;
    this.conversationService.rename(conv.id, title.trim()).subscribe({
      next: () => {
        this.notifications.success('Conversation renamed');
        this.load();
      },
    });
  }

  delete(conv: ConversationItem): void {
    if (!confirm(`Delete "${conv.title}"?`)) return;
    this.conversationService.delete(conv.id).subscribe({
      next: () => {
        this.notifications.success('Conversation deleted');
        this.load();
      },
    });
  }
}
