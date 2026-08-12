import { DatePipe, DecimalPipe } from '@angular/common';
import {
  AfterViewChecked,
  Component,
  ElementRef,
  OnInit,
  ViewChild,
  inject,
} from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSelectModule } from '@angular/material/select';
import { MatListModule } from '@angular/material/list';
import { AgentItem, ConversationItem, MessageItem } from '../../core/models/api.models';
import { ChatService } from '../../core/services/chat.service';
import { ConversationService } from '../../core/services/conversation.service';
import { NotificationService } from '../../core/services/notification.service';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    DatePipe,
    DecimalPipe,
    RouterLink,
    ReactiveFormsModule,
    MatButtonModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatProgressSpinnerModule,
    MatSelectModule,
    MatListModule,
    MarkdownPipe,
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
})
export class ChatComponent implements OnInit, AfterViewChecked {
  @ViewChild('messagePane') messagePane?: ElementRef<HTMLDivElement>;

  private readonly conversationService = inject(ConversationService);
  private readonly chatService = inject(ChatService);
  private readonly notifications = inject(NotificationService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly fb = inject(FormBuilder);

  conversations: ConversationItem[] = [];
  messages: MessageItem[] = [];
  agents: AgentItem[] = [];
  models: string[] = [];
  selectedAgentId = 'claims_assistant';
  selectedModel = 'llama3.2';
  activeConversationId: string | null = null;
  loadingConversations = true;
  loadingMessages = false;
  sending = false;
  private shouldScroll = false;

  form = this.fb.group({
    question: ['', [Validators.required, Validators.maxLength(4000)]],
  });

  ngOnInit(): void {
    this.chatService.listAgents().subscribe({
      next: (res) => {
        this.agents = res.agents;
        this.models = res.models.length ? res.models : [res.defaultModel];
        this.selectedAgentId = res.defaultAgentId;
        this.selectedModel = res.defaultModel;
      },
      error: () => {
        this.agents = [
          {
            id: 'claims_assistant',
            name: 'Claims Assistant',
            description: 'General Q&A across claim documents with citations.',
          },
        ];
        this.models = ['llama3.2'];
      },
    });

    this.refreshConversations(() => {
      const requested = this.route.snapshot.queryParamMap.get('c');
      if (requested) {
        this.selectConversation(requested);
      }
    });
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll && this.messagePane) {
      const el = this.messagePane.nativeElement;
      el.scrollTop = el.scrollHeight;
      this.shouldScroll = false;
    }
  }

  agentOptions() {
    return { agentId: this.selectedAgentId, model: this.selectedModel };
  }

  refreshConversations(done?: () => void): void {
    this.loadingConversations = true;
    this.conversationService.list().subscribe({
      next: (items) => {
        this.conversations = items;
        this.loadingConversations = false;
        done?.();
      },
      error: () => {
        this.loadingConversations = false;
      },
    });
  }

  newConversation(): void {
    this.conversationService.create('New Conversation').subscribe({
      next: (conv) => {
        this.conversations = [conv, ...this.conversations];
        this.selectConversation(conv.id);
      },
    });
  }

  selectConversation(id: string): void {
    this.activeConversationId = id;
    this.loadingMessages = true;
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { c: id },
      queryParamsHandling: 'merge',
    });
    this.conversationService.getMessages(id).subscribe({
      next: (messages) => {
        this.messages = messages;
        this.loadingMessages = false;
        this.shouldScroll = true;
      },
      error: () => {
        this.loadingMessages = false;
      },
    });
  }

  send(): void {
    if (this.form.invalid || this.sending) return;
    const question = this.form.value.question!.trim();
    if (!question) return;

    this.sending = true;
    this.form.reset();
    this.messages = [
      ...this.messages,
      {
        id: `temp-user-${Date.now()}`,
        conversationId: this.activeConversationId || '',
        role: 'user',
        content: question,
        createdAt: new Date().toISOString(),
        sources: [],
      },
    ];
    this.shouldScroll = true;

    this.chatService.ask(question, this.activeConversationId, this.agentOptions()).subscribe({
      next: (res) => {
        this.activeConversationId = res.conversationId;
        this.sending = false;
        this.selectConversation(res.conversationId);
        this.refreshConversations();
      },
      error: () => {
        this.sending = false;
      },
    });
  }

  rename(conv: ConversationItem, event: Event): void {
    event.stopPropagation();
    const title = prompt('Rename conversation', conv.title);
    if (!title?.trim()) return;
    this.conversationService.rename(conv.id, title.trim()).subscribe({
      next: (updated) => {
        this.conversations = this.conversations.map((c) =>
          c.id === updated.id ? updated : c
        );
      },
    });
  }

  deleteConversation(conv: ConversationItem, event: Event): void {
    event.stopPropagation();
    if (!confirm(`Delete conversation "${conv.title}"?`)) return;
    this.conversationService.delete(conv.id).subscribe({
      next: () => {
        this.conversations = this.conversations.filter((c) => c.id !== conv.id);
        if (this.activeConversationId === conv.id) {
          this.activeConversationId = null;
          this.messages = [];
        }
        this.notifications.success('Conversation deleted');
      },
    });
  }

  clearConversation(): void {
    if (!this.activeConversationId) return;
    if (!confirm('Clear all messages in this conversation?')) return;
    this.conversationService.clearMessages(this.activeConversationId).subscribe({
      next: () => {
        this.messages = [];
        this.notifications.success('Conversation cleared');
      },
    });
  }

  copyAnswer(content: string): void {
    navigator.clipboard.writeText(content);
    this.notifications.success('Answer copied');
  }

  regenerate(message: MessageItem): void {
    if (!this.activeConversationId || this.sending) return;
    const idx = this.messages.findIndex((m) => m.id === message.id);
    const priorUser = [...this.messages]
      .slice(0, idx)
      .reverse()
      .find((m) => m.role === 'user');
    if (!priorUser) return;

    this.sending = true;
    this.chatService
      .regenerate(
        priorUser.content,
        this.activeConversationId,
        message.id,
        this.agentOptions()
      )
      .subscribe({
        next: () => {
          this.sending = false;
          this.selectConversation(this.activeConversationId!);
          this.refreshConversations();
        },
        error: () => {
          this.sending = false;
        },
      });
  }
}
