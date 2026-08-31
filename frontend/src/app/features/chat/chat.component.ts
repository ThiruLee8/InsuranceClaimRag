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
  streamStage: 'idle' | 'retrieving' | 'generating' = 'idle';
  inspectRetrieval = false;
  inspectQuestion = '';
  inspectAnswer = '';
  inspectSources: MessageItem['sources'] = [];
  inspectSearchQuery = '';
  inspectSearchMode = '';
  private shouldScroll = false;
  private streamAssistantId: string | null = null;

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
    return {
      agentId: this.selectedAgentId,
      model: this.selectedModel,
      debug: this.inspectRetrieval,
    };
  }

  toggleInspect(): void {
    this.inspectRetrieval = !this.inspectRetrieval;
  }

  private resetInspect(question: string): void {
    if (!this.inspectRetrieval) return;
    this.inspectQuestion = question;
    this.inspectAnswer = '';
    this.inspectSources = [];
    this.inspectSearchQuery = '';
    this.inspectSearchMode = '';
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
    this.streamStage = 'retrieving';
    this.form.reset();
    this.resetInspect(question);
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
    this.startStream(question);
  }

  private startStream(
    question: string,
    options?: { regenerateMessageId?: string; reuseLastUserMessage?: boolean }
  ): void {
    const assistantId = `temp-assistant-${Date.now()}`;
    this.streamAssistantId = assistantId;
    this.messages = [
      ...this.messages.filter((m) => !m.streaming && !m.failed),
      {
        id: assistantId,
        conversationId: this.activeConversationId || '',
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
        sources: [],
        streaming: true,
        failed: false,
        errorMessage: null,
      },
    ];

    this.chatService
      .askStream(question, this.activeConversationId, {
        ...this.agentOptions(),
        regenerateMessageId: options?.regenerateMessageId || null,
        reuseLastUserMessage: !!options?.reuseLastUserMessage,
      })
      .subscribe({
        next: (event) => {
          if (event.type === 'meta') {
            this.activeConversationId = event.conversationId;
            this.router.navigate([], {
              relativeTo: this.route,
              queryParams: { c: event.conversationId },
              queryParamsHandling: 'merge',
            });
            return;
          }
          if (event.type === 'status') {
            this.streamStage = event.stage === 'generating' ? 'generating' : 'retrieving';
            if (this.inspectRetrieval) {
              if (event.originalQuestion) this.inspectQuestion = event.originalQuestion;
              if (event.searchQuery) this.inspectSearchQuery = event.searchQuery;
              if (event.searchMode) this.inspectSearchMode = event.searchMode;
            }
            return;
          }
          if (event.type === 'sources') {
            this.patchStreamingAssistant({ sources: event.sources || [] });
            if (this.inspectRetrieval) {
              this.inspectSources = event.sources || [];
            }
            return;
          }
          if (event.type === 'token') {
            this.streamStage = 'generating';
            const current = this.messages.find((m) => m.id === this.streamAssistantId);
            const nextContent = (current?.content || '') + event.content;
            this.patchStreamingAssistant({ content: nextContent });
            if (this.inspectRetrieval) {
              this.inspectAnswer = nextContent;
            }
            this.shouldScroll = true;
            return;
          }
          if (event.type === 'done') {
            this.activeConversationId = event.conversationId;
            this.patchStreamingAssistant({
              id: event.messageId,
              conversationId: event.conversationId,
              content: event.answer,
              sources: event.sources || [],
              modelName: event.model || null,
              streaming: false,
              failed: false,
              errorMessage: null,
              originalQuestion: event.originalQuestion || null,
              searchQuery: event.searchQuery || null,
              searchMode: event.searchMode || null,
            });
            if (this.inspectRetrieval) {
              this.inspectQuestion = event.originalQuestion || this.inspectQuestion;
              this.inspectSearchQuery = event.searchQuery || this.inspectSearchQuery;
              this.inspectSearchMode = event.searchMode || this.inspectSearchMode;
              this.inspectSources = event.sources || [];
              this.inspectAnswer = event.answer;
            }
            this.streamAssistantId = null;
            this.sending = false;
            this.streamStage = 'idle';
            this.refreshConversations();
            this.shouldScroll = true;
          }
        },
        error: (err) => {
          const message =
            err instanceof Error && err.message
              ? err.message
              : 'Generation failed. You can retry this message.';
          this.patchStreamingAssistant({
            streaming: false,
            failed: true,
            errorMessage: message,
            content:
              this.messages.find((m) => m.id === this.streamAssistantId)?.content ||
              'Something went wrong while generating the answer.',
          });
          this.sending = false;
          this.streamStage = 'idle';
          this.streamAssistantId = null;
          this.notifications.error('Chat response failed — use Retry to try again');
        },
      });
  }

  private patchStreamingAssistant(patch: Partial<MessageItem>): void {
    const targetId = this.streamAssistantId;
    if (!targetId) return;
    this.messages = this.messages.map((m) => (m.id === targetId ? { ...m, ...patch } : m));
  }

  retryAssistant(message: MessageItem): void {
    if (this.sending) return;
    const idx = this.messages.findIndex((m) => m.id === message.id);
    if (idx < 0) return;
    const priorUser = [...this.messages]
      .slice(0, idx)
      .reverse()
      .find((m) => m.role === 'user');
    if (!priorUser) return;

    this.sending = true;
    this.streamStage = 'retrieving';
    this.resetInspect(priorUser.content);
    this.messages = this.messages.slice(0, idx);

    const isPersistedId = !message.id.startsWith('temp-');
    if (message.failed || !isPersistedId) {
      this.startStream(priorUser.content, { reuseLastUserMessage: true });
      return;
    }
    this.startStream(priorUser.content, { regenerateMessageId: message.id });
  }

  retryUser(message: MessageItem): void {
    if (this.sending || message.role !== 'user') return;
    const idx = this.messages.findIndex((m) => m.id === message.id);
    if (idx < 0) return;

    const following = this.messages[idx + 1];
    this.sending = true;
    this.streamStage = 'retrieving';
    this.resetInspect(message.content);

    if (following?.role === 'assistant') {
      this.messages = this.messages.slice(0, idx + 1);
      const isPersistedId = !following.id.startsWith('temp-') && !following.failed;
      if (isPersistedId) {
        this.startStream(message.content, { regenerateMessageId: following.id });
      } else {
        this.startStream(message.content, { reuseLastUserMessage: true });
      }
      return;
    }

    // No assistant reply yet — reuse the persisted user message.
    this.startStream(message.content, { reuseLastUserMessage: true });
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
    this.retryAssistant(message);
  }
}
