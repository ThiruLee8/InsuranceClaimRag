import { Routes } from '@angular/router';
import { ShellComponent } from './layout/shell.component';

export const routes: Routes = [
  {
    path: '',
    component: ShellComponent,
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
      {
        path: 'dashboard',
        loadComponent: () =>
          import('./features/dashboard/dashboard.component').then((m) => m.DashboardComponent),
      },
      {
        path: 'documents',
        loadComponent: () =>
          import('./features/documents/documents.component').then((m) => m.DocumentsComponent),
      },
      {
        path: 'documents/:id',
        loadComponent: () =>
          import('./features/documents/detail/document-detail.component').then(
            (m) => m.DocumentDetailComponent
          ),
      },
      {
        path: 'chat',
        loadComponent: () =>
          import('./features/chat/chat.component').then((m) => m.ChatComponent),
      },
      {
        path: 'agent',
        loadComponent: () =>
          import('./features/agent/agent.component').then((m) => m.AgentComponent),
      },
      {
        path: 'conversations',
        loadComponent: () =>
          import('./features/conversations/conversations.component').then(
            (m) => m.ConversationsComponent
          ),
      },
    ],
  },
  { path: '**', redirectTo: 'dashboard' },
];
