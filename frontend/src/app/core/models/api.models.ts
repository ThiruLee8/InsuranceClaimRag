export type DocumentStatus =
  | 'Uploaded'
  | 'Processing'
  | 'Processed'
  | 'Failed'
  | 'Deleted';

export type MessageRole = 'user' | 'assistant' | 'system';

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
  errorCode?: string;
}

export interface DocumentItem {
  id: string;
  fileName: string;
  originalFileName: string;
  blobContainerName: string;
  blobName: string;
  blobUrl: string;
  contentType: string;
  fileSize: number;
  fileHash: string;
  status: DocumentStatus;
  processingError?: string | null;
  pageCount?: number | null;
  chunkSize?: number | null;
  chunkOverlap?: number | null;
  uploadedAt: string;
  processedAt?: string | null;
  createdBy?: string | null;
  chunkCount?: number | null;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
}

export interface DocumentStatusResponse {
  id: string;
  status: DocumentStatus;
  processingError?: string | null;
  pageCount?: number | null;
  chunkCount: number;
  processedAt?: string | null;
}

export interface ConversationItem {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  createdBy?: string | null;
  messageCount?: number | null;
}

export interface RagSource {
  documentId?: string | null;
  chunkId?: string | null;
  fileName?: string | null;
  pageNumber?: number | null;
  relevanceScore?: number | null;
}

export interface MessageItem {
  id: string;
  conversationId: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  tokenCount?: number | null;
  modelName?: string | null;
  sources: RagSource[];
}

export interface ChatResponse {
  conversationId: string;
  messageId: string;
  answer: string;
  sources: RagSource[];
  agentId?: string | null;
  model?: string | null;
}

export interface AgentItem {
  id: string;
  name: string;
  description: string;
}

export interface AgentsResponse {
  agents: AgentItem[];
  models: string[];
  defaultAgentId: string;
  defaultModel: string;
}

export interface VectorStoreActionResponse {
  cleared: boolean;
  queued: number;
  documentIds: string[];
}

export interface DashboardStats {
  totalDocuments: number;
  processedDocuments: number;
  processingDocuments: number;
  failedDocuments: number;
  totalConversations: number;
  totalQuestions: number;
  recentDocuments: DocumentItem[];
  recentConversations: ConversationItem[];
}
