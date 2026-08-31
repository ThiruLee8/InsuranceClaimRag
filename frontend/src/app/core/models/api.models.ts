export type DocumentStatus =
  | 'Uploaded'
  | 'Queued'
  | 'Processing'
  | 'ExtractingText'
  | 'Chunking'
  | 'GeneratingEmbeddings'
  | 'Indexing'
  | 'Completed'
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
  progressPercentage?: number;
  currentStep?: string | null;
  totalChunks?: number;
  processedChunks?: number;
  startedAt?: string | null;
  uploadedAt: string;
  processedAt?: string | null;
  updatedAt?: string | null;
  retryCount?: number;
  correlationId?: string | null;
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
  progressPercentage: number;
  currentStep?: string | null;
  totalChunks: number;
  processedChunks: number;
  startedAt?: string | null;
  updatedAt?: string | null;
  retryCount: number;
}

export interface DocumentQueueAccepted {
  documentId: string;
  status: string;
  message: string;
  correlationId?: string | null;
}

export interface ProcessAllAccepted {
  status: string;
  totalDocumentsFound: number;
  documentsQueued: number;
  documentsSkipped: number;
  correlationId?: string | null;
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
  content?: string | null;
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
  streaming?: boolean;
  failed?: boolean;
  errorMessage?: string | null;
  originalQuestion?: string | null;
  searchQuery?: string | null;
  searchMode?: string | null;
}

export interface ChatResponse {
  conversationId: string;
  messageId: string;
  answer: string;
  sources: RagSource[];
  agentId?: string | null;
  model?: string | null;
  originalQuestion?: string | null;
  searchQuery?: string | null;
  searchMode?: string | null;
}

export interface RetrieveResponse {
  originalQuestion: string;
  searchQuery: string;
  searchMode?: string | null;
  sources: RagSource[];
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
