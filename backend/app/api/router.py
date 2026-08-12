from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.routes import agents, chat, conversations, documents, health
from app.db.database import get_db
from app.db.models import DocumentStatus, MessageRole
from app.db.repositories import ConversationRepository, DocumentRepository
from app.schemas import ApiResponse, DashboardStats, DocumentOut, ConversationOut

router = APIRouter()
router.include_router(health.router)
router.include_router(agents.router)
router.include_router(documents.router)
router.include_router(conversations.router)
router.include_router(chat.router)


@router.get("/dashboard", response_model=ApiResponse[DashboardStats])
def dashboard(db: Session = Depends(get_db)):
    doc_repo = DocumentRepository(db)
    conv_repo = ConversationRepository(db)
    status_counts = doc_repo.count_by_status()
    recent_docs, _ = doc_repo.list(limit=5)
    recent_convs = conv_repo.list(limit=5)

    total_docs = sum(status_counts.values())
    stats = DashboardStats(
        totalDocuments=total_docs,
        processedDocuments=status_counts.get(DocumentStatus.Processed.value, 0),
        processingDocuments=status_counts.get(DocumentStatus.Processing.value, 0)
        + status_counts.get(DocumentStatus.Uploaded.value, 0),
        failedDocuments=status_counts.get(DocumentStatus.Failed.value, 0),
        totalConversations=conv_repo.count_conversations(),
        totalQuestions=conv_repo.count_user_questions(),
        recentDocuments=[
            DocumentOut.model_validate(d).model_copy(update={"chunkCount": doc_repo.chunk_count(d.Id)})
            for d in recent_docs
        ],
        recentConversations=[
            ConversationOut.model_validate(c).model_copy(
                update={"messageCount": conv_repo.message_count(c.Id)}
            )
            for c in recent_convs
        ],
    )
    return ApiResponse(data=stats)
