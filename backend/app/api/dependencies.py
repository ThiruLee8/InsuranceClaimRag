from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import DocumentRepository
from app.services.agent_loop import ClaimAgentRunner
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService
from app.services.multi_agent import TeamRunner


def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    return DocumentService(db)


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    return ConversationService(db)


def get_claim_agent_runner(db: Session = Depends(get_db)) -> ClaimAgentRunner:
    return ClaimAgentRunner(document_repo=DocumentRepository(db))


def get_team_runner(db: Session = Depends(get_db)) -> TeamRunner:
    return TeamRunner(single=get_claim_agent_runner(db))
