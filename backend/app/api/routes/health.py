from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import ApiResponse, HealthOut
from app.services.blob_service import BlobService
from app.services.llm_service import OllamaLLMService
from app.services.vector_store_service import VectorStoreService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=ApiResponse[HealthOut])
async def health(db: Session = Depends(get_db)) -> ApiResponse[HealthOut]:
    sql_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        sql_status = "unavailable"

    azurite_status = BlobService().health_check()
    chroma_status = VectorStoreService().health_check()
    ollama_status = await OllamaLLMService().health_check()

    overall = "ok" if all(
        s == "ok" for s in [sql_status, azurite_status, chroma_status, ollama_status]
    ) else "degraded"

    return ApiResponse(
        data=HealthOut(
            status=overall,
            sqlServer=sql_status,
            azurite=azurite_status,
            chroma=chroma_status,
            ollama=ollama_status,
        )
    )
