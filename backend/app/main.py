from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.blob_service import BlobService


configure_logging()
logger = get_logger(__name__)
settings = get_settings()


def _load_mcp_http():
    try:
        from app.mcp.http_app import build_mcp_http_app

        return build_mcp_http_app()
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_http_unavailable", error=str(exc))
        return None, None


mcp_mount, mcp_inner = _load_mcp_http()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup")
    try:
        BlobService().ensure_container()
    except Exception as exc:  # noqa: BLE001
        logger.warning("blob_init_failed", error=str(exc))
    try:
        from app.services.document_queue_service import DocumentQueueService

        DocumentQueueService().ensure_queue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("queue_init_failed", error=str(exc))

    inner_cm = None
    if mcp_inner is not None and getattr(mcp_inner, "lifespan", None) is not None:
        inner_cm = mcp_inner.lifespan(app)
        await inner_cm.__aenter__()

    if settings.mcp_enabled:
        try:
            from app.mcp.gateway import get_shared_gateway

            await get_shared_gateway().ensure_connected()
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp_host_connect_failed", error=str(exc))

    try:
        yield
    finally:
        try:
            from app.mcp.gateway import get_shared_gateway

            await get_shared_gateway().close()
        except Exception:  # noqa: BLE001
            pass
        if inner_cm is not None:
            await inner_cm.__aexit__(None, None, None)
        logger.info("application_shutdown")


app = FastAPI(
    title="Insurance Claims RAG API",
    description="Document Q&A for insurance claims using RAG. MCP tools are at /mcp.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
if mcp_mount is not None:
    app.mount("/mcp", mcp_mount)
    logger.info("mcp_http_mounted", path="/mcp")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "errorCode" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "errorCode": "HTTP_ERROR",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "errorCode": "VALIDATION_ERROR",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "errorCode": "INTERNAL_SERVER_ERROR",
        },
    )
