from __future__ import annotations

from typing import Protocol

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents import CLAIMS_ASSISTANT_PROMPT

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = CLAIMS_ASSISTANT_PROMPT


class LLMProvider(Protocol):
    async def generate(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> str: ...


class OllamaLLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.model = self.settings.ollama_model

    async def generate(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> str:
        selected_model = (model or self.model).strip() or self.model
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if system:
            payload["system"] = system

        logger.info("ollama_request", model=selected_model)
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        answer = (data.get("response") or "").strip()
        logger.info("ollama_response_received", model=selected_model, chars=len(answer))
        return answer

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            models = sorted(
                {
                    str(item.get("name")).strip()
                    for item in (data.get("models") or [])
                    if item.get("name")
                }
            )
            if self.model and self.model not in models:
                models.insert(0, self.model)
            return models
        except Exception as exc:  # noqa: BLE001
            logger.warning("ollama_list_models_failed", error=str(exc))
            return [self.model] if self.model else []

    async def health_check(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            return "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("ollama_health_failed", error=str(exc))
            return "unavailable"
