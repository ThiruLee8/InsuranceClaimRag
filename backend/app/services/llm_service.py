from __future__ import annotations

import json
from collections.abc import AsyncIterator
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

    def generate_stream(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> AsyncIterator[str]: ...


class OllamaLLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.model = self.settings.ollama_model

    def _payload(
        self, *, prompt: str, system: str | None, model: str | None, stream: bool
    ) -> dict:
        selected_model = (model or self.model).strip() or self.model
        payload: dict = {
            "model": selected_model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": 0.1},
        }
        if system:
            payload["system"] = system
        return payload

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                return str(data["error"])
        except Exception:  # noqa: BLE001
            pass
        return (response.text or "").strip()[:300]

    def _raise_for_ollama(self, response: httpx.Response, *, model: str) -> None:
        if response.is_success:
            return
        detail = self._error_detail(response)
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{model}' is not installed. "
                f"From the project folder run: docker compose exec ollama ollama pull {model}"
                + (f" ({detail})" if detail else "")
            )
        raise RuntimeError(
            f"Ollama request failed with HTTP {response.status_code}"
            + (f": {detail}" if detail else "")
        )

    async def generate(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> str:
        selected_model = (model or self.model).strip() or self.model
        payload = self._payload(prompt=prompt, system=system, model=model, stream=False)

        logger.info("ollama_request", model=selected_model, stream=False)
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            self._raise_for_ollama(response, model=selected_model)
            data = response.json()
        answer = (data.get("response") or "").strip()
        logger.info("ollama_response_received", model=selected_model, chars=len(answer))
        return answer

    async def generate_stream(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> AsyncIterator[str]:
        selected_model = (model or self.model).strip() or self.model
        payload = self._payload(prompt=prompt, system=system, model=model, stream=True)

        logger.info("ollama_request", model=selected_model, stream=True)
        chars = 0
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                self._raise_for_ollama(response, model=selected_model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    token = data.get("response") or ""
                    if token:
                        chars += len(token)
                        yield token
                    if data.get("done"):
                        break
        logger.info("ollama_stream_complete", model=selected_model, chars=chars)

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
                data = response.json()
            names = {
                str(item.get("name") or "").strip()
                for item in (data.get("models") or [])
                if item.get("name")
            }
            if self.model and not any(
                name == self.model or name.startswith(f"{self.model}:") for name in names
            ):
                return "model_missing"
            return "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("ollama_health_failed", error=str(exc))
            return "unavailable"
