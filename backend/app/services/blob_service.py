from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import PurePosixPath

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BlobService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = BlobServiceClient.from_connection_string(
            self.settings.azure_storage_connection_string
        )
        self.container_name = self.settings.azure_storage_container

    def ensure_container(self) -> None:
        try:
            self._client.create_container(self.container_name)
            logger.info("blob_container_created", container=self.container_name)
        except ResourceExistsError:
            logger.info("blob_container_exists", container=self.container_name)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        name = PurePosixPath(filename.replace("\\", "/")).name
        name = re.sub(r"[^\w.\- ]+", "_", name).strip()
        return name or f"document-{uuid.uuid4().hex}"

    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def upload_bytes(
        self,
        *,
        content: bytes,
        original_filename: str,
        content_type: str,
    ) -> tuple[str, str, str]:
        safe_name = self.sanitize_filename(original_filename)
        blob_name = f"{uuid.uuid4().hex}/{safe_name}"
        blob_client = self._client.get_blob_client(self.container_name, blob_name)
        blob_client.upload_blob(
            content,
            overwrite=False,
            content_settings=ContentSettings(content_type=content_type),
        )
        url = blob_client.url
        logger.info(
            "blob_uploaded",
            container=self.container_name,
            blob_name=blob_name,
            size=len(content),
        )
        return blob_name, url, safe_name

    def download_bytes(self, blob_name: str) -> bytes:
        blob_client = self._client.get_blob_client(self.container_name, blob_name)
        data = blob_client.download_blob().readall()
        logger.info("blob_downloaded", blob_name=blob_name, size=len(data))
        return data

    def delete_blob(self, blob_name: str) -> None:
        blob_client = self._client.get_blob_client(self.container_name, blob_name)
        blob_client.delete_blob(delete_snapshots="include")
        logger.info("blob_deleted", blob_name=blob_name)

    def health_check(self) -> str:
        try:
            self.ensure_container()
            return "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("azurite_health_failed", error=str(exc))
            return "unavailable"
