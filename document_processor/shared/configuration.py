from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    azure_storage_connection_string: str = Field(
        default=(
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
            "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
            "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
            "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
        ),
        validation_alias=AliasChoices(
            "AZURE_STORAGE_CONNECTION_STRING",
            "BLOB_STORAGE_CONNECTION_STRING",
            "QUEUE_STORAGE_CONNECTION_STRING",
            "AzureWebJobsStorage",
        ),
    )
    document_container_name: str = Field(
        default="insurance-documents",
        validation_alias=AliasChoices("DOCUMENT_CONTAINER_NAME", "AZURE_STORAGE_CONTAINER"),
    )
    document_processing_queue_name: str = Field(
        default="document-processing",
        validation_alias=AliasChoices("DOCUMENT_PROCESSING_QUEUE_NAME"),
    )

    sql_server_host: str = "localhost"
    sql_server_port: int = 1433
    sql_server_database: str = "InsuranceRagDb"
    sql_server_username: str = "sa"
    sql_server_password: str = "Your_Str0ng_SA_Password!"
    database_url: str | None = None

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "insurance_claims_documents"

    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    max_retry_count: int = 3
    processing_batch_size: int = 25
    log_level: str = "INFO"

    # Retrieval: semantic | keyword | hybrid
    search_mode: str = "hybrid"
    enable_rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    candidate_multiplier: int = 4
    rrf_k: int = 60

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        password = quote_plus(self.sql_server_password)
        return (
            f"mssql+pyodbc://{self.sql_server_username}:{password}"
            f"@{self.sql_server_host}:{self.sql_server_port}/{self.sql_server_database}"
            f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
