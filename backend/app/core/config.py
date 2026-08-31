from functools import lru_cache
from typing import List
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SQL Server
    sql_server_host: str = "localhost"
    sql_server_port: int = 1433
    sql_server_database: str = "InsuranceRagDb"
    sql_server_username: str = "sa"
    sql_server_password: str = "Your_Str0ng_SA_Password!"
    database_url: str | None = None

    # Azurite / Azure Blob + Queue
    azure_storage_connection_string: str = (
        "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
        "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
    )
    azure_storage_container: str = "insurance-documents"
    document_processing_queue_name: str = "document-processing"
    max_retry_count: int = 3
    processing_batch_size: int = 25

    # Azure Functions document-processor (owns all Chroma / vector access)
    document_processor_base_url: str = "http://localhost:7071"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embeddings defaults are used by Functions; kept here only for upload chunk defaults docs
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    similarity_threshold: float = 0.3
    # semantic | keyword | hybrid — forwarded to document-processor search
    search_mode: str = "hybrid"
    enable_rerank: bool = True

    # Error analysis — complete chat traces (JSONL) for open coding
    enable_trace_logging: bool = True
    traces_dir: str = ""  # empty → eval/error_analysis/traces (local) or /app/data/traces

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:4200"
    max_upload_size_mb: int = 100
    log_level: str = "INFO"

    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt"]
    )

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

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
