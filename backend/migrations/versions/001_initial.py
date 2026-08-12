"""Initial schema for Insurance RAG

Revision ID: 001_initial
Revises:
Create Date: 2026-08-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "Documents",
        sa.Column("Id", sa.Uuid(), nullable=False),
        sa.Column("FileName", sa.String(length=512), nullable=False),
        sa.Column("OriginalFileName", sa.String(length=512), nullable=False),
        sa.Column("BlobContainerName", sa.String(length=256), nullable=False),
        sa.Column("BlobName", sa.String(length=512), nullable=False),
        sa.Column("BlobUrl", sa.String(length=1024), nullable=False),
        sa.Column("ContentType", sa.String(length=128), nullable=False),
        sa.Column("FileSize", sa.BigInteger(), nullable=False),
        sa.Column("FileHash", sa.String(length=128), nullable=False),
        sa.Column(
            "Status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column("ProcessingError", sa.Text(), nullable=True),
        sa.Column("PageCount", sa.Integer(), nullable=True),
        sa.Column("UploadedAt", sa.DateTime(timezone=True), server_default=sa.text("SYSUTCDATETIME()"), nullable=False),
        sa.Column("ProcessedAt", sa.DateTime(timezone=True), nullable=True),
        sa.Column("CreatedBy", sa.String(length=256), nullable=True),
        sa.Column("IsDeleted", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("Id"),
        sa.UniqueConstraint("BlobName"),
    )
    op.create_index("ix_Documents_FileHash", "Documents", ["FileHash"])

    op.create_table(
        "Conversations",
        sa.Column("Id", sa.Uuid(), nullable=False),
        sa.Column("Title", sa.String(length=512), nullable=False),
        sa.Column("CreatedAt", sa.DateTime(timezone=True), server_default=sa.text("SYSUTCDATETIME()"), nullable=False),
        sa.Column("UpdatedAt", sa.DateTime(timezone=True), server_default=sa.text("SYSUTCDATETIME()"), nullable=False),
        sa.Column("CreatedBy", sa.String(length=256), nullable=True),
        sa.Column("IsDeleted", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("Id"),
    )

    op.create_table(
        "DocumentChunks",
        sa.Column("Id", sa.Uuid(), nullable=False),
        sa.Column("DocumentId", sa.Uuid(), nullable=False),
        sa.Column("ChunkIndex", sa.Integer(), nullable=False),
        sa.Column("PageNumber", sa.Integer(), nullable=True),
        sa.Column("Content", sa.Text(), nullable=False),
        sa.Column("CharacterCount", sa.Integer(), nullable=False),
        sa.Column("CreatedAt", sa.DateTime(timezone=True), server_default=sa.text("SYSUTCDATETIME()"), nullable=False),
        sa.ForeignKeyConstraint(["DocumentId"], ["Documents.Id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("Id"),
    )
    op.create_index("ix_DocumentChunks_DocumentId", "DocumentChunks", ["DocumentId"])

    op.create_table(
        "Messages",
        sa.Column("Id", sa.Uuid(), nullable=False),
        sa.Column("ConversationId", sa.Uuid(), nullable=False),
        sa.Column("Role", sa.String(length=32), nullable=False),
        sa.Column("Content", sa.Text(), nullable=False),
        sa.Column("CreatedAt", sa.DateTime(timezone=True), server_default=sa.text("SYSUTCDATETIME()"), nullable=False),
        sa.Column("TokenCount", sa.Integer(), nullable=True),
        sa.Column("ModelName", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(["ConversationId"], ["Conversations.Id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("Id"),
    )
    op.create_index("ix_Messages_ConversationId", "Messages", ["ConversationId"])

    op.create_table(
        "RAGSources",
        sa.Column("Id", sa.Uuid(), nullable=False),
        sa.Column("MessageId", sa.Uuid(), nullable=False),
        sa.Column("DocumentId", sa.Uuid(), nullable=True),
        sa.Column("ChunkId", sa.Uuid(), nullable=True),
        sa.Column("PageNumber", sa.Integer(), nullable=True),
        sa.Column("RelevanceScore", sa.Float(), nullable=True),
        sa.Column("CreatedAt", sa.DateTime(timezone=True), server_default=sa.text("SYSUTCDATETIME()"), nullable=False),
        sa.ForeignKeyConstraint(["ChunkId"], ["DocumentChunks.Id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["DocumentId"], ["Documents.Id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["MessageId"], ["Messages.Id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("Id"),
    )
    op.create_index("ix_RAGSources_MessageId", "RAGSources", ["MessageId"])


def downgrade() -> None:
    op.drop_index("ix_RAGSources_MessageId", table_name="RAGSources")
    op.drop_table("RAGSources")
    op.drop_index("ix_Messages_ConversationId", table_name="Messages")
    op.drop_table("Messages")
    op.drop_index("ix_DocumentChunks_DocumentId", table_name="DocumentChunks")
    op.drop_table("DocumentChunks")
    op.drop_table("Conversations")
    op.drop_index("ix_Documents_FileHash", table_name="Documents")
    op.drop_table("Documents")
