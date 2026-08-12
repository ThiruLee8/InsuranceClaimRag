"""Add document processing progress tracking fields and statuses

Revision ID: 003_document_processing_progress
Revises: 002_document_chunk_settings
Create Date: 2026-08-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003_document_processing_progress"
down_revision: Union[str, None] = "002_document_chunk_settings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "Documents",
        sa.Column("ProgressPercentage", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("Documents", sa.Column("CurrentStep", sa.String(length=128), nullable=True))
    op.add_column(
        "Documents",
        sa.Column("TotalChunks", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "Documents",
        sa.Column("ProcessedChunks", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("Documents", sa.Column("StartedAt", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "Documents",
        sa.Column("UpdatedAt", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "Documents",
        sa.Column("RetryCount", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("Documents", sa.Column("CorrelationId", sa.String(length=64), nullable=True))

    # Normalize legacy Processed → Completed for the new status vocabulary.
    op.execute("UPDATE Documents SET Status = N'Completed' WHERE Status = N'Processed'")


def downgrade() -> None:
    op.execute("UPDATE Documents SET Status = N'Processed' WHERE Status = N'Completed'")
    op.drop_column("Documents", "CorrelationId")
    op.drop_column("Documents", "RetryCount")
    op.drop_column("Documents", "UpdatedAt")
    op.drop_column("Documents", "StartedAt")
    op.drop_column("Documents", "ProcessedChunks")
    op.drop_column("Documents", "TotalChunks")
    op.drop_column("Documents", "CurrentStep")
    op.drop_column("Documents", "ProgressPercentage")
