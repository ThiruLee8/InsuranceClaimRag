"""Add per-document chunk size and overlap settings

Revision ID: 002_document_chunk_settings
Revises: 001_initial
Create Date: 2026-08-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_document_chunk_settings"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("Documents", sa.Column("ChunkSize", sa.Integer(), nullable=True))
    op.add_column("Documents", sa.Column("ChunkOverlap", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("Documents", "ChunkOverlap")
    op.drop_column("Documents", "ChunkSize")
