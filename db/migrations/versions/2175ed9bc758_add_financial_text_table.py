"""Add financial_text table

Revision ID: 2175ed9bc758
Revises: 020460a30212
Create Date: 2026-08-25 14:59:59.071872

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2175ed9bc758'
down_revision: Union[str, Sequence[str], None] = '020460a30212'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "financial_text",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("published_at", sa.DateTime(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("clean_text", sa.Text(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_financial_text_id",
        "financial_text",
        ["id"],
        unique=False,
    )

    op.create_index(
        "ix_financial_text_ticker",
        "financial_text",
        ["ticker"],
        unique=False,
    )

    op.create_index(
        "ix_financial_text_published_at",
        "financial_text",
        ["published_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_financial_text_published_at",
        table_name="financial_text",
    )

    op.drop_index(
        "ix_financial_text_ticker",
        table_name="financial_text",
    )

    op.drop_index(
        "ix_financial_text_id",
        table_name="financial_text",
    )

    op.drop_table("financial_text")
