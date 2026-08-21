"""Add feature_store table

Revision ID: 020460a30212
Revises: e142c7ad3a98
Create Date: 2026-08-21 09:08:23.557042

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '020460a30212'
down_revision: str | Sequence[str] | None = 'e142c7ad3a98'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feature_store",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("return_1d", sa.Float(), nullable=True),
        sa.Column("return_5d", sa.Float(), nullable=True),
        sa.Column("rsi_14", sa.Float(), nullable=True),
        sa.Column("macd", sa.Float(), nullable=True),
        sa.Column("volatility_20d", sa.Float(), nullable=True),
        sa.Column("regime", sa.Integer(), nullable=True),
        sa.UniqueConstraint(
            "ticker",
            "date",
            name="_ticker_date_uc",
        ),
    )

    op.create_index(
        "ix_feature_store_id",
        "feature_store",
        ["id"],
    )

    op.create_index(
        "ix_feature_store_ticker",
        "feature_store",
        ["ticker"],
    )

    op.create_index(
        "ix_feature_store_date",
        "feature_store",
        ["date"],
    )


def downgrade() -> None:
    op.drop_index("ix_feature_store_date", table_name="feature_store")
    op.drop_index("ix_feature_store_ticker", table_name="feature_store")
    op.drop_index("ix_feature_store_id", table_name="feature_store")
    op.drop_table("feature_store")