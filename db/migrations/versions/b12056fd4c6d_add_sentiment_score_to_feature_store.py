"""Add sentiment_score to feature_store

Revision ID: b12056fd4c6d
Revises: 2175ed9bc758
Create Date: 2026-08-25 20:44:29.080889
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b12056fd4c6d"
down_revision: Union[str, Sequence[str], None] = "2175ed9bc758"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "feature_store",
        sa.Column(
            "sentiment_score",
            sa.Float(),
            nullable=True,
            server_default=sa.text("0.0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("feature_store", "sentiment_score")