"""Add graph features to feature_store

Revision ID: a5184ee06927
Revises: b12056fd4c6d
Create Date: 2026-08-26 15:11:26.264254
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a5184ee06927"
down_revision: Union[str, Sequence[str], None] = "b12056fd4c6d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add graph-derived features to FeatureStore."""
    op.add_column(
        "feature_store",
        sa.Column("degree_centrality", sa.Float(), nullable=True),
    )
    op.add_column(
        "feature_store",
        sa.Column("pagerank", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Remove graph-derived features from FeatureStore."""
    op.drop_column("feature_store", "pagerank")
    op.drop_column("feature_store", "degree_centrality")