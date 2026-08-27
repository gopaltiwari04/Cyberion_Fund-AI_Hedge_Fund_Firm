"""Add expanded market features

Revision ID: 408bf28f87df
Revises: c785004616a7
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "408bf28f87df"
down_revision: Union[str, Sequence[str], None] = "c785004616a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add expanded market features to feature_store."""

    op.add_column(
        "feature_store",
        sa.Column("return_10d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("return_20d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("return_60d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("volatility_5d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("volatility_60d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("atr_14_pct", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("sma_20_distance", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("sma_50_distance", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("sma_200_distance", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("drawdown_20d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("drawdown_60d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("volume_change_5d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("volume_zscore_20d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("market_return_5d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("relative_return_5d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("beta_60d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("correlation_spy_60d", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("intraday_range", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("close_position", sa.Float(), nullable=True),
    )

    op.add_column(
        "feature_store",
        sa.Column("gap_return", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Remove expanded market features from feature_store."""

    op.drop_column("feature_store", "gap_return")
    op.drop_column("feature_store", "close_position")
    op.drop_column("feature_store", "intraday_range")
    op.drop_column("feature_store", "correlation_spy_60d")
    op.drop_column("feature_store", "beta_60d")
    op.drop_column("feature_store", "relative_return_5d")
    op.drop_column("feature_store", "market_return_5d")
    op.drop_column("feature_store", "volume_zscore_20d")
    op.drop_column("feature_store", "volume_change_5d")
    op.drop_column("feature_store", "drawdown_60d")
    op.drop_column("feature_store", "drawdown_20d")
    op.drop_column("feature_store", "sma_200_distance")
    op.drop_column("feature_store", "sma_50_distance")
    op.drop_column("feature_store", "sma_20_distance")
    op.drop_column("feature_store", "atr_14_pct")
    op.drop_column("feature_store", "volatility_60d")
    op.drop_column("feature_store", "volatility_5d")
    op.drop_column("feature_store", "return_60d")
    op.drop_column("feature_store", "return_20d")
    op.drop_column("feature_store", "return_10d")