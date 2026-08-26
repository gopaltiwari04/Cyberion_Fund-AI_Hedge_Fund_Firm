"""Add portfolio allocation and risk tables

Revision ID: c785004616a7
Revises: a5184ee06927
Create Date: 2026-08-26 15:36:15.274168

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c785004616a7"
down_revision: Union[str, Sequence[str], None] = "a5184ee06927"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create portfolio allocation and risk metric tables."""

    # ---------------------------------------------------------
    # Portfolio Allocations
    # ---------------------------------------------------------
    op.create_table(
        "portfolio_allocations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("strategy_name", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("expected_return", sa.Float(), nullable=True),
        sa.Column("risk_contribution", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_portfolio_allocations_id",
        "portfolio_allocations",
        ["id"],
        unique=False,
    )

    op.create_index(
        "ix_portfolio_allocations_date",
        "portfolio_allocations",
        ["date"],
        unique=False,
    )

    op.create_index(
        "ix_portfolio_allocations_strategy_name",
        "portfolio_allocations",
        ["strategy_name"],
        unique=False,
    )

    op.create_index(
        "ix_portfolio_allocations_ticker",
        "portfolio_allocations",
        ["ticker"],
        unique=False,
    )

    # ---------------------------------------------------------
    # Portfolio Risk Metrics
    # ---------------------------------------------------------
    op.create_table(
        "portfolio_risk_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("strategy_name", sa.String(), nullable=False),
        sa.Column("expected_annual_return", sa.Float(), nullable=True),
        sa.Column("expected_annual_volatility", sa.Float(), nullable=True),
        sa.Column("sharpe_ratio", sa.Float(), nullable=True),
        sa.Column("var_95", sa.Float(), nullable=True),
        sa.Column("cvar_95", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_portfolio_risk_metrics_id",
        "portfolio_risk_metrics",
        ["id"],
        unique=False,
    )

    op.create_index(
        "ix_portfolio_risk_metrics_date",
        "portfolio_risk_metrics",
        ["date"],
        unique=False,
    )

    op.create_index(
        "ix_portfolio_risk_metrics_strategy_name",
        "portfolio_risk_metrics",
        ["strategy_name"],
        unique=False,
    )


def downgrade() -> None:
    """Remove portfolio allocation and risk metric tables."""

    # ---------------------------------------------------------
    # Portfolio Risk Metrics
    # ---------------------------------------------------------
    op.drop_index(
        "ix_portfolio_risk_metrics_strategy_name",
        table_name="portfolio_risk_metrics",
    )

    op.drop_index(
        "ix_portfolio_risk_metrics_date",
        table_name="portfolio_risk_metrics",
    )

    op.drop_index(
        "ix_portfolio_risk_metrics_id",
        table_name="portfolio_risk_metrics",
    )

    op.drop_table("portfolio_risk_metrics")

    # ---------------------------------------------------------
    # Portfolio Allocations
    # ---------------------------------------------------------
    op.drop_index(
        "ix_portfolio_allocations_ticker",
        table_name="portfolio_allocations",
    )

    op.drop_index(
        "ix_portfolio_allocations_strategy_name",
        table_name="portfolio_allocations",
    )

    op.drop_index(
        "ix_portfolio_allocations_date",
        table_name="portfolio_allocations",
    )

    op.drop_index(
        "ix_portfolio_allocations_id",
        table_name="portfolio_allocations",
    )

    op.drop_table("portfolio_allocations")