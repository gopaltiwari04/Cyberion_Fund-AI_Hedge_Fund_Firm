from sqlalchemy import Column, Date, Float, Integer, String, UniqueConstraint, DateTime, Text
from sqlalchemy.orm import declarative_base


# 1. Define Base ONCE at the top
Base = declarative_base()

class FinancialText(Base):
    __tablename__ = "financial_text"

    id = Column(Integer, primary_key=True, index=True)

    ticker = Column(String, index=True, nullable=False)
    source = Column(String, nullable=False)  # news / sec_8k

    published_at = Column(DateTime, index=True, nullable=False)

    title = Column(String)
    raw_text = Column(Text)
    clean_text = Column(Text)

    # Keep this nullable for now.
    # We'll populate it once the embedding pipeline runs.
    embedding = Column(Text, nullable=True)

class FeatureStore(Base):
    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, index=True)

    ticker = Column(String, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)

    return_1d = Column(Float)
    return_5d = Column(Float)

    rsi_14 = Column(Float)
    macd = Column(Float)

    volatility_20d = Column(Float)

    regime = Column(Integer)

    sentiment_score = Column(Float, default=0.0)
    degree_centrality = Column(Float, default=0.0)
    pagerank = Column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "date",
            name="_ticker_date_uc"
        ),
    )

# 2. Your Week 1 Table
class Asset(Base):
    __tablename__ = 'assets'
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    asset_class = Column(String)
    sector = Column(String)

# 3. Your New Week 2 Table
class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


    
class PortfolioAllocation(Base):
    __tablename__ = "portfolio_allocations"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    strategy_name = Column(String, index=True, nullable=False)
    ticker = Column(String, index=True, nullable=False)

    weight = Column(Float, nullable=False)
    expected_return = Column(Float)
    risk_contribution = Column(Float)


class PortfolioRiskMetrics(Base):
    __tablename__ = "portfolio_risk_metrics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    strategy_name = Column(String, index=True, nullable=False)

    expected_annual_return = Column(Float)
    expected_annual_volatility = Column(Float)
    sharpe_ratio = Column(Float)

    var_95 = Column(Float)
    cvar_95 = Column(Float)