from sqlalchemy import Column, Integer, String, Float, Date, create_engine
from sqlalchemy.orm import declarative_base

# 1. Define Base ONCE at the top
Base = declarative_base()

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