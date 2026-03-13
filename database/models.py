from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import DATABASE_URL


class Base(DeclarativeBase):
    pass


class ScrapedResult(Base):
    """Raw cached response from a scraper source, keyed by (source, make, model, year)."""

    __tablename__ = "scraped_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(100), nullable=False, index=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    raw_json = Column(Text, nullable=False)
    scraped_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Problem(Base):
    """Normalized problem record aggregated from one or more sources."""

    __tablename__ = "problems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    source = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(Float, nullable=True)
    safety_impact = Column(Float, nullable=True)
    mileage_low = Column(Integer, nullable=True)
    mileage_high = Column(Integer, nullable=True)
    repair_cost_low = Column(Float, nullable=True)
    repair_cost_high = Column(Float, nullable=True)
    complaint_count = Column(Integer, default=0)
    user_reports = Column(Text, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Recall(Base):
    """Vehicle recall record from NHTSA or other sources."""

    __tablename__ = "recalls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    source = Column(String(100), nullable=False)
    campaign_number = Column(String(50), nullable=True)
    component = Column(String(200), nullable=True)
    summary = Column(Text, nullable=True)
    consequence = Column(Text, nullable=True)
    remedy = Column(Text, nullable=True)
    report_date = Column(String(50), nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class VehicleRating(Base):
    """Aggregated rating for a vehicle from a specific source."""

    __tablename__ = "vehicle_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    source = Column(String(100), nullable=False)
    overall_rating = Column(Float, nullable=True)
    reliability_rating = Column(Float, nullable=True)
    safety_rating = Column(Float, nullable=True)
    rating_details = Column(Text, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine, checkfirst=True)


def get_session() -> Session:
    return SessionLocal()
