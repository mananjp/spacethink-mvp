"""SQLAlchemy models + engine/session setup for local Alpine Postgres."""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, DateTime, JSON, Integer, create_engine, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://spacethink:spacethink_local@localhost:5432/spacethink",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class RunManifestORM(Base):
    __tablename__ = "run_manifest"
    run_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    dataset = Column(String)
    detector_name = Column(String)
    twin_name = Column(String)
    llm_name = Column(String)
    notes = Column(Text, default="")


class EventOfInterestORM(Base):
    __tablename__ = "event_of_interest"
    id = Column(String, primary_key=True)
    run_id = Column(String, index=True)
    channel = Column(String)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)
    score = Column(Float)
    severity = Column(String)
    detector_name = Column(String)
    metadata_json = Column(JSON, default={})


class HypothesisORM(Base):
    __tablename__ = "hypothesis"
    id = Column(String, primary_key=True)
    event_id = Column(String, index=True)
    text = Column(Text)
    mechanism = Column(String)
    fault_params_json = Column(JSON, default=[])
    prior = Column(Float)
    generator = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class SimResultORM(Base):
    __tablename__ = "sim_result"
    id = Column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id = Column(String, index=True)
    distance = Column(Float)
    posterior = Column(Float)
    n_sims = Column(Integer)
    diagnostics_json = Column(JSON, default={})


def init_db() -> None:
    """Create all tables. Run once against the local alpine-postgres container."""
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
