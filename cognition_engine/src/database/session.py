"""
Database session management for the Cognition Engine.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
try:
    engine = create_engine(settings.DATABASE_URL)
    logger.info(f"Database engine created with URL: {settings.DATABASE_URL}")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()

def init_db():
    """Initialize the database by creating all tables."""
    try:
        from src.database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def get_db():
    """
    Get a database session.
    
    This function is used as a dependency in FastAPI endpoints.
    It creates a new session for each request and closes it when the request is complete.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 