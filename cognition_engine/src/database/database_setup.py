"""
Database setup for the Cognition Engine.

This module provides functions to set up and initialize the database.
"""

import logging
import os
from pathlib import Path

from src.config import settings
from src.database.session import init_db

# Configure logging
logger = logging.getLogger(__name__)

def create_db_and_tables():
    """Create database and tables if they don't exist."""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path(settings.DATA_DIR)
        data_dir.mkdir(exist_ok=True)
        
        # Create agents directory if it doesn't exist
        agents_dir = Path(settings.AGENTS_DIR)
        agents_dir.mkdir(exist_ok=True)
        
        # Create vector_db directory if it doesn't exist
        vector_db_dir = Path(settings.VECTOR_DB_DIR)
        vector_db_dir.mkdir(exist_ok=True)
        
        # Initialize database tables
        init_db()
        
        logger.info("Database and tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database and tables: {e}")
        raise

def get_db():
    """Import and re-export the get_db function from session.py for convenience."""
    from src.database.session import get_db
    return get_db

if __name__ == "__main__":
    # This allows running this script directly to create tables
    print(f"Attempting to create database tables for: {settings.DATABASE_URL}")
    create_db_and_tables() 