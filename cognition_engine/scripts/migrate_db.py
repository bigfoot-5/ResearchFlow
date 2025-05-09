#!/usr/bin/env python
"""
Database migration script for Cognition Engine.

This script updates the database schema to match the current models.
"""

import logging
import sys
import os

# Add the project root to Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import create_engine, MetaData, Table, Column, Boolean, Integer, text, JSON
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_agent_executions_table():
    """Add missing columns to agent_executions table."""
    try:
        # Create SQLAlchemy engine
        engine = create_engine(settings.DATABASE_URL)
        
        # Create a metadata object
        metadata = MetaData()
        
        # Reflect the agent_executions table
        agent_executions = Table('agent_executions', metadata, autoload_with=engine)
        
        # Check if the columns already exist
        columns_to_add = []
        
        if 'reflection_enabled' not in agent_executions.columns:
            columns_to_add.append(("reflection_enabled", "BOOLEAN", "DEFAULT FALSE"))
            logger.info("Need to add reflection_enabled column")
        
        if 'reflection_steps' not in agent_executions.columns:
            columns_to_add.append(("reflection_steps", "INTEGER", "DEFAULT 0"))
            logger.info("Need to add reflection_steps column")
            
        if 'reflection_applied' not in agent_executions.columns:
            columns_to_add.append(("reflection_applied", "BOOLEAN", "DEFAULT FALSE"))
            logger.info("Need to add reflection_applied column")
            
        if 'meta_data' not in agent_executions.columns:
            columns_to_add.append(("meta_data", "JSONB", "DEFAULT '{}'::jsonb"))
            logger.info("Need to add meta_data column")

        if 'context' not in agent_executions.columns:
            columns_to_add.append(("context", "JSONB", "DEFAULT '{}'::jsonb"))
            logger.info("Need to add context column")
            
        if 'result' not in agent_executions.columns:
            columns_to_add.append(("result", "JSONB", "DEFAULT '{}'::jsonb"))
            logger.info("Need to add result column")
        
        # Execute ALTER TABLE statements if needed
        if columns_to_add:
            with engine.begin() as conn:
                for column_name, column_type, default_value in columns_to_add:
                    # Create and execute the ALTER TABLE statement using text()
                    alter_stmt = text(f"ALTER TABLE agent_executions ADD COLUMN IF NOT EXISTS {column_name} {column_type} {default_value};")
                    conn.execute(alter_stmt)
                    logger.info(f"Added column {column_name} to agent_executions table")
            
            logger.info("Database migration completed successfully")
        else:
            logger.info("No columns need to be added")
        
    except Exception as e:
        logger.error(f"Error during database migration: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting database migration...")
    migrate_agent_executions_table()
    logger.info("Database migration completed") 