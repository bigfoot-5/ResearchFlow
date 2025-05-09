"""
Configuration module for the Cognition Engine.

This module defines the configuration settings for the application.
"""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional
import os

# Load environment variables from .env file
# Make sure this is called before Settings class definition if .env is in a non-standard location
# or if you want to ensure it's loaded. For standard locations, pydantic-settings might load it automatically.
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./data/cognition_engine.db"
    
    # LLM API settings
    OPENAI_API_KEY: Optional[str] = None  # API key for OpenAI (used by reflection agent)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:1b"  # Default model, can be overridden by .env
    
    # Application settings
    ENABLE_REFLECTION: bool = True
    DEFAULT_REFLECTION_STEPS: int = 1
    API_TIMEOUT_SECONDS: int = 120
    
    # Vector store settings
    DEFAULT_CHROMA_COLLECTION: str = "crm_data"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Default SentenceTransformer model
    
    # Paths
    DATA_DIR: str = "data"
    VECTOR_DB_DIR: str = "data/vector_db"
    AGENTS_DIR: str = "data/agents"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Ensure data directories exist
for dir_path in [settings.DATA_DIR, settings.VECTOR_DB_DIR, settings.AGENTS_DIR]:
    os.makedirs(dir_path, exist_ok=True) 