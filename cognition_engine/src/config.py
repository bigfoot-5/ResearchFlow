"""
Configuration module for the Cognition Engine.

This module defines the configuration settings for the application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
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
    db_name: str = Field(default="cognition_db", env="DB_NAME")
    db_user: str = Field(default="cognition_user", env="DB_USER")
    db_password: str = Field(default="cognition_password", env="DB_PASSWORD")
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: str = Field(default="5432", env="DB_PORT")
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct the database URL from individual components."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    # LLM API settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="gemma3:1b", env="OLLAMA_MODEL")
    OLLAMA_HOST: Optional[str] = Field(default=None, env="OLLAMA_HOST")
    HUBSPOT_API_KEY: Optional[str] = Field(default=None, env="HUBSPOT_API_KEY")
    GONG_ACCESS_KEY: Optional[str] = Field(default=None, env="GONG_ACCESS_KEY")
    GONG_ACCESS_SECRET: Optional[str] = Field(default=None, env="GONG_ACCESS_SECRET")
    
    # Application settings
    ENABLE_REFLECTION: bool = Field(default=True, env="ENABLE_REFLECTION")
    DEFAULT_REFLECTION_STEPS: int = Field(default=1, env="DEFAULT_REFLECTION_STEPS")
    API_TIMEOUT_SECONDS: int = Field(default=120, env="API_TIMEOUT_SECONDS")
    
    # Vector store settings
    DEFAULT_CHROMA_COLLECTION: str = Field(default="crm_data", env="DEFAULT_CHROMA_COLLECTION")
    CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, env="CHROMA_PORT")
    DEFAULT_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="DEFAULT_EMBEDDING_MODEL")
    
    # Paths
    DATA_DIR: str = Field(default="data", env="DATA_DIR")
    VECTOR_DB_DIR: str = Field(default="data/vector_db", env="VECTOR_DB_DIR")
    AGENTS_DIR: str = Field(default="data/agents", env="AGENTS_DIR")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"  # Allow extra fields in the settings
    )

# Create settings instance
settings = Settings()

# Ensure data directories exist
for dir_path in [settings.DATA_DIR, settings.VECTOR_DB_DIR, settings.AGENTS_DIR]:
    os.makedirs(dir_path, exist_ok=True) 