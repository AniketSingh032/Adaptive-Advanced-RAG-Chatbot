"""
Application Configuration Management Module.

This module handles all configuration settings for the application using Pydantic.
It loads environment variables and provides a structured way to access configuration
settings across the application.

Attributes:
    settings (Settings): Global settings instance for application-wide configuration.

Example:
    from SRC.config.config import settings
    
    model_name = settings.GROQ_MODEL
    api_key = settings.GROQ_API_KEY
"""
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv(override=True)

class Settings(BaseSettings):
    """
    Application settings management using Pydantic.
    
    This class handles all configuration settings for the application,
    including API keys, model configurations, and storage settings.
    
    Attributes:
        GROQ_API_KEY (str): Authentication key for Groq API access.
        GROQ_MODEL (str): Model identifier used for Groq API calls.
        EMBEDDINGS_MODEL (str): HuggingFace model name for embeddings generation.
        PERSIST_DIRECTORY (str): File system path for vector store persistence.
    """
    
    # LLM Configuration
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY", description="API key for Groq service authentication")
    GROQ_MODEL: str = Field(..., env="GROQ_MODEL", description="Model identifier for Groq API calls")
    
    # Embedding Configuration
    EMBEDDINGS_MODEL: str = Field(..., env="EMBEDDINGS_MODEL", description="HuggingFace embeddings model identifier")
    
    # Vector Store Configuration
    PERSIST_DIRECTORY: str = Field(..., env="PERSIST_DIRECTORY", description="Directory path for vector store persistence")
    
    # Langsmith Configuration
    LANGSMITH_API_KEY: str = Field(..., env="LANGSMITH_API_KEY")
    LANGSMITH_ENDPOINT: str = Field(..., env="LANGSMITH_ENDPOINT")
    LANGSMITH_PROJECT: str = Field(..., env="LANGSMITH_PROJECT")
    LANGSMITH_TRACING: bool = Field(..., env="LANGSMITH_TRACING")


    class Config:
        """Pydantic model configuration settings."""
        env_file = ".env"

settings = Settings()