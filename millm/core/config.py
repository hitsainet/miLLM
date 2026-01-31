"""
Configuration management using Pydantic Settings.

All configuration is loaded from environment variables,
with support for .env files.
"""

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/millm"

    # Model cache directory (matches docker-compose volume mount)
    MODEL_CACHE_DIR: str = "/app/model_cache"

    # SAE cache directory (matches docker-compose volume mount)
    SAE_CACHE_DIR: str = "/app/sae_cache"

    # HuggingFace
    HF_TOKEN: Optional[str] = None

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS
    CORS_ORIGINS: str = "*"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "console"

    # Threading
    MAX_DOWNLOAD_WORKERS: int = 2
    MAX_LOAD_WORKERS: int = 1

    # Timeouts (seconds)
    GRACEFUL_UNLOAD_TIMEOUT: float = 30.0
    DOWNLOAD_TIMEOUT: float = 3600.0  # 1 hour max for large models

    # Redis (optional, for distributed state)
    REDIS_URL: Optional[str] = None

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


# Global settings instance
settings = Settings()
