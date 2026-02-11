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

    # Auto-load model on startup (model ID or name, empty to disable)
    AUTO_LOAD_MODEL: Optional[str] = None

    # Performance: Inference concurrency
    MAX_CONCURRENT_REQUESTS: int = 2
    MAX_PENDING_REQUESTS: int = 10

    # Performance: torch.compile
    TORCH_COMPILE: bool = False
    TORCH_COMPILE_MODE: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    # Performance: KV cache
    KV_CACHE_MODE: str = "static"  # "static" or "dynamic"

    # Performance: Prefix caching
    ENABLE_PREFIX_CACHE: bool = True
    PREFIX_CACHE_MAX_ENTRIES: int = 5

    # Performance: Speculative decoding
    SPECULATIVE_MODEL: Optional[str] = None  # HF model ID for draft model
    SPECULATIVE_NUM_TOKENS: int = 5

    # Performance: Continuous Batching (Phase 4)
    ENABLE_CONTINUOUS_BATCHING: bool = False  # Opt-in, starts CBM on model load
    CBM_MAX_QUEUE_SIZE: int = 256
    CBM_DEFAULT_TEMPERATURE: float = 0.7
    CBM_DEFAULT_TOP_P: float = 0.95
    CBM_DEFAULT_MAX_TOKENS: int = 512

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


# Global settings instance
settings = Settings()
