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

    # ── Cluster import (Feature 8) ──────────────────────────────────────
    CLUSTER_HUB_TAG: str = "mistudio-cluster-definition"
    CLUSTER_HUB_CACHE_TTL_S: int = 300
    # Fallback lambda bounds when a definition lacks budget.intensity_range
    # (also used by the per-request dial, Feature 10).
    CLUSTER_INTENSITY_MIN: float = 0.5
    CLUSTER_INTENSITY_MAX: float = 1.5

    # ── Circuit import (Feature 13) ────────────────────────────────────
    CIRCUIT_HUB_TAG: str = "mistudio-circuit-definition"
    CIRCUIT_MAX_LAYERS: int = 16          # == contract MAX_SAES
    CIRCUIT_MAX_EDGES: int = 200          # == contract MAX_EDGES
    CIRCUIT_MAX_MEMBERS_PER_LAYER: int = 20

    # ── Multi-SAE circuit serving (Feature 12) ─────────────────────────
    # ADVISORY budget for the attached SAE steering set — a "you may not have
    # intended this much" hint, NOT a capacity limit. Real capacity is enforced
    # in attach_set against live free VRAM (torch.cuda.mem_get_info) with a 10%
    # headroom margin, which refuses with InsufficientMemoryError.
    #
    # This was originally 200 MB: the close-out TARGET from the two-SAE spike
    # (two Gemma-2-2B SAEs = 128 MB fp16 / 256 MB fp32), not a capacity figure.
    # A 5-SAE circuit on a 24 GB card sits at ~640 MB — entirely fine, but it
    # tripped an "over the VRAM envelope" warning that read like a refusal.
    # A documentation number must not masquerade as an operational limit.
    #
    # 4096 MB ≈ 32 SAEs at the measured 128 MB fp16 each, comfortably past the
    # 16-layer contract maximum while still flagging a genuine runaway.
    MULTISAE_VRAM_ENVELOPE_MB: int = 4096
    # Dtype for the attached steering-weight set (fp16 ≈ 64 MB/SAE measured).
    MULTISAE_ATTACH_DTYPE: str = "float16"
    # Global circuit intensity (λ) bounds — shared with the Feature 14 dial.
    CIRCUIT_INTENSITY_MIN: float = 0.0
    CIRCUIT_INTENSITY_MAX: float = 2.0
    #: Feature 19. Several circuits may serve at once when their claim sets are
    #: disjoint. Defaults FALSE for one release (BR-011a) so the split is
    #: reversible in the field, with a dated flip commitment recorded in the
    #: BRD — an unflipped flag makes a shipped capability unreachable, which is
    #: the defect class this increment exists to eliminate.
    #:
    #: Flag OFF REFUSES LOUDLY, naming configuration as the reason. It must NOT
    #: fall back to the silent single-active disarm this feature replaces: that
    #: silent fallback IS the bug (CLAIM-M4).
    CIRCUIT_ALLOW_CONCURRENT: bool = False

    # Co-activation sensing (Feature 11)
    SENSING_EPSILON: float = 0.1              # theta_i = max(floor, eps*max_act_i)
    SENSING_THETA_FLOOR: float = 0.0
    SENSING_CONTEXT_TOKENS: int = 16          # +-K context window; hard max 64
    SENSING_MAX_EVENTS_PER_REQUEST: int = 20
    SENSING_MAX_EVENTS_PER_CLUSTER: int = 1000
    SENSING_MAX_AGE_DAYS: int = 7
    SENSING_FORCE_SERIAL: bool = True         # armed sensing forces serial routing
    SENSING_DEDUP_HISTORY: bool = True        # report re-read chat history once, not per turn
    SENSING_MAX_OVERHEAD_MS: float = 5.0      # warn threshold per request

    # --- Feature 15: circuit edge sensing -------------------------------
    #: Max tokens between an upstream fire and its downstream partner for the
    #: pair to count as one edge observation. Too wide and unrelated fires get
    #: attributed to each other; too narrow and real multi-token effects are
    #: missed. 8 is the authored default, overridable per circuit.
    CIRCUIT_SENSING_MAX_TOKEN_LAG: int = 8
    CIRCUIT_SENSING_EPSILON: float = 0.1
    CIRCUIT_SENSING_THETA_FLOOR: float = 0.0
    CIRCUIT_SENSING_CONTEXT_TOKENS: int = 16
    CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST: int = 20
    CIRCUIT_SENSING_MAX_EVENTS_PER_CIRCUIT: int = 1000
    CIRCUIT_SENSING_MAX_AGE_DAYS: int = 7
    CIRCUIT_SENSING_FORCE_SERIAL: bool = True
    CIRCUIT_SENSING_MAX_OVERHEAD_MS: float = 5.0

    # Performance: Inference concurrency.
    # MUST stay 1 for correctness of everything built on the global SAE
    # state: per-request steering overrides (Features 8/10), monitoring
    # attribution, and co-activation sensing (Feature 11) all serialize on
    # the request queue. Raising it re-introduces cross-request races the
    # reviews closed (011 R1 top finding: the old default of 2 let two
    # generations interleave apply/restore and share the sensing buffer).
    MAX_CONCURRENT_REQUESTS: int = 1
    MAX_PENDING_REQUESTS: int = 10

    # Performance: torch.compile
    # None  → auto-detect: enabled for CUDA models that don't use bitsandbytes
    # True  → always attempt compilation (loader still skips for bitsandbytes)
    # False → never compile
    TORCH_COMPILE: Optional[bool] = None
    # "default" deliberately. "reduce-overhead" enables CUDA Graphs, which broke
    # this generate path in production (2026-07-27): compile and warmup both
    # succeeded, then every request after the first raised "accessing tensor
    # output of CUDAGraphs that has been overwritten by a subsequent run".
    # Changing this back needs a multi-request soak on hardware, not a warmup.
    TORCH_COMPILE_MODE: str = "default"  # "default" | "reduce-overhead" | "max-autotune"

    # Performance: KV cache
    KV_CACHE_MODE: str = "dynamic"  # "static" (requires C compiler for triton) or "dynamic"

    # Performance: Speculative decoding
    SPECULATIVE_MODEL: Optional[str] = None  # HF model ID for draft model
    SPECULATIVE_NUM_TOKENS: int = 5

    # Performance: Continuous Batching
    ENABLE_CONTINUOUS_BATCHING: bool = False  # Opt-in, starts CBM on model load
    CBM_MAX_QUEUE_SIZE: int = 256
    # CBM fixes its sampling parameters at manager creation, and any request
    # whose temperature/top_p differ FALLS BACK TO THE SERIAL PATH
    # (cbm_routing_fallback_to_serial). So these values decide WHICH workload
    # gets batched — they are not cosmetic defaults.
    #
    # 0.0 to match bulk labeling, which is the workload continuous batching was
    # turned on for. It runs temperature 0 throughout. With the previous 0.7,
    # every labeling request mismatched and fell back to serial: the stated
    # beneficiary was the one workload excluded (observed live 2026-07-27).
    #
    # Interactive traffic at other temperatures still falls back to serial —
    # i.e. exactly the behaviour it had before CBM existed, so this costs it
    # nothing. Only one sampling profile can be batched at a time.
    CBM_DEFAULT_TEMPERATURE: float = 0.0
    CBM_DEFAULT_TOP_P: float = 1.0
    CBM_DEFAULT_MAX_TOKENS: int = 512
    # When True, requests with SAE monitoring enabled are routed through the serial
    # path instead of CBM, ensuring accurate per-request activation attribution.
    # Trades throughput for monitoring fidelity. Default False (batch-level monitoring).
    CBM_FORCE_SERIAL_MONITORING: bool = False

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Return the process-wide Settings instance.

    Most of the codebase imports the module-level ``settings`` singleton
    directly; this accessor exists for callers (and tests) that prefer a
    function-style dependency.
    """
    return settings
