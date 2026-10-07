"""
Configuration management using Pydantic Settings.

All configuration is loaded from environment variables,
with support for .env files.
"""

import warnings
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_gguf_tensor_split(value: Optional[str]) -> Optional[list[float]]:
    """GGUF_TENSOR_SPLIT as proportions, or None when unset.

    Comma-separated, one non-negative number per GPU a split uses, in index
    order ("3,1"). Raises ValueError for anything else, so a typo fails at
    startup instead of silently splitting by free memory.
    """
    text = (value or "").strip()
    if not text:
        return None
    try:
        proportions = [float(part) for part in text.split(",")]
    except ValueError as e:
        raise ValueError(
            f"GGUF_TENSOR_SPLIT must be comma-separated numbers such as '3,1', got {value!r}"
        ) from e
    if any(p < 0 or p != p or p == float("inf") for p in proportions):
        raise ValueError(f"GGUF_TENSOR_SPLIT proportions must be finite and >= 0, got {value!r}")
    if not any(p > 0 for p in proportions):
        raise ValueError(f"GGUF_TENSOR_SPLIT needs at least one proportion above 0, got {value!r}")
    return proportions


#: The context, in tokens, every card of a transformers load must have KV-cache
#: room for (TRANSFORMERS_MIN_CONTEXT). Decision 7, 2026-09-14.
TRANSFORMERS_MIN_CONTEXT_DEFAULT = 4096

#: Memory, in MB, each card of a transformers load keeps for its CUDA context
#: (TRANSFORMERS_CUDA_CONTEXT_MB). Measured on the node (hardware acceptance,
#: 2026-09-14): 250-256 MiB idle, ~330 MiB after a first generation, growing to
#: ~400 MiB over a session, on both the RTX 3080 Ti and the RTX 3090. 500 covers it.
#: A request's prefill activations and the caching allocator's share are sized per
#: model and per card (millm/ml/working_memory.py), not taken from this.
TRANSFORMERS_CUDA_CONTEXT_MB_DEFAULT = 500

#: Seconds a transformers model's cards stay idle before torch's unused cached blocks
#: are returned to them (TRANSFORMERS_IDLE_CACHE_RELEASE_S). See the setting.
TRANSFORMERS_IDLE_CACHE_RELEASE_S_DEFAULT = 5.0


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
    # Dtype for the attached steering-weight set. "model" = the precision the model was LOADED at
    # (`ml/native_dtype.py`), as the single-SAE path already does. This was "float16", so circuits
    # and clusters steered with float16 weights added to a bfloat16 residual stream. Bytes per SAE
    # are unchanged at 16 bits (~64 MB measured); an FP32 model doubles them, and the envelope
    # sizes from the element width. An explicit dtype name still overrides.
    MULTISAE_ATTACH_DTYPE: str = "model"
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
    #: Feature 26 (FTASKS 0.4): batch generation rows' sensing events, capped apart from live.
    SENSING_MAX_BATCH_EVENTS_PER_CLUSTER: int = 50000
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
    CIRCUIT_SENSING_MAX_BATCH_EVENTS_PER_CIRCUIT: int = 50000
    CIRCUIT_SENSING_MAX_AGE_DAYS: int = 7
    CIRCUIT_SENSING_FORCE_SERIAL: bool = True
    CIRCUIT_SENSING_MAX_OVERHEAD_MS: float = 5.0

    # --- Feature 24: probe monitors -------------------------------------
    #: Armed probes force the serial path. ⚠ This is a SETTING, so the CBM path must also mark a
    #: request `not_scored` with a reason rather than relying on this being true — a probe never
    #: goes silently quiet (BR-006), and an absent verdict is exactly that.
    PROBE_FORCE_SERIAL: bool = True
    #: Bounds per-request overhead. Adjustable; 8 is the authored default.
    PROBE_MAX_ARMED: int = 8
    #: Floor on the parity tolerance. A definition may ask for tighter, never looser; miStudio's
    #: acceptance measured its vectors reproducing at 0.000e+00 from the recorded token_ids, so
    #: this is absorbing nothing on the reference path.
    PROBE_PARITY_TOLERANCE: float = 0.001
    #: The floor for the GATE's tolerance, on the combined score.
    #:
    #: ⚠ Not a loosening of `PROBE_PARITY_TOLERANCE`, which still governs how closely a build is
    #: expected to track. This is the point below which a refusal says more about float precision
    #: than about the probe: miStudio scores in fp16, miLLM serves bf16, and over 16 real vectors
    #: that costs a worst-case 0.098 on the score against a threshold of 2.879 — 3.4%. Set from
    #: measurement on 2026-09-27, not chosen. See `ParityReport`.
    PROBE_PARITY_SCORE_TOLERANCE: float = 0.10
    #: The gate's floor when the definition's `model.load_dtype` EQUALS the precision this server
    #: loaded at, so no cross-precision gap remains — only two implementations' bfloat16 noise.
    #:
    #: The ABSOLUTE minimum of the matched floor; the probe-relative part is below.
    PROBE_PARITY_MATCHED_DTYPE_FLOOR: float = 0.10
    #: The matched floor's RELATIVE part, as a fraction of the probe's own bar: the floor is
    #: max(PROBE_PARITY_MATCHED_DTYPE_FLOOR, this x |decision.threshold|).
    #:
    #: MEASURED 2026-10-03 on three retrained bfloat16 probes (48 vectors), miStudio scoring one
    #: input at a time vs this server: worst combined Δ 0.0679 (L11 rolling w=64, bar 25.29),
    #: 0.0278 (L16 mean, bar 17.24), 0.0856 (L21 rolling w=32, bar 46.78) — 0.16-0.27% of the bar,
    #: and 15 of 48 vectors reproduced EXACTLY. Relative because the rules score on different
    #: scales and the noise tracks the bar more tightly (1.7x spread) than it tracks an absolute
    #: number (3.1x). 0.6% gives 2.2-3.6x headroom on all three; a wrong detector misses by whole
    #: units. Chosen by the operator over keeping 0.10, which left the L21 probe 14% headroom.
    PROBE_PARITY_MATCHED_RELATIVE_FLOOR: float = 0.006
    #: How much of a vector must be reproducible for its comparison to count.
    #:
    #: ⚠ A k-sparse JumpReLU probe has positions whose gate NO independent implementation can
    #: reproduce (see `ParityReport`); those are set aside rather than tolerated. This is the
    #: point below which setting them aside would leave a comparison resting on a handful of
    #: tokens — such a vector is refused, not passed on the remainder. Measured at real width
    #: (2048 -> 16384, k = 128, the reference probe's own sparsity), 86% of positions survive the
    #: three-sigma band, so this is nowhere near binding on a healthy probe; it binds on a probe
    #: whose selected features sit on their own thresholds, which is a probe to refuse. A SMALL
    #: robust subset is noisy, not blind: a basis error is present at EVERY position, so it
    #: survives subsetting (measured 0.945 over 94% of positions, 0.356 over 86%).
    PROBE_PARITY_MIN_ROBUST_FRACTION: float = 0.25
    # ⚠ THE BUDGET IS PER FORWARD PASS, NOT PER REQUEST, because that is the unit the cost is
    # incurred in. Measured on Llama-3.1-8B, one probe, layer 11, varying ONLY `max_tokens` on an
    # identical prompt: 4 tokens -> 0.837 ms, 30 -> 3.425 ms, 120 -> 11.223 ms. Fits
    # 0.45 ms + ~0.09 ms per pass. Meanwhile 2820 PROMPT tokens with 8 generated cost 1.94 ms —
    # twenty-one times the tokens for a third of the overhead, because prefill scores the whole
    # prompt in ONE call while decode scores one token per call. A prompt token is ~300x cheaper
    # than a generated one.
    #
    # So the old per-REQUEST threshold of 5 ms was crossed by any completion longer than ~50
    # tokens on ANY model, including the LFM2 it was specified against — it warned on normal use,
    # which is noise rather than signal. It also made the criterion measurable on its cheapest
    # case: "under 5 ms at 4k-token contexts" is nearly free.
    PROBE_MAX_OVERHEAD_MS_PER_PASS: float = 0.25   # warn threshold per forward pass
    # An absolute per-request backstop, kept live rather than reported-and-unused. It exists for
    # pathology only: at ~0.09 ms/pass a 2000-token completion is ~180 ms, so 500 ms means
    # "something is wrong", not "this was a long answer". The regression this guards against is
    # real — copying the whole residual to the CPU once measured 25-37 ms for two probes.
    PROBE_MAX_OVERHEAD_MS: float = 500.0      # absolute per-request backstop

    # --- Feature 27: stateless probe scoring and per-request activations ---
    #: Inputs per `POST /api/probes/score` call. Bounds the work one request can queue; each input
    #: takes its own admission slot, so this bounds how long a call runs, not how long another
    #: request waits (one input's forward). 64 = miStudio's `millm_score_probes` cap (034 FTDD §9).
    PROBE_SCORE_MAX_INPUTS: int = 64
    #: Probes per scoring call. Every probe is read in the same forward, so this bounds the
    #: per-input scoring work; matches PROBE_MAX_ARMED, the live runtime's own bound.
    PROBE_SCORE_MAX_PROBES: int = 8
    #: Largest `top_k` a `return_sae_activations` request may ask for, per position.
    SAE_ACTIVATIONS_MAX_TOP_K: int = 64
    #: Worst-case (positions x top_k) entries one activation request may return, counted BEFORE
    #: generation (FR-27.2f) so a request is never refused after it has generated.
    SAE_ACTIVATIONS_MAX_ENTRIES: int = 65536
    #: Positions encoded per chunk when capturing activations: bounds the transient
    #: (positions x d_sae) encode on a long prefill against a wide SAE.
    SAE_ACTIVATIONS_ENCODE_CHUNK: int = 512
    PROBE_MAX_EVENTS_PER_PROBE: int = 5000
    #: Feature 26 (T-70): events from BATCH generation rows have their own cap, so a 50,000-row
    #: labelling run never evicts the live-traffic history PROBE_MAX_EVENTS_PER_PROBE keeps.
    PROBE_MAX_BATCH_EVENTS_PER_PROBE: int = 50000
    PROBE_MAX_AGE_DAYS: int = 30
    PROBE_EVENT_CONTEXT_TOKENS: int = 24      # +-K decoded tokens around the top firing position
    PROBE_HUB_TAG: str = "mistudio-probe-definition"
    PROBE_HUB_CACHE_TTL_S: int = 300
    PROBE_MAX_IMPORT_BYTES: int = 2_097_152   # 2 MB, the circuit importer's hostile-payload cap

    # Performance: Inference concurrency.
    # MUST stay 1 for correctness of everything built on the global SAE
    # state: per-request steering overrides (Features 8/10), monitoring
    # attribution, and co-activation sensing (Feature 11) all serialize on
    # the request queue. Raising it re-introduces cross-request races the
    # reviews closed (011 R1 top finding: the old default of 2 let two
    # generations interleave apply/restore and share the sensing buffer).
    MAX_CONCURRENT_REQUESTS: int = 1
    MAX_PENDING_REQUESTS: int = 10
    # Feature 30 (FR-30.3.6, T-93): maximum strings in one /v1/embeddings request. A capped
    # request holds the only admission slot (MAX_CONCURRENT_REQUESTS = 1), so this bounds how
    # long a chat waits behind it. To be set from measured latency: the largest power of two
    # with cap x p95 seconds-per-input <= 30 s (S3-11), in [64, 2048]. MEASURED on the RTX 3090
    # (030_FTASKS 8.3, 2026-10-07) at 512 tokens per input over 64 inputs: LFM2.5-1.2B-Instruct
    # p95 0.0313 s -> 512; Llama-3.1-8B-Instruct p95 0.1636 s -> 128 (256 would be ~42 s). One
    # global setting must hold for the largest model served, so 128.
    EMBEDDINGS_MAX_INPUTS: int = Field(default=128, ge=1, le=2048)

    # --- Feature 26: the Batch API (026 FTID §9) ---
    #: Where uploaded and assembled batch file BYTES live (metadata is in PostgreSQL). k8s sets
    #: /data/batch_files, on the data volume, so the bytes survive a pod restart (FR-26.3.7).
    BATCH_FILES_DIR: str = "/app/batch_files"
    #: OpenAI's documented limits (FR-26.8.1). A file over either is refused at upload, whole.
    BATCH_MAX_ROWS: int = 50000
    BATCH_MAX_FILE_BYTES: int = 209_715_200
    #: Per input line: a longer line is an invalid line (`line_too_large`), never parsed.
    BATCH_MAX_LINE_BYTES: int = 1_048_576
    #: T-63: scoring and embedding rows are packed unless the acceptance-7 measurement finds a
    #: packed row whose top token differs from its single-row answer; then this becomes false.
    #: ⚠ An unparseable value FAILS TO THE DEFAULT (true), never to false (`_bool_to_default`).
    BATCH_PACK_DEFAULT: bool = True
    #: Single-row forwards per chunk (one admission slot each chunk). Acceptance 9 bounds it.
    BATCH_CHUNK_ROWS: int = 8
    BATCH_PACK_MAX_ROWS: int = 16
    BATCH_PACK_MAX_TOKENS: int = 16384
    #: T-65: `completion_window` accepts whole hours 1..this (OpenAI's own value is "24h").
    BATCH_MAX_COMPLETION_WINDOW_HOURS: int = 168
    #: T-68: input files expire this many days after upload. Output/error files follow the
    #: batch's `output_expires_after` (default the same 30 days, FR-26.9.1).
    BATCH_FILE_RETENTION_DAYS: int = 30
    #: The batch's OWN lease TTL; renewed when a third of it has passed. ≤ LEASE_MAX_TTL_SECONDS.
    BATCH_LEASE_TTL_S: int = 900
    #: How often a waiting batch retries its lease (`model_not_resident`, `lease_unavailable`).
    BATCH_WAIT_POLL_S: float = 10.0
    #: `batch:progress` at most this often per batch, plus every status transition.
    BATCH_PROGRESS_MIN_INTERVAL_S: float = 1.0
    #: Validation errors listed in the batch object's `errors.data` (the error FILE has all).
    BATCH_ERRORS_SHOWN: int = 100
    #: How often expired batch files are pruned (also once at startup, FR-26.9.4).
    BATCH_RETENTION_INTERVAL_S: float = 3600.0

    # Feature 29: model lease (process memory only; a restart ends every lease, X-01).
    # The default TTL is also the maximum (2 hours); a TTL outside 1..LEASE_MAX_TTL_SECONDS
    # is refused with 400 INVALID_LEASE_REQUEST, never clamped.
    LEASE_DEFAULT_TTL_SECONDS: int = 7200
    LEASE_MAX_TTL_SECONDS: int = 7200
    LEASE_HOLDER_MAX_CHARS: int = 128
    LEASE_REASON_MAX_CHARS: int = 512
    # Ended leases remembered (by digest) so renew/release of a seen ID answers 409 with
    # its end reason rather than 404. Lost on restart, deliberately.
    LEASE_ENDED_MEMORY: int = 64
    # Feature 29: slot-holding durations kept for /api/health/detailed's estimated wait.
    QUEUE_DURATION_WINDOW: int = 50
    # Feature 29: Retry-After seconds per 503 code (029 FTDD §5.3). The fallback is
    # DISTINCT on purpose: it is set only by RetryAfterMiddleware, for a 503 whose
    # builder forgot the header, and that path also logs `retry_after_defaulted`.
    RETRY_AFTER_QUEUE_DEFAULT_S: int = 5
    RETRY_AFTER_MAX_S: int = 60
    RETRY_AFTER_LOAD_S: int = 15
    RETRY_AFTER_UNLOAD_S: int = 5
    RETRY_AFTER_NOT_LOADED_S: int = 30
    RETRY_AFTER_MEMORY_S: int = 30
    RETRY_AFTER_READINESS_S: int = 5
    RETRY_AFTER_FALLBACK_S: int = 10

    # Feature 25 (request validation and structured output).
    # Compiled JSON-Schema grammars kept per loaded model (LRU). A compile costs
    # 0.1-0.7 s on the served tokenizers and runs outside the request slot; the
    # cache is dropped on unload, because a grammar belongs to one tokenizer.
    STRUCTURED_OUTPUT_GRAMMAR_CACHE: int = 64
    # Upper bound, in bytes, on the X-miLLM-Ignored-Fields response header. Field
    # names are client-chosen JSON keys; past the bound the header ends with
    # "+N more" rather than growing without limit.
    IGNORED_FIELDS_HEADER_MAX_BYTES: int = 1024

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

    # Load GGUF models with embedding output enabled, so /v1/embeddings works
    # on them without a second load.
    #
    # ON by default because the alternative is a server that cannot embed the
    # model it is serving, and the cost is small: MEASURED on the RTX 3090 with
    # zora-v1.13 Q5_K_M, generation went 113.5 -> 105.9 tok/s, a 6.7% loss, with
    # no change in VRAM. llama.cpp needs this at CONSTRUCTION — there is no way
    # to turn it on later — so the choice is made here or not at all.
    #
    # Turn it off to buy that 6.7% back on a deployment that never embeds.
    GGUF_ENABLE_EMBEDDINGS: bool = True

    # CEILING on the context window for GGUF models, in tokens — NOT a target.
    #
    # The loader reads what the model itself declares (`n_ctx_train`, via a
    # 1.2-second CPU probe that costs no VRAM) and starts the ladder at
    # min(declared, this). It was previously the starting point outright, which
    # served an 8192 window to a model trained for 262144 and said nothing:
    # ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF declares 262144 and creates a
    # context at 131072 on this 24 GiB card, so the old default threw away 16x
    # the window the hardware could actually hold.
    #
    # A ceiling is still needed in BOTH directions. llama.cpp's own default is
    # 512, which truncates almost any real conversation. Unbounded is the other
    # trap: the whole KV cache is allocated at context creation, so a
    # quarter-million-token window on a 31B model is tens of gigabytes — after
    # ~19.9 GiB of Q5_K_S weights on a 24 GiB card the load simply fails, and
    # even when it succeeds it reserves a GPU that miStudio's extraction,
    # training and steering work shares.
    #
    # 32768 is chosen to clear the labeling prompt (~4700 tokens, which 8192
    # cleared only in principle and 4096 did not clear at all) with room for
    # long documents, while leaving the card usable. Raise it when the GPU is
    # dedicated to serving; 0 means "whatever the file declares", bounded only
    # by what fits.
    GGUF_CONTEXT_LENGTH: int = 32768

    # HOW A GGUF LAYER SPLIT DIVIDES ITS LAYERS, when a model needs more than
    # one GPU. Empty (the default) splits in proportion to the free memory the
    # plan found on each card used, largest first. Otherwise one proportion per
    # card the split uses, in CUDA index order: "1,3" puts a quarter of the
    # layers on the lower-index card. llama.cpp indexes its list by device, so
    # miLLM maps these onto the cards used and gives every other card 0.
    #
    # A value with the wrong number of entries for a load is REFUSED, not
    # stretched: guessing would put layers on cards nobody chose. It has no
    # effect on a model that fits one card.
    GGUF_TENSOR_SPLIT: str = ""

    # KV-CACHE PRECISION. This is the single biggest lever on how much context
    # a GGUF model can hold, and it sat at llama.cpp's F16 default unexamined.
    #
    # gemma-4-31b spends 630 KB PER TOKEN on KV cache at F16 (60 layers x 16 KV
    # heads x 168 head_dim x 2 tensors x 2 bytes). At 4096 tokens that is 2.6
    # GiB — more than a third of the free VRAM on a 24 GiB card — to hold about
    # three thousand words. The context ladder was searching for a context that
    # fit while the cost of a token went unquestioned.
    #
    # MEASURED on gemma-4-31b IQ4_XS on the RTX 3090:
    #   f16  + FA : 8192 FAILS, 4096 is the ceiling
    #   q8_0 + FA : 12288 loads          <- 3x, and clears the ~4700-token
    #                                       labeling prompt that f16 could not
    #   q4_0 + FA : 16384 loads
    #
    # q8_0 is the default because it is near-lossless. q4_0 buys another third
    # of a window at a real accuracy cost, which is the wrong trade for a
    # judge — discrimination is the job. Set it only when a long window matters
    # more than precision.
    #
    # "f16" restores the previous behaviour exactly.
    GGUF_KV_CACHE_TYPE: str = "q8_0"

    # REQUIRED by quantized KV cache, not merely an optimisation: MEASURED,
    # q8_0 without flash attention fails at 8192 where q8_0 with it reaches
    # 12288. The loader refuses to pair a quantized cache with this off.
    GGUF_FLASH_ATTENTION: bool = True

    # THE CONTEXT EVERY TRANSFORMERS LOAD MUST HAVE ROOM FOR, in tokens.
    #
    # A transformers load — a split, Auto's one-card-or-split choice, or a named
    # card — is accepted only when EACH card it uses has room, beside the weights
    # transformers' own device map puts there, for its CUDA context
    # (TRANSFORMERS_CUDA_CONTEXT_MB) and the KV cache of the layers it holds at
    # this many tokens. It replaced a 20% slack on the weight estimate, which grew
    # with the weights, not with what each card needs: on 11.5 + 23.5 GB free it
    # refused Qwen2.5-14B at FP16 (room for about 8k tokens on both cards),
    # accepted OLMo-2-13B (whose cuda:0 cannot hold an 8k cache), and split models
    # that fit one card. Decision 7, 2026-09-14.
    #
    # An admission floor, not a limit on requests: a card with more room serves
    # longer contexts. GGUF models keep their own context prediction
    # (GGUF_CONTEXT_LENGTH).
    TRANSFORMERS_MIN_CONTEXT: int = TRANSFORMERS_MIN_CONTEXT_DEFAULT

    # MEMORY EACH CARD OF A TRANSFORMERS LOAD KEEPS FOR ITS CUDA CONTEXT, in MB.
    # Measured on the node at 250-400 MiB a card (TRANSFORMERS_CUDA_CONTEXT_MB_DEFAULT).
    # The context only: a request's prefill activations and the caching allocator's
    # share are sized per model and per card (millm/ml/working_memory.py). Hardware
    # acceptance, 2026-09-14: this was the only allowance, and OLMo-2-13B admitted
    # with 11 MiB to spare ran a 3,879-token request out of memory.
    TRANSFORMERS_CUDA_CONTEXT_MB: int = TRANSFORMERS_CUDA_CONTEXT_MB_DEFAULT

    # SECONDS A TRANSFORMERS MODEL'S CARDS STAY IDLE BEFORE TORCH'S CACHE IS RETURNED.
    #
    # torch keeps the blocks a finished request freed, and nvidia-smi — which
    # miStudio's placement and every other tenant of the node read — counts them
    # as used. After three requests the RTX 3080 Ti sat at 12,004 MiB used / 155
    # MiB free until the model was unloaded (hardware acceptance, 2026-09-14).
    # Once no request has held or waited for a slot for this long, the model's
    # cards get their unused cached blocks back (torch.cuda.empty_cache, taken
    # through the request queue so it never runs during a request). A request that
    # arrives in the meantime cancels it. The cost is paid by the next request,
    # which reserves those segments again: for a 2,000-token request on OLMo-2-13B's
    # cuda:0 that is 29 cudaMalloc calls (1,070 MiB), for Qwen2.5-7B 13. 0 releases
    # as soon as the queue is idle; a negative value never releases. Nothing is
    # released while continuous batching runs: its manager generates without a
    # queue slot, so the queue reads idle during a CBM request.
    TRANSFORMERS_IDLE_CACHE_RELEASE_S: float = TRANSFORMERS_IDLE_CACHE_RELEASE_S_DEFAULT

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

    @field_validator("BATCH_PACK_DEFAULT", mode="before")
    @classmethod
    def _bool_to_default(cls, value: Any) -> Any:
        """A boolean setting fails to its DEFAULT, not to False (this suite's `dry_run` lesson).

        `BATCH_PACK_DEFAULT=ture` must not quietly turn packing off — nor raise and stop the
        server starting over a typo in a switch whose default is already safe.
        """
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        warnings.warn(
            f"BATCH_PACK_DEFAULT={value!r} is not a boolean; using the default (true)",
            stacklevel=2,
        )
        return True

    @field_validator("GGUF_TENSOR_SPLIT")
    @classmethod
    def _validate_gguf_tensor_split(cls, value: str) -> str:
        parse_gguf_tensor_split(value)
        return value

    @field_validator("TRANSFORMERS_MIN_CONTEXT")
    @classmethod
    def _validate_transformers_min_context(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"TRANSFORMERS_MIN_CONTEXT must be at least 1 token, got {value}")
        return value

    @field_validator("TRANSFORMERS_CUDA_CONTEXT_MB")
    @classmethod
    def _validate_transformers_cuda_context_mb(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"TRANSFORMERS_CUDA_CONTEXT_MB must be 0 or more, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_batch_lease_ttl(self) -> "Settings":
        """The batch's own lease is a Feature 29 lease: its TTL must be one 029 would grant."""
        if not 1 <= self.BATCH_LEASE_TTL_S <= self.LEASE_MAX_TTL_SECONDS:
            raise ValueError(
                f"BATCH_LEASE_TTL_S must be between 1 and LEASE_MAX_TTL_SECONDS "
                f"({self.LEASE_MAX_TTL_SECONDS}), got {self.BATCH_LEASE_TTL_S}"
            )
        return self

    @model_validator(mode="after")
    def _validate_lease_ttls(self) -> "Settings":
        """A default TTL above the maximum would make every TTL-less grant a refusal."""
        if self.LEASE_MAX_TTL_SECONDS < 1:
            raise ValueError(
                f"LEASE_MAX_TTL_SECONDS must be at least 1, got {self.LEASE_MAX_TTL_SECONDS}"
            )
        if not 1 <= self.LEASE_DEFAULT_TTL_SECONDS <= self.LEASE_MAX_TTL_SECONDS:
            raise ValueError(
                "LEASE_DEFAULT_TTL_SECONDS must be between 1 and LEASE_MAX_TTL_SECONDS "
                f"({self.LEASE_MAX_TTL_SECONDS}), got {self.LEASE_DEFAULT_TTL_SECONDS}"
            )
        return self

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
