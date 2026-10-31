"""
Model service for orchestrating model operations.

This service coordinates between the repository, downloader, and loader
components to manage model lifecycle operations.
"""

import asyncio
import functools
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

# Configuration constants
MAX_DOWNLOAD_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds
UNLOAD_TIMEOUT = 30.0  # seconds

from millm.api.schemas.model import ModelDownloadRequest, ModelPreviewRequest
from millm.core.errors import (
    DownloadCancelledError,
    GpuNotFoundError,
    ModelAlreadyExistsError,
    ModelAlreadyLoadedError,
    ModelBusyError,
    ModelLoadError,
    ModelLockedError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelNotResidentError,
    InvalidLeaseRequestError,
    LoadThroughWeightsApiError,
    WeightsTornError,
    WeightsUnsavedError,
)
from millm.core.errors import GgufTensorSplitError
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.memory_utils import estimate_memory_mb
from millm.ml.gpu_placement import (
    GpuRequest,
    list_gpus,
    parse_gpu_request,
    project_free_after_unload,
)
from millm.ml.gguf_catalog import coarse_quantization
from millm.ml.task_heads import task_of_model_dir
from millm.ml.model_downloader import ModelDownloader, _safe_variant
from millm.ml.model_loader import (
    GGUF_MIN_CONTEXT,
    ModelLoader,
    _gguf_shard_rule,
    _gguf_tensor_split,
    plan_gguf_placement,
    plan_transformers_load,
    preflight_split,
)
from millm.services.model_lease import (
    END_MODEL_UNLOADED,
    EndedLease,
    LeaseGrant,
    LeaseRecord,
    LeaseRegistry,
    get_lease_registry,
    refuse_if_leased,
)
from millm.sockets.progress import ProgressEmitter

logger = structlog.get_logger()


def resolve_cache_path(cache_path: str) -> str:
    """A model row's cache path as the loader opens it.

    The database stores either an absolute path or one relative to
    MODEL_CACHE_DIR, depending on when the model was downloaded. The pre-unload
    check and the background load must read the SAME checkpoint, so both resolve
    it here.
    """
    from millm.core.config import settings

    if os.path.isabs(cache_path):
        return cache_path
    return f"{settings.MODEL_CACHE_DIR}/{cache_path}"


#: Download progress, keyed by model id, SHARED ACROSS SERVICE INSTANCES.
#:
#: This has to live outside the instance because `get_model_service` constructs
#: a brand-new `ModelService` for every request. A dict on `self` is therefore
#: written by the instance that started the download and read by a different,
#: empty one — so `GET /api/models` reported `download_progress: null` for the
#: entire duration of every download that has ever run. Observed on a 51.8 GB
#: Qwen3.8-27B pull: the bytes were landing at 110 MB/s and the API said
#: nothing, which is indistinguishable from a download that has died.
#:
#: In-process is sufficient and durable enough. The download runs in a thread of
#: this same process, and a restart ends it anyway — startup resets any model
#: left in DOWNLOADING (STALE_STATE_RESETS in main.py; that claim was made here
#: long before anything implemented it, and a row sat stuck for half an hour
#: because of it). Persisting to a column would add a write every poll interval
#: to reconstruct state that cannot outlive the work it describes.
_DOWNLOAD_PROGRESS: dict[int, int] = {}


#: The model whose load is in progress, SHARED ACROSS SERVICE INSTANCES, for the
#: reason `_DOWNLOAD_PROGRESS` is: `get_model_service` builds a new ModelService
#: per request. The one-load-at-a-time slot lived on `self`, so it was only ever
#: seen by the request that claimed it. A double-clicked Switch, or two OpenAI
#: requests naming different models, each got a fresh instance whose slot was
#: empty, both passed the busy check and both loads ran at once. Review round 3,
#: 2026-09-14. Read and written only through `ModelService._loading_model_id`.
_LOAD_SLOT: dict[str, Optional[int]] = {"model_id": None}


class _LoadInProgress:
    """`ModelService.is_loading`: whether ANY service instance has claimed the load slot.

    A descriptor rather than a plain `property` so it also reads from the class: the weight
    service probes it with no `ModelService` instance at hand (031 review D3; addendum §8
    replaces that service's private read of `_LOAD_SLOT`).
    """

    def __get__(self, obj: Any, owner: Any = None) -> bool:
        return _LOAD_SLOT["model_id"] is not None


#: The sentence every D-01 torn refusal carries: what to do next (addendum D-01).
_TORN_RECOVERY = (
    "The resident weights were left mid-write and are not served. Recover by loading a "
    "checkpoint, or the base model by id, through POST /api/weights/load."
)


class ModelService:
    """
    Orchestration layer for model operations.

    Coordinates between repository, downloader, and loader components.
    Uses a thread pool for long-running operations like downloads and loads.
    """

    def __init__(
        self,
        repository: ModelRepository,
        downloader: ModelDownloader,
        loader: Optional[ModelLoader] = None,
        emitter: Optional[ProgressEmitter] = None,
        inference_service: Optional[Any] = None,
    ) -> None:
        """
        Initialize the model service.

        Args:
            repository: Model database repository
            downloader: Model downloader for HuggingFace
            loader: Model loader for GPU loading
            emitter: Progress event emitter for WebSocket updates
            inference_service: Optional InferenceService ref for CBM lifecycle hooks.
                When provided, avoids importing from millm.api (layer violation).
        """
        self.repository = repository
        self.downloader = downloader
        self.loader = loader or ModelLoader()
        self.emitter = emitter
        self._inference_service = inference_service

        # Thread pool for background tasks
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="millm-",
        )

        # Track active downloads for cancellation
        self._active_downloads: dict[int, Future[str]] = {}
        self._cancelled_downloads: set[int] = set()

        # Active loads are tracked in the module-level `_LOAD_SLOT` (see
        # `_loading_model_id`). NOT reset here: every request constructs a
        # service, so resetting in __init__ would release a load in progress.

        # Reference to main event loop for thread-safe async operations
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Progress lives in the module-level `_DOWNLOAD_PROGRESS`, not here —
        # see the note on that dict. An instance attribute is invisible to the
        # next request, which gets a different ModelService.

    @property
    def _loading_model_id(self) -> Optional[int]:
        """The model whose load is in progress, in ANY service instance."""
        return _LOAD_SLOT["model_id"]

    @_loading_model_id.setter
    def _loading_model_id(self, model_id: Optional[int]) -> None:
        _LOAD_SLOT["model_id"] = model_id

    def _release_load_slot(self, model_id: int) -> None:
        """Release the slot only if `model_id` still holds it."""
        if _LOAD_SLOT["model_id"] == model_id:
            _LOAD_SLOT["model_id"] = None

    #: Whether a load is in progress in any instance (addendum §8; readable from the class).
    is_loading = _LoadInProgress()

    # =========================================================================
    # Attachable weights (Features 31 and 32; addendum §8)
    #
    # The weight service and the barrier are process-wide singletons. Every guard below is
    # synchronous, so a caller places it with no `await` before the claim it guards, beside
    # `_refuse_if_leased`. Order in a load: torn first (D-01; before the row-status check, so the
    # torn resident row names its recovery, RX F2), then status, lease, attachment.
    # =========================================================================

    @staticmethod
    def _refuse_if_attached(operation: str) -> None:
        """Refuse a load, unload or swap while a trainer is attached, active or stale
        (FR-31.5.2, FR-31.5.3): its handles point at the resident tensors."""
        from millm.services.weight_attachment import get_weight_service

        get_weight_service().refuse_if_attached(operation)

    @staticmethod
    def _refuse_if_torn(operation: str) -> None:
        """D-01: while the weights are `torn`, a Feature-1 load and the `/v1` auto-load are
        refused; recovery is `POST /api/weights/load`. Unload is allowed (nothing resident is
        `clean` by construction)."""
        from millm.services.weight_barrier import get_weight_barrier

        if get_weight_barrier().snapshot().weights_state == "torn":
            raise WeightsTornError(
                _TORN_RECOVERY,
                details={"operation": operation, "weights_state": "torn",
                         "recovery": "POST /api/weights/load"},
            )

    def _refuse_if_unsaved_commits(self, operation: str, model: Model) -> None:
        """D-45 (review R2-A F-A): the `/v1` auto-load never swaps out resident weights that
        carry commits no checkpoint records. Once a trainer detaches, `_refuse_if_attached` no
        longer applies, and a request naming another row would otherwise unload the trained
        weights and lose every commit since the last save.

        Refused until a save records them (it moves `commits_at_identity` to the counter) or an
        explicit operator load replaces them (`POST /api/weights/load`, the management load —
        neither calls this). The count is `resident_identity.unsaved_commits`, from the
        barrier's counter and the resident identity: no state of its own.
        """
        from millm.services import resident_identity
        from millm.services.weight_barrier import get_weight_barrier

        loaded = self.loader.state.current
        if loaded is None:
            return
        steps = get_weight_barrier().snapshot().steps_since_load
        unsaved = resident_identity.unsaved_commits(steps, loaded)
        if unsaved <= 0:
            return
        resident = resident_identity.served_name(loaded) or loaded.model_name
        recovery = [
            "POST /api/weights/save (as the attachment holder: re-attach first)",
            f'POST /api/weights/load {{"model_id": {model.id}}}',
            f"POST /api/models/{model.id}/load",
        ]
        logger.info("auto_load_refused_unsaved_commits", resident=resident,
                    resident_model_id=loaded.model_id, unsaved_commits=unsaved,
                    requested_model_id=model.id, operation=operation)
        raise WeightsUnsavedError(
            f"The resident model '{resident}' carries {unsaved} commit(s) not saved to a "
            f"checkpoint; loading '{model.name}' on demand would discard them, so it is "
            "refused. Save them first (POST /api/weights/save, as the attachment holder), or "
            "replace them deliberately with POST /api/weights/load "
            f'{{"model_id": {model.id}}} or POST /api/models/{model.id}/load.',
            details={
                "operation": operation,
                "resident": resident,
                "resident_model_id": loaded.model_id,
                "unsaved_commits": unsaved,
                "requested_model": model.name,
                "requested_model_id": model.id,
                "recovery": recovery,
            },
        )

    @staticmethod
    def _require_barrier_slot(operation: str) -> None:
        """`under_barrier=True` is legal only while the barrier holds the admission slot
        (031 G3). Raised, never `assert`ed: an assertion vanishes under `python -O`, and a
        drain skipped outside the barrier moves weights under a running request."""
        from millm.services.weight_barrier import get_weight_barrier

        if not get_weight_barrier().holds_slot():
            raise RuntimeError(
                f"{operation}(under_barrier=True) called while the weight barrier does not hold "
                "the admission slot"
            )

    @staticmethod
    def _refuse_weights_origin(model: Model, operation: str) -> None:
        """A weights-origin row loads only through `POST /api/weights/load`, which verifies its
        files against the manifest (addendum §8; 032 FR-32.6.4)."""
        # Only a non-empty string counts (the column allows nothing else); a stand-in row
        # without the attribute is a plain row.
        origin = getattr(model, "weights_origin", None)
        if isinstance(origin, str) and origin:
            raise LoadThroughWeightsApiError(
                f"Model '{model.name}' holds weights from a checkpoint and is loaded only "
                "through POST /api/weights/load, where its files are verified.",
                details={"model_id": model.id, "weights_origin": origin,
                         "operation": operation},
            )

    @staticmethod
    def _parameter_set_replaced(reason: str) -> None:
        """Tell the barrier a Feature-1 load or unload replaced the parameter set: `clean`,
        steps 0, a torn set's leftover hold released (031 D17, review D10). A no-op inside an
        `internal_write`, which makes its own transition (031 review X5)."""
        from millm.services.weight_barrier import get_weight_barrier

        get_weight_barrier().on_parameter_set_replaced(reason)

    @staticmethod
    def _attachable_load(quantization: str, gguf_file: Optional[str]) -> bool:
        """D-03: an unquantized 16-bit transformers load is allocated attachable when at least
        one `trainer` token is configured; otherwise not, so default deployments are unchanged.

        D-37: FP32 is excluded. Attach requires fp16/bf16 parameters, so an FP32 row can never
        attach and spending the attachable pool (expandable segments off) on it buys nothing.
        """
        if gguf_file is not None or quantization.upper() != "FP16":
            return False
        from millm.core.weights_auth import current_token_set

        return any(entry.role == "trainer" for entry in current_token_set().entries)

    # =========================================================================
    # Model lease (Feature 29)
    #
    # The registry is process-wide (`get_lease_registry`), for the reason `_LOAD_SLOT` is:
    # every request builds a new ModelService. The RULES are here — residency, a load in
    # progress, the lift — so a route cannot forget a check it never makes (FR-29.2.2).
    # =========================================================================

    @property
    def _leases(self) -> LeaseRegistry:
        return get_lease_registry()

    @staticmethod
    def _validate_lease_ttl(ttl_seconds: Any) -> int:
        """`ttl_seconds` omitted → the default; else an integer in 1..max, never clamped."""
        from millm.core.config import settings

        maximum = settings.LEASE_MAX_TTL_SECONDS
        if ttl_seconds is None:
            return settings.LEASE_DEFAULT_TTL_SECONDS
        # bool is an int subclass: `true` is not a TTL.
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise InvalidLeaseRequestError(
                f"ttl_seconds must be an integer from 1 to {maximum} seconds, "
                f"got {ttl_seconds!r}.",
                details={"param": "ttl_seconds", "min": 1, "max": maximum},
            )
        if not 1 <= ttl_seconds <= maximum:
            raise InvalidLeaseRequestError(
                f"ttl_seconds must be from 1 to {maximum} seconds, got {ttl_seconds}; "
                "it is refused, never clamped.",
                details={"param": "ttl_seconds", "min": 1, "max": maximum},
            )
        return ttl_seconds

    @staticmethod
    def _validate_lease_text(name: str, value: Any, limit: int) -> str:
        """A required, stripped, non-empty string of at most `limit` characters."""
        if not isinstance(value, str) or not value.strip():
            raise InvalidLeaseRequestError(
                f"{name} is required: a non-empty string of at most {limit} characters.",
                details={"param": name, "max_chars": limit},
            )
        stripped = value.strip()
        if len(stripped) > limit:
            raise InvalidLeaseRequestError(
                f"{name} is {len(stripped)} characters; the limit is {limit}.",
                details={"param": name, "max_chars": limit},
            )
        return stripped

    async def acquire_lease(
        self, model_id: int, holder: Any, reason: Any, ttl_seconds: Any = None
    ) -> LeaseGrant:
        """Lease the resident model for `holder` (FR-29.1). No approval step (P-05).

        Order: inputs (400) → the model exists (404) → no load or unload running (busy) →
        the model is the resident, LOADED one (409 MODEL_NOT_RESIDENT, T-85) → no other live
        lease (409 MODEL_LEASED, X-08). `get_model` is the only await and it comes FIRST:
        nothing awaits between the residency checks and the grant, so a load cannot claim the
        slot in between.
        """
        from millm.core.config import settings

        holder_s = self._validate_lease_text("holder", holder, settings.LEASE_HOLDER_MAX_CHARS)
        reason_s = self._validate_lease_text("reason", reason, settings.LEASE_REASON_MAX_CHARS)
        ttl = self._validate_lease_ttl(ttl_seconds)
        model = await self.get_model(model_id)

        if self._loading_model_id is not None:
            raise ModelBusyError(
                f"Model {self._loading_model_id} is being loaded; take the lease once it is "
                "resident.",
                details={"loading_model_id": self._loading_model_id},
            )
        if self.loader.is_unloading is True:
            raise ModelBusyError(
                f"Model {self.loader.loaded_model_id} is being unloaded; no lease can be taken "
                "on it.",
                details={"unloading_model_id": self.loader.loaded_model_id, "unloading": True},
            )
        resident_id = self.loader.loaded_model_id
        if resident_id != model_id or model.status != ModelStatus.LOADED:
            resident_name = self.loader.model_name if resident_id is not None else None
            raise ModelNotResidentError(
                f"Model {model_id} is not resident; only the resident model can be leased. "
                f"Resident model: {resident_name or 'none'}.",
                details={
                    "model_id": model_id,
                    "resident_model_id": resident_id,
                    "resident_model_name": resident_name,
                },
            )
        return self._leases.grant(model_id, model.name, holder_s, reason_s, ttl)

    async def renew_lease(
        self, model_id: int, lease_id: Optional[str], ttl_seconds: Any = None
    ) -> LeaseRecord:
        """Renew by ID; the new expiry is now plus the TTL (FR-29.1.7)."""
        ttl = self._validate_lease_ttl(ttl_seconds)
        self._require_lease_header(lease_id)
        return self._leases.renew(model_id, lease_id, ttl)

    async def release_lease(self, model_id: int, lease_id: Optional[str]) -> EndedLease:
        """Release by ID; the lease ends at once."""
        self._require_lease_header(lease_id)
        return self._leases.release(model_id, lease_id)

    async def get_lease(
        self, model_id: int
    ) -> tuple[Optional[LeaseRecord], Optional[EndedLease]]:
        """The live lease on the model (or None) and the last one that ended. 404 if no model."""
        await self.get_model(model_id)
        return self._leases.current(model_id), self._leases.last_ended(model_id)

    def resolve_lease(self, lease_id: Optional[str]) -> Optional[LeaseRecord]:
        """The live lease this ID proves, on any model (Feature 26: T-64, FR-26.7.7)."""
        return self._leases.resolve(lease_id)

    @staticmethod
    def _require_lease_header(lease_id: Optional[str]) -> None:
        if not lease_id or not lease_id.strip():
            raise InvalidLeaseRequestError(
                "The lease ID is required in the X-miLLM-Lease header.",
                details={"param": "X-miLLM-Lease"},
            )

    def _refuse_if_leased(
        self, operation: str, target_model_id: int, lease_id: Optional[str]
    ) -> None:
        """Refuse a load, unload or swap while ANOTHER holder leases the resident model.

        THE one enforcement function (029 FTDD §7.2), called from `load_model`,
        `unload_model` and `load_model_and_wait`. It reads the RESIDENT model's lease, not
        the target's: the lease protects whatever is resident, whatever the request asks to
        load. Synchronous on purpose — a caller places it with no `await` before the claim it
        guards. A matching lease ID lifts only this refusal; `locked`, the load slot and
        placement still apply (FR-29.3.2).
        """
        refuse_if_leased(
            self._leases,
            operation=operation,
            resident_model_id=self.loader.loaded_model_id,
            target_model_id=target_model_id,
            lease_id=lease_id,
        )

    def _run_async_from_thread(self, coro: Any) -> Any:
        """
        Run an async coroutine from a background thread.

        Uses asyncio.run_coroutine_threadsafe() to safely schedule the coroutine
        on the main event loop and wait for its result.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        if self._main_loop is None:
            raise RuntimeError("Main event loop not set")
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        return future.result(timeout=30.0)  # 30 second timeout for DB operations

    def get_download_progress(self, model_id: int) -> int | None:
        """
        Get the current download progress for a model.

        Args:
            model_id: The model's database ID

        Returns:
            Progress percentage (0-100) if downloading, None otherwise.
        """
        return _DOWNLOAD_PROGRESS.get(model_id)

    async def list_models(self) -> list[Model]:
        """
        Get all models from the database.

        Returns:
            List of all models ordered by creation date descending.
        """
        return await self.repository.get_all()

    async def get_model(self, model_id: int) -> Model:
        """
        Get a single model by ID.

        Args:
            model_id: The model's database ID

        Returns:
            The model if found

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(
                f"Model with ID {model_id} not found",
                details={"model_id": model_id},
            )
        return model

    async def preview_model(self, request: ModelPreviewRequest) -> dict[str, Any]:
        """
        Get model info from HuggingFace without downloading.

        Args:
            request: Preview request with repo_id and optional token

        Returns:
            Dict with model metadata (name, params, architecture, etc.)
        """
        return self.downloader.get_model_info(
            repo_id=request.repo_id,
            token=request.hf_token,
            revision=request.revision,
        )

    # =========================================================================
    # Download Operations
    # =========================================================================

    async def _unique_served_name(self, name: str, request: ModelDownloadRequest) -> str:
        """The name a new registration is served under: one row per name (2026-10-18, D56).

        A non-GGUF registration at any precision but the native one is tagged like a GGUF
        quant, `repo:Q8` / `repo:Q4` / `repo:FP32`: two precisions of one repository are
        different models. Untagged, `jevify-gemma4-e4b` named both the FP16 row 13 and the Q8
        row 7 — `/v1/models` listed it once, the fingerprint named the other row's precision,
        an auto-load picked one by heap order, and miDataworks could not pin either.

        The native (FP16) registration keeps the bare name, which is what every existing client
        sends. If a row registered before this rule already holds it (a legacy untagged Q8),
        the new row is tagged instead (`repo:FP16`) — the existing name never moves under its
        clients. A name the caller TYPED (`custom_name`) is never altered: a clash is refused
        409 `MODEL_NAME_TAKEN`, as is a tagged name that is still taken.
        """
        from millm.core.errors import ModelNameTakenError

        tagged = bool(request.gguf_label)
        if not request.custom_name and not tagged and request.quantization != QuantizationType.FP16:
            name = f"{name}:{request.quantization.value}"
            tagged = True
        holder = await self.repository.find_by_exact_name(name)
        if holder is not None and not request.custom_name and not tagged:
            name = f"{name}:{request.quantization.value}"
            holder = await self.repository.find_by_exact_name(name)
        if holder is not None:
            raise ModelNameTakenError(
                f"The model name '{name}' is already used by model {holder.id}. "
                "Give this registration its own name (custom_name).",
                details={"name": name, "model_id": holder.id},
            )
        return name

    async def download_model(self, request: ModelDownloadRequest) -> Model:
        """
        Start downloading a model from HuggingFace.

        Creates a database record and starts the download in a background thread.
        Progress updates are sent via WebSocket.

        Args:
            request: Download request with repo_id, quantization, etc.

        Returns:
            The created model record (status will be DOWNLOADING)

        Raises:
            ModelAlreadyExistsError: If model with same repo_id/quantization exists
        """
        # The coarse level is DERIVED from the GGUF label, never taken from the
        # request. It is a function of the label, so letting a caller send both
        # lets them disagree — and every GGUF download would otherwise be filed
        # under whatever the quantization dropdown defaulted to.
        if request.gguf_label:
            derived = QuantizationType(coarse_quantization(request.gguf_label))
            if derived != request.quantization:
                logger.info(
                    "gguf_quantization_derived",
                    gguf_label=request.gguf_label,
                    requested=request.quantization.value,
                    derived=derived.value,
                )
            request = request.model_copy(update={"quantization": derived})

        # Check for existing model with same repo_id and quantization
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            existing = await self.repository.find_by_repo_quantization(
                repo_id=request.repo_id,
                quantization=request.quantization,
                gguf_label=request.gguf_label or "",
            )
            if existing:
                described = request.gguf_label or request.quantization.value
                raise ModelAlreadyExistsError(
                    f"Model {request.repo_id} with {described} quantization already exists",
                    details={
                        "repo_id": request.repo_id,
                        "quantization": request.quantization.value,
                        "gguf_label": request.gguf_label or "",
                        "existing_model_id": existing.id,
                    },
                )

        # Get model info for metadata
        model_info = None
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            try:
                model_info = self.downloader.get_model_info(
                    repo_id=request.repo_id,
                    token=request.hf_token,
                )
            except Exception as e:
                logger.warning(
                    "failed_to_get_model_info",
                    repo_id=request.repo_id,
                    error=str(e),
                )

        # Generate cache path or use local path
        cache_path = ""
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            safe_name = request.repo_id.replace("/", "--") + f"--{request.quantization.value}"
            # Must match ModelDownloader._get_local_dir, including the variant.
            # Q4_K_M, Q4_K_S and Q4_0 are all "Q4"; without the suffix they share
            # one directory and deleting any one destroys the others.
            if request.gguf_label:
                safe_name += f"--{_safe_variant(request.gguf_label)}"
            cache_path = f"huggingface/{safe_name}"
        elif request.source == ModelSource.LOCAL and request.local_path:
            # For local models, use the local path directly as cache_path
            cache_path = request.local_path

        # Extract name from repo_id or local_path
        name = ""
        if request.custom_name:
            name = request.custom_name
        elif request.repo_id:
            name = request.repo_id.split("/")[-1]
            # `repo:QUANT`, the convention Ollama already taught everyone.
            #
            # Several quantizations of one repository can coexist since the
            # uniqueness constraint was widened, and they are DIFFERENT MODELS —
            # different sizes, different quality. Leaving them to share a name
            # made `find_by_name` raise "Multiple rows were found" and every
            # OpenAI request naming that model return 500, with the model
            # loaded and serving perfectly the whole time.
            #
            # It also makes both quantizations separately selectable in an
            # OpenAI client: previously one shadowed the other in /v1/models.
            if request.gguf_label:
                name = f"{name}:{request.gguf_label}"
        elif request.local_path:
            name = request.local_path.rstrip("/").split("/")[-1]
        name = await self._unique_served_name(name, request)

        # The model's TASK, for the generation endpoints' refusal of non-generative models. A Hub
        # row carries the repo's pipeline tag; a local folder has no Hub tag, so a classifier
        # registered from disk was not refused for chat — it loaded, reached generate() and
        # failed with a 500. Its config.json names the head; the loader's task-head table maps
        # that to the same tag a Hub row would carry. (Rows registered before this are covered
        # at check time, `openai.errors.non_generative_task`.)
        architecture = model_info.get("architecture") if model_info else None
        if request.source == ModelSource.LOCAL and request.local_path:
            architecture = task_of_model_dir(request.local_path) or architecture

        # Estimate memory requirement
        estimated_memory = 0
        if model_info and model_info.get("params"):
            estimated_memory = estimate_memory_mb(
                model_info["params"],
                request.quantization.value,
            )

        # Create database record in DOWNLOADING state
        model = await self.repository.create(
            name=name,
            source=request.source,
            repo_id=request.repo_id,
            local_path=request.local_path,
            quantization=request.quantization,
            cache_path=cache_path,
            trust_remote_code=request.trust_remote_code,
            status=ModelStatus.DOWNLOADING,
            params=model_info.get("params") if model_info else None,
            architecture=architecture,
            estimated_memory_mb=estimated_memory if estimated_memory > 0 else None,
            # '' not None — the column is part of uq_repo_quantization, and
            # Postgres treats NULLs as distinct, which would defeat it.
            gguf_label=request.gguf_label or "",
            gguf_files=list(request.gguf_files) if request.gguf_files else None,
            revision=request.revision,
        )

        # For local models, mark as ready immediately (no download needed)
        if request.source == ModelSource.LOCAL and request.local_path:
            import os
            if not os.path.isdir(request.local_path):
                from millm.core.errors import InvalidLocalPathError
                raise InvalidLocalPathError(
                    f"Local path does not exist or is not a directory: {request.local_path}",
                    details={"path": request.local_path},
                )

            # Calculate disk size
            total_size = sum(
                f.stat().st_size
                for f in Path(request.local_path).rglob("*")
                if f.is_file()
            ) if Path(request.local_path).exists() else 0

            model = await self.repository.update(
                model.id,
                status=ModelStatus.READY,
                cache_path=request.local_path,
                disk_size_mb=int(total_size / (1024 * 1024)),
            )
            logger.info(
                "local_model_registered",
                model_id=model.id,
                local_path=request.local_path,
            )
            return model

        logger.info(
            "download_started",
            model_id=model.id,
            repo_id=request.repo_id,
            quantization=request.quantization.value,
        )

        # Start background download
        # Store the main loop for thread-safe async operations
        loop = asyncio.get_running_loop()
        self._main_loop = loop
        future = loop.run_in_executor(
            self._executor,
            self._download_worker,
            model.id,
            request,
        )
        self._active_downloads[model.id] = future

        return model

    def _download_worker(
        self,
        model_id: int,
        request: ModelDownloadRequest,
    ) -> str:
        """
        Background worker for downloading models.

        Runs in thread pool. Updates database and emits events upon completion/failure.
        Implements retry logic with exponential backoff for transient failures.
        """
        last_error: Optional[Exception] = None

        # Initialize progress tracking
        _DOWNLOAD_PROGRESS[model_id] = 0

        # Create progress callback that emits WebSocket events
        def on_progress(
            pct: float, downloaded: int, total: int, speed_bps: float = 0.0
        ) -> None:
            progress = int(pct)
            _DOWNLOAD_PROGRESS[model_id] = progress

            # Emit WebSocket progress event
            if self.emitter and self._main_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.emitter.emit_download_progress(
                            model_id=model_id,
                            progress=progress,
                            downloaded_bytes=downloaded,
                            total_bytes=total if total > 0 else None,
                            speed_bps=int(speed_bps) if speed_bps > 0 else None,
                        ),
                        self._main_loop,
                    )
                except Exception:
                    pass  # Don't let emit errors break downloading

            logger.debug(
                "download_progress_update",
                model_id=model_id,
                progress=progress,
                downloaded_bytes=downloaded,
                total_bytes=total,
                speed_bps=int(speed_bps),
            )

        try:
            for attempt in range(MAX_DOWNLOAD_RETRIES):
                try:
                    # Check if cancelled before starting
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(
                            request.repo_id, request.quantization.value, request.gguf_label
                        )
                        raise DownloadCancelledError("Download was cancelled")

                    logger.info(
                        "download_attempt",
                        model_id=model_id,
                        attempt=attempt + 1,
                        max_attempts=MAX_DOWNLOAD_RETRIES,
                    )

                    # Perform download with progress callback
                    cache_path = self.downloader.download(
                        repo_id=request.repo_id,
                        quantization=request.quantization.value,
                        progress_callback=on_progress,
                        token=request.hf_token,
                        trust_remote_code=request.trust_remote_code,
                        file_paths=request.gguf_files,
                        variant=request.gguf_label,
                        revision=request.revision,
                    )

                    # Check if cancelled after download
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(
                            request.repo_id,
                            request.quantization.value,
                            request.gguf_label,
                        )
                        raise DownloadCancelledError("Download was cancelled")

                    # Mark progress as 100%
                    _DOWNLOAD_PROGRESS[model_id] = 100

                    # Get disk size
                    disk_size_bytes = self.downloader.get_cache_size(
                        request.repo_id,
                        request.quantization.value,
                        request.gguf_label,
                    )
                    disk_size_mb = disk_size_bytes // (1024 * 1024)

                    # Update database (thread-safe async call)
                    self._run_async_from_thread(
                        self._update_download_complete(
                            model_id=model_id,
                            cache_path=cache_path,
                            disk_size_mb=disk_size_mb,
                        )
                    )

                    logger.info(
                        "download_complete",
                        model_id=model_id,
                        cache_path=cache_path,
                        disk_size_mb=disk_size_mb,
                    )

                    return cache_path

                except DownloadCancelledError:
                    self._run_async_from_thread(self._update_download_cancelled(model_id))
                    raise

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "download_attempt_failed",
                        model_id=model_id,
                        attempt=attempt + 1,
                        error=str(e),
                        retrying=attempt < MAX_DOWNLOAD_RETRIES - 1,
                    )

                    # Check if error is retryable (network errors, timeouts)
                    if not self._is_retryable_error(e):
                        break

                    # Check if cancelled while waiting to retry
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(
                            request.repo_id, request.quantization.value, request.gguf_label
                        )
                        raise DownloadCancelledError("Download was cancelled")

                    # Calculate backoff delay
                    if attempt < MAX_DOWNLOAD_RETRIES - 1:
                        delay = min(
                            RETRY_BASE_DELAY * (2 ** attempt),
                            RETRY_MAX_DELAY,
                        )
                        logger.info(
                            "download_retry_delay",
                            model_id=model_id,
                            delay_seconds=delay,
                        )
                        time.sleep(delay)

                        # Clean up partial download before retry
                        self._cleanup_partial_download(
                            request.repo_id, request.quantization.value, request.gguf_label
                        )

            # All retries exhausted - clean up and report failure
            self._cleanup_partial_download(
                request.repo_id, request.quantization.value, request.gguf_label
            )

            if last_error:
                logger.error(
                    "download_failed",
                    model_id=model_id,
                    error=str(last_error),
                    attempts=MAX_DOWNLOAD_RETRIES,
                )
                self._run_async_from_thread(
                    self._update_download_error(
                        model_id=model_id,
                        error_code=getattr(last_error, "code", "DOWNLOAD_FAILED"),
                        error_message=str(last_error),
                    )
                )
                raise last_error

            return ""

        finally:
            # Clean up download tracking
            self._active_downloads.pop(model_id, None)
            self._cancelled_downloads.discard(model_id)
            _DOWNLOAD_PROGRESS.pop(model_id, None)

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable (transient network issues).

        Args:
            error: The exception to check

        Returns:
            True if the error is transient and should be retried
        """
        # Non-retryable errors
        non_retryable_codes = {
            "REPO_NOT_FOUND",
            "GATED_MODEL_NO_TOKEN",
            "INVALID_HF_TOKEN",
            "INVALID_LOCAL_PATH",
        }

        error_code = getattr(error, "code", "")
        if error_code in non_retryable_codes:
            return False

        # Check for common retryable error types
        error_name = type(error).__name__
        retryable_names = {
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "IOError",
            "HTTPError",
            "RequestException",
        }

        return error_name in retryable_names or "timeout" in str(error).lower()

    def _cleanup_partial_download(
        self, repo_id: str, quantization: str, variant: Optional[str] = None
    ) -> None:
        """
        Clean up partially downloaded files.

        Args:
            repo_id: HuggingFace repository ID
            quantization: Quantization type
            variant: The quant label, when several downloads share a coarse
                quantization. This must reach delete_cached_model: without it
                a cancelled Q4_K_M download deletes whatever "Q4" resolves to,
                which is a DIFFERENT, complete model.
        """
        try:
            self.downloader.delete_cached_model(repo_id, quantization, variant)
            logger.info(
                "partial_download_cleaned",
                repo_id=repo_id,
                quantization=quantization,
                variant=variant,
            )
        except Exception as e:
            logger.warning(
                "partial_download_cleanup_failed",
                repo_id=repo_id,
                quantization=quantization,
                error=str(e),
            )

    async def _update_download_complete(
        self,
        model_id: int,
        cache_path: str,
        disk_size_mb: int,
    ) -> None:
        """Update database and emit event when download completes."""
        await self.repository.update(
            model_id,
            cache_path=cache_path,
            disk_size_mb=disk_size_mb,
            status=ModelStatus.READY,
        )

        if self.emitter:
            await self.emitter.emit_download_complete(
                model_id=model_id,
                local_path=cache_path,
            )

    async def _update_download_cancelled(self, model_id: int) -> None:
        """Update database and emit event when download is cancelled."""
        await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message="Download cancelled by user",
        )

        if self.emitter:
            await self.emitter.emit_download_error(
                model_id=model_id,
                error_code="DOWNLOAD_CANCELLED",
                error_message="Download cancelled by user",
            )

    async def _update_download_error(
        self,
        model_id: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """Update database and emit event when download fails."""
        await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message=error_message,
        )

        if self.emitter:
            await self.emitter.emit_download_error(
                model_id=model_id,
                error_code=error_code,
                error_message=error_message,
            )

    async def cancel_download(self, model_id: int) -> Model:
        """
        Cancel an in-progress download.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.get_model(model_id)

        if model.status != ModelStatus.DOWNLOADING:
            return model

        self._cancelled_downloads.add(model_id)

        future = self._active_downloads.get(model_id)
        if future and not future.done():
            future.cancel()

        model = await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message="Download cancelled by user",
        )

        logger.info("download_cancelled", model_id=model_id)

        return model

    # =========================================================================
    # Load/Unload Operations
    # =========================================================================

    async def load_model(
        self,
        model_id: int,
        gpu: GpuRequest = None,
        *,
        lease_id: Optional[str] = None,
        under_barrier: bool = False,
        allow_weights_origin: bool = False,
    ) -> Model:
        """
        Load a model into GPU memory.

        If another model is loaded, it will be automatically unloaded first.
        Progress updates are sent via WebSocket.

        Args:
            model_id: The model's database ID
            gpu: None or "auto" (the most-free card that fits), a CUDA index,
                or a GPU UUID. A named card that does not exist, or that cannot
                hold the model even with the resident model's memory given
                back, is refused here — before anything is unloaded.
            under_barrier: the caller (032's weights load) runs inside the weight
                barrier's `internal_write`, which holds the admission slot: the
                resident model's unload skips its drain, and the torn refusal does
                not apply. Legal only while the barrier holds the slot.
            allow_weights_origin: only the 032 weights service passes it; every
                other caller is refused a weights-origin row.

        Returns:
            The updated model record (status will be LOADED)

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelBusyError: If another load is in progress
        """
        if under_barrier:
            self._require_barrier_slot("load_model")
        model = await self.get_model(model_id)
        if not allow_weights_origin:
            self._refuse_weights_origin(model, "load")
        # D-01, BEFORE the status check (RX F2): the torn resident row is LOADED, and answering
        # MODEL_ALREADY_LOADED ("ready to use") for a set that is refused on every request would
        # hide the recovery. The torn refusal names it.
        if not under_barrier:
            self._refuse_if_torn("model_load")

        # Check if model is ready to load
        self._refuse_load_status(model)

        # One load at a time, and the slot is CLAIMED here, with no await
        # between the check and the claim. It used to be set only after the
        # resident model's unload was awaited (which drains pending inference
        # for up to five seconds), so a second load arriving in that window —
        # a double-clicked Switch, or two OpenAI requests naming different
        # models — passed this same check. Both were submitted to the
        # two-worker executor, ran at once, and the second replaced the first
        # in LoadedModelState without unloading it.
        # Feature 29: a foreign lease on the resident model refuses the swap BEFORE the slot
        # is claimed, with no await between this check and the claim below, so a refused
        # load claims nothing and moves no row (FR-29.2.6).
        self._refuse_if_leased("load", model_id, lease_id)
        # Feature 31: an attached trainer's handles point at the resident tensors (FR-31.5.2).
        self._refuse_if_attached("model_load")
        self._refuse_if_load_slot_taken()
        self._loading_model_id = model_id
        # Whether this request has set the row to LOADING, so a failure before
        # the background load is submitted can put it back (see the except).
        marked_loading = False

        try:
            self._refuse_if_unloading(model_id)
            # The card, checked NOW: before any state below changes, and above
            # all before the resident model is unloaded to make room. The load
            # itself runs in the background, so a refusal found only there
            # arrived as an ERROR status after the served model was gone.
            wanted = self._parse_gpu(gpu)

            def _precheck() -> None:
                self._precheck_placement(model, wanted)

            # IN A WORKER THREAD. The pre-check reads the cards through
            # nvidia-smi, which can take up to its five-second timeout; called
            # inline it stalled the event loop, and every request with it, on
            # each load. Review round 3, 2026-09-14.
            await asyncio.to_thread(_precheck)

            if model.status == ModelStatus.ERROR:
                # Allow retry from error state - reset to ready first
                logger.info(
                    "retrying_errored_model",
                    model_id=model_id,
                    previous_error=model.error_message,
                )
                model = await self.repository.update_status(
                    model_id, status=ModelStatus.READY, error_message=None
                )

            # Unload any currently loaded model
            if self.loader.is_loaded:
                current_model_id = self.loader.loaded_model_id
                logger.info(
                    "auto_unloading_model",
                    current_model_id=current_model_id,
                    new_model_id=model_id,
                )
                # The same lease ID: the holder's own swap must not be refused by its lease.
                await self.unload_model(
                    current_model_id, lease_id=lease_id, under_barrier=under_barrier
                )

            # Update status to LOADING
            model = await self.repository.update_status(model_id, status=ModelStatus.LOADING)
            marked_loading = True

            logger.info(
                "load_started",
                model_id=model_id,
                cache_path=model.cache_path,
                quantization=model.quantization.value,
            )

            # Emit progress event
            if self.emitter:
                await self.emitter.emit_load_progress(
                    model_id=model_id,
                    stage="initializing",
                    progress=0,
                )

            # Start background load
            # Store the main loop for thread-safe async operations
            loop = asyncio.get_running_loop()
            self._main_loop = loop
            loop.run_in_executor(
                self._executor,
                # A load started inside an internal write: the 032 service owns the barrier's
                # transition, even if this worker finishes after that write ended (R2-B O2).
                functools.partial(self._load_worker, under_barrier=under_barrier),
                model_id,
                model.name,
                model.cache_path,
                model.quantization.value,
                model.estimated_memory_mb or 0,
                model.trust_remote_code,
                # The chosen GGUF file, when this row is one. Its presence is
                # what routes the load to llama.cpp instead of transformers.
                (model.gguf_files or [None])[0],
                wanted,
            )
        except BaseException as exc:
            # Nothing was submitted, so no worker's `finally` will release the
            # slot. Without this a refused or failed load left the service
            # refusing every later load as busy.
            self._release_load_slot(model_id)
            if marked_loading:
                # ...and no worker will move the row off LOADING, which this
                # method refuses as busy: that model could not be loaded again
                # until miLLM restarted. A cancelled start is not a failure.
                # Review round 3, 2026-09-14.
                cancelled = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                try:
                    await self.repository.update_status(
                        model_id,
                        status=ModelStatus.READY if cancelled else ModelStatus.ERROR,
                        error_message=None if cancelled else f"Load did not start: {exc}",
                    )
                except Exception as restore_error:  # noqa: BLE001 - never mask the original error
                    logger.warning(
                        "load_status_restore_failed",
                        model_id=model_id,
                        error=str(restore_error),
                    )
            raise

        return model

    @staticmethod
    def _refuse_load_status(model: Model, *, reloading_resident: bool = False) -> None:
        """A row still downloading or loading is busy; a LOADED row is already loaded, unless
        the caller is reloading the resident (it unloads it first: `precheck_load`)."""
        if model.status == ModelStatus.DOWNLOADING:
            raise ModelBusyError(
                f"Model {model.id} is still downloading",
                details={"model_id": model.id, "status": model.status.value},
            )
        if model.status == ModelStatus.LOADING:
            raise ModelBusyError(
                f"Model {model.id} is already being loaded",
                details={"model_id": model.id},
            )
        if model.status == ModelStatus.LOADED and not reloading_resident:
            raise ModelAlreadyLoadedError(
                f"Model {model.id} is already loaded",
                details={"model_id": model.id},
            )

    def _refuse_if_load_slot_taken(self) -> None:
        """One load at a time, across every service instance (`_LOAD_SLOT`)."""
        if self._loading_model_id is not None:
            raise ModelBusyError(
                f"Another model ({self._loading_model_id}) is currently being loaded",
                details={"loading_model_id": self._loading_model_id},
            )

    def _refuse_if_unloading(self, model_id: int) -> None:
        """The resident model is already being unloaded: a second unload would race the first.
        Retry once it finishes. Hardware acceptance, 2026-09-14, item 11."""
        if self.loader.is_unloading is True:
            raise ModelBusyError(
                f"Model {self.loader.loaded_model_id} is being unloaded; retry once the "
                "unload finishes",
                details={
                    "loading_model_id": model_id,
                    "unloading_model_id": self.loader.loaded_model_id,
                },
            )

    @staticmethod
    def _parse_gpu(gpu: GpuRequest) -> Any:
        try:
            return parse_gpu_request(gpu)
        except ValueError as e:
            raise GpuNotFoundError(str(e), details={"requested": gpu}) from e

    async def precheck_load(
        self,
        model_id: int,
        gpu: GpuRequest = None,
        *,
        lease_id: Optional[str] = None,
        under_barrier: bool = False,
        allow_weights_origin: bool = False,
    ) -> Model:
        """Every refusal `load_model` can make before it changes anything, made WITHOUT
        changing anything: no slot claimed, no row moved, nothing unloaded.

        For a caller that must not START a replacement it cannot finish: 032's full reload
        marks the weights' first write before it calls `load_model`, so a refusal found only
        inside `load_model` (the placement or memory pre-check, a row still loading, a load
        in progress) left the weights `torn` while the resident model was never touched
        (W2-G tracked debt). It runs this first, before `mark_first_write`.

        `load_model` makes every check again and its answer is authoritative (free memory can
        change in between); this only refuses early what it would refuse. Naming the
        RESIDENT row is allowed: the caller reloads it, unloading it first, so its LOADED
        status is not a refusal and the placement check already counts its memory as free.

        Raises:
            Every refusal `load_model` raises before its background load is submitted.
        """
        if under_barrier:
            self._require_barrier_slot("precheck_load")
        model = await self.get_model(model_id)
        if not allow_weights_origin:
            self._refuse_weights_origin(model, "load")
        if not under_barrier:
            self._refuse_if_torn("model_load")  # before the status check, as `load_model`
        self._refuse_load_status(
            model, reloading_resident=self.loader.loaded_model_id == model_id
        )
        self._refuse_if_leased("load", model_id, lease_id)
        self._refuse_if_attached("model_load")
        self._refuse_if_load_slot_taken()
        self._refuse_if_unloading(model_id)
        wanted = self._parse_gpu(gpu)
        await asyncio.to_thread(self._precheck_placement, model, wanted)
        return model

    def _precheck_placement(self, model: Model, wanted: Any) -> None:
        """Refuse, before unloading anything, a load that cannot be placed.

        Judged against what each card WILL have once the resident model is
        gone: live free memory plus that model's recorded usage on the card.
        Without the add-back, switching from a model on the 3090 to a bigger
        one on the same card would be refused while it fits.

        The background load decides again after the unload, against live
        memory, and that decision is authoritative — free memory can change in
        between. This check only refuses early what would be refused there.

        Raises:
            GpuNotFoundError: the named card is not visible.
            InsufficientMemoryError: the named card cannot hold the model, a
                requested split across every card ('all') cannot, no split
                across the cards can hold a transformers model, or the split's
                real device map (preflight_split) puts part of it on the CPU or
                disk.
            UnsupportedQuantizationError: Q2 on a transformers checkpoint that
                is not pre-quantized.
            GgufTensorSplitError: GGUF_TENSOR_SPLIT names fewer cards than the
                GGUF split must use, or more than could take part in any split.
        """
        from millm.core.config import parse_gguf_tensor_split, settings

        resident = self.loader.state.current if self.loader.is_loaded else None
        gpus = project_free_after_unload(
            list_gpus(), getattr(resident, "memory_by_device_mb", None)
        )
        # The checkpoint the background load will open, resolved the same way.
        cache_path = resolve_cache_path(model.cache_path) if model.cache_path else None

        gguf_file = (model.gguf_files or [None])[0]
        if gguf_file:
            # GGUF on Auto is refused here only for a GGUF_TENSOR_SPLIT no split
            # could match. With no card that has room it runs on the CPU; a
            # model every card together is short of is still attempted with
            # every layer on the GPUs (there is no partial CPU offload — the
            # claim that it "spills" was wrong; review round 2). A named card is
            # sized on weights and overhead at the smallest context with no KV
            # term — a lower bound on what the loader asks for, so this cannot
            # refuse a card the loader would accept. Reading the KV size would
            # load the file's metadata inside the request.
            weights_mb = 0
            if cache_path:
                path = Path(cache_path) / gguf_file
                if path.is_file():
                    weights_mb = int(path.stat().st_size / (1024 * 1024))
            placement = plan_gguf_placement(weights_mb, None, 0, requested=wanted, gpus=gpus)
            configured = parse_gguf_tensor_split(settings.GGUF_TENSOR_SPLIT)
            if configured is not None and placement.is_shard:
                # The loader sizes its split with the KV cache on top of these
                # weights, so it spans at least these cards and at most every
                # card with any room — and it refuses a GGUF_TENSOR_SPLIT whose
                # length differs from the cards it spans. Found there, the served
                # model was already unloaded. Review round 1 refused a list
                # SHORTER than the lower bound; a list LONGER than every card
                # that could take part ("1,1,1" on two GPUs) matches no load
                # either. Between the bounds the loader decides. Review round 2.
                card_limit_mb = _gguf_shard_rule(weights_mb, None, GGUF_MIN_CONTEXT).limit_mb
                most_cards = sum(1 for gpu in gpus if card_limit_mb(gpu) > 0)
                if len(configured) > most_cards:
                    raise GgufTensorSplitError(
                        f"GGUF_TENSOR_SPLIT has {len(configured)} value(s), but at most "
                        f"{most_cards} GPU(s) can take part in a split here. Give one "
                        "proportion per card used, in index order, or leave it empty to "
                        "split by free memory.",
                        details={
                            "tensor_split": configured,
                            "max_cards": most_cards,
                            "placement": placement.to_dict(),
                        },
                    )
                if len(configured) < len(placement.gpu_indices):
                    _gguf_tensor_split(placement, configured)
            return

        placement = plan_transformers_load(
            model.estimated_memory_mb or 0,
            model.quantization.value,
            requested=wanted,
            gpus=gpus,
            cache_path=cache_path,
            trust_remote_code=bool(getattr(model, "trust_remote_code", False)),
        )
        # The map itself, computed from the checkpoint's config with no weight
        # read: a split the estimate accepts can still map to disk, and the load
        # would find that only after the unload. Review round 2, 2026-09-14.
        preflight_split(
            model.name,
            cache_path,
            model.quantization.value,
            placement,
            trust_remote_code=bool(getattr(model, "trust_remote_code", False)),
        )

    def _load_worker(
        self,
        model_id: int,
        model_name: str,
        cache_path: str,
        quantization: str,
        estimated_memory_mb: int,
        trust_remote_code: bool,
        gguf_file: Optional[str] = None,
        gpu: GpuRequest = None,
        *,
        under_barrier: bool = False,
    ) -> None:
        """
        Background worker for loading models.

        Runs in thread pool. Updates database and emits events upon completion/failure.

        `under_barrier`: the load was started inside the weight barrier's `internal_write`
        (032's full reload), which owns the barrier's transition. Its completion must not call
        `on_parameter_set_replaced`: inside the write that is a no-op anyway, but a load that
        OUTLIVES its write (the 032 wait timed out and the write ended `torn`) would otherwise
        turn `torn` into `clean` over a checkpoint nothing verified (review R2-B O2).
        """
        try:
            # Load the model
            from millm.core.config import settings

            # Absolute or relative to MODEL_CACHE_DIR; the pre-unload check
            # resolves it the same way, so both read the same checkpoint.
            full_cache_path = resolve_cache_path(cache_path)

            # Resolve TORCH_COMPILE=None (auto-detect) to a concrete bool.
            # Auto: enable for CUDA + non-bitsandbytes models; disable otherwise.
            # A model that ENDS UP split across devices is also not compiled,
            # but that is only knowable after from_pretrained, so the loader
            # (ModelLoadContext.load) enforces it rather than this resolution.
            torch_compile_setting = settings.TORCH_COMPILE
            if torch_compile_setting is None:
                try:
                    import torch as _torch
                    is_bnb = quantization.upper() in ("Q4", "Q8", "Q2")
                    torch_compile_resolved = _torch.cuda.is_available() and not is_bnb
                except Exception:
                    torch_compile_resolved = False
                if torch_compile_resolved:
                    logger.info(
                        "torch_compile_auto_enabled",
                        quantization=quantization,
                    )
                else:
                    logger.info(
                        "torch_compile_auto_disabled",
                        reason="bitsandbytes_quantization" if quantization.upper() in ("Q4", "Q8", "Q2") else "no_cuda",
                        quantization=quantization,
                    )
            else:
                torch_compile_resolved = torch_compile_setting

            loaded = self.loader.load(
                model_id=model_id,
                model_name=model_name,
                cache_path=full_cache_path,
                quantization=quantization,
                estimated_memory_mb=estimated_memory_mb,
                trust_remote_code=trust_remote_code,
                torch_compile=torch_compile_resolved,
                torch_compile_mode=settings.TORCH_COMPILE_MODE,
                gguf_file=gguf_file,
                gpu=gpu,
                attachable=self._attachable_load(quantization, gguf_file),
            )

            # Update database (thread-safe async call)
            self._run_async_from_thread(
                self._update_load_complete(
                    model_id=model_id,
                    memory_used_mb=loaded.memory_used_mb,
                    under_barrier=under_barrier,
                )
            )

            logger.info(
                "load_complete",
                model_id=model_id,
                memory_used_mb=loaded.memory_used_mb,
            )

            # Notify inference service (starts CBM if enabled)
            try:
                svc = self._inference_service
                if svc is None:
                    from millm.api.dependencies import get_inference_service
                    svc = get_inference_service()
                svc.on_model_loaded()
            except Exception:
                pass

        except Exception as e:
            logger.error("load_failed", model_id=model_id, error=str(e))
            self._run_async_from_thread(
                self._update_load_error(
                    model_id=model_id,
                    error_code=getattr(e, "code", "MODEL_LOAD_FAILED"),
                    error_message=str(e),
                )
            )

        finally:
            self._release_load_slot(model_id)

    async def _update_load_complete(
        self,
        model_id: int,
        memory_used_mb: int,
        *,
        under_barrier: bool = False,
    ) -> None:
        """Update database and emit event when load completes."""
        await self.repository.update(
            model_id,
            status=ModelStatus.LOADED,
            loaded_at=datetime.utcnow(),
        )
        # A new parameter set is resident (031 D17); a no-op inside an internal write. On the
        # event loop, where the barrier's hold lives. Never for a load an internal write
        # started (R2-B O2): the write owns that transition even after it has ended.
        if not under_barrier:
            self._parameter_set_replaced("model_load")

        if self.emitter:
            await self.emitter.emit_load_complete(
                model_id=model_id,
                memory_used_mb=memory_used_mb,
            )

    async def _update_load_error(
        self,
        model_id: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """Update database and emit event when load fails."""
        await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message=error_message,
        )

        if self.emitter:
            await self.emitter.emit_load_error(
                model_id=model_id,
                error_code=error_code,
                error_message=error_message,
            )

    async def unload_model(
        self,
        model_id: int,
        timeout: float = UNLOAD_TIMEOUT,
        *,
        lease_id: Optional[str] = None,
        under_barrier: bool = False,
    ) -> Model:
        """
        Unload a model from GPU memory with graceful timeout.

        Args:
            model_id: The model's database ID
            timeout: Maximum time in seconds to wait for unload (default 30s)
            under_barrier: the weight barrier holds the admission slot for the
                caller (032's full reload): skip the drain, which would wait on
                the barrier's own hold (031 G3). Legal only while it holds it.

        Returns:
            The updated model record (status will be READY)

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelNotLoadedError: If model is not loaded
        """
        if under_barrier:
            self._require_barrier_slot("unload_model")
        model = await self.get_model(model_id)

        # Check if this model is actually loaded
        if self.loader.loaded_model_id != model_id:
            raise ModelNotLoadedError(
                f"Model {model_id} is not currently loaded",
                details={"model_id": model_id},
            )

        # Feature 29: a foreign lease refuses the unload before anything is marked.
        self._refuse_if_leased("unload", model_id, lease_id)
        # Feature 31: never move tensors an attached trainer holds handles to (FR-31.5.2).
        self._refuse_if_attached("model_unload")
        # Nor tensors the barrier's own operation is reading or writing (R2-B O1). Only the
        # caller of that operation may unload under it (`under_barrier`); a torn set's
        # leftover holder hold is not an operation and stays unloadable (review D10).
        if not under_barrier:
            self._refuse_if_barrier_operation(model_id)

        if self.loader.is_unloading is True:
            raise ModelBusyError(
                f"Model {model_id} is already being unloaded",
                # `unloading` marks an UNLOAD in progress, so Retry-After is the shorter
                # unload value (Feature 29, backpressure.retry_after_for).
                details={"model_id": model_id, "unloading": True},
            )

        logger.info("unload_started", model_id=model_id, timeout=timeout)

        # Refuse new work on the model BEFORE anything moves. The worker below
        # moves the weights to the CPU while the loader still reports the model
        # as loaded, and a /v1 request arriving in that window passed the route's
        # "is it loaded?" check and ran on a half-moved model: 500, "index is on
        # cuda:0, different from other tensors on cpu" (hardware acceptance,
        # 2026-09-14, item 11). InferenceService._admit reads this mark.
        self.loader.begin_unload()
        moving = False
        try:
            # Stop CBM before unloading model
            try:
                svc = self._inference_service
                if svc is None:
                    from millm.api.dependencies import get_inference_service
                    svc = get_inference_service()
                svc.on_model_unloading()
            except Exception:
                pass

            # Wait for the requests already admitted to finish. No request is
            # admitted after the mark above, so this only ever waits for fewer.
            # GRACEFUL_UNLOAD_TIMEOUT was documented for exactly this and read by
            # nothing; the drain was a hard-coded five-second poll, after which
            # the weights moved under a generation still running.
            try:
                from millm.core.config import settings

                svc = self._inference_service
                if svc is None:
                    from millm.api.dependencies import get_inference_service
                    svc = get_inference_service()
                queue = svc.request_queue
                # The barrier's own hold is not in-flight work: draining would wait on the very
                # slot it keeps — the caller's (`under_barrier`), or a torn set's left behind by
                # a holder that is gone, which this unload's success then releases (031 G3,
                # review D10).
                skip_drain = under_barrier or self._barrier_holds_leftover_hold()
                # `occupied_count`, not `pending_count`: a batch chunk waits and holds OUTSIDE the
                # interactive count (Feature 26), and the weights must not move under it.
                if not skip_drain and queue.occupied_count > 0:
                    grace = float(settings.GRACEFUL_UNLOAD_TIMEOUT)
                    logger.info(
                        "waiting_for_pending_inference",
                        pending=queue.pending_count,
                        occupied=queue.occupied_count,
                        timeout=grace,
                    )
                    if not await queue.wait_idle(grace):
                        logger.warning(
                            "unload_drain_timeout",
                            model_id=model_id,
                            pending=queue.pending_count,
                            timeout=grace,
                        )
            except Exception:
                pass  # Don't block unload if queue check fails

            # Unload from GPU with timeout
            moving = True
            try:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._unload_worker, model_id),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "unload_timeout",
                    model_id=model_id,
                    timeout=timeout,
                )
                # Force unload anyway
                self.loader.unload()
        except BaseException:
            # An unload that stopped before its worker started (cancelled while
            # draining) leaves the model serving. Once the worker has started the
            # weights may be moving, and the mark stays until clear() ends it.
            if not moving and self.loader.loaded_model_id == model_id:
                self.loader.cancel_unload()
            raise

        # The model is no longer resident, so neither is its lease (FR-29.1.9, FR-29.3.4).
        # First, before any database write below that could fail.
        self._leases.end_for_model(model_id, END_MODEL_UNLOADED)
        # Nothing resident is `clean` by construction (addendum D-01); a no-op inside an
        # internal write, which makes its own transition.
        self._parameter_set_replaced("model_unload")

        # The loader cleared the probe hooks with the model; the rows must say so, or status
        # reports probes armed that nothing will ever score again.
        from millm.db.base import async_session_factory
        from millm.services.probe_arming import UNLOAD_REASON, mark_armed_rows_disarmed

        await mark_armed_rows_disarmed(
            async_session_factory, UNLOAD_REASON, event="probes_disarmed_by_unload"
        )

        # Auto-unlock on unload
        model = await self.repository.update(
            model_id,
            status=ModelStatus.READY,
            loaded_at=None,
            locked=False,
        )

        # Emit event
        if self.emitter:
            await self.emitter.emit_unload_complete(model_id=model_id)

        logger.info("unload_complete", model_id=model_id)

        return model

    @staticmethod
    def _refuse_if_barrier_operation(model_id: int) -> None:
        """Refuse an unload while a barrier operation (`internal:<op>`: a 032 save snapshot,
        in-place copy or full reload, a 033 adapter change) holds or is taking the admission
        slot (R2-B O1).

        Without this the drain waited the full grace on the barrier's own hold (it never
        drains), and then the unload freed the weights under the running copy or reload.
        """
        from millm.services.weight_barrier import get_weight_barrier

        owner = get_weight_barrier().snapshot().hold_owner
        if owner is not None and owner.startswith("internal:"):
            operation = owner.split(":", 1)[1]
            raise ModelBusyError(
                f"Model {model_id} cannot be unloaded while the weights operation "
                f"'{operation}' is running; retry once it finishes",
                details={"model_id": model_id, "reason": "weights_operation",
                         "operation": operation},
            )

    @staticmethod
    def _barrier_holds_leftover_hold() -> bool:
        """Whether the barrier holds the slot for a holder-owned hold. While nothing is
        attached (the unload's guard), that is a `torn` set's hold whose holder is gone."""
        from millm.services.weight_barrier import get_weight_barrier

        barrier = get_weight_barrier()
        return barrier.holds_slot() and barrier.snapshot().hold_owner == "holder"

    def _unload_worker(self, model_id: int) -> None:
        """
        Background worker for unloading models.

        This allows the unload to be properly awaited with a timeout.
        """
        logger.debug("unload_worker_started", model_id=model_id)
        self.loader.unload()
        logger.debug("unload_worker_complete", model_id=model_id)

    async def get_loaded_model_id(self) -> Optional[int]:
        """
        Get the ID of the currently loaded model.

        Returns:
            Model ID if a model is loaded, None otherwise.
        """
        return self.loader.loaded_model_id

    def get_loaded_model_info(self) -> Optional[dict]:
        """
        Get runtime info for the currently loaded model.

        Returns:
            Dict with num_parameters, memory_footprint, device, dtype if loaded,
            None otherwise.
        """
        if not self.loader.is_loaded or self.loader.state.current is None:
            return None

        loaded = self.loader.state.current
        return {
            "model_id": loaded.model_id,
            "num_parameters": loaded.num_parameters,
            "memory_footprint": loaded.memory_used_mb * 1024 * 1024,  # Convert MB to bytes
            "device": loaded.device,
            "dtype": loaded.dtype,
            # Additive: which card(s), how the choice was made, and what the
            # load consumed on each card.
            "gpu_indices": list(loaded.gpu_indices),
            "memory_by_device_mb": dict(loaded.memory_by_device_mb),
            "placement": (
                {**loaded.placement, "memory_by_device_mb": dict(loaded.memory_by_device_mb)}
                if loaded.placement
                else None
            ),
        }

    # =========================================================================
    # Lock/Unlock Operations
    # =========================================================================

    async def lock_model(self, model_id: int) -> Model:
        """
        Lock a model to prevent auto-unload (used for steering).

        The model must be in LOADED state. Only one model can be locked at a time.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelNotLoadedError: If model is not loaded
            ModelLockedError: If another model is already locked
        """
        model = await self.get_model(model_id)

        if model.status != ModelStatus.LOADED:
            raise ModelNotLoadedError(
                f"Model {model_id} must be loaded before it can be locked",
                details={"model_id": model_id, "status": model.status.value},
            )

        # Check if another model is already locked
        locked = await self.repository.get_locked_model()
        if locked and locked.id != model_id:
            raise ModelLockedError(
                f"Model '{locked.name}' (ID {locked.id}) is already locked. Unlock it first.",
                details={"locked_model_id": locked.id, "locked_model_name": locked.name},
            )

        # Through the exclusive writer even though the check above has already
        # ruled out a rival lock: that check now ignores locks held by models
        # that are not LOADED, so a leaked flag on a non-resident row can no
        # longer block a legitimate lock — and this is what then clears it.
        model = await self.repository.set_exclusive_lock(model_id)
        logger.info("model_locked", model_id=model_id)
        return model

    async def unlock_model(self, model_id: int) -> Model:
        """
        Unlock a model to allow auto-unload.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.get_model(model_id)
        model = await self.repository.update(model_id, locked=False)
        logger.info("model_unlocked", model_id=model_id)
        return model

    async def get_locked_model(self) -> Model | None:
        """
        Get the currently locked model (if any).

        Returns:
            The locked model or None.
        """
        return await self.repository.get_locked_model()

    async def load_model_and_wait(
        self,
        model_id: int,
        timeout: float = 180.0,
        *,
        lease_id: Optional[str] = None,
        reload: bool = False,
        under_barrier: bool = False,
    ) -> Model:
        """
        Load a model and wait for it to complete loading.

        Used by the OpenAI-compatible endpoints for auto-load on demand.
        Calls load_model() then polls until status becomes LOADED or ERROR.

        Args:
            model_id: The model's database ID
            timeout: Maximum time in seconds to wait for load (default 180s)
            reload: the row is the resident's carrier but the resident weights are
                no longer its own (032 resolver rule 3): unload it, then load the
                row's own weights again.
            under_barrier: as `load_model` (the torn refusal does not apply).

        Returns:
            The loaded model record

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelBusyError: If another load is in progress
            ModelLockedError: If a different model is locked
            AlreadyAttachedError: If a trainer is attached to the resident weights
            WeightsTornError: If the resident weights are torn (addendum D-01)
            WeightsUnsavedError: If the resident weights carry commits no checkpoint records
                (addendum D-45)
            asyncio.TimeoutError: If loading exceeds timeout
        """
        if under_barrier:
            self._require_barrier_slot("load_model_and_wait")
        model = await self.get_model(model_id)

        # If already loaded, return immediately — unless the resident weights are no longer
        # this row's own (a swap back).
        resident = model.status == ModelStatus.LOADED and self.loader.loaded_model_id == model_id
        if resident and not reload:
            return model
        self._refuse_weights_origin(model, "auto_load")

        # Feature 29: the lease answer comes BEFORE `locked` (FPRD D9), because it names a
        # holder and an expiry; `locked` names neither.
        self._refuse_if_leased("auto_load", model_id, lease_id)
        # Feature 31, then D-01: never swap under an attached trainer; never auto-load over a
        # torn set (recovery is the weights API's load, which verifies what it loads).
        self._refuse_if_attached("auto_load")
        if not under_barrier:
            self._refuse_if_torn("auto_load")
            # D-45 (F-A): never discard commits no checkpoint records on a client's say-so.
            self._refuse_if_unsaved_commits("auto_load", model)

        # Check if locked model prevents loading this one
        locked = await self.repository.get_locked_model()
        if locked and locked.id != model_id:
            raise ModelLockedError(
                f"Model '{locked.name}' is locked for steering. "
                f"Cannot auto-load '{model.name}'.",
                details={"locked_model_id": locked.id, "requested_model_id": model_id},
            )

        if resident and reload:
            # The carrier row is LOADED, which `load_model` refuses: unload it first, then
            # load its own weights (032 FTDD §5.5).
            await self.unload_model(model_id, lease_id=lease_id, under_barrier=under_barrier)

        # Start loading (this returns immediately with status=LOADING)
        try:
            await self.load_model(model_id, lease_id=lease_id, under_barrier=under_barrier)
        except ModelAlreadyLoadedError as exc:
            # The row says LOADED and the loader does not hold it (checked
            # above): an unload has cleared the loader and not yet written the
            # row. As ModelAlreadyLoadedError the OpenAI routes answered 400
            # invalid_request_error "already loaded" — telling the caller to
            # change a request that will succeed on retry. Review round 4,
            # 2026-09-14.
            raise ModelBusyError(
                f"Model {model_id} is still being unloaded; retry once it finishes",
                details={"model_id": model_id, "unloading": True},
            ) from exc

        return await self.wait_until_loaded(model_id, timeout)

    async def wait_until_loaded(self, model_id: int, timeout: float = 180.0) -> Model:
        """Wait for a load `load_model` started to finish (addendum §8; 032 calls `load_model`
        and then this, not `load_model_and_wait`).

        Raises:
            ModelNotFoundError: the row disappeared.
            ModelLoadError: the load failed.
            asyncio.TimeoutError: it did not finish within `timeout`.
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            await asyncio.sleep(0.5)
            updated = await self.repository.get_by_id(model_id)
            if updated is None:
                raise ModelNotFoundError(
                    f"Model {model_id} disappeared during loading",
                    details={"model_id": model_id},
                )
            if updated.status == ModelStatus.LOADED:
                return updated
            if updated.status == ModelStatus.ERROR:
                # A load that FAILED, not one still running: as ModelBusyError
                # the OpenAI routes answered it "Another model load is already
                # in progress; retry once it finishes" — for a load that had
                # already finished, unsuccessfully, and would fail again.
                # Review round 3, 2026-09-14.
                raise ModelLoadError(
                    f"Model failed to load: {updated.error_message}",
                    details={"model_id": model_id, "error": updated.error_message},
                )

        raise asyncio.TimeoutError(
            f"Model {model_id} did not finish loading within {timeout}s"
        )

    async def find_model_by_name(self, name: str) -> Model | None:
        """
        Find a model by its display name.

        Args:
            name: The model's display name

        Returns:
            The model or None if not found
        """
        return await self.repository.find_by_name(name)

    async def get_available_models(self) -> list[Model]:
        """
        Get all models available for use (READY, LOADED, LOADING).

        `/v1/models` drops idle weights-origin carriers itself (`listed_name`), against the
        IN-PROCESS resident, not the row's status: a carrier whose row still reads LOADING
        while it is already resident must be listed, and one whose row is stale LOADED while
        nothing holds it must not. A row-status filter here would disagree with both, so the
        service offers none (review R2-A removed an `exclude_idle_weights_origin` keyword that
        no production caller passed).

        Returns:
            List of available models
        """
        return await self.repository.get_available_models()

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_model(self, model_id: int) -> bool:
        """
        Delete a model from the database and disk.

        Args:
            model_id: The model's database ID

        Returns:
            True if model was deleted

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelBusyError: If model is currently loaded
        """
        model = await self.get_model(model_id)

        # Check if model is loaded
        if model.status == ModelStatus.LOADED or self.loader.loaded_model_id == model_id:
            raise ModelBusyError(
                f"Cannot delete model {model_id} while it is loaded. Unload it first.",
                details={"model_id": model_id},
            )

        # Cancel any active download
        if model.status == ModelStatus.DOWNLOADING:
            await self.cancel_download(model_id)

        # Delete cached files if from HuggingFace
        if model.source == ModelSource.HUGGINGFACE and model.repo_id:
            # The variant MUST be passed. The cache directory is
            # repo--quantization[--gguf_label], so resolving it without the
            # label computes a path that does not exist: deleting a GGUF model
            # would remove the row and leave the files — 5.4 GB in the case that
            # found this — orphaned on disk with nothing left pointing at them.
            self.downloader.delete_cached_model(
                model.repo_id,
                model.quantization.value,
                model.gguf_label or None,
            )

        # Delete from database
        deleted = await self.repository.delete(model_id)

        logger.info("model_deleted", model_id=model_id)

        return deleted

    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        # Unload any loaded model
        if self.loader.is_loaded:
            self.loader.unload()

        # Shutdown executor
        self._executor.shutdown(wait=False)
