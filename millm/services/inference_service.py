"""
Inference service for OpenAI-compatible generation.

Provides the core generation logic for chat completions, text completions,
and embeddings. Handles streaming via TextIteratorStreamer.

Implementation notes:
1. Thread-based streaming (Transformers generate() is blocking)
2. TextIteratorStreamer bridges generate() to async iteration
3. Request queue prevents GPU memory conflicts
4. Steering integration is transparent to API layer
"""

import uuid
from datetime import datetime
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import torch

from millm.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    TextCompletionChoice,
    TextCompletionRequest,
    TextCompletionResponse,
    Usage,
)
from millm.core.logging import get_logger
from millm.ml.generation_config import GenerationConfig
from millm.ml.model_loader import LoadedModelState
from millm.ml.prefix_cache import PrefixCache
from millm.services.request_queue import RequestQueue

if TYPE_CHECKING:
    from millm.services.model_service import ModelService
    from millm.services.monitoring_service import MonitoringService

logger = get_logger(__name__)


class LoadedModelInfo:
    """Information about the currently loaded model."""

    def __init__(self, name: str, model_id: int, loaded_at: datetime) -> None:
        self.name = name
        self.model_id = model_id
        self.loaded_at = loaded_at


class InferenceService:
    """
    Handles inference for OpenAI-compatible endpoints.

    Thread safety notes:
    - One generation at a time via request queue
    - Model/tokenizer access is thread-safe for inference
    - Steering values applied via hooks (not thread-local)

    Attributes:
        request_queue: The request queue for managing concurrency
    """

    def __init__(
        self,
        model_service: Optional["ModelService"] = None,
        steering_service: Any = None,
        max_concurrent: int = 1,
        max_pending: int = 5,
        kv_cache_mode: str = "dynamic",
        enable_prefix_cache: bool = True,
        prefix_cache_max_entries: int = 5,
        speculative_model: Optional[str] = None,
        speculative_num_tokens: int = 5,
        enable_cbm: bool = False,
        cbm_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the inference service.

        Args:
            model_service: Reference to ModelService for model info
            steering_service: Optional SteeringService for feature steering
            max_concurrent: Maximum concurrent GPU operations
            max_pending: Maximum pending requests in queue
            kv_cache_mode: KV cache mode ("static" or "dynamic")
            enable_prefix_cache: Whether to enable prefix caching
            prefix_cache_max_entries: Maximum prefix cache entries
            speculative_model: HF model ID for draft model (speculative decoding)
            speculative_num_tokens: Number of tokens for draft model to propose
            enable_cbm: Whether to enable continuous batching backend
            cbm_config: Configuration dict for CBM backend
        """
        self._model_service = model_service
        self._steering_service = steering_service
        self._request_queue = RequestQueue(max_concurrent, max_pending)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_state = LoadedModelState()
        self._kv_cache_mode = kv_cache_mode
        self._prefix_cache = PrefixCache(
            max_entries=prefix_cache_max_entries,
            enabled=enable_prefix_cache,
        )
        self._speculative_model_id = speculative_model
        self._speculative_num_tokens = speculative_num_tokens
        self._draft_model: Any = None  # Lazy-loaded on first use

        # Continuous Batching (Phase 4)
        self._cbm_backend: Any = None
        if enable_cbm:
            from millm.services.cbm_backend import ContinuousBatchingBackend

            self._cbm_backend = ContinuousBatchingBackend(**(cbm_config or {}))

    @property
    def request_queue(self) -> RequestQueue:
        """Get the request queue."""
        return self._request_queue

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_state.is_loaded

    def get_loaded_model_info(self) -> Optional[LoadedModelInfo]:
        """
        Get info about the currently loaded model.

        Returns:
            LoadedModelInfo if a model is loaded, None otherwise
        """
        if not self._model_state.is_loaded:
            return None

        loaded = self._model_state.current
        if loaded is None:
            return None

        return LoadedModelInfo(
            name=loaded.model_name,
            model_id=loaded.model_id,
            loaded_at=loaded.loaded_at,
        )

    @property
    def _model(self) -> Any:
        """Get the loaded model."""
        if not self._model_state.is_loaded:
            raise RuntimeError("No model is loaded")
        return self._model_state.current.model

    @property
    def _tokenizer(self) -> Any:
        """Get the loaded tokenizer."""
        if not self._model_state.is_loaded:
            raise RuntimeError("No model is loaded")
        return self._model_state.current.tokenizer

    @property
    def prefix_cache(self) -> PrefixCache:
        """Get the prefix cache."""
        return self._prefix_cache

    def _use_cbm(self) -> bool:
        """Whether to use continuous batching for generation."""
        return self._cbm_backend is not None and self._cbm_backend.is_running

    def on_model_loaded(self) -> None:
        """Called after model is loaded. Starts CBM if enabled."""
        if self._cbm_backend is not None and self._model_state.is_loaded:
            try:
                model = self._model_state.current.model
                tokenizer = self._model_state.current.tokenizer
                self._cbm_backend.start(model, tokenizer)
            except Exception as e:
                logger.warning("cbm_start_failed", error=str(e))

    def on_model_unloading(self) -> None:
        """Called before model unload. Stops CBM if running."""
        if self._cbm_backend is not None and self._cbm_backend.is_running:
            self._cbm_backend.stop()

    def _is_sae_attached(self) -> bool:
        """Check if an SAE is currently attached (steering active)."""
        try:
            from millm.services.sae_service import AttachedSAEState
            return AttachedSAEState().is_attached
        except Exception:
            return False

    def _get_draft_model(self) -> Any:
        """
        Lazy-load the draft model for speculative decoding.

        Returns the draft model if configured and SAE is not attached,
        None otherwise (speculative decoding auto-disables with steering).
        """
        # Auto-disable speculative decoding when SAE is attached
        # (SAE hooks on main model don't apply to draft model's speculations)
        if self._is_sae_attached():
            return None

        if self._speculative_model_id is None:
            return None

        if self._draft_model is None:
            try:
                from transformers import AutoModelForCausalLM

                logger.info(
                    "loading_draft_model",
                    model_id=self._speculative_model_id,
                )
                self._draft_model = AutoModelForCausalLM.from_pretrained(
                    self._speculative_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self._draft_model.eval()
                logger.info("draft_model_loaded", model_id=self._speculative_model_id)
            except Exception as e:
                logger.warning(
                    "draft_model_load_failed",
                    error=str(e),
                    model_id=self._speculative_model_id,
                )
                self._speculative_model_id = None  # Disable future attempts
                return None

        return self._draft_model

    # =========================================================================
    # Generation Helpers
    # =========================================================================

    def _get_system_prompt(self, messages: list) -> Optional[str]:
        """Extract the system prompt from messages if present."""
        for msg in messages:
            if msg.role == "system":
                return msg.content
        return None

    def _try_prefix_cache(
        self, system_prompt: str, gen_config: GenerationConfig
    ) -> Optional[tuple[Any, int]]:
        """
        Try to get cached KV states for a system prompt.

        Returns (past_key_values, token_count) or None if not cached.
        """
        if not self._prefix_cache.enabled:
            return None

        steering_hash = PrefixCache.get_steering_hash()
        entry = self._prefix_cache.get(system_prompt, steering_hash)
        if entry is not None:
            return entry.past_key_values, entry.prompt_token_count
        return None

    def _cache_prefix(
        self, system_prompt: str, past_key_values: Any, token_count: int
    ) -> None:
        """Store prefix KV states in cache."""
        if not self._prefix_cache.enabled:
            return

        steering_hash = PrefixCache.get_steering_hash()
        self._prefix_cache.put(system_prompt, steering_hash, past_key_values, token_count)

    def _build_generate_kwargs(
        self, gen_config: GenerationConfig, inputs: dict
    ) -> dict:
        """
        Build kwargs for model.generate() from GenerationConfig.

        Uses to_generate_kwargs() for proper penalty mapping, then adds
        tokenizer-specific pad/eos tokens and KV cache mode.
        """
        # Inject cache mode from server config if not already set
        if gen_config.cache_implementation is None and self._kv_cache_mode == "static":
            gen_config = GenerationConfig(
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
                stop_sequences=gen_config.stop_sequences,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                cache_implementation="static",
            )
        kwargs = gen_config.to_generate_kwargs()
        kwargs.update({k: v.to(self._device) for k, v in inputs.items()})
        kwargs["pad_token_id"] = (
            self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        )
        kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        # Speculative decoding: use draft model when SAE is not attached
        draft_model = self._get_draft_model()
        if draft_model is not None:
            kwargs["assistant_model"] = draft_model
            kwargs["num_assistant_tokens"] = self._speculative_num_tokens

        return kwargs

    def _notify_monitoring(self, request_id: Optional[str] = None) -> None:
        """
        Forward captured activations to the monitoring service.

        Reads last feature activations from the attached SAE (captured
        during the forward hook) and sends them to MonitoringService.
        """
        try:
            from millm.services.sae_service import AttachedSAEState
            from millm.api.dependencies import _monitoring_service

            sae_state = AttachedSAEState()
            sae = sae_state.attached_sae
            if sae is None or not sae.is_monitoring_enabled:
                return

            activations = sae.get_last_feature_activations()
            if activations is None or _monitoring_service is None:
                return

            _monitoring_service.on_activation(
                activations, request_id=request_id
            )
        except Exception as e:
            # Never let monitoring errors affect inference
            logger.debug("monitoring_notification_failed", error=str(e))

    def _check_context_length(self, prompt_tokens: int, max_new_tokens: int) -> None:
        """
        Validate that prompt + generation fits within model context.

        Raises:
            ValueError: If context length would be exceeded.
        """
        max_length = getattr(
            getattr(self._model, "config", None), "max_position_embeddings", None
        )
        if max_length is None:
            return  # Can't validate without config

        total = prompt_tokens + max_new_tokens
        if total > max_length:
            from millm.api.routes.openai.errors import context_length_exceeded_error
            raise ValueError(
                f"Context length exceeded: {prompt_tokens} prompt + "
                f"{max_new_tokens} max_tokens = {total} > {max_length}"
            )

    def _determine_finish_reason(
        self, generated_token_count: int, max_new_tokens: int
    ) -> str:
        """
        Determine finish_reason per OpenAI spec.

        Returns "length" if generation hit max_tokens, "stop" otherwise.
        """
        if generated_token_count >= max_new_tokens:
            return "length"
        return "stop"

    def _apply_stop_sequences(
        self, text: str, stop_sequences: Optional[list[str]]
    ) -> tuple[str, bool]:
        """
        Truncate text at the first occurrence of any stop sequence.

        Returns:
            Tuple of (truncated_text, was_stopped).
        """
        if not stop_sequences:
            return text, False

        earliest_pos = len(text)
        found = False
        for seq in stop_sequences:
            pos = text.find(seq)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                found = True

        if found:
            return text[:earliest_pos], True
        return text, False

    # =========================================================================
    # Chat Completions
    # =========================================================================

    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Create non-streaming chat completion.

        Supports n > 1 for multiple completions per request.

        Args:
            request: The chat completion request

        Returns:
            ChatCompletionResponse with generated text

        Raises:
            RuntimeError: If no model is loaded
        """
        # Delegate to CBM if active
        if self._use_cbm():
            return await self._cbm_chat_completion(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Format messages to prompt
        prompt = self._format_chat_messages(request.messages)
        n = getattr(request, "n", 1) or 1

        choices: list[ChatCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._request_queue.acquire():
            # Tokenize input
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            prompt_tokens = inputs.input_ids.shape[1]

            # Build generation config
            gen_config = GenerationConfig.from_request(request)
            self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

            # Try prefix caching for system prompts
            system_prompt = self._get_system_prompt(request.messages)
            cached_prefix = None
            prefix_token_count = 0
            if system_prompt:
                cached_prefix_result = self._try_prefix_cache(system_prompt, gen_config)
                if cached_prefix_result:
                    cached_prefix, prefix_token_count = cached_prefix_result

            for i in range(n):
                # Generate
                with torch.no_grad():
                    generate_kwargs = self._build_generate_kwargs(gen_config, inputs)

                    # Use cached prefix KV states if available
                    if cached_prefix is not None and prefix_token_count > 0:
                        generate_kwargs["past_key_values"] = cached_prefix
                        # Only pass continuation tokens (after the cached prefix)
                        generate_kwargs["input_ids"] = inputs.input_ids[:, prefix_token_count:]
                        # Attention mask must cover ALL tokens (cached + continuation)
                        generate_kwargs["attention_mask"] = inputs.attention_mask

                    outputs = self._model.generate(**generate_kwargs)

                    # Cache system prompt prefix on first miss
                    if (
                        system_prompt
                        and cached_prefix is None
                        and self._prefix_cache.enabled
                        and hasattr(outputs, "past_key_values")
                    ):
                        try:
                            self._cache_prefix(
                                system_prompt,
                                outputs.past_key_values,
                                prefix_token_count,
                            )
                        except Exception:
                            pass  # Cache failure should never affect generation

                # Notify monitoring after generation
                self._notify_monitoring(request_id=completion_id)

                # Decode output
                generated_ids = outputs[0][prompt_tokens:]
                completion_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                completion_tokens = len(generated_ids)

                # Apply stop sequences
                completion_text, stopped_by_sequence = self._apply_stop_sequences(
                    completion_text, gen_config.stop_sequences
                )

                # Determine finish reason
                if stopped_by_sequence:
                    finish_reason = "stop"
                else:
                    finish_reason = self._determine_finish_reason(
                        completion_tokens, gen_config.max_new_tokens
                    )

                choices.append(
                    ChatCompletionChoice(
                        index=i,
                        message=ChatMessage(role="assistant", content=completion_text),
                        finish_reason=finish_reason,
                    )
                )

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion via SSE.

        Yields SSE-formatted strings: "data: {json}\\n\\n"
        First chunk has role, middle chunks have content, last has finish_reason.
        Always ends with "data: [DONE]\\n\\n".

        Args:
            request: The chat completion request

        Yields:
            SSE-formatted strings for streaming
        """
        # Delegate to CBM if active
        if self._use_cbm():
            async for chunk in self._cbm_stream_chat_completion(request):
                yield chunk
            return

        from transformers import TextIteratorStreamer

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        # Format messages to prompt
        prompt = self._format_chat_messages(request.messages)

        async with self._request_queue.acquire():
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            # Set up streamer
            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Build generation kwargs
            gen_config = GenerationConfig.from_request(request)
            prompt_tokens = inputs["input_ids"].shape[1]
            self._check_context_length(prompt_tokens, gen_config.max_new_tokens)
            generation_kwargs = self._build_generate_kwargs(gen_config, inputs)
            generation_kwargs["streamer"] = streamer

            # Start generation thread with error capture
            thread_error: list[Exception] = []
            thread = Thread(
                target=self._generate_in_thread,
                args=(generation_kwargs, thread_error),
            )
            thread.start()

            try:
                # Send first chunk with role
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                # Stream tokens with stop sequence checking
                token_count = 0
                accumulated_text = ""
                stop_sequences = gen_config.stop_sequences
                stopped_by_sequence = False

                for token in streamer:
                    if token:
                        # Check if accumulated text contains a stop sequence
                        if stop_sequences:
                            accumulated_text += token
                            truncated, found = self._apply_stop_sequences(
                                accumulated_text, stop_sequences
                            )
                            if found:
                                # Yield only the portion before stop sequence
                                remaining = truncated[
                                    len(accumulated_text) - len(token) :
                                ]
                                if remaining:
                                    chunk = ChatCompletionChunk(
                                        id=completion_id,
                                        created=created,
                                        model=model_name,
                                        choices=[
                                            ChatCompletionChunkChoice(
                                                index=0,
                                                delta=ChatCompletionChunkDelta(
                                                    content=remaining
                                                ),
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                                stopped_by_sequence = True
                                break

                        token_count += 1
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=token),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                # Notify monitoring after generation completes
                self._notify_monitoring(request_id=completion_id)

                # Check for thread errors
                if thread_error:
                    error_msg = str(thread_error[0])
                    logger.error("generation_failed_during_stream", error=error_msg)
                    # Send error as final SSE event before closing
                    import json

                    error_event = json.dumps(
                        {
                            "error": {
                                "message": f"Generation error: {error_msg}",
                                "type": "server_error",
                                "code": "generation_error",
                            }
                        }
                    )
                    yield f"data: {error_event}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Determine finish reason
                if stopped_by_sequence:
                    finish_reason = "stop"
                else:
                    finish_reason = self._determine_finish_reason(
                        token_count, gen_config.max_new_tokens
                    )

                # Send final chunk with finish_reason
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error("streaming_error", error=str(e))
                # Try to send error in SSE format
                import json

                try:
                    error_event = json.dumps(
                        {
                            "error": {
                                "message": f"Streaming error: {str(e)}",
                                "type": "server_error",
                                "code": "streaming_error",
                            }
                        }
                    )
                    yield f"data: {error_event}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    pass
            finally:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning("generation_thread_did_not_complete_cleanly")

    # =========================================================================
    # Text Completions
    # =========================================================================

    async def create_text_completion(
        self, request: TextCompletionRequest
    ) -> TextCompletionResponse:
        """
        Create non-streaming text completion.

        Args:
            request: The text completion request

        Returns:
            TextCompletionResponse with generated text
        """
        # Delegate to CBM if active
        if self._use_cbm():
            return await self._cbm_text_completion(request)

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Handle prompt as string or list
        prompts = (
            request.prompt
            if isinstance(request.prompt, list)
            else [request.prompt]
        )

        choices: list[TextCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._request_queue.acquire():
            gen_config = GenerationConfig.from_request(request)

            for i, prompt_text in enumerate(prompts):
                # Tokenize input
                inputs = self._tokenizer(prompt_text, return_tensors="pt").to(
                    self._device
                )
                prompt_tokens = inputs.input_ids.shape[1]
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

                # Generate
                with torch.no_grad():
                    generate_kwargs = self._build_generate_kwargs(
                        gen_config, inputs
                    )
                    outputs = self._model.generate(**generate_kwargs)

                # Notify monitoring after generation
                self._notify_monitoring(request_id=completion_id)

                # Decode output
                generated_ids = outputs[0][prompt_tokens:]
                completion_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                completion_tokens = len(generated_ids)

                # Apply stop sequences
                completion_text, stopped_by_sequence = (
                    self._apply_stop_sequences(
                        completion_text, gen_config.stop_sequences
                    )
                )

                # Determine finish reason
                if stopped_by_sequence:
                    finish_reason = "stop"
                else:
                    finish_reason = self._determine_finish_reason(
                        completion_tokens, gen_config.max_new_tokens
                    )

                choices.append(
                    TextCompletionChoice(
                        index=i,
                        text=completion_text,
                        finish_reason=finish_reason,
                    )
                )

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return TextCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    # =========================================================================
    # Embeddings
    # =========================================================================

    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Create embeddings for input text.

        Uses the model's last hidden layer with mean pooling.
        Supports float and base64 encoding formats.

        Args:
            request: The embedding request

        Returns:
            EmbeddingResponse with embeddings
        """
        import base64
        import struct

        # Normalize input to list
        inputs = (
            request.input if isinstance(request.input, list) else [request.input]
        )

        encoding_format = getattr(request, "encoding_format", "float") or "float"

        embeddings_data: list[EmbeddingData] = []
        total_tokens = 0

        async with self._request_queue.acquire():
            for i, text in enumerate(inputs):
                # Tokenize
                encoded = self._tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                ).to(self._device)
                total_tokens += encoded.input_ids.shape[1]

                # Get embeddings from last hidden layer
                with torch.no_grad():
                    outputs = self._model(
                        **encoded, output_hidden_states=True
                    )

                # Extract last hidden layer and mean pool
                last_hidden = outputs.hidden_states[-1]
                embedding = last_hidden.mean(dim=1).squeeze().cpu().tolist()

                # Ensure embedding is a list
                if isinstance(embedding, float):
                    embedding = [embedding]

                # Encode as base64 if requested
                if encoding_format == "base64":
                    packed = struct.pack(f"<{len(embedding)}f", *embedding)
                    embedding = base64.b64encode(packed).decode("ascii")

                embeddings_data.append(EmbeddingData(index=i, embedding=embedding))

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return EmbeddingResponse(
            data=embeddings_data,
            model=model_name,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    # =========================================================================
    # CBM Generation Methods (Continuous Batching)
    # =========================================================================

    async def _cbm_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion via ContinuousBatchingManager."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        prompt = self._format_chat_messages(request.messages)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        gen_config = GenerationConfig.from_request(request)

        generated_ids, finish_reason = await self._cbm_backend.generate(
            input_ids=input_ids,
            max_new_tokens=gen_config.max_new_tokens,
            request_id=completion_id,
        )

        self._notify_monitoring(request_id=completion_id)

        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        text, stopped = self._apply_stop_sequences(text, gen_config.stop_sequences)
        if stopped:
            finish_reason = "stop"

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(generated_ids),
                total_tokens=len(input_ids) + len(generated_ids),
            ),
        )

    async def _cbm_stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion via CBM."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        prompt = self._format_chat_messages(request.messages)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        gen_config = GenerationConfig.from_request(request)

        # First chunk: role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

        # Stream tokens from CBM
        token_count = 0
        async for new_token_ids in self._cbm_backend.generate_stream(
            input_ids=input_ids,
            max_new_tokens=gen_config.max_new_tokens,
            request_id=completion_id,
        ):
            text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
            if text:
                token_count += len(new_token_ids)
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        self._notify_monitoring(request_id=completion_id)

        # Final chunk with finish_reason
        finish_reason = self._determine_finish_reason(
            token_count, gen_config.max_new_tokens
        )
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    async def _cbm_text_completion(
        self, request: TextCompletionRequest
    ) -> TextCompletionResponse:
        """Text completion via ContinuousBatchingManager."""
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        prompts = (
            request.prompt
            if isinstance(request.prompt, list)
            else [request.prompt]
        )

        choices: list[TextCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        gen_config = GenerationConfig.from_request(request)

        for i, prompt_text in enumerate(prompts):
            input_ids = self._tokenizer.encode(
                prompt_text, return_tensors="pt"
            )[0].tolist()
            prompt_tokens = len(input_ids)

            generated_ids, finish_reason = await self._cbm_backend.generate(
                input_ids=input_ids,
                max_new_tokens=gen_config.max_new_tokens,
                request_id=f"{completion_id}-{i}",
            )

            self._notify_monitoring(request_id=completion_id)

            completion_text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            completion_tokens = len(generated_ids)

            completion_text, stopped = self._apply_stop_sequences(
                completion_text, gen_config.stop_sequences
            )
            if stopped:
                finish_reason = "stop"

            choices.append(
                TextCompletionChoice(
                    index=i,
                    text=completion_text,
                    finish_reason=finish_reason,
                )
            )

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return TextCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _generate_in_thread(
        self, generation_kwargs: dict, errors: Optional[list] = None
    ) -> None:
        """
        Run generation in thread for streaming.

        Must be called in separate thread because generate() is blocking.
        Errors are captured in the errors list so the caller can check them.
        """
        try:
            with torch.no_grad():
                self._model.generate(**generation_kwargs)
        except Exception as e:
            logger.error("generation_thread_error", error=str(e))
            if errors is not None:
                errors.append(e)

    def _format_chat_messages(self, messages: list[ChatMessage]) -> str:
        """
        Format chat messages into prompt string.

        Uses model's chat template if available, otherwise falls back
        to Gemma-style format with turn markers.

        Args:
            messages: List of chat messages

        Returns:
            Formatted prompt string
        """
        # Log incoming messages for debugging template issues
        for i, m in enumerate(messages):
            logger.debug(
                "chat_message",
                index=i,
                role=m.role,
                content_preview=m.content[:200] if m.content else "",
            )

        # Prefer model's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                # Check if chat_template is actually set
                if self._tokenizer.chat_template:
                    formatted = self._tokenizer.apply_chat_template(
                        [{"role": m.role, "content": m.content} for m in messages],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    logger.debug(
                        "formatted_prompt",
                        length=len(formatted),
                        preview=formatted[:500],
                    )
                    return formatted
            except Exception as e:
                logger.warning(
                    "chat_template_failed_using_fallback", error=str(e)
                )

        # Fallback: Gemma-style format with turn markers
        # This format works well with Gemma 2 and similar models
        parts = []
        pending_system = None
        for msg in messages:
            if msg.role == "system":
                # Buffer system message to prepend to next user turn
                pending_system = msg.content
            elif msg.role == "user":
                if pending_system:
                    parts.append(
                        f"<start_of_turn>user\n{pending_system}\n\n{msg.content}<end_of_turn>"
                    )
                    pending_system = None
                else:
                    parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>")
            elif msg.role == "assistant":
                parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>")

        # If there's a dangling system message with no user turn after it
        if pending_system:
            parts.append(f"<start_of_turn>user\n{pending_system}<end_of_turn>")

        # Add generation prompt
        parts.append("<start_of_turn>model")
        return "\n".join(parts)
