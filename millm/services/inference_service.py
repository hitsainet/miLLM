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
from millm.services.request_queue import RequestQueue

if TYPE_CHECKING:
    from millm.services.model_service import ModelService

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
    ) -> None:
        """
        Initialize the inference service.

        Args:
            model_service: Reference to ModelService for model info
            steering_service: Optional SteeringService for feature steering
            max_concurrent: Maximum concurrent GPU operations
            max_pending: Maximum pending requests in queue
        """
        self._model_service = model_service
        self._steering_service = steering_service
        self._request_queue = RequestQueue(max_concurrent, max_pending)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_state = LoadedModelState()

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

    # =========================================================================
    # Generation Helpers
    # =========================================================================

    def _build_generate_kwargs(
        self, gen_config: GenerationConfig, inputs: dict
    ) -> dict:
        """
        Build kwargs for model.generate() from GenerationConfig.

        Uses to_generate_kwargs() for proper penalty mapping, then adds
        tokenizer-specific pad/eos tokens.
        """
        kwargs = gen_config.to_generate_kwargs()
        kwargs.update({k: v.to(self._device) for k, v in inputs.items()})
        kwargs["pad_token_id"] = (
            self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        )
        kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        return kwargs

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

        Args:
            request: The chat completion request

        Returns:
            ChatCompletionResponse with generated text

        Raises:
            RuntimeError: If no model is loaded
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Format messages to prompt
        prompt = self._format_chat_messages(request.messages)

        async with self._request_queue.acquire():
            # Tokenize input
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            prompt_tokens = inputs.input_ids.shape[1]

            # Build generation config
            gen_config = GenerationConfig.from_request(request)

            # Generate
            with torch.no_grad():
                # Apply steering if active
                if self._steering_service and getattr(
                    self._steering_service, "is_active", False
                ):
                    self._steering_service.prepare_generation()

                generate_kwargs = self._build_generate_kwargs(gen_config, inputs)
                outputs = self._model.generate(**generate_kwargs)

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

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
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
            generation_kwargs = self._build_generate_kwargs(gen_config, inputs)
            generation_kwargs["streamer"] = streamer

            # Apply steering if active
            if self._steering_service and getattr(
                self._steering_service, "is_active", False
            ):
                self._steering_service.prepare_generation()

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

                # Stream tokens, counting for finish_reason
                token_count = 0
                for token in streamer:
                    if token:
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

            for i, prompt in enumerate(prompts):
                # Tokenize input
                inputs = self._tokenizer(prompt, return_tensors="pt").to(
                    self._device
                )
                prompt_tokens = inputs.input_ids.shape[1]

                # Generate
                with torch.no_grad():
                    generate_kwargs = self._build_generate_kwargs(
                        gen_config, inputs
                    )
                    outputs = self._model.generate(**generate_kwargs)

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

        Args:
            request: The embedding request

        Returns:
            EmbeddingResponse with embeddings
        """
        # Normalize input to list
        inputs = (
            request.input if isinstance(request.input, list) else [request.input]
        )

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
        # Prefer model's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                # Check if chat_template is actually set
                if self._tokenizer.chat_template:
                    return self._tokenizer.apply_chat_template(
                        [{"role": m.role, "content": m.content} for m in messages],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
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
