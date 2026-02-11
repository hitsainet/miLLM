"""
Continuous Batching backend using transformers ContinuousBatchingManager.

Provides high-throughput inference by batching multiple requests into
single forward passes. Opt-in via ENABLE_CONTINUOUS_BATCHING=true.

Limitations vs. queue backend:
- Per-request sampling parameters (temperature, top_p) are fixed at
  manager creation time. Only max_new_tokens varies per request.
- Manager converts model attention to paged implementation on init.
- SAE hooks fire correctly (steering works), but monitoring captures
  batch-level activations (acceptable for uniform steering).

Usage:
    backend = ContinuousBatchingBackend(max_queue_size=256, ...)
    backend.start(model, tokenizer)  # After model load
    tokens, reason = await backend.generate(input_ids, max_new_tokens, req_id)
    backend.stop()  # Before model unload
"""

import asyncio
import logging
from threading import Thread
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)


class ContinuousBatchingBackend:
    """
    Continuous batching backend using transformers ContinuousBatchingManager.

    Encapsulates all CBM interaction. Bridges the blocking CBM API to
    async FastAPI handlers via run_in_executor and asyncio.Queue.
    """

    def __init__(
        self,
        max_queue_size: int = 256,
        default_temperature: float = 0.7,
        default_top_p: float = 0.95,
        default_max_tokens: int = 512,
    ) -> None:
        self._manager: Any = None
        self._max_queue_size = max_queue_size
        self._default_temperature = default_temperature
        self._default_top_p = default_top_p
        self._default_max_tokens = default_max_tokens
        self._started = False
        self._tokenizer: Any = None

    @property
    def is_running(self) -> bool:
        """Check if the CBM is running."""
        return self._started and self._manager is not None

    def start(self, model: Any, tokenizer: Any) -> None:
        """
        Create and start the ContinuousBatchingManager.

        Called after model load. This permanently modifies the model's
        attention implementation to paged attention.

        Args:
            model: The loaded HuggingFace model.
            tokenizer: The loaded tokenizer.
        """
        from transformers import ContinuousBatchingManager
        from transformers import GenerationConfig as HFGenerationConfig

        self._tokenizer = tokenizer

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id

        hf_config = HFGenerationConfig(
            max_new_tokens=self._default_max_tokens,
            temperature=self._default_temperature,
            top_p=self._default_top_p,
            do_sample=self._default_temperature > 0,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        self._manager = ContinuousBatchingManager(
            model=model,
            generation_config=hf_config,
            max_queue_size=self._max_queue_size,
        )
        self._manager.start()
        self._started = True

        logger.info(
            "cbm_started: max_queue_size=%d, temperature=%.2f, top_p=%.2f, max_tokens=%d",
            self._max_queue_size,
            self._default_temperature,
            self._default_top_p,
            self._default_max_tokens,
        )

    def stop(self) -> None:
        """
        Stop the ContinuousBatchingManager.

        Called before model unload. Waits for in-flight requests to complete.
        """
        if self._manager and self._started:
            try:
                self._manager.stop(block=True, timeout=10)
            except Exception as e:
                logger.warning("cbm_stop_error: %s", str(e))
            self._manager = None
            self._started = False
            self._tokenizer = None
            logger.info("cbm_stopped")

    async def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        request_id: str,
        timeout: float = 300.0,
    ) -> tuple[list[int], str]:
        """
        Non-streaming generation via CBM.

        Submits the request and blocks (in executor) until complete.

        Args:
            input_ids: Pre-tokenized input token IDs.
            max_new_tokens: Maximum tokens to generate.
            request_id: Unique request identifier.
            timeout: Maximum wait time in seconds.

        Returns:
            Tuple of (generated_token_ids, finish_reason).

        Raises:
            RuntimeError: If generation fails or times out.
        """
        if not self.is_running:
            raise RuntimeError("ContinuousBatchingManager is not running")

        req_id = self._manager.add_request(
            input_ids=input_ids,
            request_id=request_id,
            max_new_tokens=max_new_tokens,
        )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._manager.get_result, req_id, timeout
        )

        if result is None:
            raise RuntimeError(
                f"Generation timed out after {timeout}s for request {request_id}"
            )

        if result.error:
            raise RuntimeError(f"Generation failed: {result.error}")

        finish_reason = (
            "length"
            if len(result.generated_tokens) >= max_new_tokens
            else "stop"
        )
        return result.generated_tokens, finish_reason

    async def generate_stream(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        request_id: str,
    ) -> AsyncGenerator[list[int], None]:
        """
        Streaming generation via CBM.

        Yields new token IDs incrementally as they are generated.
        Bridges the blocking CBM iterator to async via a background
        thread and asyncio.Queue.

        Args:
            input_ids: Pre-tokenized input token IDs.
            max_new_tokens: Maximum tokens to generate.
            request_id: Unique request identifier.

        Yields:
            Lists of newly generated token IDs.

        Raises:
            RuntimeError: If generation fails.
        """
        if not self.is_running:
            raise RuntimeError("ContinuousBatchingManager is not running")

        req_id = self._manager.add_request(
            input_ids=input_ids,
            request_id=request_id,
            max_new_tokens=max_new_tokens,
            streaming=True,
        )

        # Bridge blocking CBM iterator to async via thread + asyncio.Queue
        token_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _stream_worker() -> None:
            """Background thread that reads from CBM and feeds asyncio queue."""
            try:
                prev_len = 0
                for result in self._manager.request_id_iter(req_id):
                    new_tokens = result.generated_tokens[prev_len:]
                    if new_tokens:
                        loop.call_soon_threadsafe(
                            token_queue.put_nowait, new_tokens
                        )
                    prev_len = len(result.generated_tokens)
                    if result.is_finished():
                        break
                    if result.error:
                        loop.call_soon_threadsafe(
                            token_queue.put_nowait,
                            RuntimeError(f"Generation failed: {result.error}"),
                        )
                        return
            except Exception as e:
                loop.call_soon_threadsafe(token_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        thread = Thread(target=_stream_worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await token_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Ensure thread completes
            thread.join(timeout=5.0)
