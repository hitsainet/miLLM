"""
Generation configuration mapping.

Maps OpenAI API parameters to Transformers generate() parameters.

Mapping reference:
- max_tokens → max_new_tokens
- temperature → temperature (0 means greedy)
- top_p → top_p
- stop → stopping_criteria (custom implementation)
- frequency_penalty → repetition_penalty (approximate)
- presence_penalty → (not directly supported, uses repetition_penalty)
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

from millm.api.schemas.openai import ChatCompletionRequest, TextCompletionRequest


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Provides a bridge between OpenAI-style request parameters and
    Transformers generate() kwargs.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 = greedy decoding)
        top_p: Nucleus sampling probability threshold
        do_sample: Whether to use sampling (False for greedy)
        stop_sequences: List of sequences that stop generation
        frequency_penalty: Penalty for token frequency (OpenAI-style)
        presence_penalty: Penalty for token presence (OpenAI-style)
    """

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    stop_sequences: Optional[list[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @classmethod
    def from_request(
        cls, request: Union[ChatCompletionRequest, TextCompletionRequest]
    ) -> "GenerationConfig":
        """
        Create configuration from OpenAI-style request.

        Handles both ChatCompletionRequest and TextCompletionRequest.

        Implementation notes:
        - temperature=0 → do_sample=False (greedy decoding)
        - max_tokens=None → use default 512
        - stop can be string or list

        Args:
            request: Chat or text completion request

        Returns:
            GenerationConfig instance
        """
        # Normalize stop sequences
        stop_sequences: Optional[list[str]] = None
        if request.stop:
            if isinstance(request.stop, str):
                stop_sequences = [request.stop]
            else:
                stop_sequences = list(request.stop)

        # Temperature=0 means greedy decoding
        do_sample = request.temperature > 0

        return cls(
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=do_sample,
            stop_sequences=stop_sequences,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )

    def to_generate_kwargs(self) -> dict[str, Any]:
        """
        Convert to transformers generate() kwargs.

        Returns a dictionary suitable for passing to model.generate(**kwargs).

        Note: stop_sequences requires custom StoppingCriteria,
        which is not handled here. The InferenceService should
        handle stop sequence logic separately.

        Returns:
            Dictionary of kwargs for generate()
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            # Only set temperature and top_p when sampling
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p

        # Approximate frequency_penalty with repetition_penalty
        # OpenAI frequency_penalty: -2.0 to 2.0
        # Transformers repetition_penalty: typically 1.0 to 2.0
        # Map positive values to increased repetition penalty
        if self.frequency_penalty > 0:
            # Map 0-2 → 1.0-1.5 (conservative)
            kwargs["repetition_penalty"] = 1.0 + (self.frequency_penalty * 0.25)
        elif self.frequency_penalty < 0:
            # Negative penalty encourages repetition (not well supported)
            # Use a small decrease from 1.0
            kwargs["repetition_penalty"] = max(0.8, 1.0 + (self.frequency_penalty * 0.1))

        # presence_penalty can also contribute to repetition penalty
        if self.presence_penalty > 0 and "repetition_penalty" not in kwargs:
            kwargs["repetition_penalty"] = 1.0 + (self.presence_penalty * 0.25)

        # Default repetition penalty to prevent degenerate loops,
        # especially important for smaller models (2B-7B)
        if "repetition_penalty" not in kwargs:
            kwargs["repetition_penalty"] = 1.1

        return kwargs

    def with_max_tokens(self, max_tokens: int) -> "GenerationConfig":
        """Return a new config with updated max_tokens."""
        return GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            stop_sequences=self.stop_sequences,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

    def with_stop_sequences(
        self, stop_sequences: Optional[list[str]]
    ) -> "GenerationConfig":
        """Return a new config with updated stop sequences."""
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            stop_sequences=stop_sequences,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
