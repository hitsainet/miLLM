"""
Generation configuration mapping.

Maps OpenAI API parameters to Transformers generate() parameters.

Mapping reference:
- max_tokens        → max_new_tokens  (None → 512)
- temperature       → temperature     (0 = greedy)
- top_p             → top_p
- stop              → post-generation truncation (InferenceService)
- frequency_penalty ↘ combined linearly into repetition_penalty
- presence_penalty  ↗ (see to_generate_kwargs for formula)

Known approximations:
- repetition_penalty is a single scalar applied uniformly to all seen tokens;
  OpenAI's frequency/presence penalties have distinct semantics.  The mapping
  is symmetric and principled but not a perfect equivalence.
- max_tokens for the legacy /v1/completions endpoint intentionally deviates
  from the OpenAI default of 16 tokens — see TextCompletionRequest.
"""

from dataclasses import dataclass, replace
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
    cache_implementation: Optional[str] = None  # "static" or None for dynamic
    #: Feature 25: the request's seed (FR-25.13). Applied by InferenceService inside the
    #: admission slot under torch.random.fork_rng; never emitted into generate() kwargs.
    seed: Optional[int] = None
    #: Feature 25: the `response_format` constraint ("json_object"/"json_schema" and its schema),
    #: or None. InferenceService compiles it and appends a logits processor per generate().
    constraint: Any = None

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

        seed = getattr(request, "seed", None)
        response_format = getattr(request, "response_format", None)
        constraint = None
        if response_format is not None and getattr(response_format, "type", "text") != "text":
            constraint = response_format

        return cls(
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=do_sample,
            stop_sequences=stop_sequences,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
            constraint=constraint,
        )

    def to_generate_kwargs(self) -> dict[str, Any]:
        """
        Convert to transformers generate() kwargs.

        Returns a dictionary suitable for passing to model.generate(**kwargs).

        Note: stop_sequences is NOT emitted here — transformers' `stop_strings`
        also requires the tokenizer, which this pure config object does not
        have. InferenceService._build_generate_kwargs adds both, and keeps its
        post-generation truncation as well (needed for the streaming path, and
        it makes the boundary exact when a stop string spans a token).

        Returns:
            Dictionary of kwargs for generate()
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        # Static KV cache for faster decoding (works with torch.compile)
        if self.cache_implementation:
            kwargs["cache_implementation"] = self.cache_implementation

        if self.do_sample:
            # Only set temperature and top_p when sampling
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p

        # Map OpenAI-style penalties to Transformers repetition_penalty.
        #
        # OpenAI exposes two independent concepts:
        #   frequency_penalty (-2..+2): discourages repeating exact token sequences
        #   presence_penalty  (-2..+2): encourages introducing new topics/tokens
        #
        # Transformers has a single repetition_penalty scalar that multiplies
        # the logits of already-generated tokens (>1 = discourage, <1 = encourage).
        # This is an approximation; a perfect mapping is not possible.
        #
        # Combined formula (symmetric):
        #   combined = frequency_penalty + 0.5 * presence_penalty
        #   repetition_penalty = 1.0 + combined * 0.25
        #   clamped to [0.8, 1.8] to stay within the practical safe range
        #
        # Previous implementation used 0.25x for positive frequency but only 0.1x
        # for negative, and silently discarded presence_penalty when frequency was
        # non-zero.  This version treats both directions and both penalties
        # consistently.
        combined_penalty = self.frequency_penalty + (self.presence_penalty * 0.5)
        if combined_penalty != 0.0:
            raw = 1.0 + (combined_penalty * 0.25)
            kwargs["repetition_penalty"] = max(0.8, min(1.8, raw))

        return kwargs

    def with_max_tokens(self, max_tokens: int) -> "GenerationConfig":
        """Return a new config with updated max_tokens.

        `dataclasses.replace`, not a hand-written constructor call: the hand-written copy listed
        every field by name and would have silently dropped any field added later — the seed and
        the constraint, in Feature 25.
        """
        return replace(self, max_new_tokens=max_tokens)

    def with_stop_sequences(
        self, stop_sequences: Optional[list[str]]
    ) -> "GenerationConfig":
        """Return a new config with updated stop sequences (see `with_max_tokens`)."""
        return replace(self, stop_sequences=stop_sequences)
