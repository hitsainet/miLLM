"""Constrained decoding for `response_format` on the transformers engine (Feature 25, FR-25.10).

THE GUARANTEE: a constrained generation that the model ends itself parses and validates against
the request's schema, or the request fails — never a 200 carrying invalid JSON. A generation cut
off by the token budget reports `finish_reason: "length"`, never "stop" (FR-25.12).

Why miLLM's own processor and not `xgrammar.contrib.hf` (025_FTDD §3.1): that processor checks
token acceptance with `assert`, which `python -O` strips (a rejected token would pass silently);
it copies the bitmask host-to-device on every step; and it is never called after the final token,
so matcher state cannot say whether a document is complete. Completeness here is derived from the
LAST GENERATED TOKEN being a stop token, and every complete output is validated with
`jsonschema` (the library compiles some keywords it does not enforce).

`xgrammar` is imported lazily so this module imports in a process without it (and tests can
monkeypatch it). Feature 26 reuses `validate_output` for constrained batch lines.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import LogitsProcessor

from millm.core.errors import ConstrainedOutputInvalidError


def constraint_kind(response_format: Any) -> Optional[str]:
    kind = (
        response_format.get("type") if isinstance(response_format, dict)
        else getattr(response_format, "type", None)
    )
    return kind if kind in ("json_object", "json_schema") else None


def schema_of(response_format: Any) -> Optional[dict]:
    """The user's JSON Schema for `json_schema`, else None."""
    if constraint_kind(response_format) != "json_schema":
        return None
    spec = getattr(response_format, "json_schema", None)
    return getattr(spec, "schema_", None)


def schema_name(response_format: Any) -> Optional[str]:
    spec = getattr(response_format, "json_schema", None)
    return getattr(spec, "name", None)


def constrained_header(response_format: Any) -> Optional[str]:
    """`X-miLLM-Constrained`: `json_object`, or `json_schema;name="judge_v1"` (FR-25.12.3)."""
    kind = constraint_kind(response_format)
    if kind == "json_schema":
        return f'json_schema;name="{schema_name(response_format)}"'
    return kind


def stop_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """The model's declared EOS ids plus the tokenizer's, deduplicated — so a chat model's
    end-of-turn token counts as a stop, as it does in generate()."""
    ids: list[int] = []
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for v in (raw if isinstance(raw, (list, tuple)) else [raw]):
        if isinstance(v, int) and not isinstance(v, bool):
            ids.append(v)
    tok = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok, int) and not isinstance(tok, bool):
        ids.append(tok)
    return list(dict.fromkeys(ids))


@dataclass
class CompiledConstraint:
    """What `_build_generate_kwargs` needs to constrain one generate() call."""

    response_format: Any
    grammar: Any
    vocab_size: int
    stop_ids: list[int]
    header: Optional[str] = None
    processors: list = field(default_factory=list)  # every processor built, for mask timing


class GrammarCache:
    """Compiled grammars for ONE loaded model (its tokenizer), LRU-bounded.

    Built on first use after a load and dropped on unload: a grammar belongs to one tokenizer's
    vocabulary, so a reload must never reuse a stale one (keyed by the model's identity and
    `loaded_at` in the service).
    """

    def __init__(self, tokenizer: Any, vocab_size: int, stop_ids: list[int], capacity: int):
        import xgrammar as xgr

        self.vocab_size = int(vocab_size)
        self.stop_ids = list(stop_ids)
        self.capacity = max(1, int(capacity))
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=self.vocab_size, stop_token_ids=self.stop_ids or None
        )
        self._compiler = xgr.GrammarCompiler(info)
        self._grammars: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def compile(self, response_format: Any) -> Any:
        kind = constraint_kind(response_format)
        schema = schema_of(response_format)
        key = json.dumps({"type": kind, "schema": schema}, sort_keys=True, separators=(",", ":"))
        if key in self._grammars:
            self._grammars.move_to_end(key)
            self.hits += 1
            return self._grammars[key]
        self.misses += 1
        if kind == "json_schema":
            grammar = self._compiler.compile_json_schema(json.dumps(schema))
        else:
            # NOT compile_builtin_json_grammar(): that grammar accepts ANY JSON value (an array,
            # a bare number), and `json_object` promises an object. Found by the end-to-end test,
            # where validate_output refused a complete `[…]` with a 500.
            grammar = self._compiler.compile_json_schema('{"type": "object"}')
        self._grammars[key] = grammar
        while len(self._grammars) > self.capacity:
            self._grammars.popitem(last=False)
        return grammar


class JsonConstraintProcessor(LogitsProcessor):
    """A `transformers` logits processor: one xgrammar matcher per batch row.

    One per generate() call — matchers are stateful (the serial `n` loop and every batched chunk
    build a fresh one through `_build_generate_kwargs`).
    """

    def __init__(self, grammar: Any, batch_size: int, vocab_size: int):
        import xgrammar as xgr

        self._xgr = xgr
        self._matchers = [xgr.GrammarMatcher(grammar) for _ in range(batch_size)]
        self._bitmask = xgr.allocate_token_bitmask(batch_size, vocab_size)
        self._device_bitmask: Optional[torch.Tensor] = None
        self._started = False
        self.mask_ms = 0.0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        if self._started:
            for i, matcher in enumerate(self._matchers):
                if matcher.is_terminated():
                    continue
                token = int(input_ids[i, -1])
                if not matcher.accept_token(token):
                    # Never `assert`: python -O strips it and the token would pass silently.
                    raise ConstrainedOutputInvalidError(
                        "A token outside the response_format grammar was generated; the "
                        "constraint was not applied to this step.",
                        details={"row": i},
                    )
        self._started = True
        for i, matcher in enumerate(self._matchers):
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self._bitmask, i)
        if scores.device.type == "cpu":
            mask = self._bitmask
        else:
            # One device buffer per request, refreshed in place — not a new allocation per step.
            if self._device_bitmask is None or self._device_bitmask.device != scores.device:
                self._device_bitmask = torch.empty_like(self._bitmask, device=scores.device)
            self._device_bitmask.copy_(self._bitmask, non_blocking=True)
            mask = self._device_bitmask
        self._xgr.apply_token_bitmask_inplace(scores, mask)
        self.mask_ms += (time.perf_counter() - t0) * 1000.0
        return scores


def validate_output(text: str, response_format: Any) -> None:
    """A COMPLETE constrained output must parse, and for `json_schema` validate (FR-25.10.6).

    Raises:
        ConstrainedOutputInvalidError: logged by the caller without the content — only the schema
            name and the output's length.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    kind = constraint_kind(response_format)
    details = {"schema_name": schema_name(response_format), "length": len(text or "")}
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ConstrainedOutputInvalidError(
            "A complete constrained output did not parse as JSON.", details=details
        ) from exc
    if kind == "json_object" and not isinstance(value, dict):
        raise ConstrainedOutputInvalidError(
            "A complete json_object output is not a JSON object.", details=details
        )
    if kind == "json_schema":
        try:
            Draft202012Validator(schema_of(response_format)).validate(value)
        except ValidationError as exc:
            raise ConstrainedOutputInvalidError(
                "A complete constrained output failed validation against the request's schema "
                f"at {list(exc.absolute_path)}.",
                details={**details, "validator": exc.validator},
            ) from exc
