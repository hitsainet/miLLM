"""Per-request SAE activations — `return_sae_activations` (Feature 27, FR-27.1 – FR-27.3).

A request-scoped capture registered on the attached `LoadedSAE` and fed by the SAE's EXISTING
forward hook, before and after `apply_steering` (`sae_hooker.py`). No extra hook per request, and
it works under suppression, where the monitoring capture does not (FR-27.3b).

⚠ **ISOLATION IS BY OWNER, NOT BY TIMING.** The capture carries an owner token; the request sets
that token in a ContextVar before its forward runs, and `asyncio.to_thread` copies the context into
the worker thread (the streaming path passes it to its plain `Thread` explicitly). The hook feeds
the capture only from a forward whose context carries the owner. So a continuous-batching
generation running concurrently on the same model — which takes no admission slot — cannot write
into this request's activations, and neither can a hung thread from an earlier request
(BRD-04 acceptance 10).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from millm.core.errors import SAENotAttachedError, SaeActivationsRefusedError
from millm.ml.sae_wrapper import CAPTURE_OWNER

__all__ = ["CAPTURE_OWNER", "RequestActivationCapture", "resolve_positions", "select_sae",
           "validate_request"]


def resolve_positions(spec: Any, n_prompt: int, max_new_tokens: int) -> int:
    """The WORST-CASE number of positions a request can report, for the pre-generation cap
    (FR-27.2f). `completion` and `all` count `max_new_tokens` in full: a request is never refused
    after it has generated."""
    positions = spec.positions
    total = n_prompt + max_new_tokens
    if positions == "last":
        return 1
    if positions == "prompt":
        return n_prompt
    if positions == "completion":
        return max_new_tokens
    if positions == "all":
        return total
    return max(0, min(int(positions.end), total) - int(positions.start))


def select_sae(spec: Any, entries: list[Any]) -> Any:
    """The attached entry the request names, or a refusal naming the SAE or the candidates."""
    if spec.sae_id is not None:
        matches = [e for e in entries if e.sae_id == spec.sae_id]
        if not matches:
            raise SAENotAttachedError(
                f"SAE {spec.sae_id!r} is not attached, so its activations cannot be returned; "
                "attach it first",
                details={"sae_id": spec.sae_id,
                         "attached": [e.sae_id for e in entries]},
            )
        if len(matches) > 1:
            raise SaeActivationsRefusedError(
                f"SAE {spec.sae_id!r} is attached at several layers "
                f"({', '.join(str(e.layer) for e in matches)}), so the request is ambiguous",
                details={"param": "return_sae_activations.sae_id",
                         "layers": [e.layer for e in matches]},
            )
        return matches[0]
    if not entries:
        raise SAENotAttachedError(
            "No SAE is attached, so there are no activations to return; attach one first",
            details={"param": "return_sae_activations"},
        )
    if len(entries) > 1:
        raise SaeActivationsRefusedError(
            "Several SAEs are attached; name one with return_sae_activations.sae_id. "
            "Candidates: " + ", ".join(f"{e.sae_id} (layer {e.layer})" for e in entries),
            details={"param": "return_sae_activations.sae_id",
                     "candidates": [{"sae_id": e.sae_id, "layer": e.layer} for e in entries]},
        )
    return entries[0]


def check_request_shape(*, n: int = 1, extra_messages: bool = False, n_prompts: int = 1) -> None:
    """T-76: shapes with no single position axis are refused (FR-27.1i)."""
    for refused, what in (
        (n > 1, "n > 1"),
        (extra_messages, "extra_messages"),
        (n_prompts > 1, "several prompts"),
    ):
        if refused:
            raise SaeActivationsRefusedError(
                f"return_sae_activations is refused with {what}: positions are per conversation, "
                "so this request has no single position axis (T-76)",
                details={"param": "return_sae_activations", "shape": what},
            )


def validate_request(spec: Any, *, entries: list[Any], n_prompt: int, max_new_tokens: int,
                     n: int = 1, extra_messages: bool = False, n_prompts: int = 1) -> Any:
    """Every refusal decidable before generation (FR-27.1i, FR-27.2a-b, f). Returns the entry."""
    from millm.core.config import settings

    check_request_shape(n=n, extra_messages=extra_messages, n_prompts=n_prompts)
    if spec.top_k > settings.SAE_ACTIVATIONS_MAX_TOP_K:
        raise SaeActivationsRefusedError(
            f"top_k {spec.top_k} exceeds the limit of {settings.SAE_ACTIVATIONS_MAX_TOP_K}",
            details={"param": "return_sae_activations.top_k",
                     "max_top_k": settings.SAE_ACTIVATIONS_MAX_TOP_K},
        )
    entry = select_sae(spec, entries)
    width = int(entry.sae.d_sae)
    bad = [f for f in (spec.features or ()) if f >= width]
    if bad:
        raise SaeActivationsRefusedError(
            f"feature index {bad[0]} is outside SAE {entry.sae_id!r}, which has {width} features",
            details={"param": "return_sae_activations.features", "d_sae": width},
        )
    worst = resolve_positions(spec, n_prompt, max_new_tokens) * spec.top_k
    if worst > settings.SAE_ACTIVATIONS_MAX_ENTRIES:
        raise SaeActivationsRefusedError(
            f"this request could return up to {worst} activation entries (positions x top_k), "
            f"over the limit of {settings.SAE_ACTIVATIONS_MAX_ENTRIES}; ask for fewer positions "
            "or a smaller top_k",
            details={"param": "return_sae_activations", "worst_case_entries": worst,
                     "max_entries": settings.SAE_ACTIVATIONS_MAX_ENTRIES},
        )
    return entry


@dataclass
class _Kept:
    position: int
    values: list[float]
    indices: list[int]


@dataclass
class RequestActivationCapture:
    """One request's capture: per-pass observe, position filter, chunked encode, top-k on device,
    one host copy per pass."""

    spec: Any
    n_prompt: int
    sae: Any
    sae_id: str
    layer: int
    read_point: str  # post_steering | pre_steering | unsteered
    owner: str = field(default_factory=lambda: uuid.uuid4().hex)
    encode_chunk: int = 512
    kept: list[_Kept] = field(default_factory=list)
    offset: int = 0
    passes: int = 0

    @property
    def phase(self) -> str:
        """Which side of `apply_steering` this capture reads."""
        return "pre" if self.read_point == "pre_steering" else "post"

    def wanted(self, start: int, width: int) -> list[int]:
        """Absolute positions of THIS pass to keep."""
        positions = self.spec.positions
        end = start + width
        if positions == "last":
            return [end - 1] if width else []
        if positions == "prompt":
            lo, hi = 0, self.n_prompt
        elif positions == "completion":
            lo, hi = self.n_prompt, end
        elif positions == "all":
            lo, hi = 0, end
        else:
            lo, hi = positions.start, positions.end
        return list(range(max(start, lo), min(end, hi)))

    def observe(self, hidden: torch.Tensor, phase: str) -> None:
        """Feed one pass. Ignores the other phase; never raises into a forward (the hook guards)."""
        if phase != self.phase:
            return
        width = int(hidden.shape[1])
        start = self.offset
        # ⚠ THE OFFSET ADVANCES BY THE FULL PASS WIDTH, WHATEVER WAS KEPT, or every position after
        # the first filtered pass drifts.
        self.offset += width
        self.passes += 1
        keep = self.wanted(start, width)
        if not keep:
            return
        local = torch.tensor([p - start for p in keep], device=hidden.device)
        rows = hidden[0].index_select(0, local)
        values, indices = self._encode_topk(rows)
        values, indices = values.cpu(), indices.cpu()  # ONE host copy per pass
        entries = [
            _Kept(position=p, values=values[i].tolist(), indices=indices[i].tolist())
            for i, p in enumerate(keep)
        ]
        if self.spec.positions == "last":
            self.kept = entries[-1:]
        else:
            self.kept.extend(entries)

    def _encode_topk(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.spec.features
        k_all = len(features) if features else int(self.sae.d_sae)
        k = min(int(self.spec.top_k), k_all)
        cols = (torch.tensor(features, device=rows.device) if features else None)
        out_v, out_i = [], []
        with torch.no_grad():
            for begin in range(0, rows.shape[0], max(1, int(self.encode_chunk))):
                x = rows[begin: begin + self.encode_chunk]
                if x.dtype != self.sae.W_enc.dtype:
                    x = x.to(self.sae.W_enc.dtype)
                acts = self.sae.encode(x)
                if cols is not None:
                    acts = acts.index_select(1, cols)
                v, i = torch.topk(acts.float(), k, dim=1)
                if cols is not None:
                    i = cols[i]
                out_v.append(v)
                out_i.append(i)
        return torch.cat(out_v), torch.cat(out_i)

    def build(self, full_ids: Any) -> dict[str, Any]:
        """The `sae_activations` block, keyed by absolute position, with each position's token."""
        from millm.api.schemas.millm_extension import READ_POINT_NOTE

        ids = full_ids
        if hasattr(ids, "dim") and ids.dim() == 2:
            ids = ids[0]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids or [])
        return {
            "sae_id": self.sae_id,
            "layer": self.layer,
            "read_point": self.read_point,
            "positions": [
                {
                    "position": k.position,
                    "token_id": int(ids[k.position]) if k.position < len(ids) else None,
                    "features": [{"index": int(i), "value": float(v)}
                                 for v, i in zip(k.values, k.indices, strict=True)],
                }
                for k in self.kept
            ],
            "note": READ_POINT_NOTE.format(read_point=self.read_point),
        }


def refuse_before_generation(request: Any, inference: Any, *, chat: bool) -> None:
    """The route's pre-generation check for `return_sae_activations` (task 6.4). Raises a refusal;
    returns None when the request asked for nothing or passes every check.

    Runs after the model is resident (an auto-load may change what is attached) and before any
    admission slot or generation, so a refused request never generated anything (FR-27.2f).
    """
    spec = getattr(request, "return_sae_activations", None)
    if spec is None:
        return
    from millm.core.errors import EngineUnsupportedError
    from millm.ml.generation_config import GenerationConfig
    from millm.services.sae_service import AttachedSAEState

    prompt = getattr(request, "prompt", None)
    n_prompts = len(prompt) if isinstance(prompt, list) else 1
    check_request_shape(
        n=int(getattr(request, "n", 1) or 1),
        extra_messages=bool(getattr(request, "extra_messages", None)),
        n_prompts=n_prompts,
    )
    if inference._engine_is_llamacpp():
        # The request policy refuses this on a GGUF row before any auto-load; this is the
        # resident-engine half of the same rule (FR-27.2g).
        raise EngineUnsupportedError(
            "Per-request SAE activations need the transformers engine; llama.cpp exposes no "
            "layer to read.",
            details={"param": "return_sae_activations"},
        )
    validate_request(
        spec,
        entries=AttachedSAEState().entries(),
        n_prompt=inference.count_prompt_tokens(request, chat=chat),
        max_new_tokens=int(GenerationConfig.from_request(request).max_new_tokens),
        n_prompts=n_prompts,
    )
