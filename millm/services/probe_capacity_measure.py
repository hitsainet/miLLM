"""Measure a resident model's probe capacity: its own step time and a dense probe's cost line.

Operator 2026-10-21: per-model limits, with budgets expressible as a percentage of the model's
measured step time. This is the measurement that makes a percentage mean something.

What is measured, inside ONE admission slot (`InferenceService.run_model_work`, every SAE
suppressed), with NOTHING armed (refused otherwise — an armed probe's hook would be timed as the
model's own cost):

* **decode ms per token** — one-token forward steps over a KV cache primed with `context_tokens`,
  median of `steps`;
* **prefill ms per 1,000 tokens** — a `prefill_tokens` forward, median of `repeats`, scaled;
* **the dense cost line** — `probe_cost.measure` on a synthetic dense probe (zero head) at the middle
  layer, on the device that holds it: own cost per probe and the shared part per hooked layer.

Devices are labelled by GPU UUID (`probe_cost.device_label`), joined with `+` for a model split
across cards (1 + n GPUs). The k-sparse line needs a real SAE and is not measured here; it comes from
a hardware record or stays empty.

⚠ This runs on the GPU. It is built and CPU-tested; running it on the node needs an operator-
approved window, like any GPU work.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from millm.core.errors import ProbeCapacityRefusedError


@dataclass(frozen=True)
class CapacityReading:
    device: str
    decode_ms_per_token: float
    prefill_ms_per_1k: float
    dense: dict[str, float]
    method: str
    #: 039 (TD39-3): `{mean, max, attention}`, each in `dense`'s shape. `dense` is the `mean` line.
    lines: Optional[dict[str, dict[str, float]]] = None


#: 039 (TD39-3): the rules the recommendation costs a reference set in. `mean` is the existing
#: dense line; `max` and `attention` are timed the same way on real-shaped synthetic probes.
LINE_RULES = ("mean", "max", "attention")


def _line(cost: Any, layer: int) -> dict[str, float]:
    return {
        "decode_own_ms": cost.decode_ms, "decode_fixed_ms": cost.layer_fixed_ms,
        "prefill_own_ms_per_1k": cost.prefill_ms_per_1k,
        "prefill_fixed_ms_per_1k": cost.layer_fixed_prefill_ms_per_1k,
        "layer": layer,
    }


def synthetic_probe(rule: str, *, d_model: int, layer: int) -> Any:
    """A real-shaped synthetic probe for one rule: standardisation statistics always (like every
    real probe — without them `standardise` skips its work, which under-stated the dense line 3×
    on 2026-10-21), and an `attention_query` for the attention rule."""
    from millm.ml.probe_head import ProbeHead
    from millm.services.probe_runtime import ArmedProbe

    return ArmedProbe(
        probe_id=f"capacity-{rule}", name=f"capacity-{rule}",
        head=ProbeHead(
            weight=torch.zeros(d_model), bias=1.0, mean=torch.zeros(d_model),
            std=torch.ones(d_model), layer=layer,
            attention_query=torch.zeros(d_model) if rule == "attention" else None,
        ),
        rule=rule, scope="all", layer=layer, rung=2, rung_language="synthetic", threshold=2.0,
        windows=("all",),
    )


def _sync(devices: list[torch.device]) -> None:
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def _default_clock(fn: Callable[[], Any], devices: list[torch.device]) -> float:
    """Milliseconds of one call. A whole forward is timed, so a synchronise around it is the
    honest bracket here (unlike a probe's hook, nothing else shares the slot)."""
    _sync(devices)
    t = time.perf_counter()
    fn()
    _sync(devices)
    return (time.perf_counter() - t) * 1000.0


#: Seam: tests inject `(fn, devices) -> ms`.
_clock: Callable[[Callable[[], Any], list[torch.device]], float] = _default_clock


def model_devices(model: Any) -> list[torch.device]:
    seen: list[torch.device] = []
    for p in model.parameters():
        if p.device not in seen:
            seen.append(p.device)
    return seen


def _vocab(model: Any) -> int:
    config = getattr(model, "config", None)
    return int(getattr(config, "vocab_size", 0) or 1000)


def _first_device(model: Any) -> torch.device:
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None:
        return emb.weight.device
    return next(model.parameters()).device


def measure_resident(model: Any, *, prefill_tokens: int = 1024, context_tokens: int = 128,
                     steps: int = 32, repeats: int = 5, k: int = 20) -> CapacityReading:
    """The measurement itself (synchronous; the caller runs it in the model slot)."""
    from millm.ml.layer_resolution import get_layer_count, layer_device
    from millm.services import probe_cost

    devices = model_devices(model)
    first = _first_device(model)
    vocab = _vocab(model)
    gen = torch.Generator().manual_seed(0)

    def ids(n: int) -> torch.Tensor:
        return torch.randint(3, max(vocab - 1, 4), (1, n), generator=gen).to(first)

    with torch.no_grad():
        long_ids = ids(prefill_tokens)
        prefill = [_clock(lambda: model(input_ids=long_ids, use_cache=False), devices)
                   for _ in range(repeats + 1)][1:]
        ctx = model(input_ids=ids(context_tokens), use_cache=True)
        past = ctx.past_key_values
        token = ids(1)
        decode: list[float] = []

        def step() -> None:
            nonlocal past
            out = model(input_ids=token, past_key_values=past, use_cache=True)
            past = out.past_key_values

        for _ in range(steps + 1):
            decode.append(_clock(step, devices))
        decode = decode[1:]

    n_layers = get_layer_count(model)
    layer = n_layers // 2
    d_model = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "d_model"))
    # ⚠ WITH STANDARDISATION STATISTICS, like every real probe (the contract requires `norm_mean`
    # and `norm_std`). Without them `standardise` skips its subtract and divide, and the first live
    # run (2026-10-21) measured a dense probe at 0.008 ms per token against the 0.025 the hardware
    # record measured for real-shaped probes — a 3× under-statement of what a probe costs.
    dtype = next(model.parameters()).dtype
    device = layer_device(model, layer)
    # 039 (TD39-3): one line per rule — `mean` (the existing dense line), `max` and `attention`
    # (an attention probe also carries its query) — through the same `probe_cost.measure`.
    lines = {
        rule: _line(probe_cost.measure(synthetic_probe(rule, d_model=d_model, layer=layer),
                                       device=device, d_model=d_model, k=k, dtype=dtype), layer)
        for rule in LINE_RULES
    }
    label = "+".join(probe_cost.device_label(d) for d in devices)
    return CapacityReading(
        device=label,
        decode_ms_per_token=float(statistics.median(decode)),
        prefill_ms_per_1k=float(statistics.median(prefill)) * 1000.0 / prefill_tokens,
        dense=lines["mean"],
        lines=lines,
        method=(
            f"measured in the model slot with nothing armed: decode = median of {steps} one-token "
            f"steps over a {context_tokens}-token KV cache; prefill = median of {repeats} "
            f"{prefill_tokens}-token forwards; cost lines = probe_cost.measure (K={k}) of "
            f"synthetic mean, max and attention probes at layer {layer}"
        ),
    )


async def measure_and_record(db: Any, inference: Any, model_row: Any, *,
                             steps: Optional[int] = None) -> Any:
    """Refusals first, then the measurement in the model slot, then ONE appended row."""
    from millm.core.config import settings
    from millm.db.models.probe_capacity import ProbeCapacityMeasurement
    from millm.ml.model_loader import LoadedModelState
    from millm.services.probe_runtime import ProbeRuntimeState

    current = LoadedModelState().current
    if current is None or current.model_id != model_row.id:
        raise ProbeCapacityRefusedError(
            "Capacity is measured on the RESIDENT model: load this model first",
            details={"reason": "not_resident", "model_id": model_row.id},
        )
    if not getattr(current, "supports_hooks", True):
        raise ProbeCapacityRefusedError(
            "This engine exposes no module tree, so probes cannot run on it",
            details={"reason": "engine_unsupported"},
        )
    if ProbeRuntimeState().has_armed():
        raise ProbeCapacityRefusedError(
            "Disarm every probe first: an armed probe's hook would be timed as the model's own cost",
            details={"reason": "probes_armed"},
        )
    reading = await inference.run_model_work(lambda: measure_resident(
        current.model, steps=int(steps or settings.PROBE_CAPACITY_DECODE_STEPS),
        k=int(settings.PROBE_COST_SAMPLES),
    ))
    q = getattr(model_row, "quantization", None)
    row = ProbeCapacityMeasurement(
        repo_id=model_row.repo_id or model_row.name,
        quantization=str(getattr(q, "value", q) or ""),
        device=reading.device,
        decode_ms_per_token=reading.decode_ms_per_token,
        prefill_ms_per_1k=reading.prefill_ms_per_1k,
        dense=reading.dense,
        sparse=None,
        lines=reading.lines,
        method=reading.method,
        source="POST /api/models/{id}/probe-capacity/measure",
    )
    db.add(row)
    await db.commit()
    return row
