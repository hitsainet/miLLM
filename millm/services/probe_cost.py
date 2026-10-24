"""What one armed probe costs per forward pass, measured at arm time (Feature 37, FR-37.2).

⚠ MEASURED THROUGH THE SERVING FUNCTION, NOT A COPY OF IT. The candidate is timed by calling
`ProbeRequestContext.observe` — the exact code the layer hook runs on every pass — on synthetic
hidden states placed on the device that holds the probe's layer. A re-implementation of "the
arithmetic" here would drift from what serves the first time `observe` changed.

Two shapes, the two halves of the amended SC-4 budget (D45):

* decode  — `[1, 1, d_model]`, one generated token: milliseconds per pass;
* prefill — `[1, 1024, d_model]`: reported per 1,000 prompt tokens.

The LAYER'S SHARED PART (the float32 cast, the window, the dispatch — paid once per pass however
many probes share the layer, `ProbeRuntimeState.arm`'s one-hook-per-layer design) is separated from
the probe's own arithmetic by timing the candidate alone and the candidate twice: own = t2 − t1,
fixed = t1 − own. `project` charges each probe its own part and each distinct (layer, device) its
fixed part once (TC-37.2).

⚠ TIMING: CUDA events on a CUDA device, never a `synchronize()` pair around the work — on the
hardware record that pair was 99% of the cost it measured (024 10.4). The timer is a module seam
(`_timer`) so unit tests inject figures; no wall-clock number is asserted in CI.

1 + n GPUs: a device is LABELLED by its UUID (`cuda:GPU-…`), never an index, through the existing
`weight_handles.uuid_for_index` (no context is created on any other card).
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

import torch

#: Prefill shape: positions per synthetic prompt.
PREFILL_TOKENS = 1024


@dataclass(frozen=True)
class ProbeCost:
    """One probe's measured cost on one device."""

    decode_ms: float
    prefill_ms_per_1k: float
    #: The layer's shared part: charged once per (layer, device), not per probe.
    layer_fixed_ms: float
    layer_fixed_prefill_ms_per_1k: float
    layer: int
    device: str
    k: int
    measured_at: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _cuda_timer(fn: Callable[[], None], k: int, device: torch.device) -> float:
    """Per call, the LARGER of the GPU time and the host time — the rule the product's own
    per-pass timer applies (`ProbeRequestContext.overhead_ms`).

    ⚠ GPU EVENT TIME ALONE UNDER-PREDICTED (037 hardware record, 2026-10-21): during decode the
    card waits on the host to launch the probe's kernels, so the Python dispatch IS the cost. With
    events alone the projection ran 1.15–1.32× under the measured hook cost for dense probes on
    the 3090 (FR-37.8's 1.25 limit). The host clock is read without any synchronise inside the
    bracket; the sync after the end event only waits to read it.
    """
    stream = torch.cuda.current_stream(device)
    samples = []
    fn()  # warm: first call allocates the head's device copy and the window
    for _ in range(k):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        t = time.perf_counter()
        fn()
        host_ms = (time.perf_counter() - t) * 1000.0
        end.record(stream)
        end.synchronize()  # AFTER the end event: waits for the bracketed work, not inside it
        samples.append(max(start.elapsed_time(end), host_ms))
    return float(statistics.median(samples))


def _host_timer(fn: Callable[[], None], k: int, device: torch.device) -> float:
    samples = []
    fn()
    for _ in range(k):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000.0)
    return float(statistics.median(samples))


def _default_timer(fn: Callable[[], None], k: int, device: torch.device) -> float:
    """Median milliseconds of `k` calls of `fn` on `device`."""
    if device.type == "cuda":
        return _cuda_timer(fn, k, device)
    return _host_timer(fn, k, device)


#: Seam: tests inject `(fn, k, device) -> ms`.
_timer: Callable[[Callable[[], None], int, torch.device], float] = _default_timer


def _uuid_for_index(index: int) -> Optional[str]:
    from millm.services.weight_handles import uuid_for_index

    return uuid_for_index(index)


def device_label(device: Any) -> str:
    """`cuda:GPU-…` for a card (by UUID, never index), else the device type (`cpu`, `meta`)."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        # ⚠ No "current device" fallback: that is GPU 0 on a 1 + n node. A layer's device always
        # carries its index (`layer_resolution.layer_device`); one without it is labelled as such.
        if dev.index is None:
            return "cuda:unindexed"
        uuid = _uuid_for_index(dev.index)
        return f"cuda:{uuid}" if uuid else "cuda:unknown"
    return dev.type


def _observe_fn(armed: Any, probes: list[Any], hidden: torch.Tensor) -> Callable[[], None]:
    """One timed call: a module forward that fires a prepended forward hook, which extracts the
    hidden states and calls `ProbeRequestContext.observe` — the serving path's dispatch, not only
    its arithmetic (037 hardware record: the arithmetic alone under-predicted a lone probe). The
    module is an `nn.Identity`, so its own forward adds nothing; the hook is registered on it
    directly (not through `ProbeHooker.install`, which resets dynamo for the SERVED model)."""
    from torch import nn

    from millm.ml.probe_hooker import extract_hidden_states
    from millm.services.probe_runtime import ProbeRequestContext

    context = ProbeRequestContext("probe-cost", probes)
    module = nn.Identity()

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        context.observe(armed.layer, extract_hidden_states(output))
        return None

    module.register_forward_hook(hook, prepend=True)

    def run() -> None:
        module(hidden)
        if context.not_scored_reason is not None:
            raise RuntimeError(f"cost measurement could not score: {context.not_scored_reason}")

    return run


def measure(
    armed: Any,
    *,
    device: Any,
    d_model: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    first_on_layer: bool = True,
) -> ProbeCost:
    """Time `armed` on `device`. Raises when it cannot be measured (the gate refuses then).

    `first_on_layer` is recorded for the refusal's wording only: the fixed part is always measured,
    and `project` decides whether it is charged.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    twin = dataclasses.replace(armed, probe_id=f"{armed.probe_id}#cost-twin")
    figures: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for name, n_tokens in (("decode", 1), ("prefill", PREFILL_TOKENS)):
            hidden = torch.randn(1, n_tokens, int(d_model), dtype=dtype, device=dev)
            one = _timer(_observe_fn(armed, [armed], hidden), int(k), dev)
            two = _timer(_observe_fn(armed, [armed, twin], hidden), int(k), dev)
            own = max(two - one, 0.0)
            figures[name] = (own, max(one - own, 0.0))
    scale = 1000.0 / PREFILL_TOKENS
    return ProbeCost(
        decode_ms=figures["decode"][0],
        prefill_ms_per_1k=figures["prefill"][0] * scale,
        layer_fixed_ms=figures["decode"][1],
        layer_fixed_prefill_ms_per_1k=figures["prefill"][1] * scale,
        layer=int(armed.layer),
        device=device_label(dev),
        k=int(k),
        measured_at=datetime.now(timezone.utc).isoformat(),
    )


def project(costs: Iterable[ProbeCost]) -> dict[str, float]:
    """The armed set's cost: every probe's own part, plus each distinct (layer, device)'s fixed
    part ONCE — one hook per layer, shared (TC-37.2)."""
    decode = prefill = 0.0
    seen: set[tuple[int, str]] = set()
    for cost in costs:
        decode += cost.decode_ms
        prefill += cost.prefill_ms_per_1k
        key = (cost.layer, cost.device)
        if key not in seen:
            seen.add(key)
            decode += cost.layer_fixed_ms
            prefill += cost.layer_fixed_prefill_ms_per_1k
    return {"decode": decode, "prefill": prefill}


def cost_from_dict(data: Optional[dict[str, Any]]) -> Optional[ProbeCost]:
    if not isinstance(data, dict):
        return None
    try:
        return ProbeCost(**{f.name: data[f.name] for f in dataclasses.fields(ProbeCost)})
    except (KeyError, TypeError):
        return None
