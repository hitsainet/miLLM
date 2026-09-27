"""Armed probes, and the per-request context that scores them (FR-24.5, FR-24.6, FR-24.10).

## One scoring path, not two

The live path collects per-token scores and calls `combine()` at the end — the same function the
parity engine calls. `OnlineRule` exists and is proven equal to `combine` on every streamable rule,
but it is **not** what serves a request. Using the online form live and the batch form for parity
would be two implementations of one definition, and parity would be verifying the wrong one.

The cost is holding the per-token scores for the request: a 4k context across the 8-probe limit is
about 128 KB. That is not a trade worth making for a drift risk.

## One hook per layer, one device-to-host copy per pass

The budget is `PROBE_MAX_OVERHEAD_MS` (5 ms). It is only reachable if the hidden states cross to
the host **once** per forward pass, with every probe on that layer reading the same CPU tensor. A
`.item()` or `.tolist()` per probe per token is what caused an earlier sensing regression here.

## The request slot

`current_request()` is a single slot, not a map. Serial execution guarantees it is safe:
`MAX_CONCURRENT_REQUESTS` is 1 and `PROBE_FORCE_SERIAL` keeps continuous batching off while
anything is armed. A second concurrent `begin` raises rather than overwriting, because silently
replacing the slot would attribute one request's activations to another's verdict.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from millm.ml.probe_head import ProbeHead, combine
from millm.ml.probe_hooker import ProbeHooker, is_single_row
from millm.services.probe_scope import window as scope_window

logger = logging.getLogger(__name__)

#: How many of the highest-scoring positions an event records.
TOP_POSITIONS = 5


@dataclass(frozen=True)
class ArmedProbe:
    """Everything needed to score one probe, resolved once at arm time."""

    probe_id: str
    name: str
    head: ProbeHead
    rule: str
    scope: str
    layer: int
    rung: int
    rung_language: str
    threshold: Optional[float] = None
    rule_params: dict[str, Any] = field(default_factory=dict)
    #: For a k-sparse probe: maps (tokens, d_model) -> (tokens, k). `None` for a dense probe.
    encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


@dataclass
class Verdict:
    """What one probe concluded about one request."""

    probe_id: str
    name: str
    rung: int
    rung_language: str
    scored: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    #: ⚠ `None` means the probe has SAID NOTHING — either it was not scored, or no threshold was
    #: placed so it ranks without deciding. It is not `False`.
    fires: Optional[bool] = None
    not_scored_reason: Optional[str] = None
    n_scored_tokens: int = 0
    top_positions: list[int] = field(default_factory=list)


class ProbeRequestContext:
    """One request, as every armed probe sees it."""

    def __init__(self, request_id: str, probes: list[ArmedProbe]) -> None:
        self.request_id = request_id
        self.probes = probes
        #: Milliseconds this request has spent doing PROBE work — the scoring in `observe` plus
        #: the rules in `finish`. Reported by `GET /api/probes/status` and compared against
        #: `PROBE_MAX_OVERHEAD_MS` (FR-24.14, SC-4).
        #:
        #: ⚠ It was declared here and written by NOTHING, so `last_request_overhead_ms` was
        #: `null` for every request ever served, the above-threshold warning could never fire,
        #: and SC-4 was unmeasurable from the product itself. Found on the node, by measuring.
        self.overhead_ms = 0.0
        self._not_scored_reason: Optional[str] = None
        #: probe_id -> per-token scores, in arrival order.
        self._scores: dict[str, list[float]] = {p.probe_id: [] for p in probes}
        #: probe_id -> per-token attention logits, for the `attention` rule only.
        self._logits: dict[str, list[float]] = {p.probe_id: [] for p in probes}
        self._mask: dict[str, list[bool]] = {p.probe_id: [] for p in probes}
        self._position = 0

    @property
    def not_scored_reason(self) -> Optional[str]:
        return self._not_scored_reason

    def mark_not_scored(self, reason: str) -> None:
        """Record why this request cannot be scored.

        ⚠ The FIRST reason wins. A request that hit speculative decoding and then also batched is
        most usefully described by what happened first; overwriting would report the last symptom
        rather than the cause.
        """
        if self._not_scored_reason is None:
            self._not_scored_reason = reason
            logger.debug("probe_request_not_scored id=%s reason=%s", self.request_id, reason)

    def observe(self, layer: int, hidden: torch.Tensor, mask: Optional[list[bool]] = None) -> None:
        """Score one forward pass's hidden states for every probe on `layer`.

        `mask` is per-position over this pass. `None` means every position in the pass is scored,
        which is what `scope="all"` produces.
        """
        if self._not_scored_reason is not None:
            return
        if not is_single_row(hidden):
            self.mark_not_scored("batched_request")
            return

        here = [p for p in self.probes if p.layer == layer]
        if not here:
            return

        # ⚠ SYNCHRONISE BEFORE STARTING THE CLOCK, or this measures the MODEL.
        #
        # The hook fires during the forward pass, so the model's own kernels are still in flight
        # when we arrive. The first thing the probe does that touches the result — `.tolist()` —
        # blocks until they finish, and a naive timer therefore charges the model's remaining
        # work to the probe.
        #
        # Measured on the node at 4k tokens with one probe: the timer reported **116 ms** while
        # the true added cost, wall-clock armed against disarmed, was **11.2 ms**. A tenfold
        # over-report in the number an operator reads to decide whether probes are affordable,
        # and one that would keep the above-threshold warning permanently lit.
        #
        # Syncing here moves that wait to where it belongs. It costs nothing the forward was not
        # going to pay anyway: the pass must complete before its output is used.
        if hidden.is_cuda:
            torch.cuda.synchronize(hidden.device)
        started = time.perf_counter()

        # ⚠ **SCORED WHERE THE TENSOR ALREADY IS, AND ONLY THE SCORES COME BACK.**
        #
        # This used to be `hidden[0].detach().to(torch.float32).cpu()` — the whole residual
        # upcast to float32 and copied to the host, then scored there. At 4k tokens and
        # d_model 2048 that is **33.6 MB per forward pass** and the arithmetic then runs
        # single-threaded on the CPU: measured at **25-37 ms for two probes**, against SC-4's
        # 5 ms budget. The comment above it read "THE ONE DEVICE-TO-HOST COPY", which was
        # true and was not the point — one copy of 33.6 MB is the cost.
        #
        # Now the heads follow the tensor's device, the matvec runs there, and what crosses
        # to the host is the (T,) score vector: **16 kB instead of 33.6 MB**, a 2000x
        # reduction, with the arithmetic bit-identical (`ProbeHead.token_scores` explains why
        # the tempting fold and the fp16 matvec were both rejected).
        #
        # `.detach()` still, and no `.cpu()`: a `.cpu()` here would undo the whole thing.
        row = hidden[0].detach()
        n_tokens = row.shape[0]

        for probe in here:
            try:
                # ⚠ THE ENCODER IS INSIDE THE TRY. It was outside, so a k-sparse probe whose
                # encode raised propagated to the hook — which swallows callback exceptions by
                # design, because a probe must never break generation. The request then had no
                # scores at all, and `finish()` reported `no_scored_tokens`, which means "the
                # probe never looked", not "the probe broke". On hardware that turned a
                # one-line configuration error into sixteen vectors of a misleading reason,
                # with the real traceback only in the worker log.
                basis = probe.encoder(row) if probe.encoder is not None else row
            except Exception as exc:  # noqa: BLE001 - reported, never raised at the hook
                logger.error("probe_encode_failed id=%s error=%s", probe.probe_id, exc)
                self.mark_not_scored(f"encoder_failed: {exc}"[:200])
                return
            try:
                # `.tolist()` below is the one host transfer, of (T,) floats.
                scores = probe.head.token_scores(basis)
            except ValueError as exc:
                # A width mismatch here means arming let through a probe that cannot read this
                # model. Refusing the request is right; scoring it would be scoring noise.
                logger.error("probe_score_failed id=%s error=%s", probe.probe_id, exc)
                self.mark_not_scored("head_mismatch")
                return
            self._scores[probe.probe_id].extend(scores.tolist())
            if probe.rule == "attention":
                self._logits[probe.probe_id].extend(probe.head.attention_logits(basis).tolist())
            self._mask[probe.probe_id].extend(
                scope_window(mask, self._position, n_tokens)
            )

        self._position += n_tokens
        self.overhead_ms += (time.perf_counter() - started) * 1000.0

    def token_scores_for(self, probe_id: str) -> list[float]:
        """The scores at the positions that were actually SCORED, in order.

        Masked-out positions are omitted rather than zeroed, because that is the shape miStudio
        records in a test vector — `forward_scores` returns the kept scores, not the full row —
        and parity compares against exactly that.
        """
        scores = self._scores.get(probe_id, [])
        mask = self._mask.get(probe_id, [])
        return [value for value, keep in zip(scores, mask) if keep]

    def finish(self) -> list[Verdict]:
        """Combine each probe's scores into a verdict.

        Every armed probe produces a verdict, scored or not. A probe that is silent about a request
        is indistinguishable from one that is broken.
        """
        started = time.perf_counter()
        verdicts: list[Verdict] = []
        for probe in self.probes:
            base = dict(
                probe_id=probe.probe_id,
                name=probe.name,
                rung=probe.rung,
                rung_language=probe.rung_language,
                threshold=probe.threshold,
            )
            if self._not_scored_reason is not None:
                verdicts.append(
                    Verdict(scored=False, not_scored_reason=self._not_scored_reason, **base)
                )
                continue

            scores = self._scores[probe.probe_id]
            mask = self._mask[probe.probe_id]
            if not scores or not any(mask):
                # No position in scope. Not an error and not a zero — the probe never looked.
                verdicts.append(Verdict(scored=False, not_scored_reason="no_scored_tokens", **base))
                continue

            score_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits = self._logits[probe.probe_id]
            logit_tensor = (
                torch.tensor(logits, dtype=torch.float32).unsqueeze(0) if logits else None
            )
            params = {
                k: v for k, v in (probe.rule_params or {}).items() if k in ("tau", "window")
            }
            value = float(
                combine(
                    probe.rule,
                    score_tensor,
                    mask=mask_tensor,
                    attention_logits=logit_tensor,
                    **params,
                ).item()
            )

            scored_positions = [i for i, keep in enumerate(mask) if keep]
            top = sorted(scored_positions, key=lambda i: scores[i], reverse=True)[:TOP_POSITIONS]
            verdicts.append(
                Verdict(
                    scored=True,
                    score=value,
                    # ⚠ `None`, not `False`, when no threshold was placed: the probe ranks but
                    # does not decide, and reporting `False` would be a verdict it never gave.
                    fires=None if probe.threshold is None else value > probe.threshold,
                    n_scored_tokens=len(scored_positions),
                    top_positions=top,
                    **base,
                )
            )
        self.overhead_ms += (time.perf_counter() - started) * 1000.0
        return verdicts


class ProbeRuntimeState:
    """Process-wide armed-probe registry and hook ownership. Singleton."""

    _instance: Optional["ProbeRuntimeState"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ProbeRuntimeState":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._armed = {}
                instance._handles = {}
                instance._hooker = ProbeHooker()
                instance._request = None
                cls._instance = instance
        return cls._instance

    # --- registry ---------------------------------------------------------------------

    def has_armed(self) -> bool:
        return bool(self._armed)

    def armed(self) -> list[ArmedProbe]:
        return list(self._armed.values())

    def probes_at(self, layer: int) -> list[ArmedProbe]:
        return [p for p in self._armed.values() if p.layer == layer]

    def layers(self) -> set[int]:
        return {p.layer for p in self._armed.values()}

    def arm(self, probe: ArmedProbe, model: Any) -> None:
        """Register a probe and make sure its layer has a hook.

        One hook per layer, shared. Arming a second probe on a layer that already has one installs
        nothing new — which is what keeps the device-to-host budget at one copy per pass however
        many probes are armed.
        """
        self._armed[probe.probe_id] = probe
        if probe.layer not in self._handles:
            layer = probe.layer
            self._handles[layer] = self._hooker.install(
                model, layer, lambda hidden, _layer=layer: self._on_activations(_layer, hidden)
            )
            logger.info("probe_layer_hooked layer=%s", layer)

    def disarm(self, probe_id: str) -> bool:
        """Remove one probe, and its layer's hook if it was the last one there."""
        probe = self._armed.pop(probe_id, None)
        if probe is None:
            return False
        if not self.probes_at(probe.layer):
            handle = self._handles.pop(probe.layer, None)
            if handle is not None:
                self._hooker.remove(handle)
                logger.info("probe_layer_unhooked layer=%s", probe.layer)
        return True

    def disarm_all(self, reason: str) -> list[str]:
        """Remove every probe and every hook. Returns the ids that were armed."""
        ids = list(self._armed)
        for probe_id in ids:
            self._armed.pop(probe_id, None)
        for layer, handle in list(self._handles.items()):
            self._hooker.remove(handle)
            self._handles.pop(layer, None)
        if ids:
            logger.info("probes_disarmed count=%s reason=%s", len(ids), reason)
        return ids

    # --- the request slot -------------------------------------------------------------

    def begin_request(self, request_id: str) -> Optional[ProbeRequestContext]:
        """Open the per-request context. `None` when nothing is armed."""
        if not self._armed:
            return None
        if self._request is not None:
            raise RuntimeError(
                f"a probe request context is already open for {self._request.request_id!r}; "
                f"serial execution is supposed to guarantee this cannot happen, and silently "
                f"replacing it would attribute one request's activations to another's verdict"
            )
        self._request = ProbeRequestContext(request_id, self.armed())
        return self._request

    def current_request(self) -> Optional[ProbeRequestContext]:
        return self._request

    def end_request(self) -> Optional[ProbeRequestContext]:
        context, self._request = self._request, None
        return context

    def _on_activations(self, layer: int, hidden: torch.Tensor) -> None:
        context = self._request
        if context is not None:
            context.observe(layer, hidden, getattr(context, "mask", None))

    # --- tests ------------------------------------------------------------------------

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disarm_all("reset_for_tests")
                cls._instance._request = None
            cls._instance = None
