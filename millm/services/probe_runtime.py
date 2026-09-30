"""Armed probes, and the per-request context that scores them (FR-24.5, FR-24.6, FR-24.10).

## One scoring path, not two

The live path collects per-token scores and calls `combine()` at the end — the same function the
parity engine calls. `OnlineRule` exists and is proven equal to `combine` on every streamable rule,
but it is **not** what serves a request. Using the online form live and the batch form for parity
would be two implementations of one definition, and parity would be verifying the wrong one.

The cost is holding the per-token scores for the request: a 4k context across the 8-probe limit is
about 128 KB, as float32 **tensors**. That is not a trade worth making for a drift risk.

## One hook per layer, and ONE host crossing per probe per REQUEST

The budget is `PROBE_MAX_OVERHEAD_MS_PER_PASS` (0.25 ms) — **per forward pass, not per request**,
because that is the unit the cost is incurred in. Measured 2026-09-30 on Llama-3.1-8B with one
probe, varying only `max_tokens`: 4 generated tokens cost 0.837 ms, 30 cost 3.425, 120 cost
11.223, while 2820 PROMPT tokens with 8 generated cost 1.94. Prefill scores the whole prompt in
one call and decode scores one token per call, so a prompt token is ~300x cheaper than a
generated one and the total is essentially the pass count times a constant. Judging the TOTAL
therefore means warning on long answers; the old 5 ms per-request threshold fired above roughly
50 generated tokens on any model.

The way the budget is met is that a forward pass costs the probe no host-side work proportional
to the tokens in it. The row crosses to the host
once, in `finish()`, where `_row` explains why it crosses at all rather than staying on the card.

Scores, attention logits and the scope mask are accumulated as **tensors on the activations' own
device**, one small tensor per pass, and `torch.cat` runs once per probe in `finish()`. What used
to happen instead — measured on the 3090 at 4096 tokens, per probe, with `nvidia-smi` idle:

    scores.tolist()                     0.051 ms   (D2H, then 4096 Python floats)
    [True] * 4096                       0.006 ms
    torch.tensor(py_scores)  in finish  0.133 ms
    torch.tensor(py_mask)    in finish  0.192 ms
    sorted(4096, key=...)    in finish  0.431 ms   <- the top-5 positions
                                        -------
                                        0.813 ms   per probe, x2 probes = 1.6 ms

against a 5 ms budget for the whole request, for bookkeeping around 0.42 ms of arithmetic. The
`sorted()` is the one that surprises: it is a Python sort over every scored position to keep five
of them, and it cost more than the matvec it was sorting.

A `.item()` or `.tolist()` per probe per token is what caused an earlier sensing regression here;
there is now no `.tolist()` on the pass path at all.

## The head follows the activations, and is resolved ONCE per request

`ProbeHead.to_device` builds a new head every call, so calling it inside `token_scores` re-uploaded
`weight`, `mean` and `std` to the card **on every forward pass** — 0.035 ms per probe per pass,
which over a 192-token completion with two probes is 13 ms of pure re-upload, more than twice the
whole budget. The context resolves the device head once and keeps it.

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
from millm.services.probe_scope import scored_mask
from millm.services.probe_scope import window as scope_window

logger = logging.getLogger(__name__)

#: How many of the highest-scoring positions an event records.
TOP_POSITIONS = 5


def _row(parts: list[torch.Tensor]) -> torch.Tensor:
    """The per-pass tensors as one row, ON THE HOST.

    A prefill-only request has exactly one part, and `torch.cat` on a single tensor COPIES it,
    so the common case skips the cat.

    ⚠ **AND THE ROW COMES BACK TO THE HOST HERE, ONCE PER PROBE PER REQUEST**, which is the only
    device-to-host transfer left on the probe path: 16 kB of scores and 4 kB of mask at 4k tokens,
    against the 33.6 MB per forward pass this module used to copy.

    It is not an oversight that it is not left on the card. `combine()` is the function the parity
    gate calls, and the parity gate scores its test vectors on the **host**. A reduction over 4k
    float32 values does not associate the same way on both, so leaving the row on the card makes
    serving and parity the same CODE over different arithmetic — a weaker form of the two
    implementations this module's docstring exists to prevent. Measured: the `mean` over a
    4160-token row differed by **4.8e-07** between the two, deterministic and reproducible across
    rounds. That is four orders inside the 1e-3 parity tolerance and it is still a number nobody
    asked for, bought back for about 0.05 ms per probe per request.
    """
    row = parts[0] if len(parts) == 1 else torch.cat(parts)
    return row if row.device.type == "cpu" else row.cpu()


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

    def __init__(
        self,
        request_id: str,
        probes: list[ArmedProbe],
        *,
        collect_flip_risk: bool = False,
    ) -> None:
        self.request_id = request_id
        self.probes = probes
        #: ⚠ **OFF FOR EVERY SERVED REQUEST, ON FOR PARITY ONLY.** `flip_risk` re-derives the
        #: pre-activations, which roughly doubles a k-sparse probe's encode cost — affordable
        #: sixteen times before arming, not on the hot path with a 5 ms budget. It is a
        #: constructor argument rather than a setting so that nothing can turn it on globally.
        self.collect_flip_risk = collect_flip_risk
        #: Milliseconds this request has spent doing PROBE work — the scoring in `observe` plus
        #: the rules in `finish`. Reported by `GET /api/probes/status` and compared against
        #: `PROBE_MAX_OVERHEAD_MS` (FR-24.14, SC-4).
        #:
        #: ⚠ It was declared here and written by NOTHING, so `last_request_overhead_ms` was
        #: `null` for every request ever served, the above-threshold warning could never fire,
        #: and SC-4 was unmeasurable from the product itself. Found on the node, by measuring.
        self.overhead_ms = 0.0
        #: How many forward passes did probe work. THE DENOMINATOR OF THE BUDGET: the cost is
        #: per-call dispatch, so a 120-token completion pays it 121 times (one prefill, 120
        #: decode steps) while a 2820-token prompt pays it once. Reported per pass by
        #: `GET /api/probes/status` and compared against `PROBE_MAX_OVERHEAD_MS_PER_PASS`.
        self.n_passes = 0
        self._not_scored_reason: Optional[str] = None
        #: probe_id -> one (n_tokens,) float32 score tensor PER FORWARD PASS, in arrival order.
        #: Tensors, not floats: see the module docstring. `finish()` cats them once.
        self._scores: dict[str, list[torch.Tensor]] = {p.probe_id: [] for p in probes}
        #: probe_id -> the same, for per-token attention logits, for the `attention` rule only.
        self._logits: dict[str, list[torch.Tensor]] = {p.probe_id: [] for p in probes}
        #: probe_id -> per-token "this position's gate is not reproducible", parity only.
        #: A Python list and not a tensor, deliberately: it is collected only when
        #: `collect_flip_risk` is set, which is never on a served request, so it is not on the
        #: path the 5 ms budget covers.
        self._flip_risk: dict[str, list[bool]] = {p.probe_id: [] for p in probes}
        #: probe_id -> one (n_tokens,) bool scope window per pass.
        self._mask: dict[str, list[torch.Tensor]] = {p.probe_id: [] for p in probes}
        #: probe_id -> that probe's head, already on the activations' device. Resolved on the
        #: first pass and reused, because `to_device` copies every weight when it is called.
        self._heads: dict[str, ProbeHead] = {}
        self._position = 0
        #: Where the prompt ends, so `prompt`/`response` scopes can be honoured.
        #:
        #: ⚠ SET EXPLICITLY BY THE CALLER, NOT INFERRED FROM THE FIRST PASS. Taking the prefill
        #: length would be right for an ordinary generation and silently wrong under chunked
        #: prefill or speculative decoding — and the failure is a probe scoring the model's own
        #: output under the name of the user's prompt, which is the exact confusion the role
        #: mask exists to prevent. `None` means nobody said, and a non-`all` probe then reports
        #: `prompt_boundary_unknown` rather than guessing.
        self._n_prompt_tokens: Optional[int] = None

    def set_prompt_length(self, n_prompt_tokens: int) -> None:
        """Record where the prompt ends. Idempotent; a conflicting second call is refused.

        A second, different value means two callers disagree about the boundary, and scoring
        under either would be a guess. Refusing is the honest outcome — the verdict says why.
        """
        n = int(n_prompt_tokens)
        if n < 0:
            self.mark_not_scored(f"prompt_boundary_invalid: {n}")
            return
        if self._n_prompt_tokens is not None and self._n_prompt_tokens != n:
            self.mark_not_scored(
                f"prompt_boundary_conflict: {self._n_prompt_tokens} then {n}"
            )
            return
        self._n_prompt_tokens = n

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

        # Counted where the work is DONE, after both early returns — a pass that scored nothing
        # must not dilute the per-pass figure, or the budget flatters itself on a probe that is
        # armed for another layer.
        self.n_passes += 1

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

        # The scope window for THIS pass, built once and shared by every probe on this layer —
        # it depends on the position and the length, not on the probe. As a tensor on the
        # activations' device, because `combine()` masks against the scores and a CPU mask would
        # drag the scores back off the card.
        #
        # `scope_window` still owns the padding rule (short mask pads CLOSED), so there is one
        # definition of it and not two. When `mask` is None — which is every request the runtime
        # serves today, `scope="all"` — there is no Python list at all.
        #
        # ⚠ PER SCOPE, NOT ONCE PER PASS. This built a single window and shared it across every
        # probe on the layer, which is correct only while they all agree — and scope is a
        # per-probe field. Two probes on one layer with different scopes would both have read
        # the first one's window. Cached by scope within the pass, because probes usually do
        # agree and `scored_mask` should not be rebuilt per probe.
        windows: dict[str, torch.Tensor] = {}

        def window_for(scope: str) -> Optional[torch.Tensor]:
            if scope in windows:
                return windows[scope]
            if scope == "all" and mask is None:
                built = torch.ones(n_tokens, dtype=torch.bool, device=row.device)
            else:
                spec = mask
                if spec is None:
                    if self._n_prompt_tokens is None:
                        return None
                    # `scored_mask` is the DEFINITION of which positions a scope admits, and it
                    # is called here so there is one of them rather than a second arithmetic
                    # copy that can drift. `n_generated` counts only as far as this pass, which
                    # is all that is needed: positions beyond it are not being scored yet.
                    generated = max(0, self._position + n_tokens - self._n_prompt_tokens)
                    spec = scored_mask(
                        scope=scope,
                        n_prompt_tokens=self._n_prompt_tokens,
                        n_generated=generated,
                    )
                built = torch.as_tensor(
                    scope_window(spec, self._position, n_tokens),
                    dtype=torch.bool,
                    device=row.device,
                )
            windows[scope] = built
            return built

        for probe in here:
            window = window_for(probe.scope)
            if window is None:
                self.mark_not_scored("prompt_boundary_unknown")
                return
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
            # The head, on the activations' device, resolved ONCE per request rather than on
            # every pass. `to_device` builds a new head and copies `weight`, `mean` and `std`
            # each time it is called, so calling it per pass re-uploaded them per pass.
            head = self._heads.get(probe.probe_id)
            if head is None or head.weight.device != basis.device:
                head = probe.head.to_device(basis.device)
                self._heads[probe.probe_id] = head
            try:
                scores = head.token_scores(basis)
            except ValueError as exc:
                # A width mismatch here means arming let through a probe that cannot read this
                # model. Refusing the request is right; scoring it would be scoring noise.
                logger.error("probe_score_failed id=%s error=%s", probe.probe_id, exc)
                self.mark_not_scored("head_mismatch")
                return
            # ⚠ APPEND THE TENSOR. `.tolist()` here was 4096 Python floats per probe per pass,
            # which `finish()` then rebuilt into the tensor it needed anyway.
            self._scores[probe.probe_id].append(scores)
            if self.collect_flip_risk:
                # ⚠ A probe whose encoder cannot say (a dense probe, a relu basis, an injected
                # test double) records False, meaning "nothing here is unreproducible" — NEVER
                # True. Defaulting to True would set positions aside on no evidence, which is
                # exactly how a gate stops gating.
                risk = getattr(probe.encoder, "flip_risk", None)
                self._flip_risk[probe.probe_id].extend(
                    risk(row).tolist() if risk is not None else [False] * n_tokens
                )
            if probe.rule == "attention":
                self._logits[probe.probe_id].append(head.attention_logits(basis))
            self._mask[probe.probe_id].append(window)

        self._position += n_tokens

        # ⚠ AND SYNCHRONISE BEFORE STOPPING THE CLOCK, for the same reason it was started after
        # one. The work above is now entirely asynchronous kernel launches, so a bare
        # `perf_counter` would time the ENQUEUE and report a fraction of what the probe costs —
        # the mirror image of the 10x over-report this measurement used to carry. `.tolist()`
        # provided this sync implicitly, once per probe; this is once per pass and moves no data.
        if hidden.is_cuda:
            torch.cuda.synchronize(hidden.device)
        self.overhead_ms += (time.perf_counter() - started) * 1000.0

    def token_scores_for(self, probe_id: str) -> list[float]:
        """The scores at the positions that were actually SCORED, in order.

        Masked-out positions are omitted rather than zeroed, because that is the shape miStudio
        records in a test vector — `forward_scores` returns the kept scores, not the full row —
        and parity compares against exactly that.

        This is the ONLY place per-token scores become Python floats, and it is called by the
        parity gate at arm time, never by a served request.
        """
        score_parts = self._scores.get(probe_id) or []
        mask_parts = self._mask.get(probe_id) or []
        if not score_parts or not mask_parts:
            return []
        return _row(score_parts)[_row(mask_parts)].tolist()

    def flip_risk_for(self, probe_id: str) -> list[bool]:
        """Per SCORED position, whether a step gate could have resolved the other way.

        Aligned with `token_scores_for` — the same positions, in the same order — so the parity
        engine can pair them without re-deriving the scope mask. Empty when the collector was
        off, which callers must read as "unknown", not as "nothing at risk".

        ⚠ `_mask` holds ONE TENSOR PER FORWARD PASS, not one bool per token — so the scope has to
        be flattened through `_row` before it can be zipped against a per-token list. Zipping the
        parts directly pairs the first token's risk with a whole pass's window, which is a
        length-of-passes answer that a multi-element tensor then raises on.
        """
        risk = self._flip_risk.get(probe_id) or []
        mask_parts = self._mask.get(probe_id) or []
        if not risk or not mask_parts:
            return []
        return [value for value, keep in zip(risk, _row(mask_parts).tolist()) if keep]

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

            score_parts = self._scores[probe.probe_id]
            mask_parts = self._mask[probe.probe_id]
            if not score_parts:
                # The probe never looked. Not an error and not a zero.
                verdicts.append(Verdict(scored=False, not_scored_reason="no_scored_tokens", **base))
                continue

            # One `cat` and one host crossing per probe per request, over one tensor per pass.
            score_row = _row(score_parts)
            mask_row = _row(mask_parts)
            #: Ascending positions that are in scope. Also the `no_scored_tokens` test and the
            #: `n_scored_tokens` count, so it replaces three separate walks over the mask.
            scored_index = mask_row.nonzero(as_tuple=True)[0]
            if scored_index.numel() == 0:
                # A mask that selects nothing is not a score of 0 — the probe never looked.
                verdicts.append(Verdict(scored=False, not_scored_reason="no_scored_tokens", **base))
                continue

            score_tensor = score_row.unsqueeze(0)
            mask_tensor = mask_row.unsqueeze(0)
            logit_parts = self._logits[probe.probe_id]
            logit_tensor = _row(logit_parts).unsqueeze(0) if logit_parts else None
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

            # ⚠ `stable=True` IS LOAD-BEARING, not tidiness. This replaced
            # `sorted(scored_positions, key=scores.__getitem__, reverse=True)`, and Python's sort
            # keeps equal elements in their original (ascending-position) order. `topk`, and an
            # unstable sort, break ties arbitrarily — so on a row with repeated scores the
            # reported positions would drift between runs on the same input, which is the kind of
            # difference nobody notices and nobody can reproduce. Measured at 0.431 ms per probe
            # as a Python sort at 4k tokens, against 0.42 ms for the matvec it was sorting.
            order = torch.sort(score_row[scored_index], descending=True, stable=True).indices
            top = scored_index[order[:TOP_POSITIONS]].tolist()
            verdicts.append(
                Verdict(
                    scored=True,
                    score=value,
                    # ⚠ `None`, not `False`, when no threshold was placed: the probe ranks but
                    # does not decide, and reporting `False` would be a verdict it never gave.
                    fires=None if probe.threshold is None else value > probe.threshold,
                    n_scored_tokens=int(scored_index.numel()),
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
