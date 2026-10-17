"""Which token positions a probe is allowed to score (FR-24.6).

A probe's `scope` is part of its identity, not a display option. A probe fitted on the prompt and
run over everything is a different detector: it would score the model's own output with weights
that never saw one.

The contract's vocabulary is **`all | prompt | response`**, and that is all a runtime needs:

    all       every position
    prompt    the input tokens, [0, n_prompt)
    response  what the model produced, [n_prompt, total)

⚠ **THIS IS DELIBERATELY SIMPLER THAN miSTUDIO'S INTERNAL SCOPES, AND THE WIDENING IS
miSTUDIO'S DECISION, NOT AN APPROXIMATION MADE HERE.** 032 works in
`all | assistant | user | last_assistant` and maps them down on export — `user` becomes `prompt`,
`assistant` and `last_assistant` both become `response`. Its own note records why: the mapping is
lossy in one direction only, because *"a consumer told `response` scores more tokens than the probe
was trained on rather than fewer. Refusing would be the alternative; widening is safe, narrowing
would not be."*

So a `prompt`-scoped probe here may score a system preamble that miStudio's `user` mask excluded.
That is the contract's intent. It also means a `prompt` or `response` probe's **test vectors cannot
be reproduced exactly** — see `probe_parity.py`, which reports that rather than hiding it.

The first version of this module implemented role spans by prefix-rendering the chat template,
because I took the scope vocabulary from a miStudio TypeScript type instead of from the frozen
schema. The contract has no role scopes, so none of that machinery could ever have run.
"""

from __future__ import annotations

from typing import Optional

SCOPES = ("all", "prompt", "response")
#: The windows a probe may report. Wider than `SCOPES`, which is a probe's IDENTITY (what it was
#: trained on): a window is a reading of the same weights. `last_user` (2026-10-04) is the newest
#: user message alone — `prompt` is every turn before the reply, so on a client that resends the
#: conversation a high-stakes earlier turn kept firing on every later one.
WINDOWS = ("all", "prompt", "response", "last_user")


def scored_mask(
    *,
    scope: str,
    n_prompt_tokens: int,
    n_generated: int = 0,
) -> list[bool]:
    """A per-position mask over `n_prompt_tokens + n_generated` positions.

    Never returns `None`: every contract scope is computable from the two counts alone, which is
    the practical benefit of the contract's vocabulary over a role-based one. There is no
    `role_mask_unreliable` path here — a chat template that cannot be decomposed into roles does
    not prevent the runtime from knowing where the prompt ended.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; known: {', '.join(SCOPES)}")
    if n_prompt_tokens < 0 or n_generated < 0:
        raise ValueError(
            f"token counts cannot be negative (prompt={n_prompt_tokens}, generated={n_generated})"
        )

    total = n_prompt_tokens + n_generated
    if scope == "all":
        return [True] * total
    if scope == "prompt":
        return [True] * n_prompt_tokens + [False] * n_generated
    return [False] * n_prompt_tokens + [True] * n_generated


def window_is_calibrated(probe_scope: str, window: str) -> bool:
    """Whether this probe's threshold means anything over this window.

    A threshold is the `(1 - target_fpr)` quantile of negatives **aggregated under one window**.
    Move the window and the quantile is a quantile of something else, so the same number no longer
    names the same false-positive rate.

    ⚠ THIS IS NOT PEDANTRY, AND THE ESTATE HAS ALREADY PAID FOR THE GENERAL VERSION OF IT. A
    high-stakes probe once fired on *"What is the capital of France?"* because its 1% budget had
    been spent on plain-prose negatives and then applied to chat. The recorded lesson was that
    *"a false-positive rate is a property of the negative distribution the monitor will actually
    see. Calibrating on one distribution and serving on another does not transfer, and nothing in
    the numbers says so."* A narrower window is a different negative distribution by exactly that
    argument.

    ⚠ AND FOR `response` IT IS WORSE THAN UNCALIBRATED — IT IS UNTRAINED. miStudio's training
    corpus is plain prose wrapped as a single user turn, and its capture refuses a response scope
    outright: *"no scored tokens at all under scope 'response'; every row's mask is empty, so
    there is nothing to train on"*. Weights fitted on user statements are being pointed at model
    output.

    A window this returns `False` for, with no bar of its own, still produces a verdict and still
    fires — the operator chose alerts now over silence on 2026-09-30 — but it is marked
    `provisional` everywhere it surfaces, which is the condition that choice was made under. A
    window with its own bar (`decision.windows`) is not marked for THIS reason; it can still be
    marked by `window_weights_trained`.
    """
    return window == probe_scope


def window_weights_trained(probe_scope: str, window: str) -> bool:
    """Whether this probe's WEIGHTS were fitted on the tokens this window reads.

    ⚠ A SECOND QUESTION FROM `window_is_calibrated`, AND CONFLATING THEM SHIPPED A FALSE CLAIM.
    miStudio now cuts a bar per window from that window's own negatives, which answers the
    calibration question — and retired the `provisional` marker on the `response` window, so its
    verdicts read as measured. But the weights behind them were fitted on plain prose sent as a
    single user turn: they never saw a model reply. A calibrated bar over an untrained readout is
    still a guess, and the UI's own copy says response verdicts are reported as provisional.

    THE CRITERION, stated rather than argued (review round 2, M4):

    * The probe's OWN scope is trained by definition. It is what the probe was fitted on, what
      its metrics and evidence rung describe, and the only window parity verifies. If an `all`
      probe fitted on prose then reads chat replies in its `all` window, that is a property of
      the PROBE, stated by its evidence, not a per-window caveat.
    * Another window counts as trained only when the probe is `all` and the window is `prompt`:
      the person's words are the kind of text an `all` probe was fitted on. The contract's
      `prompt` also holds the system preamble and earlier assistant turns, so on a multi-turn
      chat this is an approximation, accepted and named here.
    * `response` is untrained unless the probe's scope IS `response`. So is any other window of a
      non-`all` probe — a `prompt` probe's `all` window reads the reply too.

    A future response-trained probe declares `response`; nothing infers it.
    """
    if window == probe_scope:
        return True
    # Spelled exactly as the docstring states it, so a window name added later is untrained until
    # someone decides otherwise, rather than trained by omission (review round 3). `last_user`
    # was decided on 2026-10-04: the newest user message is a person's words, the kind of text an
    # `all` probe was fitted on.
    return probe_scope == "all" and window in ("prompt", "last_user")


def scope_is_reproducible(scope: str) -> bool:
    """Whether a test vector under this scope can be scored exactly from its `token_ids`.

    Only `all` can. A `prompt` or `response` vector's recorded `token_scores` came from
    miStudio's narrower role mask, and the contract carries no record of which positions those
    were — only the ids. See `probe_parity.py`.
    """
    return scope == "all"


#: Scopes the RUNTIME can actually honour, which is a different question from whether a scope's
#: test vectors can be reproduced.
#:
#: ⚠ THIS WAS `{"all"}` FROM 2026-09-28 TO 2026-09-29, BECAUSE `scored_mask` HAD NO PRODUCTION
#: CALLER. The function was complete and correct the whole time; nothing built a mask from it, and
#: `ProbeRequestContext` was constructed with `mask=None`, so `window(None, ...)` returned all-True
#: and every armed probe scored every position whatever its scope said. A `prompt`-scoped probe
#: would have scored the model's own output with weights that never saw one.
#:
#: It is now wired: `ProbeRequestContext.observe` builds a per-scope window from `scored_mask`,
#: and the prompt boundary is stated explicitly by whoever tokenized the prompt rather than
#: inferred from the first pass — inference would be right for an ordinary generation and
#: silently wrong under chunked prefill. A context never told the boundary reports
#: `prompt_boundary_unknown` rather than scoring the wrong window.
#:
#: ⚠ REPRODUCIBILITY IS A SEPARATE GATE AND HAS NOT MOVED. `scope_is_reproducible` is still
#: `all` only, because miStudio's `user` mask excludes a system preamble the contract's `prompt`
#: includes, so a narrower scope's recorded vectors cannot be replayed here at all. A `prompt`
#: probe can now be SCORED correctly and still cannot be PARITY-CHECKED, and `ProbeArmingService`
#: refuses it on that second ground. The two conditions were deliberately separated so widening
#: one could not silently widen the other; this commit widens exactly one.
#:
#: `test_probe_scope.py::TestRuntimeAdmissionTracksTheWiring` holds the pair together: it fails
#: if this set narrows while `scored_mask` has a caller, and fails if it widens while it does
#: not.
RUNTIME_SCORABLE_SCOPES = frozenset(SCOPES)


def scope_is_runtime_scorable(scope: str) -> bool:
    """Whether this runtime can restrict scoring to the positions this scope names.

    Distinct from `scope_is_reproducible`, which is about replaying recorded test vectors.
    """
    return scope in RUNTIME_SCORABLE_SCOPES


def window(mask: Optional[list[bool]], start: int, length: int) -> list[bool]:
    """The slice of `mask` covering one forward pass, padded CLOSED if it runs short.

    Padding open would score positions nobody vouched for.
    """
    if mask is None:
        return [True] * length
    taken = mask[start : start + length]
    if len(taken) < length:
        taken = taken + [False] * (length - len(taken))
    return taken
