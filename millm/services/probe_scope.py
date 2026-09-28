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
#: ⚠ **THIS IS `{"all"}` BECAUSE `scored_mask` HAS NO PRODUCTION CALLER, NOT BECAUSE THE OTHER TWO
#: ARE UNCOMPUTABLE.** `scored_mask` above is complete and correct and needs nothing but the two
#: token counts — but nothing builds a mask from it, and `ProbeRequestContext` is constructed at
#: `probe_runtime.py` with `mask=None`, so `window(None, ...)` returns all-True. A `prompt`-scoped
#: probe armed today would score the model's own output with weights that never saw one, which is
#: precisely what this module's opening paragraph says makes it a different detector.
#:
#: Until 2026-09-28 the only thing stopping that was the parity gate incidentally refusing
#: non-reproducible scopes. That is protection by coincidence: it would evaporate the moment
#: parity learned to verify a `prompt` vector, and the arming path would open onto a runtime that
#: still scores everything. So the refusal is now primary and keyed on runtime capability.
#:
#: The two conditions are genuinely independent and both must widen on their own evidence —
#: reproducibility cannot be fixed here at all, because miStudio's `user` mask excludes a system
#: preamble that the contract's `prompt` includes (see the module docstring).
#:
#: `test_probe_scope.py::TestRuntimeAdmissionTracksTheWiring` fails if this set widens while
#: `scored_mask` is still uncalled, and fails if `scored_mask` gains a caller while this set stays
#: narrow — so the constant cannot drift away from the code in either direction.
RUNTIME_SCORABLE_SCOPES = frozenset({"all"})


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
