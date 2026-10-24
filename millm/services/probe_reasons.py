"""The published list of probe result reason codes (FR-38.5, FTDD 038 §4.3, D38.9, D38.10).

A probe result that is not `completed` says WHY with one of these codes, and nothing else. The
detail that used to ride in the reason (`encoder_failed: <exception text>`,
`prompt_boundary_invalid: <n>`, `prompt_boundary_conflict: <a> then <b>`) goes to the local log
only: an exception message is not a contract value, it leaked to clients through the verdict
header, and on PostgreSQL a 200-character one overflowed `not_scored_reason String(64)` and lost
every event of the request (D38.32).

⚠ THE LIST IS OPEN WITHIN v1 (D38.10). A new code is added here, with its outcome and one line of
meaning, and documented; consumers read an unknown code as "other". Removing or renaming one needs
v2. `tests/unit/services/test_probe_result_outcomes.py` collects every literal passed to
`mark_not_scored`, `Verdict(not_scored_reason=…)`, `_probe_begin_detached` and
`set_last_user_span`, and every return of `last_user_token_span`, by AST over `millm/`, and fails
on any code missing here or on any f-string in those positions.
"""

from __future__ import annotations

#: code -> (outcome, one line of meaning).
REASON_TABLE: dict[str, tuple[str, str]] = {
    # ── skipped: the request was served, the probe did not look, and that is not a fault ──
    "judge_traffic": ("skipped", "a judge's own call (X-miLLM-Origin: judge); never user traffic"),
    "batched_request": ("skipped", "more than one row in the forward pass (n > 1 or a batch)"),
    "continuous_batching": ("skipped", "served by the continuous-batching engine, which has no hook"),
    "engine_unsupported": ("skipped", "served by an engine with no forward hooks (llama.cpp)"),
    "speculative_decoding": ("skipped", "a draft model proposed tokens; positions are not served"),
    "prompt_boundary_unknown": ("skipped", "the window needs the prompt boundary and none was stated"),
    "last_user_span_unknown": ("skipped", "the last_user window needs a span and none was stated"),
    "last_user_span_unresolved": ("skipped", "the newest user message could not be located"),
    "no_user_turn": ("skipped", "the conversation has no user message"),
    "no_user_header": ("skipped", "the template's user header could not be found in the ids"),
    "no_chat_template": ("skipped", "the tokenizer has no chat template to locate a turn with"),
    "text_completion_has_no_user_turn": ("skipped", "a text completion has no user message"),
    "no_scored_tokens": ("skipped", "the window selected no position the probe observed"),
    # ── failed: the probe tried and something was wrong ─────────────────────────────────
    "prompt_boundary_invalid": ("failed", "a negative prompt length was stated (detail in the log)"),
    "prompt_boundary_conflict": ("failed", "two different prompt lengths were stated (detail in the log)"),
    "encoder_failed": ("failed", "the k-sparse encoder raised (detail in the log)"),
    "head_mismatch": ("failed", "the head's width does not match the activations"),
    "non_finite_score": ("failed", "the score was NaN or infinite, so no score is reported"),
    # ── 039 S2: rule signals (`kind: "rule"`, FTDD §5.5) ─────────────────────────────────
    "rule_worker_unavailable": ("skipped", "operator code is scored off the live path and no "
                                "runner result is attached (D41)"),
    "rule_timeout": ("failed", "the request's rule time budget ran out before this signal ran"),
    "rule_error": ("failed", "a validator raised while the signal ran (detail in the log)"),
    "rule_set_vectors_failed": ("failed", "the rule set disagrees with its own cases under this "
                                "build, so it is not evaluated"),
    "rule_signals_disabled": ("failed", "rule signals are switched off (RULE_SIGNALS_ENABLED)"),
    # ── interrupted: the request did not finish ─────────────────────────────────────────
    "client_disconnected": ("interrupted", "the client went away before the stream finished"),
    "cancelled": ("interrupted", "the request was cancelled before it finished"),
    "generation_error": ("interrupted", "generation raised before it finished"),
    "generation_thread_hung": ("interrupted", "the generation thread did not stop in time"),
}

#: Every published code.
REASONS: frozenset[str] = frozenset(REASON_TABLE)

#: code -> outcome (`skipped`, `failed` or `interrupted`).
OUTCOME_OF: dict[str, str] = {code: outcome for code, (outcome, _m) in REASON_TABLE.items()}

#: The outcomes a result can have. `completed` has no reason.
OUTCOMES: tuple[str, ...] = ("completed", "skipped", "failed", "interrupted")

#: What a reason code may look like (the open schema's pattern for codes added later).
REASON_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
