# Probe served-render follow-ups — 2026-10-08

Branch `fix/probe-render-followups` from `origin/main` `1f33643`. Two miLLM defects were found
while moving probes onto miStudio's served render. Each definition now carries an optional
`render` block, `{"generation_prompt": true, "add_special_tokens": false}`. Neither defect changed
a verdict.

## Defect 1: parity's `messages` round-trip reported 0 of 16 on documents that reproduce

Seen live on `pr_f90227264893` and `pr_7de1e85e2d16`. miStudio's own build-time check on the same
documents records `messages_reproduce_token_ids: true`.

There were **two** causes, and either one alone was enough to cause the 0 of 16:

1. Every vector was re-rendered with `add_generation_prompt=False`. A served-render user-ended
   vector carries the generation prompt.
2. **The older cause:** the round-trip called `apply_chat_template(..., tokenize=True)`. On
   transformers 5 (pinned `>=5.15.1`) that returns a `BatchEncoding`, so `list()` of it is
   `['input_ids', 'attention_mask']`, which never equals any recorded ids. So the round-trip has
   reported 0 matched on **every** definition since transformers 5. That includes old
   no-`render` documents whose assistant-ended vectors do reproduce without the generation prompt
   (7 of 16 on `pm_f736aa73969d`, measured).

**Fix (`7920804`)**
- `probe_scoring.served_render` is now the one place the served-render rule lives in miLLM. It
  was extracted from `ProbeInputPreparer.prepare` and keeps the same branches:
  - for an assistant-ended conversation, it renders without the generation prompt and takes the
    boundary from a render with the generation prompt, accepted only when that render is a prefix;
  - otherwise it renders with the generation prompt;
  - in both cases it tokenizes with `rendered_chat_ids`, which gives exactly one BOS.
- `/api/probes/score` and parity's `_drift` both call `served_render`. The AST guard
  `test_prompt_encoding_guard.py` now requires both calls, plus `served_render` → `rendered_chat_ids`.
- **When the `render` block is absent**, the old no-generation-prompt render is kept, now
  tokenized correctly. The report carries `render: null`, `rendered_with: "no_generation_prompt"`
  and a `render_note` saying the document predates render recording. An absent block is never read
  as the served form. A block whose `generation_prompt` is not `true` is treated the same way.
- **Truncation.** miStudio records no per-vector truncation flag. Its `truncate_tokens` keeps the
  TAIL at 1,024 tokens, and its own round-trip compares the tail of the re-render. When a vector's
  recorded ids are exactly the end of a longer re-render, miLLM now lists it in
  `truncated_vectors` and adds a `truncation_note`. It is no longer counted as a mismatch. A head
  cut, or a tail with a wrong token, is still a mismatch.
- Report keys:
  - unchanged: `checked`, `messages_reproduce_token_ids` (exact matches), `mismatched_vectors`,
    `note`;
  - added: `truncated_vectors`, `mismatched`, `render`, `rendered_with`, `render_note`,
    `truncation_note`.

**Measured on the real definitions** (Llama-3.1 tokenizer, template sha `e10ca381…`, the hash the
definitions record):

| Definition | Exact | Truncated | Mismatched |
|---|---|---|---|
| `pm_f736aa73969d` (high-stakes) | 14 | 10, 15 | 0 |
| `pm_1b1f8c50d6d7` (humor) | 16 | none | 0 |

## Defect 2: `last_user` never resolved for an assistant-ended `/api/probes/score` input

`last_user_token_span` placed the span by matching the served ids against a full render that was
hard-coded `add_generation_prompt=True`. Scoring serves an assistant-ended input without the
generation prompt, so the prefix match always failed and the result was
`last_user_span_unresolved`. Live chats were unaffected: they are always rendered with the
generation prompt, including when the last turn is an assistant turn.

**Fix (`4f756c3`)**
- `generation_prompt` is now a **required** keyword-only parameter. It has no default, because a
  default is how a caller silently ends up with the other render.
- Scoring passes `served_render(...).generation_prompt`. The live hook passes `True` explicitly,
  with a comment saying why. Deriving it from the last role there would break live
  assistant-ended chats, and a test pins that.
- The span itself is unchanged and is still miStudio's construction: renders of message prefixes
  without the generation prompt, starting at the newest user message's own header. miStudio's
  `served_render` makes the same generation-prompt decision branch for branch.

## Mutation controls

Method: back up the file, apply a single-occurrence edit, run the target test file, restore, then
check the sha256 against the original and grep for the mutated line. All 14 controls went **RED**,
so there were no survivors and no negative-control re-runs were needed.

| ID | File | Mutation | Caught by |
|---|---|---|---|
| M1 | probe_parity | `served_form = False` (ignore the block) | served-render reproduce test |
| M2 | probe_parity | absent block read as served | absent-block test |
| M3 | probe_parity | engine does not pass `definition["render"]` (wiring) | served-render reproduce test |
| M4 | probe_parity | always render with the generation prompt (drop `served_render`) | assistant-ended vector |
| M5 | probe_parity | truncation branch removed | keep-tail test |
| M6 | probe_parity | truncation accepts the HEAD | keep-tail test |
| M7 | probe_scoring | `served_render` tokenizes with the default specials, so the BOS is duplicated (reverts the previous round's one-BOS fix) | served-render reproduce test |
| M8 | probe_turns | `served_generation_prompt` always True | served-render reproduce test |
| M9 | probe_parity | truncated counted as matched, not named | keep-tail test |
| N1 | probe_turns | span full render hard-coded True again (the defect) | assistant-ended span test |
| N2 | probe_scoring | preparer passes `generation_prompt=True` | assistant-ended span test |
| N3 | inference_service | live hook derives the flag from the last role | live assistant-ended test |
| N4 | probe_turns | default `= True` reinstated | no-default test |
| N5 | probe_scoring | `served_render` reports `generation_prompt=True` always | assistant-ended span test |

One helper false alarm, recorded so nobody mistakes it for a real problem. M9's replacement,
`matched += 1`, already occurs elsewhere in the file, so the helper's "mutated text absent after
restore" grep flagged a failed restore. Verifying by hand showed sha256 `d9ed29c235a0` equal to the
original and the original line present once. The restore was correct; the grep was wrong.

## Tests and suite

- New test files:
  - `tests/unit/services/test_probe_parity_render.py`: 10 WordLevel tests plus 3 tests on real
    definitions;
  - `tests/unit/services/test_probe_last_user_assistant_ended.py`: two tokenizer variants plus a
    real-tokenizer test.
- New fixture: `tests/fixtures/probe_served_render_definitions.json`, reduced from the two real
  definitions (messages and token_ids only).
- The real-tokenizer tests find a tokenizer **by template hash**, via `MILLM_REAL_TOKENIZERS` or
  the HuggingFace cache. They skip loudly when none is found; CI has none.
- Existing tests updated:
  - `TestTokenizationDrift` (its fake now renders text, then tokenizes);
  - 12 `last_user_token_span` call sites now pass `generation_prompt=True`.
- `tests/unit`, serial run with the real tokenizer available: **5064 passed, 12 skipped, 0 failed**.

## Hardware check after deploy (operator)

1. Re-run parity on `pr_f90227264893`. Expect `tokenization_drift.messages_reproduce_token_ids` = 14,
   `truncated_vectors` = [10, 15] and `mismatched` = 0.
2. Re-run parity on `pr_7de1e85e2d16`. Expect 16 matched, with `truncated_vectors` and
   `mismatched` both empty.
3. Send `/api/probes/score` an assistant-ended `messages` input with `windows` including
   `last_user`. Expect the `last_user` verdict to be scored, with no `not_scored_reason` of
   `last_user_span_unresolved`.
