# Served-render shared cases (2026-10-08) — pointer

The full record lives in miStudio:
`0xcc/reviews/served_render_cases_2026-10-08.md` (branch `fix/render-cases-and-band-count`).

This repo's part, on branch `feat/served-render-cases`:

- **`docs/schemas/served-render-cases.json`.** It is byte-identical to miStudio's copy and survives
  `sync-to-clean`, which keeps `docs/schemas/`.
- **`tests/unit/services/test_served_render_cases.py`.** It covers:
  - `served_render` and `ProbeInputPreparer.prepare`: ids, `generation_prompt`, `prompt_tokens`
    and the `last_user` span;
  - parity's `_drift` on the keep-tail truncation case;
  - byte identity with miStudio, under `MILLM_REQUIRE_CROSS_REPO_CHECKS` / `MISTUDIO_REPO`;
  - the real Llama-3.1 cases, which skip loudly without `MILLM_REAL_TOKENIZERS` or the HF cache.
- **`last-user-span-cases.json` gained two assistant-ended cases.** A new
  `test_the_scored_span_matches_the_shared_cases` runs every span case through the
  `/api/probes/score` preparer.

Controls on this side, every one killed, each restore verified by sha256 and `git diff`:

| ID | Mutation | Result |
|---|---|---|
| L1 | the generation prompt is always on | killed |
| L2 | a duplicate BOS (`rendered_chat_adds_special_tokens` always True) | killed |
| L3 | defect 2 reverted (the span placed over the generation-prompt render) | killed by the new scored-span test |
| L4 | the boundary is accepted without the prefix check | killed |
| B2 | one byte of the case file changed | killed by the identity test in both repos |
| L6 | one byte of the span-case file changed | killed by the identity test in both repos |

**Known divergence (miStudio's, not this repo's).** A chat template can write no BOS while its
tokenizer adds one, as TinyLlama-Chat's does. On that form this server serves one BOS and miStudio
trains on none. The case asserts the rule here, and miStudio runs it as a strict xfail pending an
operator decision.

Suite: `tests/unit` **5099 passed / 12 skipped / 0 failed**.
