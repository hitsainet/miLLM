# miLLM dependency review — 2026-08-23

Companion to the miStudio review of 2026-08-22, prompted by `google/gemma-4-*`
and widened to every pin and open alert.

## gemma-4 already worked, by accident

The deployed pod was already running transformers **5.15.0** with
`gemma4_unified` present, so gemma-4 loads today. But nothing asked for that
version. `pyproject.toml` said `transformers>=4.47.0`, and the image installs
whatever is newest at build time — so this project went 4.57 → 5.5 → 5.15
without a single commit recording it.

That is not a near miss to shrug at. transformers 5 removed `output_attentions`
from the decoder forward, and in the sibling repo that silently broke attention
capture: nothing raised until a capture indexed an empty tuple and died with
`tuple index out of range`. The same open `>=` was pointed at miLLM the whole
time.

**Found in passing, and worse:** the local venv was on transformers **5.0.0**
while production ran **5.15.0**. Fifteen minor versions apart — the test suite
was proving things about a version nobody deploys. The venv has been aligned to
5.15.1 and the suite re-run against it.

## Changed

| declaration | from | to |
|---|---|---|
| transformers | `>=4.47.0` | `>=5.15.1,<6` |
| huggingface-hub | `>=1.23.0,<2.0` | `>=1.28.0,<2.0` |
| safetensors | `>=0.5.0` | `>=0.8.0` (transformers 5.15's floor) |
| torch | `>=2.5.0` | `>=2.10.0,<3` |
| accelerate | `>=1.2.0` | `>=1.14.0,<2` |

Every major is now capped and every floor is a version that has actually been
run. A cap turns a surprise major into a resolver failure at build time, where
it is cheap, instead of a behaviour change in production.

`bitsandbytes` keeps a bare floor: it is pre-1.0, where a major cap says
nothing useful.

## npm

* `admin-ui`: `npm audit fix` — **7 → 0**, `package.json` untouched, so no
  direct dependency moved. tsc clean, 306 tests, build clean.
* `manual`: `npm audit fix` plus Docusaurus 3.9.2 → 3.10.2 (a minor within v3)
  — 25 → 18. Builds clean.

## Accepted, no fix available

**`image-size`** — all 18 remaining alerts trace to this one root, reached only
through `@docusaurus/mdx-loader`. `image-size@2.0.2` is the latest published
version and two advisories cover `<= 2.0.2` with `firstPatchedVersion: NONE`.
Build-time denial of service on a malformed ICNS/JXL/HEIF; reaching it needs a
crafted image committed to this repo, and nothing ships to a user at runtime.
Identical situation to miStudio's manual. Revisit when upstream patches.

## Verified

Suite **2191 passed** on transformers 5.15.1 — the 4 remaining failures are the
pre-existing performance timings and 17 e2e `/app` permission errors, unchanged
from before this review.

## Worth knowing

Dependabot reports **no Python alerts** for miLLM, and that is a consequence of
the open ranges rather than good health: with no version declared, there is
nothing for it to match an advisory against. The pins above should make Python
alerts start appearing — that is the point of them.
