# Technical Implementation Document: Inline Steering and Steering-State Header
## miLLM Feature 28

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** `028_FPRD` v1.1, `028_FTDD` v1.0 · BRD-04 §5.8 · operator decisions T-78 – T-83,
P-22, X-07, X-09
**Load-bearing points verified against** miLLM `main` @ `f5c71b6` (2026-10-06) by read-only survey.
**Re-check every line number before editing.** `inference_service.py` is over 5,000 lines, and
Features 25, 26, 27, 29 and 30 edit it in the same increment.

Terms: SAE = sparse autoencoder; CBM = continuous batching manager; SSE = Server-Sent Events;
RFC 8941 = HTTP structured field values; GGUF = llama.cpp's model file format; AST = abstract syntax
tree.

---

## 1. Implementation Overview

Three pieces, built in this order so each can be tested before the next exists:

1. **A pure module** (`millm/core/steering_state.py`) for the hash, the canonical form and the
   header string. It ships first, because miDataworks 007 pins its test vectors (X-07).
2. **Request-scoped steering**: schema fields, an inline apply and an unsteered apply that reuse the
   existing per-layer restore.
3. **The report**: a snapshot of hook-visible state taken before restore, a reader that labels it,
   and the routes and stream chunk that publish it.

**Principles:**
- **Report what ran, never what was asked.** The reader's input is the snapshot. The request record
  only helps it choose a label, and only when the snapshot's values equal the record's.
- **One function per fact.** One hash function, one serialiser, one capture-then-restore helper.
  Header, stream chunk and batch line all call the same serialiser.
- **Refuse rather than degrade.** Every ambiguous selection, conflict and unsupported engine returns
  `400` naming the cause.
- **No hand-kept lists in tests.** Generation entry points are discovered from the service.

**Integration points:** `_admit()` (`inference_service.py:649`), `_apply_request_steering`
(`:1932`), `_restore_request_profile` (`:2149`), `_has_steering_override` (`:945-956`),
`reset_steering_memo` (`:333-349`), the probe stream chunk (`:2585-2611`), the chat route
(`millm/api/routes/openai/chat.py:101`) and the completions route
(`millm/api/routes/openai/completions.py:51`).

## 2. File Structure and Organization

**New files:**
```
millm/core/steering_state.py                  pure: SteeringItem, canonical_set_form, steering_set_hash,
                                               serialize_steering_header, encode_name, format_intensity
millm/services/steering_report.py             SteeringSnapshot, RequestSteeringRecord, SteeringReport,
                                               SteeringStateReader
tests/unit/core/test_steering_state.py        test vectors TV-1..TV-4, grammar, encoding
tests/unit/services/test_inline_steering.py   apply, validate, restore, epoch, CBM routing
tests/unit/services/test_inline_steering_real_model.py   tiny real Llama + real LoadedSAE equality
tests/unit/services/test_steering_report.py   labelling, honesty (report ≠ request), changed, unknown
tests/unit/services/test_steering_report_every_path.py   discovery over generation entry points
tests/integration/api/test_steering_header_routes.py     chat, completions, stream chunk, pre-load refusals
tests/support/generation_entry_points.py      AST call-graph discovery (shared with Feature 27 FR-27.9;
                                               whichever feature lands first creates it)
```

**Modified files:**
- `millm/api/schemas/openai.py`: `InlineSteeringFeature`, `InlineSteering`; `steering` on
  `ChatCompletionRequest` (after `steering_intensity`, `:128`); `profile`, `steering_intensity`,
  `steering` and their validators on `TextCompletionRequest` (after `:247`).
- `millm/services/inference_service.py`: contextvars beside `_PROBE_VERDICTS` (`:279`); reset in
  `reset_steering_memo`; `_has_steering_override`; `_apply_inline_steering`,
  `_apply_explicit_unsteered`, `_dispatch_request_steering`, `_finish_request_steering`; the restore
  branch (`:2206`); the four generation sites; the stream terminal chunk; the llama.cpp refusal
  (`:3885`); the scoring report; the CBM and llama.cpp reports.
- `millm/api/routes/openai/chat.py`: pre-load refusals; header after generation (`:280-297`).
- `millm/api/routes/openai/completions.py`: `reset_steering_memo()`; pre-load refusals; header after
  generation (`:147-148`).
- Docs: the API reference for `/v1/chat/completions` and `/v1/completions`, and
  `manual/docs/features/` (the OpenAI-API page that documents `profile` and `steering_intensity`).

**Import rules:** `millm/core/steering_state.py` imports only the standard library and
`millm.core.steering_range`. `steering_report.py` imports `core` and lazily imports services, as
the inference service already does (for example `:1170`), to avoid import cycles.

## 3. Component Implementation Hints

### 3.1 `millm/core/steering_state.py`

```python
CANONICAL_VERSION = "millm.steering-set/v1"

def applied_set(pairs: Iterable[tuple[int, float]]) -> dict[int, float]:
    """index → clamp_steering(strength), zeros (incl. -0.0) removed."""

def canonical_set_form(sae_id: str, applied: Mapping[int, float]) -> bytes:
    lines = [CANONICAL_VERSION, f"sae={sae_id}"]
    lines += [f"{i}:{struct.pack('>d', applied[i]).hex()}" for i in sorted(applied)]
    return "".join(f"{line}\n" for line in lines).encode("utf-8")

def steering_set_hash(sae_id: str, applied: Mapping[int, float]) -> str:
    return "sha256:" + hashlib.sha256(canonical_set_form(sae_id, applied)).hexdigest()
```

- `applied_set` drops zeros *after* clamping, and `+ 0.0` is not needed because `-0.0` compares
  equal to zero and is dropped.
- Refuse an `sae_id` containing LF (`ValueError`). Real IDs never do: they are repo paths with `/`
  replaced by `--` (`millm/ml/sae_downloader.py:321-326`).
- `SteeringItem` is a frozen dataclass: `kind` plus optional fields matching FTDD §5.2.
- `serialize_steering_header(items: Sequence[SteeringItem]) -> str` writes each kind's parameters in
  the FTDD §5.2 order. String values escape `\` and `"` (RFC 8941 §4.1.6) after percent-encoding has
  removed non-ASCII. A true boolean is written bare. It raises `ValueError` if `none` or `unknown` is
  combined with another member, because that is a bug, not data.
- `format_intensity(x: float) -> str` returns `repr(float(x))`.
- `encode_name(name: str) -> str` percent-encodes per FTDD §5.2.

### 3.2 `millm/services/steering_report.py`

```python
@dataclass(frozen=True)
class EntrySnapshot:
    sae_id: str; layer: int; enabled: bool; applied: dict[int, float]   # zeros removed

@dataclass(frozen=True)
class SteeringSnapshot:
    epoch: int; entries: tuple[EntrySnapshot, ...]; failed: bool = False

    @classmethod
    def capture(cls, state) -> "SteeringSnapshot": ...   # never raises

@dataclass(frozen=True)
class RequestSteeringRecord:
    kind: str                     # "inline" | "unsteered" | "profile" | "dial" | "none"
    epoch_at_admission: int
    sae_id: str | None = None; layer: int | None = None
    applied: dict[int, float] | None = None
    clamped: int = 0
    profile_name: str | None = None
    intensity: float | None = None

class SteeringStateReader:
    def __init__(self, inference): ...      # for _steering_circuit / plan / strict composition
    async def describe(self, snapshot, record) -> SteeringReport: ...   # never raises
```

**Labelling order** (FTDD §6.2): request record, then serving circuit, then active profile, then
`manual`. An entry counts as steered only when `enabled` is true and `applied` is non-empty. That is
exactly the condition the hook tests (`sae_wrapper.py:313`), except suppression, which is per-thread
and never active at capture time on the generation paths.

**Equality** is exact dict equality of float values. Both sides use the same formula
(`clamp_steering(float(v) * lam)`, `inference_service.py:2114`, `profile_service.py:444-447`). Do not
use `math.isclose`: an approximate match would let a different setting borrow a profile's name.

### 3.3 Inference-service seams

**`_dispatch_request_steering(request, request_id) -> dict | None`** is the one place that decides
which apply runs. The four generation sites call it instead of their inline `if request.profile or
...` blocks (`:3460-3465`, `:3629-3635`, `:4349-4356`, and the new text-completion block):

```python
steering = getattr(request, "steering", None)
if steering is not None:
    return (self._apply_explicit_unsteered(request_id) if not steering.features
            else self._apply_inline_steering(steering, request_id))
if getattr(request, "profile", None) or getattr(request, "steering_intensity", None) is not None:
    return await self._apply_request_steering(request.profile, request.steering_intensity,
                                              request_id=request_id)
return None
```

It also sets `_REQUEST_STEERING` with `epoch_at_admission`. It must be called *inside* `_admit()`.

**`_apply_inline_steering(steering, request_id) -> dict`**:
1. `entries = AttachedSAEState().entries()` (`sae_service.py:544-547`).
2. Select: by `sae_id` (all entries with that ID; zero → `SAENotAttachedError`, two or more →
   `ValidationError` naming each `(sae_id, layer)`); or, with no `sae_id`, exactly one entry,
   otherwise the same refusals.
3. Validate every index against `entry.sae.d_sae`, raising `InvalidFeatureIndexError`
   (`millm/core/errors.py:355-359`), before mutating anything.
4. `applied = applied_set(...)`; count the clamps with `would_clamp` (`steering_range.py:19-21`);
   log `inline_steering_values_clamped` when the count is positive.
5. Save `{"sae_id", "layer", "values": e.sae.get_steering_values(), "enabled":
   e.sae.is_steering_enabled}` for **every** entry, plus `epoch` and `request_id`.
6. Target: `clear_steering()`, `set_steering_batch(applied)`, `enable_steering(True)`. Others:
   `enable_steering(False)`.
7. Return `{"kind": "inline", "layers": saved, "epoch": ..., "request_id": ...}`.

If a mutation raises part-way, restore from the saved shape before re-raising, as the circuit apply
does (FTDD §6.3; `inference_service.py:1841`).

**`_apply_explicit_unsteered(request_id)`**: steps 5 and 7, then `enable_steering(False)` on every
entry. Return `None` when nothing is attached.

**`_restore_request_profile`**: change `if saved.get("circuit"):` (`:2206`) to
`if saved.get("layers") is not None:`. The skip log's `path` becomes
`saved.get("kind") or ("circuit" if saved.get("circuit") else "profile")`. Check every existing
producer of the circuit shape still sets `layers` (`_apply_request_circuit_steering`, `:1678`).

**`_finish_request_steering(saved) -> SteeringSnapshot`**: capture, then
`self._restore_request_profile(saved)`. It replaces the restore call at `:3506` and `:3734` and is
used in the new text-completion block. The streaming path calls `SteeringSnapshot.capture` beside
`_probe_finish` (`:4650-4653`) and keeps both restores (`:4450`, `:4762`).

**Report publication**, non-streaming: after `async with self._admit()` exits, call
`await SteeringStateReader(self).describe(snapshot, record)` and set `_STEERING_REPORT`. Streaming:
describe inside the generator, then yield the chunk after the probe chunk (`:4656-4659`) and before
`[DONE]` (`:4660`).

**Paths without a record:** scoring sets the report to `none` directly. CBM and llama.cpp capture at
entry (epoch only) and at exit, then describe with `record=None`.

## 4. Database Implementation Approach

N/A. No table, column, index or migration. Reads only: `ProfileRepository.get_active()` and the
circuit reads `_steering_circuit()` already makes.

## 5. API Implementation Strategy

### 5.1 Schemas (`millm/api/schemas/openai.py`)

```python
class InlineSteeringFeature(BaseModel):
    index: int = Field(ge=0)
    strength: float
    model_config = {"extra": "forbid"}
    # field_validator("strength", mode="before"): refuse bool; after: refuse non-finite

class InlineSteering(BaseModel):
    sae_id: Optional[str] = Field(default=None, min_length=1, max_length=100)  # SAE.id is String(100)
    features: list[InlineSteeringFeature]
    model_config = {"extra": "forbid"}
    # model_validator(after): duplicate index → error naming it; sae_id with [] → error
```

`extra="forbid"` inside `steering` is deliberate: a misspelt `strenght` must fail, not vanish. The
request models keep their own `extra` setting, which Feature 25 owns.

On both request models, a `model_validator(mode="after")` refuses `steering` with `profile`
(FR-28.2.1) and `steering` with `steering_intensity` (T-78), naming both fields. Copy the
`steering_intensity` validators (`openai.py:182-195`) to `TextCompletionRequest` unchanged.
Feature 25's `validate_scoring_mode` (`openai.py:258-281`) gains: any steering field with scoring
→ refused (FR-25.7.2, X-09). If Feature 25 has already added that rule, extend it rather than adding
a second one.

### 5.2 Routes

**Chat** (`chat.py:101`), before the auto-load block (`:143`):
- `steering` present and the model row is GGUF (`getattr(model, "gguf_files", None)`, the test
  `completions.py:89` uses) → `validation_error(..., param="steering")`.
- `steering.features` non-empty and the requested model is not the resident one → the
  `SAE_NOT_ATTACHED` error (T-82). Use the same resident test as the route's load guard
  (`chat.py:143-144`).

After generation, next to the probe header (`:293-297`):

```python
report = get_steering_report()
response.headers["X-miLLM-Steering"] = report.header if report else "unknown;reason=read_failed"
```

For streaming, also run the in-slot validations as a dry run before returning the
`StreamingResponse`, beside `ensure_profile_exists` (`:236-237`).

**Completions** (`completions.py:51`): call `reset_steering_memo()` at the top; add the same
pre-load checks before the auto-load block (`:112`); and set the header after `await inference.create_text_completion`
(`:148`). The `X-miLLM-Backend` line stays where it is.

### 5.3 Error patterns

Reuse existing errors: `SAENotAttachedError` (`errors.py:327-331`), `InvalidFeatureIndexError`
(`millm/core/errors.py:355-359`), `EngineUnsupportedError` (`errors.py:42`), `ValidationError` for ambiguous
selection. All are `MiLLMError` subclasses, so `openai_exception_handler` (`errors.py:121`) shapes
them. Each message names the field and the value (an ID, an index), never only the rule.

## 6. Frontend Implementation Approach

N/A. No Admin UI change (FPRD §4). Open WebUI is unaffected: it ignores unknown headers and skips a
`choices: []` chunk (024 FTASKS 0.2).

## 7. Business Logic Implementation Hints

- **Clamp before hash.** The hash is over `applied_set`, never the raw request (TV-4).
- **Zeros.** A zero strength is accepted, applied as nothing, excluded from `features` and from the
  hash.
- **Selection is by registry, not by `attached_sae`.** Never call `AttachedSAEState().attached_sae`
  in the new code. It is the first entry only (`sae_service.py:509-512`), which is the recorded
  profile defect.
- **`changed`.** Compare `record.epoch_at_admission` with `snapshot.epoch`. Inline and unsteered
  applies never bump the epoch, so they cannot set it themselves.
- **Circuit grouping.** Entries covered by one serving circuit collapse into one `circuit` item. An
  entry steered while the circuit plan does not cover its layer is labelled on its own.
- **Profile `source`.** `request` only via the record (the request named it). `active` only via
  `get_active()` with equal values.
- **`profile_sae_mismatch`.** When a `profile` item is produced and `profile.sae_id` is set and
  differs from the entry's `sae_id`, log a warning with the profile, both IDs and the layer. Do not
  change targeting (FPRD D27).

## 8. Testing Implementation Approach

**Fixtures:**
- The real-model tests copy the pattern of `tests/unit/services/test_scoring_completions.py`: a tiny
  `LlamaForCausalLM` (`:48-54`) and a real `LoadedSAE` (`:416`) attached through `AttachedSAEState`.
  Use two SAEs at two layers for the selection and "others disabled" tests.
- Build the inline set and the equivalent profile from **separate literals**, so the equality test
  cannot pass because both read one variable.
- Database-backed profile reads use the existing test database fixtures in `tests/conftest.py`.

**Honesty tests** (the operator's rule that the header be computed from what the hooks applied):
1. *Record lies, snapshot wins.* Call the reader with a record claiming inline `{5: 8.0}` and a
   snapshot holding `{5: 4.0}`. Assert `manual` and the hash of `{5: 4.0}`. Run the mutation "report
   from the record" as a negative control.
2. *Clamp shows.* Inline `{3: 500.0}` → `hash == TV-4`, `clamped=1`.
3. *Operator wins mid-request.* Inside a patched `_generate_sync`, call the operator steering route's
   service method (an authoritative, epoch-bumping write). Assert the header describes the new values
   and carries `changed`.
4. *Live global profile.* Activate a profile with λ = 0.75 through `ProfileService`, send a request
   with no steering field, and assert `profile;source=active;intensity="0.75"` with a hash computed
   independently in the test.

**Discovery** (`tests/support/generation_entry_points.py`): walk the AST of `InferenceService` and
build its call graph from `self.<name>(` calls. An entry point is any public or `_cbm_` /
`_llamacpp_` method reaching `_generate_sync` (`:5492`), `_generate_in_thread` (`:5516`), the CBM
backend's generate calls, or `_llamacpp_sync` (`:3849`). This is FR-27.9's definition. For each one,
drive it with the minimal stubs and assert a report was published. Exemptions (scoring, embeddings)
live in one dict with reasons, and a test asserts each exempt method reaches no generation primitive.
Match calls in the AST, never names in the source text, or comments will satisfy the guard.

**Header grammar:** parse the serialiser's output for every kind with `http-sfv` (dev dependency),
skipping with a loud reason if it is not installed, and assert the parsed kinds and parameters.

## 9. Configuration and Environment Strategy

N/A. No setting, flag or environment variable. The ±200 bound is `STEERING_RANGE`
(`steering_range.py:11`) and is not made configurable here. Add `http-sfv` to the dev or test
extras only, pinned as the consumer pins it (`0.9.9`).

## 10. Integration Strategy

- **Feature 25.** Flip the `steering` row of its outcome table (`025_FPRD` FR-25.3) to "honoured" for
  transformers chat and completions in the same change that adds the fields. Its test asserting
  "`steering` refused" on those endpoints changes to the 028 behaviour; scoring and embeddings keep
  refusing.
- **Feature 26.** Exposes `steering_report_for_row(snapshot, record)` returning the header string.
  Feature 26 owns where the string goes in the output line.
- **Feature 27.** Shares `tests/support/generation_entry_points.py`.
- **Backwards compatibility.** `X-miLLM-Steering-Intensity` and `X-miLLM-Circuit-Rung` are
  unchanged. Requests without `steering` behave exactly as before, except that they now carry the
  header.
- **miStudio 034 / miDataworks 007.** No code here. Publish the test vectors in the API reference
  in phase 1 so they can pin them.

## 11. Utilities and Helpers Design

- `millm/core/steering_state.py` is the reusable unit. Nothing else formats a steering hash or the
  header.
- `SteeringSnapshot.capture` is the only reader of live steering for reporting.
- `_dispatch_request_steering` and `_finish_request_steering` are the only apply and finish entry
  points at generation sites. A test asserts, by AST, that no generation method calls
  `_apply_request_steering` or `_restore_request_profile` directly any more, except the streaming
  restores (`:4450`, `:4762`), which are named in that test with their reason.

## 12. Error Handling and Logging Strategy

| Event | Level | Fields |
|---|---|---|
| `request_inline_steering_applied` | info | request_id, sae_id, layer, features, clamped |
| `inline_steering_values_clamped` | warning | request_id, sae_id, indices (first 20) |
| `request_unsteered_applied` | info | request_id, entries |
| `steering_report_unknown` | warning | request_id, reason, error |
| `profile_sae_mismatch` | warning | profile, profile_sae_id, applied_sae_id, layer |
| `request_restore_skipped_superseded` | info (existing, `:2186`) | path now includes `inline` / `unsteered` |

`describe` and `capture` never raise into a request; they degrade to `unknown` with a reason
(FR-28.3.7). An apply failure is a client error (4xx) and is raised, as the profile path does.

## 13. Performance Implementation Hints

- Capture is O(entries × features), a few microseconds.
- Reuse `_steering_circuit()`'s contextvar memo (`inference_service.py:1532-1537`). Do not add a
  second circuit read.
- Read the active profile once per request, only when an entry is steered and neither the record
  nor the circuit claimed it.
- Measure: one benchmark in `tests/performance/` timing 200 serial requests with and without the
  report (target p95 delta under 5 ms). That benchmarks the path this feature changed.

## 14. Code Quality and Standards

- Black (line length 100), Ruff, MyPy strict on the new modules.
- Comments state *why*: cite T-79/T-80 at `enable_steering(False)`, X-07 at the canonical form,
  RFC 8941 §3.3.2 at `format_intensity`.
- No `getattr(..., default)` on fields this feature owns: read `request.steering` directly on both
  request types once they declare it.
- No source-scrape guards. Wiring guards use the AST or the live call path.
- Every wiring line gets a removal test asserting payload and call count (CLAUDE.md reachability
  rule).

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is answered from a cited source.

| # | Question | Answer | Source |
|---|---|---|---|
| ID1 | File organisation? | Pure core module plus one service module; existing files extended | Existing split (`millm/core/steering_range.py`, `millm/services/*`) |
| ID2 | One dispatcher or per-site `if` blocks? | One dispatcher | Four sites duplicate the condition today (`:3461`, `:3630`, `:4350`); a fifth would drift |
| ID3 | Restore reuse? | Branch on `layers` | FTDD TD1 |
| ID4 | Validation location? | Schema for shape; slot for live state | FTDD §4 |
| ID5 | `extra` inside `steering`? | `forbid` | A misspelt key inside an output-changing object must not vanish (R-04.3 spirit) |
| ID6 | Equality for labelling? | Exact | FTDD TD4; identical formula on both apply paths |
| ID7 | Discovery helper ownership? | Shared with Feature 27 | FR-27.9 definition reused |
| ID8 | RFC 8941 parser for tests? | `http-sfv` 0.9.9, dev only | 007 FTDD TQ7 (the consumer's parser) |
| ID9 | Streaming capture point? | Before the final chunk | `inference_service.py:4650-4660` (the `finally` runs after the stream closed) |

**Open items:**
1. FPRD Open Question 1 (first-SAE profile targeting) stays with the operator.
2. Whether a dev-only `http-sfv` needs a PADR §5 note (FTDD §14). **Resolved (Stage 3, 2026-10-06):**
   PADR §5 lists it as dev-only.
