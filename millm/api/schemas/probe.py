"""The ``mistudio.probe-definition/v1`` contract mirror.

miStudio owns this contract; the frozen schema is vendored at
``docs/schemas/probe-definition-v1.json`` and `tests/unit/api/test_probe_schema_sync.py` pins the
two together, byte-identically against miStudio when it is checked out.

``extra="allow"`` is deliberate on every contract model, following `circuit.py`: a newer miStudio
may emit additive v1 fields, and those must survive a round-trip rather than being silently
stripped. The typed fields below are the ones miLLM actually reads — identity, the read point, the
head, aggregation, the decision, the evidence. Everything else travels through untouched in the
stored `definition` JSON.

⚠ THE VALIDATORS HERE ARE NOT DUPLICATED FROM MISTUDIO FOR TIDINESS. Each one guards a failure
that is silent rather than loud, and a definition can reach miLLM by a route that never passed
through miStudio's endpoint — a hand-edited file, an older export, a third party. A contract that
only validates at the producer is a contract that is not validated.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

KIND = "mistudio.probe-definition/v1"

#: The hook point a probe may read. The contract admits exactly one: a probe fitted on the decoder
#: layer's output is not interpretable at a norm output, and miStudio has already shipped an
#: estate-wide defect from that exact confusion (every SAE trained before `4e334f28` learned a
#: normalised pre-MLP signal because "residual" resolved to `ffn_norm`).
HOOK_POINT = "resid_post"

RULES = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")
STREAMABLE_RULES = frozenset(RULES) - {"last"}
#: ⚠ THE CONTRACT'S VOCABULARY, NOT miSTUDIO'S INTERNAL ONE. miStudio's 032 works in
#: `all | assistant | user | last_assistant` and maps them down on export: `user` -> `prompt`,
#: `assistant` and `last_assistant` -> `response`. The mapping is lossy and deliberately widening —
#: a consumer told `response` scores more tokens than the probe was trained on, never fewer.
#: This was declared with the internal four until 2026-09-27, which would have REJECTED every real
#: document carrying `prompt` or `response`.
SCOPES = ("all", "prompt", "response")
#: The contract's distribution vocabulary, for evaluation entries.
DISTRIBUTIONS = ("in_distribution", "out_of_distribution")
#: The only authoritative input for reproducing a test vector.
AUTHORITATIVE_INPUT = "token_ids"
BASES = ("residual", "sae_features")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class ModelIdentity(_Contract):
    """What the probe was fitted on. Every field here is compared at arm time."""

    hf_id: str
    revision: str
    d_model: int
    n_layers: int
    architecture: str
    chat_template_sha256: str | None = None
    mistudio_model_id: str | None = None
    # The precision the producer's model ran at (miStudio 2026-10-03). None = NOT RECORDED —
    # the document predates the field. `probe_identity.check_identity` compares it to what this
    # server loaded; it is never read as "float16".
    load_dtype: str | None = None
    # The producer's model-row quantization (FP32/FP16/Q8/Q4/Q2). Identity alongside the
    # precision: Q4 and FP16 loads of one bfloat16 checkpoint share a load_dtype and read
    # different activations. None = not recorded.
    quantization: str | None = None


class ReadPoint(_Contract):
    layer: int = Field(ge=0)
    hook_point: str = HOOK_POINT

    @model_validator(mode="after")
    def _only_resid_post(self) -> "ReadPoint":
        if self.hook_point != HOOK_POINT:
            raise ValueError(
                f"hook_point must be {HOOK_POINT!r}, got {self.hook_point!r} — a probe fitted at "
                f"one point and read at another produces plausible scores in the wrong basis"
            )
        return self


class ProbeHeadSpec(_Contract):
    """The weights themselves.

    ⚠ `norm_std` must not contain a zero. miStudio's runtime zeroes a degenerate channel rather
    than dividing by a floor precisely because clamping a 0 to 1e-6 turns a 0.001 drift into
    1000.0; refusing it here means the situation never reaches the runtime at all.
    """

    weights: list[float] = Field(min_length=1)
    bias: float = 0.0
    norm_mean: list[float]
    norm_std: list[float]
    attention_query: list[float] | None = None

    @model_validator(mode="after")
    def _widths_agree_and_std_is_usable(self) -> "ProbeHeadSpec":
        width = len(self.weights)
        for name in ("norm_mean", "norm_std", "attention_query"):
            vector = getattr(self, name)
            if vector is not None and len(vector) != width:
                raise ValueError(
                    f"{name} has {len(vector)} entries but weights has {width} — a probe's "
                    f"normalisation must match its weights"
                )
        if any(value == 0.0 for value in self.norm_std):
            raise ValueError(
                "norm_std contains a zero; a zero standard deviation has no defined "
                "standardisation and clamping it silently amplifies drift"
            )
        return self


class Aggregation(_Contract):
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)
    streamable: bool

    @model_validator(mode="after")
    def _rule_is_known_and_streamable_is_derived(self) -> "Aggregation":
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; known: {', '.join(RULES)}")
        expected = self.rule in STREAMABLE_RULES
        if self.streamable != expected:
            raise ValueError(
                f"streamable={self.streamable} for rule {self.rule!r}, but only "
                f"{sorted(STREAMABLE_RULES)} can be computed incrementally — `last` is not defined "
                f"until the sequence ends"
            )
        return self


class Decision(_Contract):
    """`threshold: None` means NO threshold was placed — the probe ranks but does not decide.

    That is not the same as a threshold of 0, and a runtime that treats it as one turns a probe
    that has said nothing into one that fires on half its input.
    """

    threshold: float | None = None
    target_fpr: float | None = None
    realised_fpr: float | None = None
    threshold_source: str | None = None
    calibration: dict[str, Any] | None = None
    #: A threshold per contract window: `{all|prompt|response: {threshold, target_fpr, ...}}`.
    #:
    #: ⚠ `None` AND `{}` ARE DIFFERENT. `None` means the producer never calibrated per window, so
    #: a verdict on any window but the probe's own scope is judged against a bar cut for a
    #: different distribution and must be reported PROVISIONAL. `{}` would mean it tried and
    #: placed none. Only the first is a reason to assume the single `threshold` transfers.
    #:
    #: Mirrored explicitly rather than left to `extra="allow"`: this one is READ, and a field the
    #: runtime depends on should fail loudly on a shape change rather than silently vanish into
    #: an untyped bag. `circuit_contract_mirror_drift` is the recorded case of the opposite.
    windows: dict[str, dict[str, Any]] | None = None
    #: A threshold per ABSOLUTE token-length band, contiguous from 0, last one open-ended.
    #:
    #: ⚠ ONE CONSTANT THRESHOLD IS MISCALIBRATED AT EVERY LENGTH BUT THE ONE IT WAS CUT AT. A
    #: probe's score drifts with how many tokens were scored. miStudio measured a realised FPR
    #: of 5.4x its 1% budget on the longest quartile of one evaluation set, and recall falling
    #: 0.500 -> 0.297 on another — the direction that took a live monitor silent on turn four of
    #: a real conversation here.
    #:
    #: ⚠ `None` means the producer did not calibrate per length, and the runtime falls back to
    #: `threshold` — which is what every document written before 2026-10-02 carries. Mirrored
    #: explicitly for the same reason as `windows`: this field is READ, and a field the runtime
    #: depends on must fail loudly on a shape change rather than vanish into an untyped bag.
    length_bands: list[dict[str, Any]] | None = None


class Acknowledgement(_Contract):
    by: str
    at: str
    reason: str


class Evidence(_Contract):
    """The rung, and the words miLLM is allowed to use about it.

    `rung_language` travels with the definition rather than being derived here. miStudio owns the
    vocabulary; a local number→phrase map would be a second vocabulary free to drift, and the thing
    most likely to drift is a detector's language rising above its evidence.
    """

    rung: int = Field(ge=0, le=3)
    rung_language: str
    acknowledgement: Acknowledgement | None = None
    evaluations: list[dict[str, Any]] = Field(default_factory=list)
    judge: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _below_rung_two_must_be_acknowledged(self) -> "Evidence":
        if self.rung < 2 and self.acknowledgement is None:
            raise ValueError(
                "a probe below rung 2 carries no evidence that it detects anything on data "
                "unlike its training data, so the definition must record an acknowledgement; "
                "an endpoint gate alone is bypassed by a hand-edited file"
            )
        return self


class SaeReference(_Contract):
    """Where the k-sparse basis came from, and which columns of it this probe uses.

    `feature_indices` must be sorted and unique: each weight pairs with one feature BY POSITION,
    so a reordering silently re-points every weight at a different feature.
    """

    hf_repo: str
    path: str
    revision: str
    weights_sha256: str
    architecture: str
    d_model: int
    n_features: int
    normalization: dict[str, Any] = Field(default_factory=dict)
    feature_indices: list[int]
    feature_labels: list[Any] | None = None

    @model_validator(mode="after")
    def _indices_are_sorted_and_unique(self) -> "SaeReference":
        if list(self.feature_indices) != sorted(set(self.feature_indices)):
            raise ValueError(
                "feature_indices must be sorted and unique — each weight pairs with one feature "
                "by position, so a reordering re-points every weight at a different feature"
            )
        return self


class TestVector(_Contract):
    """One input with the scores miStudio recorded for it.

    ⚠ `token_ids` IS AUTHORITATIVE, not `messages`. miStudio's own acceptance measured `token_ids`
    reproducing at 0.000e+00 while re-rendering `messages` missed by up to 1.153, because
    `messages` is a reconstruction of plain prose as a single user turn and re-rendering adds
    template tokens. Parity scores from `token_ids`; `messages` divergence is reported separately
    as tokenization drift.
    """

    messages: list[dict[str, Any]]
    token_ids: list[int]
    token_scores: list[float]
    score: float
    verdict: bool | None = None


class TestVectors(_Contract):
    tolerance: float = Field(gt=0)
    authoritative_input: str = AUTHORITATIVE_INPUT
    messages_reproduce_token_ids: bool | None = None
    vectors: list[TestVector] = Field(min_length=1)


class ProbeDefinitionV1(_Contract):
    """The document miLLM imports."""

    kind: str = KIND
    name: str
    description: str | None = None
    concept: str | None = None
    model: ModelIdentity
    read: ReadPoint
    scope: str
    basis: str
    head: ProbeHeadSpec
    sae: SaeReference | None = None
    aggregation: Aggregation
    decision: Decision
    evidence: Evidence
    provenance: dict[str, Any] = Field(default_factory=dict)
    test_vectors: TestVectors
    # How the producer rendered a conversation into token ids (miStudio, 2026-10-08):
    # `{"generation_prompt": true, "add_special_tokens": <bool>, "bos_handling": {...}}`.
    # None = NOT RECORDED — the document predates the field, and for miStudio that means rendered
    # WITHOUT the generation prompt. Never read as the served form.
    #
    # `bos_handling` (additive, F1, 2026-10-08) = `{template_wrote_bos, tokenizer_added_bos,
    # bos_count}`: miStudio now tokenizes with THIS server's start-of-text rule
    # (`prompt_encoding.rendered_chat_ids`), and records what it did. `add_special_tokens` then
    # follows that rule (false when the template writes the BOS, true otherwise). A block WITHOUT
    # `bos_handling` was tokenized with add_special_tokens=false for every model: identical to the
    # rule when the template writes the BOS or the tokenizer adds nothing, one BOS short on a
    # TinyLlama-style template. Absent means not recorded — never "the tokenizer added one".
    # Kept as a dict: the vendored schema validates the shape, and parity re-renders a served-form
    # document by `probe_scoring.served_render`, which IS the rule, so it needs no field of its own.
    render: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _shapes_are_coherent(self) -> "ProbeDefinitionV1":
        if self.kind != KIND:
            raise ValueError(f"unknown kind {self.kind!r}; this importer reads {KIND!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}; known: {', '.join(SCOPES)}")
        if self.basis not in BASES:
            raise ValueError(f"unknown basis {self.basis!r}; known: {', '.join(BASES)}")

        if self.basis == "sae_features":
            if self.sae is None:
                raise ValueError("basis is sae_features but no `sae` block is present")
            if len(self.head.weights) != len(self.sae.feature_indices):
                raise ValueError(
                    f"head has {len(self.head.weights)} weights but the sae block selects "
                    f"{len(self.sae.feature_indices)} features — they pair by position"
                )
        else:
            if self.sae is not None:
                raise ValueError(
                    "basis is residual but an `sae` block is present; a definition that carries "
                    "both does not say which basis its weights are in"
                )
            if len(self.head.weights) != self.model.d_model:
                raise ValueError(
                    f"head has {len(self.head.weights)} weights but the model is "
                    f"d_model={self.model.d_model}"
                )

        if self.read.layer >= self.model.n_layers:
            raise ValueError(
                f"read.layer {self.read.layer} is outside a {self.model.n_layers}-layer model"
            )
        if self.aggregation.rule == "attention" and self.head.attention_query is None:
            raise ValueError(
                "the `attention` rule needs `head.attention_query`; without it the rule silently "
                "becomes `softmax` at tau=1, a different detector under the same name"
            )
        return self
