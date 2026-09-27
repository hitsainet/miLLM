"""`sae.normalization` is an OBJECT, and reading it as a string broke every k-sparse probe.

The field is `{"mode": "constant_norm_rescale", "source": "sae_metadata.training_hyperparameters"}`
— the mode, plus a record of where miStudio read it from. `_normalization_mode` passed the whole
dict through `str()`, so `SaeFeatureSlice` received the dict's repr and refused it as an unknown
mode. On hardware every one of the sixteen test vectors failed, and the reason reported was
`no_scored_tokens`.

⚠ **THE SLICE REFUSING IS THE DESIGN WORKING, AND IT IS WHY THIS WAS A FIVE-MINUTE DIAGNOSIS.**
A lenient loader would have fallen back to a default mode, encoded in the wrong basis, and
produced plausible features with different meanings — invisible in every metric. miStudio shipped
exactly that once: `sae_row.normalize_activations` silently guessed, and the default happened to
be right for the SAE it was found on.

Asserted against the REAL published fixture, because the shape of this field is precisely what a
hand-written one would get wrong — I wrote the bug by assuming it was a string.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from millm.core.errors import ProbeSaeMismatchError
from millm.services.probe_arm_bridge import _normalization_mode
from millm.services.probe_sae_slice import NO_NORMALIZATION, normalize

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "lfm2_probe_definition_sae.json"


class TestTheRealDocumentsField:
    def test_the_published_fixture_carries_an_OBJECT_not_a_string(self):
        """If this ever becomes a string upstream, the parser below can be simplified — but it
        must be a deliberate change, not a surprise."""
        block = json.loads(FIXTURE.read_text())["sae"]
        assert isinstance(block["normalization"], dict)
        assert block["normalization"]["mode"] == "constant_norm_rescale"

    def test_the_mode_is_read_out_of_it(self):
        block = json.loads(FIXTURE.read_text())["sae"]
        assert _normalization_mode(block) == "constant_norm_rescale"

    def test_the_result_is_a_mode_the_slice_ACCEPTS(self):
        """⚠ The assertion that would have caught the bug.

        Checking that a string comes back is not enough — `str(dict)` is a string. What matters
        is that `normalize()` accepts it, which is the contract between these two modules.
        """
        block = json.loads(FIXTURE.read_text())["sae"]
        import torch

        mode = _normalization_mode(block)
        normalize(torch.randn(4, 8), mode)  # raises ValueError on an unknown mode


class TestTheParserItself:
    def test_a_plain_string_still_works(self):
        """An older document needs no migration."""
        assert _normalization_mode({"normalization": "none"}) == "none"

    def test_a_missing_field_is_no_normalization(self):
        assert _normalization_mode({}) == NO_NORMALIZATION

    def test_an_object_with_no_mode_REFUSES_rather_than_defaulting(self):
        """Defaulting here is the miStudio defect exactly: a guess that is right often enough
        to survive review and wrong silently when it is not."""
        with pytest.raises(ProbeSaeMismatchError):
            _normalization_mode({"normalization": {"source": "somewhere"}})

    def test_the_dict_repr_is_never_returned(self):
        """The literal shape of the bug, pinned."""
        mode = _normalization_mode({"normalization": {"mode": "anthropic_rescale", "source": "x"}})
        assert mode == "anthropic_rescale"
        assert "{" not in mode and "'" not in mode


class TestAnEncoderFailureIsNamed:
    """⚠ A broken encoder must not report as "the probe never looked".

    The read hook swallows callback exceptions by design — a probe must never break generation
    — so an encoder that raises left the request with no scores and `finish()` said
    `no_scored_tokens`. That reason means the scope selected no positions. Reporting a crash as
    an empty scope sent me to the wrong question on hardware, and the actual traceback was only
    in the worker log.
    """

    def test_the_reason_names_the_encoder_and_the_error(self):
        import torch

        from millm.ml.probe_head import ProbeHead
        from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

        def exploding_encoder(_x):
            raise ValueError('unknown normalization mode "{\'mode\': ...}"')

        probe = ArmedProbe(
            probe_id="pr_k",
            name="k-sparse",
            head=ProbeHead(weight=torch.randn(4)),
            rule="mean",
            scope="all",
            layer=1,
            rung=2,
            rung_language="detects on unseen tasks",
            encoder=exploding_encoder,
        )
        context = ProbeRequestContext("req", [probe])
        context.observe(1, torch.randn(1, 6, 8))
        verdict = context.finish()[0]

        assert not verdict.scored
        assert verdict.not_scored_reason is not None
        assert "encoder_failed" in verdict.not_scored_reason, (
            f"an encoder crash reported as {verdict.not_scored_reason!r} — that reason means "
            "the scope selected no positions, which is a different thing entirely"
        )
        assert "normalization" in verdict.not_scored_reason, (
            "the reason does not carry the underlying error, so diagnosing it still needs the "
            "worker log"
        )


class TestTheCallerUsesIt:
    """⚠ A MUTATION SURVIVED HERE, and it was the caller that was uncovered.

    Reverting `build_probe_encoder` to `str(block.get("normalization"))` — the exact production
    defect — left all 36 tests above GREEN, because every one of them called
    `_normalization_mode` directly. A well-tested helper whose call site is tested by nothing is
    the shape this estate has now hit five times; the helper ships correct and unused.

    So this exercises `build_probe_encoder` end to end against the REAL published sae block, with
    a real (tiny) safetensors file on disk, and asserts the slice it returns carries a mode the
    encoder can actually run.
    """

    @pytest.mark.asyncio
    async def test_build_probe_encoder_produces_a_slice_that_ENCODES(self, tmp_path, monkeypatch):
        import torch
        from safetensors.torch import save_file

        from millm.core.config import settings
        from millm.services.probe_arm_bridge import build_probe_encoder

        block = dict(json.loads(FIXTURE.read_text())["sae"])
        d_model, k = 8, 4
        block["feature_indices"] = list(range(k))
        block.pop("weights_sha256", None)  # the tiny stand-in is not the published file

        target = tmp_path / block["hf_repo"].replace("/", "__") / block["path"]
        target.mkdir(parents=True)
        save_file(
            {
                "W_enc": torch.randn(d_model, 16),
                "b_enc": torch.zeros(16),
                "threshold": torch.zeros(16),
            },
            str(target / "sae_weights.safetensors"),
        )
        monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path), raising=False)

        class Row:
            basis = "sae_features"
            definition = {"sae": block}

        encoder = await build_probe_encoder(Row())
        assert encoder is not None
        # The mode came out of the OBJECT. `str(dict)` would leave the repr here.
        assert encoder.normalization_mode == "constant_norm_rescale"
        # And it must actually run — the assertion the direct helper tests cannot make.
        out = encoder(torch.randn(3, d_model))
        assert out.shape == (3, k)

    @pytest.mark.asyncio
    async def test_a_dense_probe_gets_no_encoder(self, tmp_path):
        """Specificity: if an encoder were built for everything, the test above proves nothing
        about the k-sparse path."""
        from millm.services.probe_arm_bridge import build_probe_encoder

        class Row:
            basis = "residual"
            definition = {}

        assert await build_probe_encoder(Row()) is None
