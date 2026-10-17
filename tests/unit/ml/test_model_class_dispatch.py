"""Picking the right AutoModel class for each architecture family.

Everything defaulted to AutoModelForCausalLM. Loading an encoder classifier
therefore died with:

    Failed to load model: Unrecognized configuration class
    'transformers.models.modernbert.configuration_modernbert.ModernBertConfig'
    for this kind of AutoModel: AutoModelForCausalLM

followed by a 200-line dump of every config transformers knows — which is what
the user saw on screen when loading p-christ/ModernBERT-large-nli.

Configs below are the REAL ones, read from the checkpoints.
"""

from types import SimpleNamespace

import pytest

from millm.ml.model_loader import _get_auto_model_class


def _cfg(architectures, model_type):
    return SimpleNamespace(architectures=architectures, model_type=model_type)


class TestEncoderClassifiers:
    def test_modernbert_nli_gets_sequence_classification(self):
        """p-christ/ModernBERT-large-nli: 3 labels, entailment/neutral/contradiction."""
        cls = _get_auto_model_class(
            _cfg(["ModernBertForSequenceClassification"], "modernbert"))
        assert cls.__name__ == "AutoModelForSequenceClassification"

    @pytest.mark.parametrize("arch", [
        "BertForSequenceClassification",
        "RobertaForSequenceClassification",
        "DebertaV2ForSequenceClassification",
        "DistilBertForSequenceClassification",
    ])
    def test_other_classifier_families(self, arch):
        assert _get_auto_model_class(_cfg([arch], "bert")).__name__ == \
            "AutoModelForSequenceClassification"

    def test_token_classification_too(self):
        assert _get_auto_model_class(
            _cfg(["BertForTokenClassification"], "bert")).__name__ == \
            "AutoModelForSequenceClassification"


class TestExistingFamiliesAreUnchanged:
    def test_t5_still_seq2seq(self):
        assert _get_auto_model_class(
            _cfg(["T5ForConditionalGeneration"], "t5")).__name__ == \
            "AutoModelForSeq2SeqLM"

    @pytest.mark.parametrize("arch,mt", [
        ("GemmaForCausalLM", "gemma"),
        ("LlamaForCausalLM", "llama"),
        ("Gemma4ForConditionalGeneration", "gemma4"),   # multimodal generative
    ])
    def test_generative_models_are_not_diverted(self, arch, mt):
        cls = _get_auto_model_class(_cfg([arch], mt))
        assert cls.__name__ in (
            "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"), (
            f"{arch} is generative and must not be loaded as a classifier"
        )

    def test_an_unknown_architecture_still_defaults_to_causal(self):
        assert _get_auto_model_class(
            _cfg(["SomethingNovelForCausalLM"], "novel")).__name__ == \
            "AutoModelForCausalLM"

    def test_missing_architectures_field_does_not_raise(self):
        assert _get_auto_model_class(_cfg(None, "mystery")) is not None


class TestClassifiersAreRefusedForGeneration:
    """Loading correctly is necessary but NOT sufficient — a classifier has no
    lm_head, so a chat request must be refused rather than reaching it."""

    def test_nli_and_zero_shot_are_non_generative(self):
        from millm.api.routes.openai.errors import is_embedding_only
        assert is_embedding_only("zero-shot-classification")
        assert is_embedding_only("text-classification")

    def test_summarization_is_still_allowed_to_generate(self):
        """T5 DOES generate; only the output slicing was ever wrong."""
        from millm.api.routes.openai.errors import is_embedding_only
        assert not is_embedding_only("summarization")
