"""A GGUF repo is a bag of mutually exclusive quantizations; read it as one.

The fixtures here are taken from REAL listings, because the two traps in this
code are both invisible on a convenient fixture:

  * every quantization on a 7B repo is a single file, so grouping looks
    unnecessary until a 70B repo splits one into numbered parts;
  * `bartowski/Qwen2.5-7B-Instruct-GGUF` ships exactly one lower-case filename
    among twenty-four, which an upper-case-only pattern leaves unlabelled.

MUTATION CONTROLS (each must turn this file red):
  * return files instead of grouped quants  -> "a split quant is ONE choice" fails
  * make the quant pattern case-sensitive   -> "reads a lower-case label" fails
  * detect GGUF via the Hub's `gguf` block  -> "detects from the file list" fails
  * sum `size` rather than the group total  -> "sums every part" fails
"""

from millm.ml.gguf_catalog import (
    GGUFFile,
    coarse_quantization,
    companion_files,
    group_gguf_files,
    has_gguf,
    is_companion,
    is_shard,
    quant_label_from_path,
)

# From bartowski/Qwen2.5-72B-Instruct-GGUF — Q6_K is genuinely two files.
SPLIT_LISTING = [
    GGUFFile("Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00001-of-00002.gguf", 34_000),
    GGUFFile("Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00002-of-00002.gguf", 30_350),
    GGUFFile("Qwen2.5-72B-Instruct-Q4_K_M.gguf", 47_000),
    GGUFFile("README.md", 12),
]


class TestReadingTheQuantLabel:
    def test_reads_ordinary_labels(self):
        assert quant_label_from_path("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
        assert quant_label_from_path("Qwen2.5-7B-Instruct-IQ2_M.gguf") == "IQ2_M"
        assert quant_label_from_path("Qwen2.5-7B-Instruct-Q8_0.gguf") == "Q8_0"

    def test_reads_a_lower_case_label(self):
        """Real repos mix cases. `-f16.gguf` sits beside `-Q4_K_M.gguf`."""
        assert quant_label_from_path("Qwen2.5-7B-Instruct-f16.gguf") == "F16"

    def test_ignores_the_shard_suffix(self):
        path = "A-Q6_K/A-Q6_K-00002-of-00002.gguf"
        assert quant_label_from_path(path) == "Q6_K"
        assert is_shard(path) is True

    def test_does_not_invent_a_label(self):
        """Returning None beats guessing; the caller falls back to the filename."""
        assert quant_label_from_path("README.md") is None
        assert quant_label_from_path("model.safetensors") is None

    def test_does_not_match_inside_the_model_name(self):
        """`Qwen2.5` starts with Q and must not read as a quantization."""
        assert quant_label_from_path("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == "Q4_K_M"


class TestGroupingIntoChoices:
    def test_a_split_quant_is_ONE_choice_that_sums_every_part(self):
        """The trap. Offering one part yields a model that cannot load."""
        quants = {q.label: q for q in group_gguf_files(SPLIT_LISTING)}

        assert set(quants) == {"Q6_K", "Q4_K_M"}
        q6 = quants["Q6_K"]
        assert len(q6.files) == 2, "the two parts must be ONE selectable quant"
        assert q6.is_split is True
        assert q6.total_size_bytes == 64_350, "the size shown must cover every part"

    def test_selecting_a_quant_names_every_file_it_needs(self):
        q6 = next(q for q in group_gguf_files(SPLIT_LISTING) if q.label == "Q6_K")
        assert q6.paths == [
            "Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00001-of-00002.gguf",
            "Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00002-of-00002.gguf",
        ], "ordered, so the first entry is the part llama.cpp is pointed at"

    def test_a_single_file_quant_is_not_marked_split(self):
        q4 = next(q for q in group_gguf_files(SPLIT_LISTING) if q.label == "Q4_K_M")
        assert q4.is_split is False
        assert len(q4.files) == 1

    def test_non_gguf_files_are_not_offered(self):
        assert all(".gguf" in p for q in group_gguf_files(SPLIT_LISTING) for p in q.paths)

    def test_every_gguf_file_lands_in_exactly_one_quant(self):
        """Accounting. A dropped file is a quant nobody can choose."""
        quants = group_gguf_files(SPLIT_LISTING)
        placed = [p for q in quants for p in q.paths]
        expected = [f.path for f in SPLIT_LISTING if f.path.endswith(".gguf")]
        assert sorted(placed) == sorted(expected)

    def test_an_unrecognised_name_stays_selectable(self):
        """Never silently drop a file: show it under its own name instead."""
        quants = group_gguf_files([GGUFFile("mystery-model.gguf", 5)])
        assert len(quants) == 1
        assert quants[0].quant_parsed is False
        assert quants[0].label == "mystery-model.gguf"

    def test_cheapest_first(self):
        labels = [q.label for q in group_gguf_files(SPLIT_LISTING)]
        assert labels == ["Q4_K_M", "Q6_K"]


# From mradermacher/gemma-4-31b-it-3MPER0RR-abliterated-GGUF, a VLM repo. The
# projector ships at two precisions, so its filenames carry quantization tokens
# and read exactly like quantizations.
MULTIMODAL_LISTING = [
    GGUFFile("m.Q8_0.gguf", 32_640_000_000),
    GGUFFile("m.mmproj-Q8_0.gguf", 810_000_000),
    GGUFFile("m.mmproj-f16.gguf", 1_200_000_000),
    GGUFFile("m.Q4_K_M.gguf", 18_690_000_000),
]


class TestCompanionFilesAreNotQuantizations:
    """A projector is not a model. Offering one is offering a broken download."""

    def test_a_projector_is_not_offered_as_a_quantization(self):
        labels = {q.label for q in group_gguf_files(MULTIMODAL_LISTING)}

        assert labels == {"Q8_0", "Q4_K_M"}
        assert "F16" not in labels, (
            "mmproj-f16.gguf is a 1.2 GB vision projector; offering it as the "
            "F16 of a 31B model presents something that cannot serve, at a size "
            "that reads like a bargain"
        )

    def test_a_projector_does_not_corrupt_the_quantization_it_shadows(self):
        """The nastier half: mmproj-Q8_0 parses to the SAME label as the model."""
        q8 = next(q for q in group_gguf_files(MULTIMODAL_LISTING) if q.label == "Q8_0")

        assert len(q8.files) == 1, "the projector must not be merged into the model"
        assert q8.total_size_bytes == 32_640_000_000, "size must be the model's alone"
        assert q8.is_split is False, (
            "a merged projector made this read as a 2-part split — "
            "indistinguishable in the UI from a genuinely sharded quantization"
        )

    def test_companions_are_reported_not_discarded(self):
        """A quant downloaded without its projector is silently text-only."""
        companions = [c.path for c in companion_files(MULTIMODAL_LISTING)]

        assert companions == ["m.mmproj-Q8_0.gguf", "m.mmproj-f16.gguf"]

    def test_an_ordinary_repo_has_no_companions(self):
        assert companion_files(SPLIT_LISTING) == []

    def test_is_companion_is_case_insensitive(self):
        assert is_companion("M.MMPROJ-F16.GGUF") is True
        assert is_companion("m.Q8_0.gguf") is False


class TestCoarseQuantization:
    """The DB bucket is a FUNCTION of the label, not an independent choice."""

    def test_maps_each_family_to_its_bucket(self):
        assert coarse_quantization("Q2_K") == "Q2"
        assert coarse_quantization("IQ2_XXS") == "Q2"
        assert coarse_quantization("Q4_K_M") == "Q4"
        assert coarse_quantization("Q3_K_M") == "Q4"
        assert coarse_quantization("Q6_K") == "Q8"
        assert coarse_quantization("Q8_0") == "Q8"
        assert coarse_quantization("F16") == "FP16"
        assert coarse_quantization("F32") == "FP32"

    def test_an_unknown_label_does_not_claim_a_quantization(self):
        """FP16 is the least wrong default; Q2 would understate memory badly."""
        assert coarse_quantization("mystery.gguf") == "FP16"
        assert coarse_quantization("") == "FP16"


class TestDetectingAGGUFRepo:
    def test_detects_from_the_file_list(self):
        """The file list is the authority, NOT HuggingFace's computed metadata.

        That block is derived by the Hub and can be absent on a repo that
        plainly holds .gguf files; branching on it would render such a repo as
        an ordinary one and offer a whole-repo download of every quant.
        """
        assert has_gguf([f.path for f in SPLIT_LISTING]) is True

    def test_an_ordinary_repo_is_not_gguf(self):
        assert has_gguf(["config.json", "model.safetensors", "tokenizer.json"]) is False
