"""The arm-time identity check, and the revision resolution the FTID got wrong.

A probe read in the wrong model's residual space produces numbers that are plausible, stable and
about nothing. There is no symptom, so the check is the only thing standing between an operator and
a monitor that reports confidently about noise.

`TestResolveRevision` is the interesting half. The specified mechanism — parse `cache_path` for
`snapshots/<40-hex>` — never matches on this deployment, so it would have reported
REVISION_UNVERIFIED forever while looking like a working check.
"""

from __future__ import annotations

import hashlib

import pytest

from millm.services.probe_identity import (
    REVISION_INCONSISTENT,
    REVISION_UNVERIFIED,
    LoadedIdentity,
    check_identity,
    resolve_revision,
)
from tests.unit.probe_fixtures import probe_definition

SHA_A = "0f604ada3f766f9f257460c4c9f0b5d6f69d431b"
SHA_B = "a09a35458c702b33eeacc393d103063234e8bc28"
TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"
TEMPLATE_SHA = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def loaded(**over) -> LoadedIdentity:
    base = dict(
        hf_id="LiquidAI/LFM2.5-1.2B-Instruct",
        d_model=8,
        n_layers=16,
        chat_template=None,
        revision=SHA_A,
        revision_source="download_metadata",
        supports_hooks=True,
    )
    base.update(over)
    return LoadedIdentity(**base)


def model_block(**over) -> dict:
    block = dict(probe_definition()["model"])
    block["chat_template_sha256"] = None  # opt in per test
    block.update(over)
    return block


def write_cache(tmp_path, commits: dict[str, str]):
    """Build a `local_dir` download layout with the given per-file commits."""
    d = tmp_path / ".cache" / "huggingface" / "download"
    d.mkdir(parents=True)
    for filename, sha in commits.items():
        (d / f"{filename}.metadata").write_text(f"{sha}\nblobhash\n1789303036.19\n")
    return tmp_path


class TestResolveRevision:
    def test_it_reads_the_commit_from_the_download_metadata(self, tmp_path):
        write_cache(tmp_path, {"config.json": SHA_A, "model.safetensors": SHA_A})
        sha, source = resolve_revision(str(tmp_path))
        assert sha == SHA_A
        assert source == "download_metadata"

    @pytest.mark.parametrize(
        "cache_path",
        [
            "/data/model_cache/huggingface/LiquidAI--LFM2.5-1.2B-Instruct--FP16",
            "/data/model_cache/huggingface/Qwen--Qwen2.5-7B-Instruct--FP16",
            "/data/model_cache/huggingface/bartowski--OLMo-2-1124-13B-Instruct-GGUF--Q4",
            "/data/model_cache/huggingface/kushalpatil--jevify-gemma4-e4b--Q8",
        ],
    )
    def test_the_hub_cache_segment_is_absent_from_every_real_cache_path(self, cache_path):
        """⚠ The FTID's mechanism, shown failing against the real paths.

        These four are the literal `cache_path` values on the node (2026-09-27). miLLM downloads
        with a `local_dir`, so none has the hub cache's `snapshots/<sha>` segment. Anything parsing
        for one finds nothing on every model and reports REVISION_UNVERIFIED — which reads as a
        working check that happens to have nothing to say.

        (The assertion deliberately does NOT use tmp_path: pytest names its temp directories after
        the test, so a test with that word in its name contains it in its own path. My first
        version did exactly that and failed on its own fixture.)
        """
        assert "/snapshots/" not in cache_path
        assert resolve_revision(cache_path) == (None, REVISION_UNVERIFIED)

    def test_disagreeing_files_are_reported_not_resolved(self, tmp_path):
        """A directory re-downloaded across two revisions is not any published revision.

        Reading one file would report a confident, wrong SHA.
        """
        write_cache(tmp_path, {"config.json": SHA_A, "model.safetensors": SHA_B})
        sha, source = resolve_revision(str(tmp_path))
        assert sha is None
        assert source == REVISION_INCONSISTENT

    def test_it_falls_back_to_the_row_when_the_row_holds_a_sha(self):
        assert resolve_revision(None, SHA_A) == (SHA_A, "model_row")

    def test_a_branch_name_in_the_row_is_not_a_revision(self):
        """`models.revision` stores what was REQUESTED, which may be a branch."""
        assert resolve_revision(None, "main") == (None, REVISION_UNVERIFIED)

    def test_metadata_wins_over_the_row(self, tmp_path):
        """The row says what was asked for; the metadata says what arrived."""
        write_cache(tmp_path, {"config.json": SHA_A})
        assert resolve_revision(str(tmp_path), SHA_B) == (SHA_A, "download_metadata")

    def test_nothing_at_all_is_unverified_not_an_error(self):
        assert resolve_revision(None, None) == (None, REVISION_UNVERIFIED)

    def test_a_non_sha_first_line_is_ignored(self, tmp_path):
        write_cache(tmp_path, {"config.json": "not-a-commit"})
        assert resolve_revision(str(tmp_path)) == (None, REVISION_UNVERIFIED)


class TestIdentityRefusals:
    def test_a_matching_model_passes(self):
        assert check_identity(model_block(), loaded()).ok

    def test_a_different_model_is_refused_by_name(self):
        report = check_identity(model_block(), loaded(hf_id="Qwen/Qwen2.5-7B-Instruct"))
        assert not report.ok
        assert report.mismatches[0]["field"] == "hf_id"
        assert report.mismatches[0]["actual"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_a_different_width_is_refused(self):
        report = check_identity(model_block(), loaded(d_model=4096))
        assert [m["field"] for m in report.mismatches] == ["d_model"]

    def test_a_different_depth_is_refused(self):
        report = check_identity(model_block(), loaded(n_layers=32))
        assert [m["field"] for m in report.mismatches] == ["n_layers"]

    def test_EVERY_mismatch_is_named_not_just_the_first(self):
        """Stopping at the first makes an operator fix one field, retry, and meet the next —
        never learning whether they loaded the wrong model or imported the wrong probe."""
        report = check_identity(
            model_block(), loaded(hf_id="other/model", d_model=4096, n_layers=32)
        )
        assert {m["field"] for m in report.mismatches} == {"hf_id", "d_model", "n_layers"}


class TestTheChatTemplate:
    def test_a_different_template_is_refused(self):
        """Same weights, different template = a different token stream. Every other field matches,
        which is what makes this one worth checking."""
        report = check_identity(
            model_block(chat_template_sha256="f" * 64), loaded(chat_template=TEMPLATE)
        )
        assert [m["field"] for m in report.mismatches] == ["chat_template_sha256"]

    def test_the_matching_template_passes(self):
        report = check_identity(
            model_block(chat_template_sha256=TEMPLATE_SHA), loaded(chat_template=TEMPLATE)
        )
        assert report.ok

    def test_a_pinned_template_against_a_model_that_has_none_is_refused(self):
        report = check_identity(model_block(chat_template_sha256="f" * 64), loaded())
        assert [m["field"] for m in report.mismatches] == ["chat_template_sha256"]

    def test_a_definition_that_pins_no_template_does_not_check_one(self):
        assert check_identity(model_block(), loaded(chat_template=TEMPLATE)).ok


class TestTheRevisionWarnsRatherThanRefusing:
    def test_an_unknown_revision_warns_and_allows(self):
        """Locked decision 10: warn and allow when miLLM cannot confirm the revision."""
        report = check_identity(
            model_block(revision=SHA_A), loaded(revision=None, revision_source=REVISION_UNVERIFIED)
        )
        assert report.ok
        assert REVISION_UNVERIFIED in report.warnings

    def test_an_inconsistent_checkout_says_so_specifically(self):
        report = check_identity(
            model_block(revision=SHA_A),
            loaded(revision=None, revision_source=REVISION_INCONSISTENT),
        )
        assert report.ok
        assert REVISION_INCONSISTENT in report.warnings

    def test_two_KNOWN_revisions_that_differ_ARE_refused(self):
        """Not knowing and knowing-they-differ are different situations.

        Decision 10 covers the first. The second is a definition fitted on a different commit of
        the same repo, which is a real mismatch.
        """
        report = check_identity(model_block(revision=SHA_A), loaded(revision=SHA_B))
        assert not report.ok
        assert [m["field"] for m in report.mismatches] == ["revision"]

    def test_a_definition_naming_a_branch_warns(self):
        report = check_identity(model_block(revision="main"), loaded())
        assert report.ok
        assert REVISION_UNVERIFIED in report.warnings


class TestTheEngine:
    def test_gguf_is_refused_before_anything_else_is_compared(self):
        """A GGUF model exposes no module tree, so there is nothing to hook and every other
        comparison is beside the point."""
        report = check_identity(model_block(), loaded(supports_hooks=False, hf_id="wrong/model"))
        assert not report.ok
        assert [m["field"] for m in report.mismatches] == ["engine"]
