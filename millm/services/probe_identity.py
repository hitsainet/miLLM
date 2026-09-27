"""Is this probe's model the model that is loaded?

⚠ **PROBES REFUSE WHERE CIRCUITS BIND.** A circuit can be bound across an identity mismatch,
because a circuit names features a human can reason about. A probe cannot: its weights are a
direction in one specific model's residual space. Read in another model's space they produce
numbers that are plausible, stable, well-behaved — and about nothing. There is no symptom.

Five fields are compared. Four refuse; the fifth warns.

The `chat_template_sha256` comparison is the least obvious and the most valuable. Same weights plus
a different template is a different token stream, so the probe reads positions that do not mean
what it learned — and every other field would match.

## Resolving the revision

⚠ The FTID's plan does not work, and would have failed silently. It said to parse `cache_path` for
`snapshots/<40-hex>`. miLLM downloads with a `local_dir`, so a real `cache_path` is
`/data/model_cache/huggingface/{repo}--{QUANT}` with no `snapshots/` segment anywhere — the parse
would fall through to `REVISION_UNVERIFIED` on every model forever, which looks exactly like a
working check that keeps finding nothing to complain about.

`models.revision` is not a substitute either: it stores what the operator *requested*, which may be
a branch name, and is NULL on half the models on this node.

What does work is the HuggingFace download metadata. A `local_dir` download leaves
`.cache/huggingface/download/<file>.metadata` whose **first line is the resolved commit SHA**.
Measured 2026-09-27 across all four model directories on the node: every one internally consistent,
and the one model with a populated `revision` column agreed with it exactly.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Recorded, not refused: miLLM could not establish which commit is on disk.
REVISION_UNVERIFIED = "REVISION_UNVERIFIED"
#: Recorded, not refused: the files on disk came from more than one commit.
REVISION_INCONSISTENT = "REVISION_INCONSISTENT"

#: Files whose metadata is read to establish the commit. Two, deliberately — see `resolve_revision`.
REVISION_WITNESS_FILES = ("config.json", "tokenizer_config.json", "model.safetensors")


@dataclass(frozen=True)
class LoadedIdentity:
    """What the loaded model actually is, as far as miLLM can establish."""

    hf_id: str
    d_model: int
    n_layers: int
    chat_template: Optional[str] = None
    revision: Optional[str] = None
    revision_source: str = REVISION_UNVERIFIED
    supports_hooks: bool = True

    @property
    def chat_template_sha256(self) -> Optional[str]:
        if self.chat_template is None:
            return None
        return hashlib.sha256(self.chat_template.encode("utf-8")).hexdigest()


@dataclass
class IdentityReport:
    """Every mismatch, not the first one.

    Stopping at the first difference makes an operator fix one field, retry, and meet the next —
    and never learn whether they loaded the wrong model or imported the wrong probe.
    """

    mismatches: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def add(self, name: str, expected: Any, actual: Any) -> None:
        self.mismatches.append({"field": name, "expected": expected, "actual": actual})

    def as_details(self) -> dict[str, Any]:
        return {"mismatches": self.mismatches, "warnings": self.warnings}


def _read_metadata_commit(path: Path) -> Optional[str]:
    try:
        first = path.read_text(errors="replace").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return first if SHA_RE.match(first) else None


def resolve_revision(
    cache_path: Optional[str], row_revision: Optional[str] = None
) -> tuple[Optional[str], str]:
    """The commit actually on disk, and how it was established.

    Returns `(sha_or_None, source)` where source is `"download_metadata"`, `"model_row"`,
    `REVISION_UNVERIFIED` or `REVISION_INCONSISTENT`.

    ⚠ **Several witness files are read, not one.** A directory re-downloaded file-by-file across
    two upstream revisions holds files from different commits; reading one file would report a
    confident, wrong SHA. Disagreement is reported rather than resolved — the checkout is then not
    any published revision, which is worth saying out loud.
    """
    if cache_path:
        download_dir = Path(cache_path) / ".cache" / "huggingface" / "download"
        if download_dir.is_dir():
            found: dict[str, str] = {}
            for name in REVISION_WITNESS_FILES:
                sha = _read_metadata_commit(download_dir / f"{name}.metadata")
                if sha:
                    found[name] = sha
            if not found:
                # Fall through: some layouts name their files differently. Take any metadata file.
                for meta in sorted(download_dir.glob("*.metadata"))[:4]:
                    sha = _read_metadata_commit(meta)
                    if sha:
                        found[meta.name] = sha
            distinct = set(found.values())
            if len(distinct) == 1:
                return distinct.pop(), "download_metadata"
            if len(distinct) > 1:
                logger.warning(
                    "probe_identity_revision_inconsistent path=%s commits=%s", cache_path, found
                )
                return None, REVISION_INCONSISTENT

    if row_revision and SHA_RE.match(row_revision.strip()):
        return row_revision.strip(), "model_row"
    return None, REVISION_UNVERIFIED


def check_identity(model_block: dict[str, Any], loaded: LoadedIdentity) -> IdentityReport:
    """Compare a definition's `model` block against the loaded model.

    Four fields refuse on mismatch: `hf_id`, `d_model`, `n_layers`, `chat_template_sha256`.
    The revision **warns** rather than refuses when it cannot be established (locked decision 10) —
    but it REFUSES when both sides are known and disagree, which is a different situation entirely
    from not knowing.
    """
    report = IdentityReport()

    if not loaded.supports_hooks:
        # Checked first: a GGUF model exposes no PyTorch module tree, so there is nothing to hook
        # and every other comparison is beside the point.
        report.add("engine", "a hookable PyTorch model", "llama.cpp / GGUF")
        return report

    expected_hf = str(model_block.get("hf_id", ""))
    if expected_hf != loaded.hf_id:
        report.add("hf_id", expected_hf, loaded.hf_id)

    expected_d = model_block.get("d_model")
    if expected_d is not None and int(expected_d) != int(loaded.d_model):
        report.add("d_model", expected_d, loaded.d_model)

    expected_layers = model_block.get("n_layers")
    if expected_layers is not None and int(expected_layers) != int(loaded.n_layers):
        report.add("n_layers", expected_layers, loaded.n_layers)

    expected_template = model_block.get("chat_template_sha256")
    actual_template = loaded.chat_template_sha256
    if expected_template:
        if actual_template is None:
            # The definition pins a template and the loaded model has none: the probe was fitted on
            # a rendered conversation this model cannot reproduce.
            report.add("chat_template_sha256", expected_template, None)
        elif expected_template != actual_template:
            report.add("chat_template_sha256", expected_template, actual_template)

    expected_revision = (model_block.get("revision") or "").strip()
    if loaded.revision is None:
        report.warnings.append(
            REVISION_INCONSISTENT
            if loaded.revision_source == REVISION_INCONSISTENT
            else REVISION_UNVERIFIED
        )
    elif expected_revision and SHA_RE.match(expected_revision):
        if expected_revision != loaded.revision:
            report.add("revision", expected_revision, loaded.revision)
    elif expected_revision:
        # The definition names a branch or tag rather than a commit; there is nothing to compare.
        report.warnings.append(REVISION_UNVERIFIED)

    return report
