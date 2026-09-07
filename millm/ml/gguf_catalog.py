"""Read a HuggingFace repo listing as a catalogue of GGUF quantizations.

A GGUF repo is a bag of MUTUALLY EXCLUSIVE quantizations of one model — twenty
of them is ordinary. You want exactly one. That makes two things true that the
rest of the model pipeline does not expect:

1. The unit a person chooses is a QUANTIZATION, not a file. On 70B-class repos
   the larger quants are split into numbered parts inside a per-quant
   subdirectory::

       Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00001-of-00002.gguf
       Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00002-of-00002.gguf

   Downloading one part yields a directory that looks populated and a model that
   cannot load. Every quant on a 7B repo happens to be a single file, so a
   fixture built from one would pass whether or not this grouping exists.

2. ``Q4`` is not a quantization. ``Q4_K_M``, ``Q4_K_S`` and ``Q4_0`` are three
   different files with different sizes and different quality, and the coarse
   ``QuantizationType`` enum cannot tell them apart.

Sizes here are MEASURED — HuggingFace returns exact per-file byte counts — never
estimated from a parameter count. For a mixed-precision GGUF quant a
bytes-per-parameter estimate is meaningless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# A GGUF quantization token, as it appears in a filename.
#
# Anchored on both sides so it cannot match inside a word: without the leading
# boundary this finds "Q4" inside a hypothetical "SEQ4", and model names are not
# a controlled vocabulary. "Qwen2.5" is safe only because a digit must follow
# the Q — which is exactly the kind of coincidence worth pinning in a test
# rather than relying on.
#
# CASE-INSENSITIVE, and the label is upper-cased on the way out. Found by
# running this against the real listing rather than a handwritten fixture:
# `bartowski/Qwen2.5-7B-Instruct-GGUF` ships `Qwen2.5-7B-Instruct-f16.gguf` in
# lower case while every other file in the same repo is upper. An
# upper-case-only pattern left exactly one of twenty-four quants unlabelled.
_QUANT_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])("
    r"I?Q\d+(?:_[A-Z0-9]+)*"  # Q4_K_M, Q8_0, IQ4_XS, IQ2_XXS, Q6_K
    r"|BF16|F16|F32|FP16|FP32"  # unquantized GGUF is still a choice
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# The shard suffix llama.cpp writes: "-00001-of-00002".
_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$")

GGUF_SUFFIX = ".gguf"


@dataclass(frozen=True)
class GGUFFile:
    """One `.gguf` blob in a repo, with its TRUE size.

    ``size_bytes`` comes from the Hub's file metadata. For an LFS file the
    pointer file is a few hundred bytes and the real size lives under ``lfs``;
    reading the wrong one understates a 40 GB download as trivial.
    """

    path: str
    size_bytes: int


@dataclass(frozen=True)
class GGUFQuant:
    """One quantization: the unit a person selects and the unit that downloads.

    ``files`` is ordered by path, so a sharded quant reads 00001, 00002, … and
    the first entry is the one llama.cpp is pointed at.
    """

    label: str
    files: tuple[GGUFFile, ...]
    total_size_bytes: int
    is_split: bool
    quant_parsed: bool

    @property
    def paths(self) -> list[str]:
        """Every path this quant needs. Downloading a subset is a broken model."""
        return [f.path for f in self.files]


def quant_label_from_path(path: str) -> str | None:
    """Extract the quantization token from a `.gguf` path, or None.

    Returns None rather than guessing. A caller that needs something to show
    falls back to the filename stem, which is at least true.
    """
    name = path.rsplit("/", 1)[-1]
    if not name.lower().endswith(GGUF_SUFFIX):
        return None
    stem = name[: -len(GGUF_SUFFIX)]
    stem = _SHARD_SUFFIX.sub("", stem)

    # LAST match, not first: the quant token is a suffix on the model name, and
    # a model name may legitimately contain something token-shaped.
    matches = _QUANT_TOKEN.findall(stem)
    # Upper-cased so `f16` and `F16` are ONE quantization rather than two rows
    # offering the same download.
    return matches[-1].upper() if matches else None


def is_shard(path: str) -> bool:
    """Whether this path is one numbered part of a split quantization."""
    name = path.rsplit("/", 1)[-1]
    if not name.lower().endswith(GGUF_SUFFIX):
        return False
    return bool(_SHARD_SUFFIX.search(name[: -len(GGUF_SUFFIX)]))


def _group_key(path: str) -> tuple[str, bool]:
    """The identity of the quantization a file belongs to.

    Sharded quants live in their own directory, so the directory is the
    authority when there is one — two different quants never share a directory,
    while two files CAN parse to the same label (a stray copy at the repo root
    beside a sharded directory of the same name). Falling back to the parsed
    label keeps single-file quants at the repo root grouped correctly.
    """
    label = quant_label_from_path(path)
    directory = path.rsplit("/", 1)[0] if "/" in path else ""
    if directory:
        return (directory, True)
    if label:
        return (label, False)
    # Unparseable and at the root: it is its own group, named by its stem, so it
    # remains selectable rather than being silently dropped from the catalogue.
    return (path.rsplit("/", 1)[-1], False)


def group_gguf_files(files: Iterable[GGUFFile]) -> list[GGUFQuant]:
    """Group `.gguf` files into selectable quantizations, largest label sorted.

    Non-`.gguf` entries are ignored. The result is sorted by size ascending, so
    the cheapest option is first — the order someone scanning for "what fits"
    reads in.
    """
    buckets: dict[tuple[str, bool], list[GGUFFile]] = {}
    for f in files:
        if not f.path.lower().endswith(GGUF_SUFFIX):
            continue
        buckets.setdefault(_group_key(f.path), []).append(f)

    quants: list[GGUFQuant] = []
    for _key, group in buckets.items():
        ordered = tuple(sorted(group, key=lambda f: f.path))
        # Label from the files themselves, never from the directory name: the
        # directory is a grouping device, the filename carries the quant.
        parsed = [quant_label_from_path(f.path) for f in ordered]
        label = next((p for p in parsed if p), None)
        quants.append(
            GGUFQuant(
                label=label or ordered[0].path.rsplit("/", 1)[-1],
                files=ordered,
                total_size_bytes=sum(f.size_bytes for f in ordered),
                is_split=len(ordered) > 1 or any(is_shard(f.path) for f in ordered),
                quant_parsed=label is not None,
            )
        )

    return sorted(quants, key=lambda q: (q.total_size_bytes, q.label))


def has_gguf(paths: Iterable[str]) -> bool:
    """Whether a repo listing contains any GGUF file.

    THIS is the detector, not HuggingFace's computed ``gguf`` metadata block.
    That block is derived by the Hub and can be absent on a repo that plainly
    holds `.gguf` files; branching on it would render such a repo as an ordinary
    one and quietly offer a whole-repo download. The file list is the authority.
    """
    return any(p.lower().endswith(GGUF_SUFFIX) for p in paths)
