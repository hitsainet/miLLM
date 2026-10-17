"""Read, verify and plan an adapter directory before anything is hooked (Feature 33).

Every refusal that can happen before the serving pause lives here, and nothing here touches the
model's modules, its parameters or any global state. `prepare_adapter` runs the checks in the
order of 033 FTDD §5.1 (steps 8–14), cheapest and most general first:

1. **Confinement (addendum D-11).** The directory resolves inside one of the checkpoint roots
   (`WEIGHTS_CHECKPOINT_DIR` plus the read-only `WEIGHTS_CHECKPOINT_READ_ROOTS`), by `realpath` and
   `commonpath`, and so does EACH of the three files — a symlinked `adapter_model.safetensors` must
   not reach outside the root through a directory that is inside it. Every file must be a regular
   file: a FIFO would hang the worker thread forever on its first read, so files are opened
   non-blocking and checked with `fstat` on the open descriptor, not with a `stat` beforehand.
2. **The schema is vendored, and the manifest validates** (`adapter_manifest`).
3. **`adapter_config.json` agrees with the manifest.** PEFT reads that file, the manifest is
   miForge's record beside it; an `r`, `lora_alpha` or `target_modules` that disagree mean the
   directory describes two different adapters, and serving either would be a guess.
4. **One read of the weights (D-11).** The file is read ONCE into memory, capped first by a bound
   derived from the shapes the targets can have, and the SHA-256 of THOSE bytes is compared with
   the manifest's `weights_sha256` (`hmac.compare_digest`). The header is parsed and the tensors
   are loaded from the same bytes — the file is never reopened, so bytes swapped on disk after the
   hash cannot reach the model. No tensor exists anywhere before the hash matches.
5. **The base matches (D-09).** `base_matches`.
6. **The plan.** `adapter_hooks.plan_targets` maps every key to a live module and checks every
   shape and the one rank; its reasons map to public codes in `PLAN_REASON_CODES`.
7. **The build.** `LoraHookSet.build` places A and B (float32, D-12) on each module's device.

The returned `AdapterArtifact` carries a `PreparedAdapter` ready for `LoraHookSet.install`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from millm.core.errors import (
    AdapterBaseMismatchError,
    AdapterHashMismatchError,
    AdapterInstallFailedError,
    AdapterInvalidError,
    AdapterManifestInvalidError,
    AdapterShapeMismatchError,
    InsufficientMemoryError,
    MiLLMError,
)
from millm.ml.adapter_hooks import (
    AdapterHookError,
    LoraHookSet,
    PreparedAdapter,
    plan_targets,
    target_entry_matches,
)
from millm.services.adapter_manifest import (
    AdapterManifestFields,
    ValidatorFactory,
    extract_fields,
    validate_manifest,
    vendored_validator,
)

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
MANIFEST_FILE = "adapter_manifest.json"
ADAPTER_FILES = (WEIGHTS_FILE, CONFIG_FILE, MANIFEST_FILE)

#: A JSON file beside the weights is a few KB; refuse anything absurd before reading it.
MAX_JSON_BYTES = 1024 * 1024
#: Allowance for the safetensors header (8-byte length + JSON) on top of the tensor bytes.
MAX_HEADER_BYTES = 16 * 1024 * 1024
#: The widest safetensors floating dtype (F64). The size bound assumes it, so it never refuses a
#: well-formed adapter on dtype alone; `LoraHookSet.build` converts to float32 anyway.
MAX_ELEMENT_BYTES = 8

#: `adapter_hooks` planning/build reasons -> the public error. Every `adapter_hooks.Reason` is
#: listed explicitly (a test enforces it), so a new reason cannot fall through unnoticed.
PLAN_REASON_CODES: dict[str, type[MiLLMError]] = {
    "rank": AdapterShapeMismatchError,
    "shape": AdapterShapeMismatchError,
    "missing_module": AdapterShapeMismatchError,
    "unsupported_module": AdapterShapeMismatchError,
    "fused_by_transformers": AdapterShapeMismatchError,
    "incomplete_pair": AdapterShapeMismatchError,
    "duplicate_target": AdapterShapeMismatchError,
    "invalid_target_entry": AdapterShapeMismatchError,
    "target_not_declared": AdapterShapeMismatchError,
    "target_unused": AdapterShapeMismatchError,
    "unsupported_tensor": AdapterInvalidError,
    "scaling": AdapterInvalidError,
    "dtype": AdapterInvalidError,
    "non_finite": AdapterInvalidError,
    "empty": AdapterInvalidError,
    "install_failed": AdapterInstallFailedError,
}


# ---------------------------------------------------------------------------------------------
# Private file helpers. Checked against 032's `checkpoint_store` at the W3 merge and KEPT, because
# none of its helpers behaves identically here:
# - `_sha256_bytes` hashes bytes ALREADY IN MEMORY: the weights are read once and the hash, the
#   header and the tensors all come from those bytes (D-11). `checkpoint_store.sha256_file`
#   reopens the path, which would let bytes swapped on disk after the hash reach the model.
# - `_inside` is a predicate; `checkpoint_store.resolve_inside_root` raises 032's
#   `INVALID_CHECKPOINT_PATH`, where an adapter refusal must be `ADAPTER_INVALID` with its reason.
# - `local_manifest_sha256` is the SAME VALUE as 032's identity (the SHA-256 of the whole
#   `manifest.json`, `sha256_file` there) — a test pins that equality — but reads through
#   `_read_regular` (one open, `O_NOFOLLOW`, regular files only, size-capped), which
#   `sha256_file` does not.
# ---------------------------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inside(root: str, path: str) -> bool:
    """`path` is `root` or below it. Both must already be realpaths."""
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # different drives / mixed absolute and relative
        return False


def _invalid(message: str, reason: str, **details: Any) -> AdapterInvalidError:
    return AdapterInvalidError(message, details={"reason": reason, **details})


_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _open_beneath(root: str, path: str, flags: int) -> int:
    """Open `path` (a realpath inside the realpath `root`) by walking from `root` one component
    at a time, each directory opened `O_NOFOLLOW | O_DIRECTORY` relative to its parent's
    descriptor and the file itself `O_NOFOLLOW`.

    `realpath` + `commonpath` prove confinement at ONE instant; opening the resolved string later
    re-walks every directory in it, so an intermediate directory swapped for a symlink in between
    would be followed out of the root (and `O_NOFOLLOW` on the open guards only the last
    component). Walking by descriptor refuses any symlink that appears after resolution, at any
    depth. Where `dir_fd` is unsupported this degrades to a plain open of the resolved path.
    """
    if os.open not in os.supports_dir_fd:  # pragma: no cover - POSIX always supports it
        return os.open(path, flags)
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if rel in (".", "") or any(p in ("", ".", "..") for p in parts):
        raise _invalid("a resolved adapter file is not below its root", "outside_root")
    fd = os.open(root, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        return os.open(parts[-1], flags, dir_fd=fd)
    finally:
        os.close(fd)


def _read_regular(
    path: str, *, max_bytes: int, label: str, beneath: Optional[str] = None
) -> bytes:
    """Read a whole regular file, at most `max_bytes`, with one open.

    Opened `O_NONBLOCK` so a FIFO cannot block the open, `O_NOFOLLOW` so a file swapped for a
    symlink after resolution is refused, and checked with `fstat` on the descriptor that is then
    read — never a `stat` of the path followed by a second open. With `beneath` (the root `path`
    was confined to) every directory component is opened by descriptor too (`_open_beneath`).
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | _O_NOFOLLOW | _O_CLOEXEC
    try:
        fd = os.open(path, flags) if beneath is None else _open_beneath(beneath, path, flags)
    except FileNotFoundError:
        raise _invalid(f"{label} is missing", "missing_file", file=label) from None
    except OSError as e:
        raise _invalid(
            f"{label} cannot be opened", "unreadable", file=label, error=type(e).__name__
        ) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _invalid(f"{label} is not a regular file", "not_regular_file", file=label)
        if info.st_size > max_bytes:
            raise _invalid(
                f"{label} is {info.st_size} bytes, above the {max_bytes}-byte bound",
                "too_large", file=label, size=info.st_size, bound=max_bytes,
            )
        if hasattr(os, "set_blocking"):
            os.set_blocking(fd, True)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:  # grew after fstat
                raise _invalid(
                    f"{label} grew above the {max_bytes}-byte bound while being read",
                    "too_large", file=label, bound=max_bytes,
                )
        return b"".join(chunks)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterLocation:
    """Where an adapter lives. Only `root_relative` and `root_label` ever reach a response."""

    root: str  # realpath of the checkpoint root that holds it
    root_label: str  # "primary" or "read:<n>"
    directory: str  # realpath of the adapter directory
    root_relative: str
    files: dict[str, str]  # file name -> realpath, each a regular file inside `directory`


def adapter_roots(settings: Any = None) -> list[tuple[str, str]]:
    """(label, path) of every checkpoint root adapters resolve under: the primary first, then
    each read-only root. Read at call time so a test or a reconfiguration needs no restart."""
    if settings is None:
        from millm.core.config import settings as _settings

        settings = _settings
    roots = [("primary", settings.WEIGHTS_CHECKPOINT_DIR)]
    roots += [(f"read:{i}", r) for i, r in enumerate(settings.weights_checkpoint_read_roots)]
    return roots


def _check_relative(relative: Any) -> str:
    if not isinstance(relative, str) or not relative.strip():
        raise _invalid("the adapter path must be a non-empty relative path", "bad_path")
    if "\x00" in relative or "\\" in relative:
        raise _invalid("the adapter path contains a forbidden character", "bad_path")
    try:
        os.fsencode(relative)  # a lone surrogate raises UnicodeEncodeError in every os call
    except UnicodeEncodeError:
        raise _invalid("the adapter path is not encodable as a file name", "bad_path") from None
    pure = PurePosixPath(relative)
    if pure.is_absolute():
        raise _invalid("the adapter path must be relative to the checkpoint root", "bad_path")
    if any(part == ".." for part in pure.parts):
        raise _invalid("the adapter path must not contain '..'", "bad_path")
    if pure.as_posix() in ("", "."):
        raise _invalid("the adapter path must name a directory below the root", "bad_path")
    return pure.as_posix()


def resolve_adapter_dir(
    relative: str, roots: Optional[Sequence[tuple[str, str]]] = None
) -> AdapterLocation:
    """Resolve `relative` to an adapter directory inside exactly one checkpoint root, and each of
    its three files to a regular file inside that directory. Raises `AdapterInvalidError`."""
    rel = _check_relative(relative)
    roots = adapter_roots() if roots is None else roots
    found: list[tuple[str, str, str]] = []
    escaped = False
    for label, root in roots:
        root_real = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(root_real, rel))
        if not _inside(root_real, candidate) or candidate == root_real:
            if os.path.lexists(os.path.join(root_real, rel)):
                escaped = True
            continue
        if os.path.isdir(candidate):
            found.append((label, root_real, candidate))
    if not found:
        if escaped:
            raise _invalid(
                f"adapter path {rel!r} resolves outside the checkpoint root", "outside_root",
                path=rel,
            )
        raise _invalid(f"no adapter directory at {rel!r}", "not_found", path=rel)
    if len(found) > 1:
        raise _invalid(
            f"adapter path {rel!r} exists in more than one checkpoint root", "ambiguous",
            path=rel, roots=[label for label, _, _ in found],
        )
    label, root_real, directory = found[0]
    files: dict[str, str] = {}
    for name in ADAPTER_FILES:
        joined = os.path.join(directory, name)
        if not os.path.lexists(joined):
            raise _invalid(f"{name} is missing from {rel!r}", "missing_file", path=rel, file=name)
        real = os.path.realpath(joined)
        if not _inside(directory, real):
            raise _invalid(
                f"{name} in {rel!r} resolves outside the adapter directory", "outside_root",
                path=rel, file=name,
            )
        files[name] = real
    return AdapterLocation(
        root=root_real, root_label=label, directory=directory, root_relative=rel, files=files
    )


# ---------------------------------------------------------------------------------------------
# Base identity (addendum D-09)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidentBase:
    """What the resident base can be matched by."""

    #: The resident's manifest hash, when it has one. When set, it IS the identity.
    manifest_sha256: Optional[str]
    #: The names a hash-less manifest may use: the row's name and its HF repo id.
    names: frozenset[str]
    #: Every name the resident base is served or addressed by (row name, repo id, the loaded
    #: model's name and its weights identity's name). An adapter may not be named any of them.
    served_names: frozenset[str] = frozenset()


def local_manifest_sha256(local_path: Optional[str]) -> Optional[str]:
    """sha256 of `<local_path>/manifest.json` for a LOCAL row, or None when there is none.

    A checkpoint served as a plain LOCAL row is the right base for an adapter naming that
    checkpoint's hash (033 review finding 2). The file is small and read once, regular files only.
    """
    if not local_path:
        return None
    path = os.path.join(local_path, "manifest.json")
    if not os.path.lexists(path):
        return None
    try:
        data = _read_regular(os.path.realpath(path), max_bytes=MAX_JSON_BYTES, label="manifest.json")
    except AdapterInvalidError as e:
        logger.warning(
            "adapter_resident_manifest_unreadable reason=%s", e.details.get("reason")
        )
        return None
    return _sha256_bytes(data)


def resident_base(row: Any, loaded: Any = None) -> ResidentBase:
    """The resident base's identity from its database row (`name`, `repo_id`, `source`,
    `local_path`) and, when present, the loaded model's `weights_identity` (032)."""
    identity = getattr(loaded, "weights_identity", None) if loaded is not None else None
    manifest_sha256 = getattr(identity, "manifest_sha256", None) if identity is not None else None
    if manifest_sha256 is None:
        source = getattr(row, "source", None)
        if getattr(source, "value", source) == "local":
            manifest_sha256 = local_manifest_sha256(getattr(row, "local_path", None))
    names = frozenset(n for n in (getattr(row, "name", None), getattr(row, "repo_id", None)) if n)
    served = set(names)
    for value in (
        getattr(loaded, "model_name", None) if loaded is not None else None,
        getattr(identity, "name", None) if identity is not None else None,
    ):
        if isinstance(value, str) and value:
            served.add(value)
    return ResidentBase(
        manifest_sha256=manifest_sha256, names=names, served_names=frozenset(served)
    )


def _norm_hash(value: Any) -> Optional[str]:
    return value.lower() if isinstance(value, str) else None


def base_matches(manifest_base: Mapping[str, Any], resident: ResidentBase) -> bool:
    """D-09. A resident with a manifest hash is matched by that hash and nothing else. A
    resident without one matches only a manifest whose `manifest_sha256` is null and whose
    `name` is the row's name or its repo id. `None` is never a wildcard on either side."""
    claimed = _norm_hash(manifest_base.get("manifest_sha256"))
    if resident.manifest_sha256 is not None:
        if claimed is None:
            return False
        return hmac.compare_digest(
            claimed.encode("utf-8"), resident.manifest_sha256.lower().encode("utf-8")
        )
    if manifest_base.get("manifest_sha256") is not None:
        return False
    return manifest_base.get("name") in resident.names


def require_base_match(manifest_base: Mapping[str, Any], resident: ResidentBase) -> None:
    """Raise `AdapterBaseMismatchError` naming both sides unless `base_matches`."""
    if base_matches(manifest_base, resident):
        return
    rule = "manifest_sha256" if resident.manifest_sha256 is not None else "name"
    raise AdapterBaseMismatchError(
        f"the adapter was trained on {manifest_base.get('name')!r}, which is not the resident base",
        details={
            "rule": rule,
            "adapter_base": {
                "name": manifest_base.get("name"),
                "manifest_sha256": manifest_base.get("manifest_sha256"),
            },
            "resident": {
                "names": sorted(resident.names),
                "manifest_sha256": resident.manifest_sha256,
            },
        },
    )


# ---------------------------------------------------------------------------------------------
# The adapter's name (BRD-01 R-01.27 bullet 4)
# ---------------------------------------------------------------------------------------------


#: The longest servable adapter name: the width of every column the served name is written to
#: (`models.name`, and `probe_events.served_model` from migration 022). A longer name would make
#: each probe event insert fail while the adapter is applied — monitoring going quietly dark —
#: so it is refused at load instead (R2-C). A test ties this to the column width.
MAX_ADAPTER_NAME_CHARS = 255

_NAME_PROBLEMS = {
    "empty": "it is empty",
    "not_printable": "it contains whitespace or non-printable characters",
    "too_long": f"it is longer than {MAX_ADAPTER_NAME_CHARS} characters",
    "base_name": "it is a name of the resident base",
    "registered_name": "it is the name of a registered model",
}


def name_problem(name: Any) -> Optional[str]:
    """Why `name` cannot be a served model name, or None. Shared with the manifest tool.

    The name is what clients address and what every response is labelled with, so it must be a
    non-empty run of printable, non-whitespace characters: a newline or a control character
    would forge log lines and header values, a space makes it un-addressable in practice."""
    if not isinstance(name, str) or not name:
        return "empty"
    if not name.isprintable() or any(c.isspace() for c in name):
        return "not_printable"
    if len(name) > MAX_ADAPTER_NAME_CHARS:
        return "too_long"
    return None


def require_adapter_name(
    name: Any, resident: ResidentBase, registered_names: Iterable[str] = ()
) -> None:
    """Refuse (`ADAPTER_MANIFEST_INVALID`, `details.reason == "name"`) a manifest `name` that is
    empty, not printable, or the name of the resident base or of any registered model.

    An adapter named like its base is served under the base's name: responses claim the base
    answered, and a request for the base's name reaches the adapter. Named like another row, it
    makes that row unreachable by name. `registered_names` is every row name the caller knows
    (the service reads them from the database); the resident's own names are always checked.
    """
    problem = name_problem(name)
    if problem is None:
        reserved = set(resident.served_names) | set(resident.names)
        reserved.update(n for n in registered_names if isinstance(n, str))
        if name in resident.served_names or name in resident.names:
            problem = "base_name"
        elif name in reserved:
            problem = "registered_name"
    if problem is None:
        return
    raise AdapterManifestInvalidError(
        f"adapter_manifest.json name {name!r} cannot be served: {_NAME_PROBLEMS[problem]}",
        details={"file": MANIFEST_FILE, "reason": "name", "problem": problem, "path": "$.name"},
    )


# ---------------------------------------------------------------------------------------------
# Manifest and config
# ---------------------------------------------------------------------------------------------


class _NotStrictJson(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise _NotStrictJson(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise _NotStrictJson(f"{name} is not JSON")


def _parse_json_object(data: bytes, label: str, error: type[MiLLMError]) -> dict:
    """Strict JSON: a duplicated key or a bare `NaN`/`Infinity` is refused.

    Python's `json` silently keeps the LAST of two equal keys, while other readers keep the first,
    so `{"weights_sha256": A, "weights_sha256": B}` would say different things to different
    tools; and `Infinity` is not JSON at all (it passed `alpha`'s `exclusiveMinimum: 0`).
    """
    try:
        parsed = json.loads(
            data, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant
        )
    except ValueError as e:
        raise error(f"{label} is not valid JSON: {e}", details={"file": label}) from None
    if not isinstance(parsed, dict):
        raise error(f"{label} is not a JSON object", details={"file": label})
    return parsed


def check_config_agrees(config: dict, fields: AdapterManifestFields) -> None:
    """`adapter_config.json` must describe the adapter the manifest describes, and be one v1 can
    serve (no rsLoRA, DoRA, biases, per-module patterns, ...). Refused `ADAPTER_MANIFEST_INVALID`."""
    from millm.tools.write_adapter_manifest import ManifestToolError, read_peft_config

    try:
        r, alpha, target_modules, _dropout = read_peft_config(config)
    except ManifestToolError as e:
        raise AdapterManifestInvalidError(
            str(e), details={"file": CONFIG_FILE, "reason": "config_unservable"}
        ) from None
    disagreements = []
    if r != fields.rank:
        disagreements.append({"field": "r", "config": r, "manifest": fields.rank})
    if float(alpha) != float(fields.alpha):
        disagreements.append({"field": "lora_alpha", "config": alpha, "manifest": fields.alpha})
    if set(target_modules) != set(fields.target_modules):
        disagreements.append(
            {
                "field": "target_modules",
                "config": sorted(set(target_modules)),
                "manifest": sorted(set(fields.target_modules)),
            }
        )
    if disagreements:
        first = disagreements[0]
        raise AdapterManifestInvalidError(
            f"adapter_config.json disagrees with adapter_manifest.json on {first['field']}",
            details={"file": CONFIG_FILE, "reason": "config_disagrees",
                     "disagreements": disagreements},
        )


# ---------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------


def weights_size_bound(model: nn.Module, rank: int, target_modules: Sequence[str]) -> int:
    """The largest file a correct adapter can be: A and B at the widest float dtype for every
    linear module a `target_modules` entry can name, plus a header allowance."""
    total = 0
    matched = 0
    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(target_entry_matches(path, e) for e in target_modules):
            total += rank * (int(module.in_features) + int(module.out_features)) * MAX_ELEMENT_BYTES
            matched += 1
    if matched == 0:
        raise AdapterShapeMismatchError(
            "no linear module in the resident model matches the adapter's target_modules",
            details={"reason": "target_unused", "target_modules": list(target_modules)},
        )
    return total + 8 + MAX_HEADER_BYTES


def safetensors_shapes(data: bytes) -> dict[str, tuple[int, ...]]:
    """Tensor key -> shape from a safetensors header held in memory. `ADAPTER_INVALID` when the
    bytes are not a safetensors file."""
    if len(data) < 8:
        raise _invalid(f"{WEIGHTS_FILE} is not a safetensors file", "unreadable", file=WEIGHTS_FILE)
    header_len = int.from_bytes(data[:8], "little")
    if header_len <= 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > len(data):
        raise _invalid(
            f"{WEIGHTS_FILE} has an implausible header length", "unreadable", file=WEIGHTS_FILE
        )
    try:
        header = json.loads(data[8:8 + header_len])
    except ValueError:
        raise _invalid(f"{WEIGHTS_FILE} header is not JSON", "unreadable", file=WEIGHTS_FILE) from None
    if not isinstance(header, dict):
        raise _invalid(f"{WEIGHTS_FILE} header is not an object", "unreadable", file=WEIGHTS_FILE)
    shapes: dict[str, tuple[int, ...]] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        shape = info.get("shape") if isinstance(info, dict) else None
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise _invalid(
                f"{WEIGHTS_FILE}: tensor {key!r} has no valid shape", "unreadable",
                file=WEIGHTS_FILE,
            )
        shapes[key] = tuple(shape)
    return shapes


def map_hook_error(error: AdapterHookError) -> MiLLMError:
    """The public error for an `adapter_hooks` refusal; every offender in `details`."""
    cls = PLAN_REASON_CODES.get(error.reason, AdapterInvalidError)
    details: dict[str, Any] = {"reason": error.reason, "offenders": list(error.offenders)}
    if error.offenders:
        details["first"] = error.offenders[0]
    return cls(error.message, details=details)


# ---------------------------------------------------------------------------------------------
# The whole preparation
# ---------------------------------------------------------------------------------------------


@dataclass
class AdapterArtifact:
    """A verified adapter, its tensors placed and ready for `LoraHookSet.install`."""

    location: AdapterLocation
    manifest: dict
    manifest_sha256: str
    fields: AdapterManifestFields
    weights_sha256: str
    weights_bytes: int
    prepared: PreparedAdapter

    @property
    def root_relative(self) -> str:
        return self.location.root_relative


def prepare_adapter(
    relative: str,
    model: nn.Module,
    resident: ResidentBase,
    *,
    validator_factory: ValidatorFactory = vendored_validator,
    roots: Optional[Sequence[tuple[str, str]]] = None,
    registered_names: Iterable[str] = (),
) -> AdapterArtifact:
    """Every pre-pause check, in FTDD §5.1 order. Blocking file I/O: run it off the event loop.

    `registered_names`: every model row name, so an adapter cannot shadow another row
    (`require_adapter_name`); the resident's own names are checked regardless."""
    location = resolve_adapter_dir(relative, roots)
    validator = validator_factory()

    manifest_bytes = _read_regular(
        location.files[MANIFEST_FILE], max_bytes=MAX_JSON_BYTES, label=MANIFEST_FILE,
        beneath=location.root,
    )
    manifest = _parse_json_object(manifest_bytes, MANIFEST_FILE, AdapterManifestInvalidError)
    validate_manifest(manifest, validator)
    fields = extract_fields(manifest)
    require_adapter_name(fields.name, resident, registered_names)

    config_bytes = _read_regular(
        location.files[CONFIG_FILE], max_bytes=MAX_JSON_BYTES, label=CONFIG_FILE,
        beneath=location.root,
    )
    check_config_agrees(
        _parse_json_object(config_bytes, CONFIG_FILE, AdapterManifestInvalidError), fields
    )

    bound = weights_size_bound(model, fields.rank, fields.target_modules)
    data = _read_regular(
        location.files[WEIGHTS_FILE], max_bytes=bound, label=WEIGHTS_FILE, beneath=location.root
    )
    actual = _sha256_bytes(data)
    if not hmac.compare_digest(actual.encode("ascii"), fields.weights_sha256.encode("ascii")):
        raise AdapterHashMismatchError(
            f"{WEIGHTS_FILE} does not match the manifest's weights_sha256",
            details={"expected": fields.weights_sha256, "actual": actual,
                     "path": location.root_relative},
        )

    require_base_match(manifest["base"], resident)

    shapes = safetensors_shapes(data)
    try:
        targets = plan_targets(
            model, shapes, rank=fields.rank, target_modules=list(fields.target_modules)
        )
    except AdapterHookError as e:
        raise map_hook_error(e) from None

    from safetensors.torch import load as load_safetensors

    try:
        tensors = load_safetensors(data)
    except (torch.OutOfMemoryError, MemoryError) as e:
        raise InsufficientMemoryError(
            "not enough memory to read the adapter's tensors",
            details={"error": type(e).__name__},
        ) from None
    except Exception as e:
        raise _invalid(
            f"{WEIGHTS_FILE} cannot be read as safetensors", "unreadable", file=WEIGHTS_FILE,
            error=type(e).__name__,
        ) from None
    del data
    try:
        prepared = LoraHookSet.build(model, targets, tensors, fields.scaling)
    except AdapterHookError as e:
        raise map_hook_error(e) from None
    except (torch.OutOfMemoryError, MemoryError) as e:
        raise InsufficientMemoryError(
            "not enough memory to place the adapter's tensors",
            details={"error": type(e).__name__},
        ) from None
    finally:
        del tensors

    return AdapterArtifact(
        location=location,
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        fields=fields,
        weights_sha256=actual,
        weights_bytes=sum(
            a.numel() * a.element_size() + b.numel() * b.element_size()
            for a, b in prepared.weights.values()
        ),
        prepared=prepared,
    )
