"""Write `adapter_manifest.json` for a plain PEFT LoRA directory (Feature 33, addendum D-08).

miLLM serves an adapter only when its directory carries a manifest in miForge's
`miforge.adapter-manifest` v1 format, because the manifest is what names the base and pins the
weights' hash (BRD-01 R-01.27). Adapters trained with `peft` directly, in a notebook or downloaded
from the Hub carry only `adapter_config.json` and `adapter_model.safetensors`. This tool reads those
two files and writes the manifest, so any plain LoRA becomes servable in one step:

    python -m millm.tools.write_adapter_manifest /data/checkpoints/my-run/adapter-500 \\
        --base-name meta-llama/Llama-3.1-8B-Instruct

`--base-name` is the operator's statement of which base the adapter was trained against: the HF
repo id for a Hub base, or the checkpoint manifest's `name` for a saved checkpoint (then also pass
`--base-manifest-sha256`). The tool cannot check it — nothing in a PEFT directory proves a base —
which is exactly why miLLM refuses the adapter against any other base at load time.

It REFUSES a configuration whose served behaviour v1 cannot express, rather than writing a manifest
that would serve a different model from the one trained: rank-stabilised scaling (`use_rslora`),
DoRA (`use_dora`), trained biases (`bias` other than "none", `lora_bias`), per-module ranks or alphas
(`rank_pattern`, `alpha_pattern`), transposed weights (`fan_in_fan_out`), extra full modules
(`modules_to_save`) and a regex or `all-linear` `target_modules` (v1 matches literal names). It also
checks the weights file's keys against `target_modules` and `r` before writing anything.

An existing manifest is never overwritten without `--force`. The result is validated against the
vendored schema (`docs/schemas/adapter-manifest-v1.json`) before it is written.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

# One RFC 3339 rule for every manifest miLLM reads or writes (addendum D-29). Re-exported here.
from millm.services.rfc3339 import (  # noqa: F401 - is_rfc3339 is re-exported for callers
    date_time_format_checker,
    is_rfc3339,
    utc_now_rfc3339,
)

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"
MANIFEST_FILE = "adapter_manifest.json"
KIND = "miforge.adapter-manifest"
SCHEMA_VERSION = "1"

VENDORED_SCHEMA = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "adapter-manifest-v1.json"

#: Caps on the safetensors header read, so a hostile or corrupt file cannot make this allocate.
MAX_HEADER_BYTES = 64 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


#: PEFT `LoraConfig` fields that, when truthy, change what the adapter computes at inference — a
#: LoRA variant with its own forward, or a change to the model's structure or parameters. The
#: hooks implement plain `x + s·B·A·x` only, so each is refused rather than served as plain LoRA.
#: (peft 0.21's variant fields: `dataclasses.fields(LoraConfig)` with `is_lora_variant`; `mica`
#: is a variant of initialisation only and serves as plain LoRA.)
_UNSERVABLE_WHEN_SET: dict[str, str] = {
    "use_qalora": "QALoRA pools the input before lora_A",
    "use_bdlora": "block-diagonal LoRA is not a plain LoRA",
    "alora_invocation_tokens": "activated LoRA applies only after its invocation tokens",
    "arrow_config": "Arrow routes between adapters per token",
    "velora_config": "VeLoRA is not a plain LoRA",
    "monteclora_config": "MonteCLoRA is not a plain LoRA",
    "kasa_config": "KaSA is not a plain LoRA",
    "layer_replication": "layer replication changes the model's structure",
    "target_parameters": "LoRA on raw parameters is not a module hook",
    "trainable_token_indices": "trainable token embeddings are not served",
}


class ManifestToolError(Exception):
    """A refusal. The message is printed to stderr and the tool exits 1."""


def _regular_file_inside(directory: Path, name: str) -> Path:
    """`directory/name`, resolved, refusing anything outside the directory or not a regular file
    (a FIFO would hang the read)."""
    root = directory.resolve()
    path = (root / name).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise ManifestToolError(f"{name} resolves outside {root}")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        raise ManifestToolError(f"{name} not found in {root}") from None
    if not stat.S_ISREG(mode):
        raise ManifestToolError(f"{name} in {root} is not a regular file")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Tensor key -> shape, from the header alone (8-byte little-endian length + JSON)."""
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ManifestToolError(f"{path.name} is not a safetensors file (too short)")
        header_len = int.from_bytes(raw_len, "little")
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            raise ManifestToolError(f"{path.name} has an implausible header length {header_len}")
        raw = fh.read(header_len)
    if len(raw) != header_len:
        raise ManifestToolError(f"{path.name} is truncated inside its header")
    try:
        header = json.loads(raw)
    except ValueError as e:
        raise ManifestToolError(f"{path.name} header is not JSON: {e}") from None
    if not isinstance(header, dict):
        raise ManifestToolError(f"{path.name} header is not an object")
    shapes: dict[str, tuple[int, ...]] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        shape = info.get("shape") if isinstance(info, dict) else None
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            raise ManifestToolError(f"{path.name}: tensor {key!r} has no valid shape")
        shapes[key] = tuple(shape)
    return shapes


def read_peft_config(config: dict) -> tuple[int, float, list[str], float]:
    """(r, lora_alpha, target_modules, lora_dropout), refusing what v1 cannot serve faithfully."""
    peft_type = config.get("peft_type")
    if peft_type is not None and str(peft_type).upper() != "LORA":
        raise ManifestToolError(f"peft_type is {peft_type!r}; only LORA adapters are servable")
    refusals = []
    if config.get("use_rslora"):
        refusals.append("use_rslora is true (rank-stabilised scaling is not schema v1's alpha/r)")
    if config.get("use_dora"):
        refusals.append("use_dora is true (DoRA is not a plain LoRA)")
    if config.get("bias", "none") not in (None, "none"):
        refusals.append(f"bias is {config.get('bias')!r} (trained biases are not served)")
    if config.get("lora_bias"):
        refusals.append("lora_bias is true (a LoRA bias is not served)")
    if config.get("rank_pattern"):
        refusals.append("rank_pattern is not empty (v1 carries one rank)")
    if config.get("alpha_pattern"):
        refusals.append("alpha_pattern is not empty (v1 carries one alpha)")
    if config.get("fan_in_fan_out"):
        refusals.append("fan_in_fan_out is true (transposed weights are not supported)")
    if config.get("modules_to_save"):
        refusals.append("modules_to_save is not empty (full module copies are not served)")
    for key, why in _UNSERVABLE_WHEN_SET.items():
        if config.get(key):
            refusals.append(f"{key} is set ({why})")
    target_modules = config.get("target_modules")
    if isinstance(target_modules, str):
        refusals.append(
            f"target_modules is the string {target_modules!r} (a regex or 'all-linear'); v1 needs "
            "an explicit list of module names"
        )
    elif not isinstance(target_modules, list) or not target_modules or not all(
        isinstance(t, str) and t for t in target_modules
    ):
        refusals.append("target_modules must be a non-empty list of module names")
    if refusals:
        raise ManifestToolError("adapter_config.json cannot be served as v1: " + "; ".join(refusals))

    r = config.get("r")
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise ManifestToolError(f"r must be a positive integer, got {r!r}")
    alpha = config.get("lora_alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not alpha > 0
    ):
        raise ManifestToolError(f"lora_alpha must be a positive finite number, got {alpha!r}")
    dropout = config.get("lora_dropout", 0.0) or 0.0
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
        raise ManifestToolError(f"lora_dropout must be between 0 and 1, got {dropout!r}")
    return r, alpha, sorted(set(target_modules)), float(dropout)


def check_weights_against_config(
    shapes: dict[str, tuple[int, ...]], rank: int, target_modules: Sequence[str]
) -> None:
    """Every key is a PEFT LoRA key, A and B pair up at rank `r`, and `target_modules` agrees
    with the keys (addendum D-17), so a manifest is never written for weights miLLM will refuse
    for a reason visible here."""
    from millm.ml.adapter_hooks import check_target_modules, parse_peft_key

    if not shapes:
        raise ManifestToolError(f"{WEIGHTS_FILE} contains no tensors")
    pairs: dict[str, dict[str, tuple[int, ...]]] = {}
    problems: list[str] = []
    for key, shape in shapes.items():
        parsed = parse_peft_key(key)
        if parsed is None:
            problems.append(f"{key}: not a PEFT LoRA key")
            continue
        path, which = parsed
        pairs.setdefault(path, {})[which] = shape
    for path, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            problems.append(f"{path}: lora_A and lora_B do not both exist")
            continue
        a, b = pair["A"], pair["B"]
        if len(a) != 2 or len(b) != 2 or a[0] != rank or b[1] != rank:
            problems.append(f"{path}: lora_A {a} / lora_B {b} are not rank {rank}")
    for offender in check_target_modules({p: (p,) for p in pairs}, list(target_modules)):
        problems.append(f"{offender.get('module') or 'target_modules'}: {offender['reason']}"
                        f" ({offender.get('detail', '')})")
    if problems:
        raise ManifestToolError(
            f"{WEIGHTS_FILE} does not match adapter_config.json:\n  " + "\n  ".join(problems)
        )


def build_manifest(
    directory: Path,
    *,
    base_name: str,
    base_manifest_sha256: Optional[str],
    run_id: Optional[str],
    step: int,
    name: Optional[str],
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    if not directory.is_dir():
        raise ManifestToolError(f"{directory} is not a directory")
    if not base_name or not base_name.strip():
        raise ManifestToolError("--base-name must not be empty")
    if base_name != base_name.strip() or not base_name.isprintable():
        raise ManifestToolError(
            f"--base-name {base_name!r} has surrounding whitespace or non-printable characters; "
            "miLLM matches it exactly"
        )
    if base_manifest_sha256 is not None and not _SHA256.match(base_manifest_sha256):
        raise ManifestToolError("--base-manifest-sha256 must be 64 lowercase hex characters")
    if isinstance(step, bool) or step < 0:
        raise ManifestToolError(f"--step must be a non-negative integer, got {step!r}")

    config_path = _regular_file_inside(directory, CONFIG_FILE)
    weights_path = _regular_file_inside(directory, WEIGHTS_FILE)
    try:
        config = json.loads(config_path.read_text())
    except ValueError as e:
        raise ManifestToolError(f"{CONFIG_FILE} is not valid JSON: {e}") from None
    if not isinstance(config, dict):
        raise ManifestToolError(f"{CONFIG_FILE} is not a JSON object")
    rank, alpha, target_modules, dropout = read_peft_config(config)
    check_weights_against_config(read_safetensors_shapes(weights_path), rank, target_modules)

    run_id = run_id or _RUN_ID_SAFE.sub("-", directory.resolve().name).strip("-") or "adapter"
    name = f"{base_name}:{run_id}:adapter-{step}" if name is None else name
    from millm.services.adapter_artifact import _NAME_PROBLEMS, name_problem

    problem = name_problem(name)
    if problem is not None:
        raise ManifestToolError(
            f"the adapter name {name!r} cannot be served: {_NAME_PROBLEMS[problem]}"
        )
    if name == base_name:
        raise ManifestToolError(
            f"the adapter name {name!r} is the base's name; miLLM would serve it as the base"
        )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "run_id": run_id,
        "step": step,
        "base": {"name": base_name, "manifest_sha256": base_manifest_sha256},
        "adapter": {
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "dropout": dropout,
            "scaling": "alpha_over_r",
        },
        "weights_sha256": sha256_file(weights_path),
        "created_at": created_at or utc_now_rfc3339(),
    }


def validate_manifest(manifest: dict, schema_path: Path = VENDORED_SCHEMA) -> None:
    """Validate against the vendored schema, with `date-time` actually checked."""
    import jsonschema

    if not schema_path.is_file():
        raise ManifestToolError(
            f"the vendored adapter manifest schema is not at {schema_path}; pass --schema PATH "
            "(a miLLM checkout has it at docs/schemas/adapter-manifest-v1.json)"
        )
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema, format_checker=date_time_format_checker())
    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors]
        raise ManifestToolError("the manifest does not validate:\n  " + "\n  ".join(lines))


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:  # pragma: no cover - platforms that cannot open a directory
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def _publish_without_link(tmp: str, target: Path) -> None:
    """No-clobber publish where `os.link` is unsupported: `O_CREAT | O_EXCL` claims the name
    (refusing if it exists), then `os.replace` swaps in the finished file atomically."""
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                     0o644)
    except FileExistsError:
        raise ManifestToolError(f"{target} already exists; pass --force to replace it") from None
    os.close(fd)
    os.replace(tmp, target)


def write_manifest(directory: Path, manifest: dict, *, force: bool) -> Path:
    """Atomic write; refuses to replace an existing manifest unless `force`.

    Without `force` the temporary file is published with `os.link`, which fails if the name exists
    at that instant — a check-then-`replace` would overwrite a manifest written in between. With
    `force`, `os.replace`. Either way the directory is fsynced so the new name survives a crash,
    and `allow_nan=False` keeps the file JSON (`Infinity` is a Python extension).
    """
    target = directory / MANIFEST_FILE
    if (target.exists() or target.is_symlink()) and not force:
        raise ManifestToolError(f"{target} already exists; pass --force to replace it")
    try:
        data = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    except ValueError as e:
        raise ManifestToolError(f"the manifest is not representable as JSON: {e}") from None
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".adapter_manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        if force:
            os.replace(tmp, target)
        else:
            try:
                os.link(tmp, target)
            except FileExistsError:
                raise ManifestToolError(
                    f"{target} already exists; pass --force to replace it"
                ) from None
            except OSError:
                # A filesystem without hard links (some network, FUSE and FAT volumes answer
                # EPERM / ENOTSUP). Claim the name exclusively, then replace the placeholder:
                # still never clobbers a manifest written in between (R2-C).
                _publish_without_link(tmp, target)
            else:
                os.unlink(tmp)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    _fsync_directory(directory)
    return target


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m millm.tools.write_adapter_manifest",
        description="Write adapter_manifest.json for a plain PEFT LoRA directory.",
    )
    parser.add_argument("directory", type=Path, help="directory holding adapter_config.json and "
                        "adapter_model.safetensors")
    parser.add_argument("--base-name", required=True, help="the base the adapter was trained "
                        "against: an HF repo id, or a saved checkpoint's manifest name")
    parser.add_argument("--base-manifest-sha256", default=None, help="sha256 of the base "
                        "checkpoint's manifest.json, when the base is a saved checkpoint")
    parser.add_argument("--run-id", default=None, help="default: the directory's name")
    parser.add_argument("--step", type=int, default=0, help="default: 0")
    parser.add_argument("--name", default=None,
                        help="default: '<base-name>:<run-id>:adapter-<step>'")
    parser.add_argument("--schema", type=Path, default=VENDORED_SCHEMA,
                        help="the vendored adapter-manifest-v1.json (default: this checkout's)")
    parser.add_argument("--force", action="store_true", help="replace an existing manifest")
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            args.directory,
            base_name=args.base_name,
            base_manifest_sha256=args.base_manifest_sha256,
            run_id=args.run_id,
            step=args.step,
            name=args.name,
        )
        validate_manifest(manifest, args.schema)
        path = write_manifest(args.directory, manifest, force=args.force)
    except ManifestToolError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print(f"name {manifest['name']}")
    print(f"weights_sha256 {manifest['weights_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
