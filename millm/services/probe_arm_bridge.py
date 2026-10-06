"""The bridge between an HTTP request and `ProbeArmingService` (FR-24.3, FR-24.4).

⚠ **THIS MODULE EXISTS BECAUSE THE ARMING SERVICE HAD NO CALLER.** Phase 4 built four gates, an
acknowledgement gate and a parity engine; phase 7 built the route module and counted its paths
against the number in its own task text. The count agreed — seven paths — while the SET was wrong:
`arm`, `parity` and the three `hub` paths the FPRD specifies were all absent, so
`ProbeArmingService.arm`, `check_identity`, `resolve_revision` and `ProbeParityEngine.run` had
between them no production caller at all. A count is not a set.

What lives here is only the part that needs the process's model state: reading the loaded model's
identity, and building the `forward` callable the parity engine runs. Both are deliberately outside
`ProbeArmingService` so that service stays testable without a GPU — which is also why every gate
could pass its tests while nothing could invoke them.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from sqlalchemy import select

from millm.ml.native_dtype import LOAD_DTYPES
from millm.core.errors import (
    ProbeHookUnsupportedError,
    ProbeNoModelLoadedError,
    ProbeSaeMismatchError,
    ProbeSaeMissingError,
)
from millm.ml.probe_hooker import ProbeHooker
from millm.services.probe_identity import LoadedIdentity, resolve_revision
from millm.services.probe_sae_slice import SaeFeatureSlice
import logging

logger = logging.getLogger(__name__)


async def loaded_identity(session: Any) -> tuple[LoadedIdentity, Any, Any]:
    """`(identity, model, tokenizer)` for the model this process currently holds.

    Raises rather than returning a partial identity. An identity assembled from defaults would
    compare cleanly against a definition and mean nothing — the probe's weights are a direction in
    one specific model's residual space, and read in another's they produce numbers that are
    plausible, stable and about nothing.
    """
    from millm.ml.model_loader import LoadedModelState

    current = LoadedModelState().current
    if current is None:
        raise ProbeNoModelLoadedError(
            "No model is loaded, so there is nothing to check this probe against"
        )
    if not current.supports_hooks:
        # llama.cpp exposes no nn.Module, so there is no layer to hook and no residual tensor
        # reachable from Python. Refused here rather than failing inside the hooker.
        raise ProbeHookUnsupportedError(
            f"This model is served by {current.engine!r}, which exposes no module tree to hook; "
            f"probes need a transformers model",
            details={"engine": current.engine},
        )

    config = getattr(current.model, "config", None)
    d_model = _config_int(config, ("hidden_size", "d_model", "n_embd"))
    n_layers = _config_int(config, ("num_hidden_layers", "n_layer", "num_layers"))
    if d_model is None or n_layers is None:
        # ⚠ NEVER a default. A fabricated d_model is how a probe gets armed against a model it was
        # never fitted on, with every gate reporting a match.
        raise ProbeNoModelLoadedError(
            "Cannot read the loaded model's width and depth from its config, so identity cannot "
            "be checked",
            details={"d_model": d_model, "n_layers": n_layers},
        )

    row = await _model_row(session, current.model_id)
    revision, revision_source = resolve_revision(
        getattr(row, "cache_path", None), getattr(row, "revision", None)
    )
    tokenizer = current.tokenizer
    identity = LoadedIdentity(
        hf_id=(getattr(row, "repo_id", None) or current.model_name or ""),
        d_model=d_model,
        n_layers=n_layers,
        chat_template=getattr(tokenizer, "chat_template", None),
        revision=revision,
        revision_source=revision_source,
        supports_hooks=True,
        # The RESOLVED precision the loader recorded (`LoadedModel.dtype`), compared against the
        # definition's `model.load_dtype`.
        dtype=current.dtype if current.dtype in LOAD_DTYPES else None,
        quantization=(
            str(getattr(row.quantization, "value", row.quantization)) if row is not None else None
        ),
    )
    logger.info(
        "probe_loaded_identity hf_id=%s d_model=%s n_layers=%s revision_source=%s",
        identity.hf_id,
        identity.d_model,
        identity.n_layers,
        identity.revision_source,
    )
    return identity, current.model, tokenizer


def _config_int(config: Any, names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


async def _model_row(session: Any, model_id: int) -> Any:
    from millm.db.models.model import Model

    result = await session.execute(select(Model).where(Model.id == model_id))
    return result.scalar_one_or_none()


def build_parity_forward(model: Any, layer: int) -> Any:
    """`forward(input_ids, context)` — one pass with a temporary hook on `layer`.

    ⚠ **A TEMPORARY hook, installed and removed per call.** Parity runs BEFORE the probe is armed,
    so `ProbeRuntimeState` has installed nothing yet; and it must stay uninstalled if a gate
    refuses, or a refused probe would leave a hook on the model with nothing tracking it.

    Delegates to `build_probe_forward` with one layer, so parity and stateless scoring run the
    SAME forward (FR-27.5e) — a second copy is how offline and live scores drift apart.
    """
    return build_probe_forward(model, [layer])


def build_probe_forward(model: Any, layers: Any) -> Any:
    """`forward(input_ids, context)` — one pass with a temporary read hook on each of `layers`.

    The parity forward generalised to several layers, so stateless scoring runs ONE forward per
    input for every requested probe whatever layer each reads (FR-27.6c). One `ProbeHooker` hook
    per DISTINCT layer, each feeding `context.observe(layer, hidden)`; every handle is removed in
    `finally`, so a failed pass leaves nothing installed.

    Runs whatever thread calls it; the caller owns the admission slot and the suppression
    (`InferenceService.run_model_work`).
    """
    hooker = ProbeHooker()
    distinct = sorted({int(layer) for layer in layers})

    def forward(input_ids: torch.Tensor, context: Any) -> None:
        device = next(model.parameters()).device
        ids = input_ids.to(device)
        handles = []
        try:
            for layer in distinct:
                handles.append(hooker.install(
                    model, layer, lambda hidden, _layer=layer: context.observe(_layer, hidden)
                ))
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                hooker.remove(handle)

    return forward


async def build_probe_encoder(probe: Any) -> Optional[Any]:
    """The k-sparse encoder for an `sae_features` probe, or `None` for a dense one.

    ⚠ **A PRIVATE SLICE, NEVER `AttachedSAEState`.** A probe must read the dictionary it was fitted
    against, at the k columns its weights pair with BY POSITION. Borrowing whatever SAE is attached
    would encode in a different basis and produce plausible features with different meanings,
    invisible in every metric — and would also make arming a probe change what the operator's own
    steering reads.

    Normalization and architecture come from the DEFINITION, never guessed from the SAE row. In
    miStudio the same guess wrote `constant_norm_rescale` over an SAE trained with `none`; the
    default happened to be right for the SAE it was found on, which is why it survived review.
    """
    if (probe.basis or "residual") != "sae_features":
        return None

    block = (probe.definition or {}).get("sae") or {}
    indices = list(block.get("feature_indices") or [])
    if not indices:
        raise ProbeSaeMismatchError(
            "This probe reads an SAE basis but its definition selects no features",
            details={"basis": probe.basis},
        )

    path = await _resolve_sae_path(block)
    slice_ = SaeFeatureSlice.load(
        path,
        indices,
        architecture=str(block.get("architecture") or "standard"),
        normalization_mode=_normalization_mode(block),
        expected_sha256=block.get("weights_sha256") or None,
    )
    logger.info(
        "probe_sae_slice_built repo=%s path=%s k=%s d_model=%s architecture=%s normalization=%s",
        block.get("hf_repo"),
        block.get("path"),
        slice_.k,
        slice_.d_model,
        slice_.architecture,
        slice_.normalization_mode,
    )
    return slice_


#: Weights filenames an SAE directory may use, in preference order.
_WEIGHT_NAMES = ("sae_weights.safetensors", "sae.safetensors", "weights.safetensors", "sae_weights.npz")


def _weights_file(target: "Path") -> str:
    """The weights file at or inside `target`.

    ⚠ **`sae.path` IS A DIRECTORY, NOT A FILE.** miStudio writes `ExternalSAE.hf_filepath`
    into it — e.g. `layer_11`, or `layer_12/width_16k/canonical` — naming the directory inside
    the repo that holds `cfg.json` and `sae_weights.safetensors`. The first version of this
    function handed the directory straight to `SaeFeatureSlice.load`, which branches on
    `.suffix == ".safetensors"` and would have fallen through to `np.load` on a directory.
    Caught before it ran, by reading what miStudio actually writes rather than what the field
    name suggests: the contract's `path` has no description, so the producer is the authority.

    A file path still works, so a future producer that names the file directly needs no change
    here.
    """
    from pathlib import Path as _Path

    target = _Path(target)
    if target.is_file():
        return str(target)
    for name in _WEIGHT_NAMES:
        found = target / name
        if found.is_file():
            return str(found)
    raise ProbeSaeMissingError(
        f"{target} holds no recognised SAE weights file "
        f"(looked for {', '.join(_WEIGHT_NAMES)})",
        details={"directory": str(target), "looked_for": list(_WEIGHT_NAMES)},
    )


def _normalization_mode(block: dict[str, Any]) -> str:
    """The SAE's normalisation MODE, out of a field that is an object, not a string.

    ⚠ `sae.normalization` is `{"mode": "constant_norm_rescale", "source": "..."}` — the mode
    plus a record of where miStudio read it from. This passed the whole dict through `str()`,
    so the slice received the dict's repr and refused it as an unknown mode. Every k-sparse
    probe failed to score, and because the read hook swallows callback exceptions by design
    ("a probe must never break generation"), parity reported `no_scored_tokens` on all sixteen
    vectors rather than the actual error. The traceback was only in the worker log.

    ⚠ **AND THE SLICE REFUSING IS THE DESIGN WORKING.** A lenient loader would have fallen
    back to a default mode, encoded in the wrong basis, and produced plausible features with
    different meanings — invisible in every metric. miStudio shipped exactly that once
    (`sae_row.normalize_activations` silently guessed, and the default happened to be right
    for the SAE it was found on). Refusing an unrecognised mode is why this was a five-minute
    diagnosis instead of a wrong answer nobody noticed.

    A plain string is still accepted, so an older document needs no migration.
    """
    raw = block.get("normalization")
    if isinstance(raw, dict):
        mode = raw.get("mode")
        if not mode:
            raise ProbeSaeMismatchError(
                "the sae block's `normalization` object carries no `mode`",
                details={"normalization": raw},
            )
        return str(mode)
    return str(raw or "none")


async def _resolve_sae_path(block: dict[str, Any]) -> str:
    """Where this probe's SAE weights are on disk, or a refusal naming what to fetch.

    Looked up by repo AND relative path: one repo publishes a dictionary per layer, and arming a
    layer-11 probe against the layer-13 file would pass every other gate.
    """
    from pathlib import Path

    from millm.core.config import settings

    repo = str(block.get("hf_repo") or "")
    rel = str(block.get("path") or "")
    if not repo or not rel:
        raise ProbeSaeMissingError(
            "The definition's sae block names no repo and path, so its weights cannot be located",
            details={"hf_repo": repo or None, "path": rel or None},
        )

    # `path` comes from an imported document. Resolve it and require it to stay inside the cache
    # directory: `../../etc/...` in a definition must not read a file outside it.
    root = Path(settings.SAE_CACHE_DIR).resolve()
    candidate = (root / repo.replace("/", "__") / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ProbeSaeMissingError(
            "The definition's sae path escapes the SAE cache directory",
            details={"path": rel},
        ) from None
    if candidate.exists():
        return _weights_file(candidate)

    # Any layout under the cache root that ends in this relative path.
    for found in sorted(root.rglob(Path(rel).name)):
        if str(found).endswith(rel) and repo.split("/")[-1] in str(found):
            return _weights_file(found)

    raise ProbeSaeMissingError(
        f"This probe's SAE is not downloaded here. Fetch {rel!r} from {repo!r} first.",
        details={"hf_repo": repo, "path": rel, "revision": block.get("revision")},
    )
