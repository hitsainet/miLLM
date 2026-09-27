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
    """
    hooker = ProbeHooker()

    def forward(input_ids: torch.Tensor, context: Any) -> None:
        device = next(model.parameters()).device
        ids = input_ids.to(device)
        handle = hooker.install(
            model, layer, lambda hidden, _layer=layer: context.observe(_layer, hidden)
        )
        try:
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
        finally:
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
        normalization_mode=str(block.get("normalization") or "none"),
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
        return str(candidate)

    # Any layout under the cache root that ends in this relative path.
    for found in sorted(root.rglob(Path(rel).name)):
        if str(found).endswith(rel) and repo.split("/")[-1] in str(found):
            return str(found)

    raise ProbeSaeMissingError(
        f"This probe's SAE is not downloaded here. Fetch {rel!r} from {repo!r} first.",
        details={"hf_repo": repo, "path": rel, "revision": block.get("revision")},
    )
