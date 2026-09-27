"""A private copy of the encoder columns a k-sparse probe reads (FR-24.15, D14).

A k-sparse probe was fitted on SAE feature activations, so miLLM has to encode the residual into
that basis before the head can read it. It does **not** attach the SAE: the slice is private to the
probe, it never steers, and attaching, detaching or steering with any SAE — including this one —
does not affect it.

## Why this cannot reuse `LoadedSAE.encode`

Two reasons, both established by spike 0.4:

1. `LoadedSAE.encode` is `relu(x @ W_enc + b_enc)` with **no training normalization**. miStudio's
   `encode_with_training_normalization` applies the SAE's own rescale first, and the reason that
   helper exists is MIS-E2E-083 — five of six call sites reached for bare `encode()` and mined
   every circuit from activations the dictionary was never trained to decode. "The features fire,
   the numbers are plausible, and the basis is wrong."
2. **miLLM has never loaded a JumpReLU threshold.** `sae_loader` reads `W_enc, b_enc, W_dec, b_dec`
   and nothing else, so the learned θ that JumpReLU's activation needs is not available through any
   existing path. `torch.relu` is not `(pre > θ) · pre`, and it destroys negative pre-activations
   before any threshold comparison.

So the normative target is **miStudio's** encode, not `LoadedSAE`'s — which is why FTASKS 4A.4 was
reworded. Compared against `LoadedSAE.encode`, "the slice equals the full encode restricted to
`idx`" is an assertion that is satisfiable *and wrong*.

## Normalization needs no new contract fields

`constant_norm_rescale` and `anthropic_rescale` are the **same** operation — a per-sample rescale
to ‖x‖ = √d. miStudio measured the two formulations agreeing to 7.2e-7 and records that the second
is an alias, not a second method (MIS-E2E-085). `none` is a no-op. So the definition's `mode` is
everything the runtime needs, and `sae.normalization` already carries it.

## Memory

k × d_model per probe. At k=128 and d=2048 in fp16 that is about 0.5 MB.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

#: Normalization modes. The first two are the same operation; see the module docstring.
RESCALE_MODES = ("constant_norm_rescale", "anthropic_rescale")
NO_NORMALIZATION = "none"

#: Architectures whose activation is a hard threshold rather than a relu.
JUMPRELU_ARCHITECTURES = ("jumprelu", "jump_relu")


def normalize(x: torch.Tensor, mode: str) -> torch.Tensor:
    """miStudio's training-time activation normalization, per sample.

    Rescales each row so ‖x‖ = √d. An all-zero row is left alone rather than divided by zero — its
    norm carries no direction to preserve.
    """
    if mode in (NO_NORMALIZATION, "", None):
        return x
    if mode not in RESCALE_MODES:
        raise ValueError(
            f"unknown normalization mode {mode!r}; known: "
            f"{', '.join((*RESCALE_MODES, NO_NORMALIZATION))}"
        )
    d = x.shape[-1]
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(
        norm > 0, (d**0.5) / norm.clamp_min(torch.finfo(x.dtype).tiny), torch.ones_like(norm)
    )
    return x * scale


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SaeFeatureSlice:
    """`W_enc[:, idx]`, `b_enc[idx]` and (for JumpReLU) `threshold[idx]` — nothing else."""

    weight: torch.Tensor          # (d_model, k)
    bias: torch.Tensor            # (k,)
    architecture: str
    normalization_mode: str = NO_NORMALIZATION
    thresholds: Optional[torch.Tensor] = None   # (k,), JumpReLU only
    feature_indices: tuple[int, ...] = ()
    #: (d_model,) — the DECODER bias, kept because the encoder centers by it. See `encode`.
    decoder_bias: Optional[torch.Tensor] = None

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.weight.shape[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d_model) -> (..., k), in the basis the probe's weights were fitted in.

        ⚠ **THE SLICE MOVES TO `x`, NEVER `x` TO THE SLICE.** `ProbeRequestContext.observe`
        scores where the model's tensor already is, so that a 4k-token residual is not copied
        to the host — 33.6 MB per forward pass. Pulling `x` to this slice's device would undo
        exactly that, and it would do it silently, because the numbers would be right.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"activations are d_model={x.shape[-1]} but this slice is d_model={self.d_model}"
            )
        here = self.to_device(x.device)
        hidden = normalize(x.to(here.weight.dtype), here.normalization_mode)
        return here._encode_on_device(hidden)

    def to_device(self, device: torch.device) -> "SaeFeatureSlice":
        """This slice with its tensors on `device`. Returns self when already there.

        The dtype is preserved: an SAE probe's weights pair with the dictionary it was fitted
        against, and re-casting them is the kind of silent basis change that produces
        plausible features with different meanings.
        """
        if self.weight.device == device:
            return self
        return SaeFeatureSlice(
            weight=self.weight.to(device),
            bias=self.bias.to(device),
            architecture=self.architecture,
            normalization_mode=self.normalization_mode,
            thresholds=None if self.thresholds is None else self.thresholds.to(device),
            feature_indices=self.feature_indices,
            decoder_bias=None if self.decoder_bias is None else self.decoder_bias.to(device),
        )

    def _encode_on_device(self, hidden: torch.Tensor) -> torch.Tensor:
        # ⚠ **CENTER BY THE DECODER BIAS FIRST.** miStudio's SAE encodes
        #     z = ReLU(W_enc @ (x - b_dec) + b_enc)
        # — the standard Bricken et al. 2023 formulation, and what the probe's weights were
        # fitted against. This omitted it, so the slice encoded in an UNCENTERED basis.
        #
        # It produced sparse, plausible, well-behaved features that meant something else, and
        # nothing about the output said so: the slice loaded, the hash matched, the right number
        # of features fired. The only thing that caught it was the parity gate, on hardware —
        # the recorded scores diverged with a per-token median of 58.3 where the dense probe's
        # was 0.96. miStudio's own notes record this exact omission costing a sparsity reading
        # of 2-3x elsewhere in its pipeline.
        if self.decoder_bias is not None:
            hidden = hidden - self.decoder_bias
        pre = hidden @ self.weight + self.bias
        if self.architecture.lower() in JUMPRELU_ARCHITECTURES:
            if self.thresholds is None:
                raise ValueError(
                    "a JumpReLU slice needs its learned thresholds; falling back to relu would "
                    "encode in a different basis than the probe was fitted in"
                )
            return torch.where(pre > self.thresholds, pre, torch.zeros_like(pre))
        return torch.relu(pre)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    # ── loading ────────────────────────────────────────────────────────────────────

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        feature_indices: Sequence[int],
        *,
        architecture: str,
        normalization_mode: str = NO_NORMALIZATION,
        dtype: torch.dtype = torch.float32,
    ) -> "SaeFeatureSlice":
        """Slice an already-loaded state dict. Keeps only the k columns this probe reads."""
        weight = _pick(state, ("W_enc", "encoder.weight", "W_e", "w_enc"))
        bias = _pick(state, ("b_enc", "encoder.bias", "b_e", "bias_enc"))
        # (d_model,) and kept WHOLE — it is subtracted before the projection, so it is not
        # sliced to k. 8 KB at d_model 2048.
        decoder_bias = _pick(state, ("b_dec", "decoder.bias", "b_d", "bias_dec"))
        if weight is None or bias is None:
            raise ValueError("the SAE weights carry no encoder (W_enc / b_enc)")

        # SAELens stores (d_model, n_features); some exports transpose. The d_model side is the
        # one that is NOT the feature count, and getting this backwards silently encodes noise.
        if weight.shape[0] == bias.shape[0] and weight.shape[1] != bias.shape[0]:
            weight = weight.t()

        index = torch.tensor(list(feature_indices), dtype=torch.long)
        thresholds = _pick(state, ("threshold", "thresholds", "jumprelu_threshold", "theta"))
        return cls(
            weight=weight.to(dtype)[:, index].contiguous(),
            bias=bias.to(dtype)[index].contiguous(),
            architecture=architecture,
            normalization_mode=normalization_mode,
            thresholds=(thresholds.to(dtype)[index].contiguous() if thresholds is not None else None),
            feature_indices=tuple(int(i) for i in feature_indices),
            # NOT sliced: it is subtracted in d_model space, before the projection.
            decoder_bias=(decoder_bias.to(dtype).contiguous() if decoder_bias is not None else None),
        )

    @classmethod
    def load(
        cls,
        weights_path: str | Path,
        feature_indices: Sequence[int],
        *,
        architecture: str,
        normalization_mode: str = NO_NORMALIZATION,
        expected_sha256: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SaeFeatureSlice":
        """Read one SAE weights file and keep only this probe's columns.

        ⚠ The SHA is verified before anything is read into the basis. A dictionary that is not the
        one the probe was fitted against produces features that are plausible and mean something
        else — the failure has no symptom, so the hash is the only thing that catches it.
        """
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"SAE weights not found at {path}")
        if expected_sha256:
            actual = sha256_of(path)
            if actual != expected_sha256:
                raise ValueError(
                    f"SAE weights sha256 is {actual}, but the probe was fitted against "
                    f"{expected_sha256}"
                )

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            import numpy as np

            data = np.load(str(path))
            state = {key: torch.from_numpy(data[key].copy()) for key in data.files}

        return cls.from_state_dict(
            state,
            feature_indices,
            architecture=architecture,
            normalization_mode=normalization_mode,
            dtype=dtype,
        )


def _pick(state: dict[str, Any], names: Sequence[str]) -> Optional[torch.Tensor]:
    for name in names:
        if name in state:
            value = state[name]
            return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return None
