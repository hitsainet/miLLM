"""An SAE attaches only where its card keeps room for the KV cache the model was admitted with.

Review round 5, 2026-09-14. A transformers load is accepted when each card holds
its weights, its CUDA context and its layers' KV cache at the admitted context
(Decision 7). The cache is not allocated at load: it grows during generation. SAEs
attach AFTER the load, on their layer's card, and both attach paths compared an
SAE against the card's free memory alone — so an attach could take the room the
admission promised the cache, and the model then ran out of memory mid-generation
at a context the load had accepted. attach_sae did not even refuse: an SAE larger
than the card's free memory logged a warning and loaded.

OLMo-2-1124-13B split on 11,500 / 23,500 MB free: cuda:0 holds 14 layers, 9,450 MiB
of weights, and after a real ~500 MB context has about 1,550 MiB free, of which its
4,096-token cache needs 1,120. What that card can still give an SAE is ~430 MiB.

Each attach now asks, per card: the SAE's projected size plus the KV cache of the
model's layers on that card (at the context the fit admitted) must fit in what
torch can still allocate there — free memory plus blocks torch has cached and not
handed out (a finished generation's cache sits there, unreleased to the card).

MUTATION CONTROLS (mutate.py, millm-p2-review5; run against this file and
test_sae_attach_device.py; each restored, sha256 verified, git diff clean):
  R5-M14 _refuse_without_room ignores the KV reserve        -> 3 red: the attach_set refusal,
         the per-card reserve test, the attach_sae refusal
  R5-M15 attach_set no longer calls _refuse_without_room    -> 4 red, two of them in
         test_sae_attach_device (the per-device gate)
  R5-M16 attach_sae no longer calls _refuse_without_room    -> test_an_sae_without_room_is_refused_not_warned_about
  R5-M17 _card_room_mb ignores torch's cached blocks         -> test_memory_torch_has_cached_counts_as_room
  R5-M18 _kv_reserve_mb counts every layer on every card    -> 8 red
  R5-M19 the reserve is sized at TRANSFORMERS_MIN_CONTEXT, not the admitted context
         -> test_the_reserve_is_capped_at_the_models_own_context
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

pytest.importorskip("transformers")
from transformers import Olmo2Config  # noqa: E402

from millm.core.config import settings  # noqa: E402
from millm.core.errors import InsufficientMemoryError  # noqa: E402
from millm.services.sae_service import AttachedSAEState, CompatibilityResult, SAEService  # noqa: E402

MB = 1024 * 1024

#: allenai/OLMo-2-1124-13B-Instruct config.json.
OLMO2_13B = dict(
    vocab_size=100_352, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=40,
    num_attention_heads=40, num_key_value_heads=40, max_position_embeddings=4_096,
    tie_word_embeddings=False,
)
#: Where transformers' map puts OLMo-2-13B on 11,500 / 23,500 MB free: 14 / 26 layers.
LAYER_DEVICE = {index: torch.device("cuda", 0 if index < 14 else 1) for index in range(40)}


@pytest.fixture(autouse=True)
def reset_registry():
    state = AttachedSAEState()
    state.reset_for_tests()
    yield
    state.reset_for_tests()


def _row(sae_id: str, d_sae: int):
    row = MagicMock()
    row.id = sae_id
    row.cache_path = f"/tmp/{sae_id}"
    row.d_in = 5_120
    row.d_sae = d_sae
    # float32 on disk: W_enc + W_dec = 2 x 5,120 x d_sae x 4 B (biases left out).
    row.file_size_bytes = 2 * 5_120 * d_sae * 4
    return row


def _service(d_sae: int):
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    svc._loader = MagicMock()
    loaded = MagicMock()
    loaded.estimate_memory_mb.return_value = 64.0
    loaded.W_enc.dtype = torch.bfloat16
    svc._loader.load.return_value = loaded
    svc._hooker = MagicMock()
    svc._hooker.layer_device.side_effect = lambda model, layer: LAYER_DEVICE[layer]
    svc._hooker.install.return_value = MagicMock()
    svc.get_sae = AsyncMock(side_effect=lambda sae_id: _row(sae_id, d_sae))
    svc.check_compatibility = AsyncMock(return_value=CompatibilityResult(compatible=True, errors=[], warnings=[]))
    svc._reset_dynamo_for_hook_change = MagicMock()
    svc.repository = MagicMock()
    svc.repository.update_status = AsyncMock()
    svc.repository.create_attachment = AsyncMock()
    return svc


def _olmo_loaded():
    state = MagicMock()
    state.is_loaded = True
    state.loaded_model_id = 7
    model = MagicMock()
    model.config = Olmo2Config(**OLMO2_13B)
    model.dtype = torch.bfloat16
    state.current.model = model
    return patch("millm.services.sae_service.LoadedModelState", return_value=state)


def _cards(free_mb: dict[int, int], cached_mb: dict[int, int] | None = None):
    """free_mb as the driver reports it; cached_mb reserved by torch and not allocated."""
    cached_mb = cached_mb or {}

    def index_of(device):
        return device if isinstance(device, int) else torch.device(device).index

    return patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        mem_get_info=lambda device: (free_mb[index_of(device)] * MB, 24_576 * MB),
        memory_reserved=lambda device=None: (cached_mb.get(index_of(device), 0) + 100) * MB,
        memory_allocated=lambda device=None: 100 * MB,
    )


class TestAttachSet:
    async def test_an_sae_that_leaves_its_card_the_admitted_cache_attaches(self):
        """16,384 latents: attach_set projects the larger of its fp16 dims (2 x 5,120 x
        16,384 x 2 B = 320 MiB) and 0.6 of its float32 file (640 MiB -> 384), x1.1 = 422.
        422 + 1,120 = 1,542 <= 1,550."""
        svc = _service(16_384)
        with _olmo_loaded(), _cards({0: 1_550, 1: 6_000}):
            result = await svc.attach_set([("sae-l10", 10)])
        assert result["attached_count"] == 1

    async def test_an_sae_that_takes_the_caches_room_is_refused_naming_the_card(self):
        """32,768 latents: 0.6 x 1,280 MiB x 1.1 = 844. 844 < 1,550 free, so the old
        check admitted it; 844 + 1,120 = 1,964 > 1,550."""
        svc = _service(32_768)
        with _olmo_loaded(), _cards({0: 1_550, 1: 6_000}):
            with pytest.raises(InsufficientMemoryError) as raised:
                await svc.attach_set([("sae-l10", 10)])

        details = raised.value.details
        assert details["device"] == "cuda:0"
        assert (details["projected_mb"], details["kv_reserve_mb"], details["kv_context_tokens"]) == (844, 1_120, 4_096)
        assert details["available_mb"] == 1_550
        assert "KV cache" in raised.value.message
        assert svc._loader.load.call_count == 0

    async def test_the_reserve_is_only_the_cache_of_that_cards_layers(self):
        """Layer 30 is on cuda:1: its 26 layers need 2,080 MiB. 844 + 2,080 = 2,924."""
        svc = _service(32_768)
        with _olmo_loaded(), _cards({0: 10, 1: 2_700}):
            with pytest.raises(InsufficientMemoryError) as raised:
                await svc.attach_set([("sae-l30", 30)])
        assert (raised.value.details["device"], raised.value.details["kv_reserve_mb"]) == ("cuda:1", 2_080)

        svc = _service(32_768)
        with _olmo_loaded(), _cards({0: 10, 1: 2_924}):
            assert (await svc.attach_set([("sae-l30", 30)]))["attached_count"] == 1

    async def test_memory_torch_has_cached_counts_as_room(self):
        """A finished generation's cache stays in torch's allocator: nvidia-smi and
        mem_get_info count it as used, and torch allocates from it first. 700 free +
        1,264 cached = 1,964 = 844 + 1,120."""
        svc = _service(32_768)
        with _olmo_loaded(), _cards({0: 700, 1: 6_000}, cached_mb={0: 1_264}):
            result = await svc.attach_set([("sae-l10", 10)])
        assert result["attached_count"] == 1

    async def test_the_reserve_is_capped_at_the_models_own_context(self):
        """OLMo-2-13B serves 4,096 tokens: under an 8,192 floor its cache on cuda:0 is
        still 1,120 MiB (2,240 at 8,192). 422 + 1,120 = 1,542 <= 1,550."""
        svc = _service(16_384)
        with _olmo_loaded(), _cards({0: 1_550, 1: 6_000}), \
                patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 8_192):
            result = await svc.attach_set([("sae-l10", 10)])
        assert result["attached_count"] == 1

    async def test_the_reserve_follows_the_admitted_context(self):
        """At a 2,048 floor the cache of cuda:0's layers is 560 MiB: 844 + 560 fits 1,550."""
        svc = _service(32_768)
        with _olmo_loaded(), _cards({0: 1_550, 1: 6_000}), \
                patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 2_048):
            result = await svc.attach_set([("sae-l10", 10)])
        assert result["attached_count"] == 1


class TestTheRequestsWorkingMemoryIsKeptFreeToo:
    """Hardware acceptance, 2026-09-14. The per-card fit now also admits each card with room for
    a request's working memory — its prefill's activations and the caching allocator's share
    (millm/ml/working_memory.py) — and that room reads as free between requests as the KV
    cache's does. The load's placement carries it per card (Placement.working_mb_by_device)."""

    async def test_an_sae_that_fits_beside_the_cache_alone_is_refused_for_the_working_memory(self):
        """The 16,384-latent attach above: 422 + 1,120 = 1,542 fits 1,550 MiB. With 400 MiB of
        working memory kept on cuda:0, 422 + 1,120 + 400 = 1,942 does not."""
        svc = _service(16_384)
        patcher = _olmo_loaded()
        state = patcher.kwargs["return_value"]
        state.current.placement = {"working_mb_by_device": {"cuda:0": 400, "cuda:1": 900}}
        with patcher, _cards({0: 1_550, 1: 6_000}):
            with pytest.raises(InsufficientMemoryError) as raised:
                await svc.attach_set([("sae-l10", 10)])

        details = raised.value.details
        assert (details["device"], details["projected_mb"], details["kv_reserve_mb"], details["working_reserve_mb"]) == (
            "cuda:0", 422, 1_120, 400,
        )
        assert "working memory there 400 MB" in raised.value.message
        assert svc._loader.load.call_count == 0

    async def test_it_attaches_once_the_card_has_room_for_all_three(self):
        svc = _service(16_384)
        patcher = _olmo_loaded()
        patcher.kwargs["return_value"].current.placement = {"working_mb_by_device": {"cuda:0": 400}}
        with patcher, _cards({0: 1_942, 1: 6_000}):
            assert (await svc.attach_set([("sae-l10", 10)]))["attached_count"] == 1

    async def test_the_reserve_belongs_to_the_model_it_was_planned_for(self):
        """A placement left on the state by another model does not reserve room for this one."""
        svc = _service(16_384)
        patcher = _olmo_loaded()
        state = patcher.kwargs["return_value"]
        state.current.placement = {"working_mb_by_device": {"cuda:0": 400}}
        other = MagicMock()
        other.config = state.current.model.config
        other.dtype = torch.bfloat16
        with patcher, _cards({0: 1_550, 1: 6_000}):
            from millm.services.sae_service import _working_reserve_mb

            assert _working_reserve_mb(other, torch.device("cuda", 0)) == 0
            assert _working_reserve_mb(state.current.model, torch.device("cuda", 0)) == 400
            assert _working_reserve_mb(state.current.model, 0) == 400


class TestAttachSae:
    async def test_an_sae_without_room_is_refused_not_warned_about(self):
        """16,384 latents stored at float32: 640 MiB x 1.2 = 768. 768 + 1,120 > 1,550.
        It used to log sae_memory_warning and load anyway."""
        svc = _service(16_384)
        with _olmo_loaded(), _cards({0: 1_550, 1: 6_000}):
            with pytest.raises(InsufficientMemoryError) as raised:
                await svc.attach_sae("sae-l10", 10)

        details = raised.value.details
        assert (details["device"], details["projected_mb"], details["kv_reserve_mb"]) == ("cuda:0", 768, 1_120)
        assert svc._loader.load.call_count == 0
        assert svc._sae_state.count == 0

    async def test_an_sae_with_room_attaches(self):
        """8,192 latents: 320 MiB x 1.2 = 384. 384 + 1,120 = 1,504 <= 1,550."""
        svc = _service(8_192)
        with _olmo_loaded(), _cards({0: 1_550, 1: 6_000}), \
                patch("millm.db.base.async_session_factory", side_effect=RuntimeError("no db")):
            result = await svc.attach_sae("sae-l10", 10)
        assert result["status"] == "attached"
        assert svc._loader.load.call_count == 1
