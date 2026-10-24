"""The CUDA caching-allocator setting miLLM runs with (see `millm/__init__.py` for why).

Pure, so the decision can be tested without a GPU and without re-importing torch.
"""

from __future__ import annotations

from collections.abc import Mapping

#: torch >= 2.9 reads PYTORCH_ALLOC_CONF and still honours the older CUDA-specific name.
NEW_NAME = "PYTORCH_ALLOC_CONF"
OLD_NAME = "PYTORCH_CUDA_ALLOC_CONF"
OPTION = "expandable_segments"


def allocator_env(environ: Mapping[str, str]) -> tuple[str, str]:
    """The (variable, value) to set so the allocator uses expandable segments.

    Edits whichever variable the operator already set (the newer name first), so their other
    options survive; an explicit `expandable_segments:False` is their decision and is kept.
    """
    name = NEW_NAME if environ.get(NEW_NAME) else OLD_NAME
    current = (environ.get(name) or "").strip().strip(",")
    if OPTION in current:
        return name, current
    return name, f"{current},{OPTION}:True" if current else f"{OPTION}:True"
