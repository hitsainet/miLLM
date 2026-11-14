"""Task-head architectures: the ONE table that says which checkpoints are not language models.

A checkpoint's `config.json` names the class it was saved with (`architectures`). A name ending
in a task-head suffix (`BartForSequenceClassification`, `DebertaV2ForSequenceClassification`,
`RobertaForQuestionAnswering`, `BertForMaskedLM`, ...) is a model that emits label, span,
choice or masked-position logits, never a continuation. Two consumers read this table and must agree:

- the loader (`model_loader._get_auto_model_class`) builds the head with its own Auto class;
- the generation endpoints refuse the model for chat/completions BEFORE a load
  (`openai.errors.non_generative_task`), with the pipeline tag in the third column.

The refusal was keyed only on the Hub `pipeline_tag` stored on the model row, so a classifier
registered from a LOCAL folder (no Hub call, no tag) slipped past it, loaded, reached
`generate()` and failed with a 500. Deriving the tag from `config.json` with THIS table closes
that, for new and already-registered rows alike, without a second hand-kept list.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Iterable, Optional

#: (architecture-name suffix, transformers Auto class, Hub pipeline tag of that head).
TASK_HEADS: tuple[tuple[str, str, str], ...] = (
    ("ForSequenceClassification", "AutoModelForSequenceClassification", "text-classification"),
    ("ForTokenClassification", "AutoModelForTokenClassification", "token-classification"),
    ("ForQuestionAnswering", "AutoModelForQuestionAnswering", "question-answering"),
    ("ForMultipleChoice", "AutoModelForMultipleChoice", "multiple-choice"),
    # A masked LM predicts a token AT a [MASK] position; it cannot continue text. Before
    # 2026-10-18 (operator decision) BertForMaskedLM fell through to AutoModelForCausalLM, which
    # built BertLMHeadModel (a decoder it never was) and served chats from it; ModernBert's
    # MaskedLM has no causal class at all.
    ("ForMaskedLM", "AutoModelForMaskedLM", "fill-mask"),
)

#: Every pipeline tag a task head maps to (all non-generative).
TASK_HEAD_PIPELINE_TAGS: frozenset[str] = frozenset(tag for _, _, tag in TASK_HEADS)


def task_of_architectures(architectures: Optional[Iterable[Any]]) -> Optional[str]:
    """The pipeline tag of the first task-head architecture named, else None."""
    for arch in architectures or ():
        if not isinstance(arch, str):
            continue
        for suffix, _auto_class, tag in TASK_HEADS:
            if arch.endswith(suffix):
                return tag
    return None


@lru_cache(maxsize=256)
def _task_of_config_file(path: str, _mtime_ns: int) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    return task_of_architectures(config.get("architectures"))


def task_of_model_dir(model_dir: Any) -> Optional[str]:
    """The task-head pipeline tag named by `<model_dir>/config.json`, else None.

    None for anything that is not a readable directory with a JSON config (a GGUF folder, a
    path that has gone away, a non-string). Cached by (path, mtime), so the per-request check
    reads the file once per change.
    """
    if not isinstance(model_dir, (str, os.PathLike)) or not str(model_dir):
        return None
    path = os.path.join(os.fspath(model_dir), "config.json")
    try:
        mtime_ns = os.stat(path).st_mtime_ns
    except OSError:
        return None
    return _task_of_config_file(path, mtime_ns)
