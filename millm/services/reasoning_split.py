"""Separate a reasoning trace from the answer, for `reasoning_content`.

`reasoning_content` is NOT in the OpenAI specification -- OpenAI never returns
reasoning text from Chat Completions. It is a de-facto convention started by
DeepSeek and adopted by vLLM, SGLang and most OSS servers, and it is what
Open WebUI renders as a collapsible "Thinking" section. miLLM follows the
convention so those clients work.

THE HARD CASE, and the reason a naive regex is wrong. granite-4.2-8b's chat
template puts the OPENING tag in the GENERATION PROMPT: with thinking enabled
the prompt ends `<|im_start|>assistant\\n<think>\\n`, so the model resumes
INSIDE the block and its completion is

    <reasoning...></think><answer...>

with no opening tag anywhere in it. A `<think>(.*?)</think>` pattern matches
nothing here and leaves the whole trace in `content` -- the exact failure that
made a labeling response read "Okay, let's see. The user says...".

THE DANGEROUS FAILURE runs the other way. If a bare `</think>`-less completion
from a NON-reasoning model were treated as reasoning, `content` would come back
empty and the model would look broken. So reasoning is only ever recognised on
positive evidence:

  * the completion itself opens `<think>`, or
  * the caller says the PROMPT opened one (`prompt_opened_think`), which is
    knowable exactly -- it is the string the template produced.

Absent both, the text is returned unchanged as content. Silence is the safe
default: a trace left in `content` is visible and annoying, an answer moved into
`reasoning_content` is invisible and looks like data loss.
"""

from __future__ import annotations

from typing import Optional

OPEN = "<think>"
CLOSE = "</think>"


def split_reasoning(
    text: Optional[str],
    prompt_opened_think: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    """Return (reasoning, content).

    `reasoning` is None when no trace was recognised; `content` is then the
    input unchanged.
    """
    if not text:
        return None, text

    if OPEN in text:
        head, _, rest = text.partition(OPEN)
        if CLOSE in rest:
            reasoning, _, answer = rest.partition(CLOSE)
            # Text before an opening tag is ordinary content; keep it.
            return reasoning.strip() or None, (head + answer).strip()
        # Opened and never closed: generation was cut short. Everything after
        # the tag is reasoning and there is no answer yet.
        return rest.strip() or None, head.strip()

    if prompt_opened_think:
        if CLOSE in text:
            reasoning, _, answer = text.partition(CLOSE)
            return reasoning.strip() or None, answer.strip()
        # Still reasoning when the budget ran out -- no answer was reached.
        return text.strip() or None, ""

    return None, text


class StreamingReasoningSplitter:
    """Incremental `split_reasoning` for token streams.

    Tokenisers split `</think>` across chunks, so a fixed tail is withheld until
    it is known not to be the start of a closing tag. Without that the tag
    arrives as `</th` + `ink>` and is never recognised.
    """

    def __init__(self, prompt_opened_think: bool = False) -> None:
        self._buf = ""
        self._in_reasoning = bool(prompt_opened_think)
        self._closed = False
        self._saw_open = False

    def feed(self, token: str) -> tuple[Optional[str], Optional[str]]:
        """Consume one token; return (reasoning_delta, content_delta)."""
        if self._closed:
            return None, token

        self._buf += token or ""

        if not self._in_reasoning and not self._saw_open:
            if OPEN in self._buf:
                before, _, rest = self._buf.partition(OPEN)
                self._buf = rest
                self._in_reasoning = True
                self._saw_open = True
                if before:
                    return None, before
            elif not _could_still_become(self._buf, OPEN):
                out, self._buf = self._buf, ""
                return None, out
            else:
                return None, None

        if CLOSE in self._buf:
            reasoning, _, answer = self._buf.partition(CLOSE)
            self._buf = ""
            self._closed = True
            return (reasoning or None), (answer or None)

        hold = len(CLOSE) - 1
        if len(self._buf) > hold:
            out, self._buf = self._buf[:-hold], self._buf[-hold:]
            return (out or None), None
        return None, None

    def flush(self) -> tuple[Optional[str], Optional[str]]:
        """Emit whatever is still buffered at end of stream."""
        rest, self._buf = self._buf, ""
        if not rest:
            return None, None
        if self._closed or not self._in_reasoning:
            return None, rest
        return rest, None

    @property
    def saw_reasoning(self) -> bool:
        return self._in_reasoning or self._saw_open


def _could_still_become(buf: str, tag: str) -> bool:
    """True if `buf` ends with a proper prefix of `tag` (tag may be splitting)."""
    tail = buf[-(len(tag) - 1):] if len(tag) > 1 else ""
    return any(tail.endswith(tag[:k]) for k in range(len(tag), 0, -1))
