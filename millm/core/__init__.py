"""
Core module for miLLM.

Contains configuration, error handling, and logging setup.
"""

from millm.core.config import settings
from millm.core.errors import MiLLMError

__all__ = ["settings", "MiLLMError"]
