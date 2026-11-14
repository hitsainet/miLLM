"""
Core module for miServer.

Contains configuration, error handling, and logging setup.
"""

from miserver.core.config import settings
from miserver.core.errors import MiLLMError

__all__ = ["settings", "MiLLMError"]
