"""
Service layer for miLLM.

Services contain business logic and coordinate between repositories and ML components.
"""

from millm.services.model_service import ModelService

__all__ = ["ModelService"]
