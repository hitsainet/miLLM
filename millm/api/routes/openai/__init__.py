"""
OpenAI-compatible API routes.

Provides endpoints compatible with the OpenAI API specification:
- POST /v1/chat/completions
- POST /v1/completions
- POST /v1/embeddings
- GET /v1/models
- GET /v1/models/{model_id}
"""

from fastapi import APIRouter

from millm.api.routes.openai.chat import router as chat_router
from millm.api.routes.openai.completions import router as completions_router
from millm.api.routes.openai.embeddings import router as embeddings_router
from millm.api.routes.openai.models import router as models_router

# Aggregate all OpenAI routes under a single router
openai_router = APIRouter(tags=["openai"])

# Include sub-routers
openai_router.include_router(chat_router)
openai_router.include_router(completions_router)
openai_router.include_router(embeddings_router)
openai_router.include_router(models_router)

__all__ = ["openai_router"]
