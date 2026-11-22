"""Shared helpers for the ShaiLearnsAI notebooks.

Teaching code lives in the notebooks. This package holds only the pieces
that several labs call in the same way: seeding, the pluggable language-model
client, and small checks used by the tests.
"""

from labs._shared.llm import get_llm
from labs._shared.seed import seed_everything

__all__ = ["get_llm", "seed_everything"]
