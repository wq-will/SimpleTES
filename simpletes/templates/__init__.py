"""
Prompt templates for SimpleTES.

All templates are string constants with placeholders for formatting.
"""

from simpletes.templates.reflection import (
    REFLECTION_PROMPT_TEMPLATE,
    SEV2_REFLECTION_PROMPT_TEMPLATE,
)
from simpletes.templates.generation import (
    GENERATION_PROMPT_TEMPLATE,
    INSPIRATION_TEMPLATE,
    FAILURE_PATTERNS_TEMPLATE,
)
from simpletes.templates.elite_selection import ELITE_SELECTION_PROMPT_TEMPLATE
from simpletes.templates.elite_context import ELITE_CONTEXT_TEMPLATE, ELITE_ENTRY_TEMPLATE
from simpletes.templates.mcts_selection import (
    MCTS_SELECTION_PROMPT_TEMPLATE,
    MCTS_INSPIRATION_ITEM_TEMPLATE,
)

__all__ = [
    # Reflection
    "REFLECTION_PROMPT_TEMPLATE",
    "SEV2_REFLECTION_PROMPT_TEMPLATE",
    # Generation
    "GENERATION_PROMPT_TEMPLATE",
    "INSPIRATION_TEMPLATE",
    "FAILURE_PATTERNS_TEMPLATE",
    # Elite selection (llm_elite)
    "ELITE_SELECTION_PROMPT_TEMPLATE",
    # Elite context (for generator)
    "ELITE_CONTEXT_TEMPLATE",
    "ELITE_ENTRY_TEMPLATE",
    # MCTS selection
    "MCTS_SELECTION_PROMPT_TEMPLATE",
    "MCTS_INSPIRATION_ITEM_TEMPLATE",
]
