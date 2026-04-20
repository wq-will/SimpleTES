"""
MCTS selection prompt template for LLM-based inspiration selection.
"""

MCTS_INSPIRATION_ITEM_TEMPLATE = """\
=== Candidate Index: {index} ===
[Code Implementation]:
{code}
[Reflection & Insight]:
{reflection}
"""

MCTS_SELECTION_PROMPT_TEMPLATE = """\
You are an AI research assistant. Below are {num_candidates} candidate solutions (Inspirations) generated in an evolutionary search. Your task is to select the top {n_select} items that are most promising, diverse, or provide the best insights for the next generation.

{inspiration_items}
Please output the indices of the exactly {n_select} selected candidates.
Your response must end with a single line in the format: #SELECTED: idx1, idx2, ...
Example: #SELECTED: 0, 3, 5"""
