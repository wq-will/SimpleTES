"""
Reflection prompt template.
"""

REFLECTION_PROMPT_TEMPLATE = """\
Instruction for an LLM:
{llm_input}

The LLM generated solution:
{code}
New solution metrics:
{metrics}

Your task is to write reflections for this solution.
Write a **concise plain-text summary with exactly two paragraphs(each with 3-5 sentences)**:
Approach: <what idea the solution tried>
Insight: <what lessons can be learned for future improvement>
Your response can only have these two paragraphs. The first paragraph must starts with 'Approach:' and the second paragraph must starts with 'Insight:'.
"""


SEV2_REFLECTION_PROMPT_TEMPLATE = """\
Instruction for an LLM:
{llm_input}

The LLM generated solution:
{code}
New solution metrics:
{metrics}

Your task is to write reflections for this solution.
Write a **concise plain-text summary with exactly two lines**:
Approach: <what idea the solution tried>
Insight: <what lessons can be learned for future improvement>
Your response can only have these two lines. The response must starts with 'Approach:'
"""
