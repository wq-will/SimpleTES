"""
Elite context template for LLMElitePolicy.
"""

ELITE_CONTEXT_TEMPLATE = """\
[ELITE POOL OVERVIEW] ({num_elites} diverse solutions, scores and insights only)
This shows the breadth of approaches already explored. Use this to:
- Understand what directions have been tried
- Avoid duplicating existing approaches
- Identify gaps for novel solutions

{entries}"""

ELITE_ENTRY_TEMPLATE = """\
#{index} [Score: {score}]
{metrics_line}{reflection_line}
"""
