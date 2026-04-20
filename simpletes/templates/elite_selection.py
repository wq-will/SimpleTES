"""
Elite selection prompt template for LLMElitePolicy.
"""

ELITE_SELECTION_PROMPT_TEMPLATE = """\
You are curating an Elite Pool to maximize DIVERSITY of approaches for future exploration.

Task: {task_instruction}

### Core Principle: DIVERSITY FIRST
The elite pool should represent as many DISTINCT approaches as possible.
Similar solutions waste pool slots - keep only ONE representative per approach.

### Pool Status
Current pool size: {current_size}/{elite_limit}

### Selection Rules (in priority order):
1. **New Best Score → ADD or REPLACE**
   - If candidate has the HIGHEST score (better than all pool entries), prefer ADD or REPLACE
   - Do NOT simply reject a new best - score improvement matters

2. **Maximize Approach Diversity**
   - Each solution should represent a fundamentally different strategy
   - Similar implementations = redundancy = should be eliminated

3. **Among Similar Solutions, Keep**:
   - The one with HIGHER score, OR
   - The one with SIMPLER/cleaner implementation (if scores are close)
   - Remove the redundant copy

4. **Reject Low-Value Candidates**:
   - Solutions too similar to others already in the pool
   - Low-scoring solutions with no unique insights
   - Solutions with fundamental errors that provide no learning value

### Current Pool:
{pool_description}
### New Candidate:
{new_candidate_description}
### Decision:
{decision_instruction}

Output format:
## ACTION {{action}}
## REASON: {{brief reason}}
"""
