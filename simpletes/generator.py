"""
Code generation module for SimpleTES.

Handles LLM client lifecycle, prompt building, and code extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

from simpletes.config import EngineConfig
from simpletes.llm import LLMBackend, create_llm_client
from simpletes.node import Node, EvolveBlockContext, extract_code_detailed, score_key
from simpletes.utils.text import summarize_error
from simpletes.templates import (
    GENERATION_PROMPT_TEMPLATE,
    INSPIRATION_TEMPLATE,
    FAILURE_PATTERNS_TEMPLATE,
)

if TYPE_CHECKING:
    pass

_PROMPT_ERROR_MAX_CHARS = 240


# ============================================================================
# Generation Task
# ============================================================================

@dataclass
class GenerationTask:
    """A task for the generation worker."""
    prompt: str
    inspiration_ids: list[str]
    k: int
    chain_idx: int
    gen_id: int
    shared_construction_id: str | None = None


# ============================================================================
# Generation Result
# ============================================================================

@dataclass
class GenerationResult:
    """Result of a single generation attempt."""
    success: bool
    code: str | None = None
    reason: str | None = None
    # LLM I/O tracking (only populated when save_llm_io is enabled)
    llm_input: str | None = None
    llm_output: str | None = None
    token_usage: dict[str, Any] | None = None


# ============================================================================
# Formatting helpers
# ============================================================================

def format_inspiration(
    index: int,
    code: str,
    score: float | None,
    metrics: dict[str, Any] | None,
    reflection: str | None,
) -> str:
    """Format a single inspiration using the template."""
    metrics_lines: list[str] = []
    for k, v in (metrics or {}).items():
        if k == "error":
            err_summary = summarize_error(str(v), _PROMPT_ERROR_MAX_CHARS)
            metrics_lines.append(f"  {k}: {err_summary}")
        elif isinstance(v, float):
            metrics_lines.append(f"  {k}: {v:.6f}")
        else:
            metrics_lines.append(f"  {k}: {v}")

    reflection_block = ""
    if reflection:
        reflection_block = f"\nReflection:\n{reflection.strip()}\n"

    return INSPIRATION_TEMPLATE.format(
        index=index,
        score=score,
        metrics_text="\n".join(metrics_lines),
        reflection_block=reflection_block,
        code=code,
    )


def format_failure_patterns(
    failure_patterns: dict[str, float],
    max_chars: int = _PROMPT_ERROR_MAX_CHARS,
) -> str:
    """Format failure patterns using the template."""
    if not failure_patterns:
        return ""

    failure_lines = "\n".join(
        f"  - {summarize_error(err, max_chars)}: {ratio*100:.1f}%"
        for err, ratio in failure_patterns.items()
    )
    return FAILURE_PATTERNS_TEMPLATE.format(failure_lines=failure_lines)


def format_failure_patterns_inline(
    failure_patterns: dict[str, float],
    max_chars: int = _PROMPT_ERROR_MAX_CHARS,
) -> str:
    """Format failure patterns with inline prose (used by ``rpucg``)."""
    if not failure_patterns:
        return ""

    failure_lines = "\n".join(
        f"  - {summarize_error(err, max_chars)}: {ratio*100:.1f}%"
        for err, ratio in failure_patterns.items()
    )
    return (
        "\n\nCommon failure patterns to avoid (error: frequency over all trials):\n"
        f"{failure_lines}\n"
    )


# ============================================================================
# Generator
# ============================================================================

class Generator:
    """Handles LLM lifecycle, prompt building, and code extraction."""

    def __init__(
        self,
        config: EngineConfig,
        instruction: str,
        evolve_context: EvolveBlockContext,
        available_packages: list[str] | None = None,
    ) -> None:
        self._config = config
        self._instruction = instruction
        self._evolve_context = evolve_context
        self._available_packages = [pkg for pkg in (available_packages or []) if pkg]
        self._gen_id_counter: int = 0

        self._llm: LLMBackend = create_llm_client(config)

    def close(self) -> None:
        """Clean up LLM resources."""
        self._llm.close()

    # ==================== Gen ID Management ====================

    def next_gen_id(self) -> int:
        gid = self._gen_id_counter
        self._gen_id_counter += 1
        return gid

    def get_gen_id_counter(self) -> int:
        return self._gen_id_counter

    def set_gen_id_counter(self, value: int) -> None:
        self._gen_id_counter = value

    # ==================== Task Creation ====================

    def create_task(
        self,
        prompt: str,
        inspiration_ids: list[str],
        k: int,
        chain_idx: int,
        shared_construction_id: str | None = None,
    ) -> GenerationTask:
        gen_id = self.next_gen_id()
        return self.create_task_with_gen_id(
            prompt=prompt,
            inspiration_ids=inspiration_ids,
            k=k,
            chain_idx=chain_idx,
            gen_id=gen_id,
            shared_construction_id=shared_construction_id,
        )

    def create_task_with_gen_id(
        self,
        prompt: str,
        inspiration_ids: list[str],
        k: int,
        chain_idx: int,
        gen_id: int,
        shared_construction_id: str | None = None,
    ) -> GenerationTask:
        return GenerationTask(
            prompt=prompt,
            inspiration_ids=inspiration_ids,
            k=k,
            chain_idx=chain_idx,
            gen_id=gen_id,
            shared_construction_id=shared_construction_id,
        )

    # ==================== Prompt Building ====================

    def build_prompt(
        self,
        inspirations: Sequence[Node],
        failure_patterns: dict[str, float] | None = None,
        policy_context: str = "",
        shared_construction_summary: str | None = None,
    ) -> str:
        """Implements ``ConstructQuery`` of Alg.~\\ref{alg:async_method}:
        combines the task instruction $x$, the inspiration set $S^{(c)}$, and
        the three prompt-side signals drawn from $\\mathcal{R}^{(c)}$
        (failure histogram, reflection context, shared-construction summary)
        into the generator query $q^{(c)}$."""
        # Format inspirations
        sorted_insps = sorted(inspirations, key=score_key, reverse=True)
        insp_chunks = [
            format_inspiration(i, node.code, node.score, node.metrics, node.reflection)
            for i, node in enumerate(sorted_insps, 1)
        ]
        insp_text = "".join(insp_chunks)

        # Format failure patterns
        failure_text = ""
        if failure_patterns and self._config.include_failure_patterns:
            if self._config.selector == "rpucg":
                failure_text = format_failure_patterns_inline(failure_patterns)
            else:
                failure_text = format_failure_patterns(failure_patterns)

        # Format available packages
        available_packages_text = ""
        if self._available_packages:
            package_lines = "\n".join(f"- {pkg}" for pkg in self._available_packages)
            available_packages_text = (
                "\nAvailable packages from requirements.txt "
                "(these are the task-local dependencies you can use):\n"
                f"{package_lines}\n"
            )

        # Format shared construction
        shared_construction_text = ""
        if shared_construction_summary:
            shared_construction_text = (
                "\nShared runtime variable available in the evaluated program:\n"
                "- `GLOBAL_BEST_CONSTRUCTION` contains the validated construction from the current chain-best solution.\n"
                f"- Summary: {shared_construction_summary}\n"
                "- You may use it as a warm start / initialization / seed if helpful.\n"
                "- Treat it as optional: handle `None`, incompatible shapes, and task-specific adaptation safely.\n"
            )

        # Format policy context
        policy_context_section = ""
        if policy_context:
            policy_context_section = f"\n{policy_context}\n"

        if self._config.selector == "rpucg":
            prompt = (
                f"Task: {self._instruction}\n\n"
                "Generation instruction (must follow exactly):\n"
                "1) Only the code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END is extracted.\n"
                "2) The final program is reconstructed as EXACT_PREFIX + evolved_block + EXACT_SUFFIX.\n"
                "3) Keep marker lines exactly as written.\n"
                "4) Return one Python code block that includes both EVOLVE-BLOCK markers.\n\n"
                "EXACT_PREFIX (kept unchanged):\n"
                f"```python\n{self._evolve_context.prefix.rstrip(chr(10))}\n```\n\n"
                "EXACT_SUFFIX (kept unchanged):\n"
                f"```python\n{self._evolve_context.suffix.rstrip(chr(10))}\n```\n"
                f"{available_packages_text}\n"
                f"{shared_construction_text}\n"
                f"In-context inspirations (sorted by score, higher is better):\n{insp_text}\n"
                f"{failure_text}"
                f"Analyze the metrics above and create an improved program with higher combined_score. "
                "Try diverse approaches to solve the problem. Think outside the box. "
            )
        else:
            prompt = GENERATION_PROMPT_TEMPLATE.format(
                instruction=self._instruction,
                prefix=self._evolve_context.prefix.rstrip("\n"),
                suffix=self._evolve_context.suffix.rstrip("\n"),
                available_packages_text=available_packages_text + shared_construction_text,
                policy_context_section=policy_context_section,
                num_inspirations=len(sorted_insps),
                inspirations_text=insp_text,
                failure_text=failure_text,
            )

        # Debug: print first N lines of prompt
        if self._config.debug_prompt_lines > 0:
            lines = prompt.split("\n")
            n = self._config.debug_prompt_lines
            preview = "\n".join(lines[:n])
            if len(lines) > n:
                preview += f"\n... ({len(lines) - n} more lines)"
            print(f"\n{'='*60}\n[Generator Prompt Preview ({n} lines)]\n{'='*60}\n{preview}\n{'='*60}\n")

        return prompt

    # ==================== Generation ====================

    async def generate(
        self,
        task: GenerationTask,
        instance_id: str,
        track_io: bool = False,
    ) -> list[GenerationResult]:
        """Call LLM and extract code from responses.

        Returns a list of GenerationResult, one per LLM response.
        May return fewer than task.k results if LLM fails.
        """
        llm_results = await self._llm.generate_batch(
            task.prompt, n=task.k, instance_id=instance_id, track_io=track_io
        )

        results: list[GenerationResult] = []
        for llm_result in llm_results:
            code, reason = extract_code_detailed(llm_result.text, self._evolve_context)
            failure_reason = getattr(llm_result, "error_reason", None) or reason
            if code:
                results.append(GenerationResult(
                    success=True,
                    code=code,
                    reason=reason,
                    llm_input=getattr(llm_result, "prompt", None),
                    llm_output=getattr(llm_result, "raw_output", None) or llm_result.text,
                    token_usage=getattr(llm_result, "token_usage", None),
                ))
            else:
                results.append(GenerationResult(
                    success=False,
                    reason=failure_reason,
                    llm_output=getattr(llm_result, "raw_output", None) or llm_result.text,
                ))
        return results
