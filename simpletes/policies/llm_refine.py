"""LLM-refinement policies: first pass uses a base selector (PUCT or
RPUCG) to shortlist ``llm_policy_pool_size`` candidates, then an
LLM second-pass prunes the shortlist down to the requested ``n``."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from litellm import completion

from simpletes.evaluator import rich_print
from simpletes.templates import (
    MCTS_INSPIRATION_ITEM_TEMPLATE,
    MCTS_SELECTION_PROMPT_TEMPLATE,
)

from .base import register_selector
from .puct import PuctPolicy
from .rpucg import RpucgPolicy

if TYPE_CHECKING:
    from simpletes.node import Node


class _LLMReranker:
    """Ask an LLM to pick ``n`` candidates out of a shortlist, parse the reply."""

    _SELECTED_RE = re.compile(r"#SELECTED:\s*([\d,\s]+)")

    def __init__(self, model: str, api_base: str, api_key: str) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key

    def select(self, candidates: list[Node], n: int) -> list[Node]:
        if not candidates or n >= len(candidates):
            return candidates

        inspiration_items = "".join(
            MCTS_INSPIRATION_ITEM_TEMPLATE.format(
                index=i, code=node.code, reflection=node.reflection,
            )
            for i, node in enumerate(candidates)
        )
        prompt = MCTS_SELECTION_PROMPT_TEMPLATE.format(
            num_candidates=len(candidates),
            n_select=n,
            inspiration_items=inspiration_items,
        )
        try:
            res = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            response_text = (res.choices[0].message.content or "").strip()
            match = self._SELECTED_RE.search(response_text)
            if match:
                raw_indices = match.group(1).replace(",", " ").split()
                indices = [int(s) for s in raw_indices if s.strip().isdigit()]
                selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
                if selected:
                    return selected[:n]
        except Exception:
            # Silent; fall through to top-n fallback below.
            pass

        rich_print(
            "[yellow]Failed to parse output from llm inspiration selector, "
            "falling back to base selector's top-n...[/yellow]"
        )
        return candidates[:n]


class _LLMRefineMixin:
    """Shared LLM-refine behaviour. Composes with either PuctPolicy or
    RpucgPolicy; the mixin supplies the LLM shortlist pass while the base
    class supplies the initial score-based shortlist."""

    selector: _LLMReranker
    llm_policy_pool_size: int

    def _init_llm_refine(self, num_inspirations: int) -> None:
        self.llm_policy_pool_size = num_inspirations
        self.selector = _LLMReranker(
            model=self.llm_policy_model,            # type: ignore[attr-defined]
            api_base=self.llm_policy_api_base,      # type: ignore[attr-defined]
            api_key=self.llm_policy_api_key,        # type: ignore[attr-defined]
        )

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()                    # type: ignore[misc]
        info["llm_pool_size"] = self.llm_policy_pool_size
        info["llm_model"] = self.selector.model
        return info

    def _select_from_chain(
        self,
        chain_idx: int,
        chain_nodes: list[Node],
        n: int,
    ) -> list[Node]:
        # Step 1: base selector picks llm_policy_pool_size shortlist by score.
        candidates = super()._select_from_chain(     # type: ignore[misc]
            chain_idx, chain_nodes, self.llm_policy_pool_size,
        )
        if not candidates:
            return []
        # Step 2: LLM prunes the shortlist to the final n.
        try:
            return self.selector.select(candidates, n)
        except Exception as e:
            rich_print(f"[red]Error during LLM inspiration selection: {e}[/red]")
            return candidates[:n]


@register_selector("llm_puct")
class LLMPuctPolicy(_LLMRefineMixin, PuctPolicy):
    """PUCT shortlist + LLM final selection."""

    def __init__(self, llm_policy_pool_size: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._init_llm_refine(llm_policy_pool_size)


@register_selector("llm_rpucg")
class LLMRpucgPolicy(_LLMRefineMixin, RpucgPolicy):
    """RPUCG shortlist + LLM final selection."""

    def __init__(self, llm_policy_pool_size: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._init_llm_refine(llm_policy_pool_size)
