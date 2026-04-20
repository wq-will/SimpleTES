"""
Selectors for SimpleTES.

Selectors pick historical nodes from the trajectory's committed history
to form the next generation proposal. Register new selectors with
``@register_selector("name")`` in their module.
"""
from .base import (
    BatchCompletion,
    TrajectoryPolicyBase,
    Selector,
    PendingFinalize,
    SELECTOR_REGISTRY,
    register_selector,
)

# Import all policy modules to trigger @register_selector side effects.
from . import balance  # noqa: F401
from . import puct     # noqa: F401
from . import rpucg    # noqa: F401
from . import llm_refine  # noqa: F401
from . import llm_elite  # noqa: F401


def available_policies() -> list[str]:
    return sorted(SELECTOR_REGISTRY)


def create_selector(name: str, **kwargs) -> Selector:
    """Look up a registered policy and instantiate it.

    Every policy's __init__ accepts **kwargs, so unused keys are ignored
    by the policy itself — we don't need runtime signature introspection.
    """
    if name not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown policy: {name}. Available: {', '.join(available_policies())}")
    return SELECTOR_REGISTRY[name](**kwargs)


__all__ = [
    "TrajectoryPolicyBase",
    "BatchCompletion",
    "Selector",
    "PendingFinalize",
    "register_selector",
    "create_selector",
    "available_policies",
    "SELECTOR_REGISTRY",
]
