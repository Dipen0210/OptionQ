# strategies/registry.py
"""Strategy registry — resolves user-selected strategy names to runner callables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .momentum import run_momentum
from .mean_reversion import run_mean_reversion
from .value import run_value
from .growth import run_growth
from .quality import run_quality


@dataclass
class StrategySpec:
    """Metadata for a registered strategy."""
    name: str
    lookback_window: int  # default trading-day lookback


# ---------------------------------------------------------------------------
# Registry map
# ---------------------------------------------------------------------------
_STRATEGY_REGISTRY: dict[str, tuple[StrategySpec, Callable]] = {
    "momentum": (
        StrategySpec(name="Momentum", lookback_window=90),
        run_momentum,
    ),
    "mean reversion": (
        StrategySpec(name="Mean Reversion", lookback_window=60),
        run_mean_reversion,
    ),
    "value": (
        StrategySpec(name="Value", lookback_window=120),
        run_value,
    ),
    "growth": (
        StrategySpec(name="Growth", lookback_window=90),
        run_growth,
    ),
    "quality": (
        StrategySpec(name="Quality", lookback_window=60),
        run_quality,
    ),
}


def resolve_strategy(name: str) -> tuple[StrategySpec, Callable]:
    """
    Look up a strategy by user-friendly name.

    Parameters
    ----------
    name : str
        Case-insensitive strategy name (e.g. "Momentum", "Mean Reversion").

    Returns
    -------
    tuple[StrategySpec, Callable]
        The strategy specification and its scoring function.

    Raises
    ------
    NotImplementedError
        If the requested strategy is not registered.
    """
    key = name.strip().lower()
    entry = _STRATEGY_REGISTRY.get(key)
    if entry is None:
        available = ", ".join(
            spec.name for spec, _ in _STRATEGY_REGISTRY.values()
        )
        raise NotImplementedError(
            f"Strategy '{name}' is not implemented. Available: {available}"
        )
    return entry
