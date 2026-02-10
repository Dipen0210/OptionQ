# pricing/payoff.py
"""
Option payoff modelling — single-leg and multi-leg strategy payoff diagrams.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def call_payoff(S: float, K: float, premium: float = 0.0) -> float:
    """
    Long call payoff at expiration.

    Parameters
    ----------
    S : float   Underlying price at expiration.
    K : float   Strike price.
    premium : float   Premium paid (cost to enter).

    Returns
    -------
    float   Net P/L per share.
    """
    return max(S - K, 0.0) - premium


def put_payoff(S: float, K: float, premium: float = 0.0) -> float:
    """
    Long put payoff at expiration.

    Parameters
    ----------
    S : float   Underlying price at expiration.
    K : float   Strike price.
    premium : float   Premium paid.

    Returns
    -------
    float   Net P/L per share.
    """
    return max(K - S, 0.0) - premium


def _leg_payoff(
    S: float,
    strike: float,
    option_type: str,
    position: str,
    premium: float,
    quantity: int,
) -> float:
    """Compute a single leg's payoff at price S."""
    if option_type.lower() == "call":
        intrinsic = max(S - strike, 0.0)
    else:
        intrinsic = max(strike - S, 0.0)

    if position.lower() in ("long", "buy"):
        pnl = intrinsic - premium
    else:
        pnl = premium - intrinsic

    return pnl * quantity


def payoff_diagram(
    strategy_legs: List[dict],
    price_range: tuple[float, float] | None = None,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Generate a multi-leg payoff diagram (at expiration).

    Parameters
    ----------
    strategy_legs : list[dict]
        Each leg dict must contain:
          - 'strike': float
          - 'option_type': 'call' or 'put'
          - 'position': 'long' or 'short'
          - 'premium': float (per-share cost)
          - 'quantity': int (number of contracts, default 1)
    price_range : tuple[float, float] | None
        Min/max underlying prices to evaluate. If None, auto-ranged
        from strikes ± 30%.
    n_points : int
        Number of price points.

    Returns
    -------
    DataFrame
        Columns: 'UnderlyingPrice', 'Payoff' (net P/L of entire structure).

    Examples
    --------
    Bull call spread:

    >>> legs = [
    ...     {"strike": 100, "option_type": "call", "position": "long",  "premium": 5.0, "quantity": 1},
    ...     {"strike": 110, "option_type": "call", "position": "short", "premium": 2.0, "quantity": 1},
    ... ]
    >>> df = payoff_diagram(legs)
    """
    if not strategy_legs:
        return pd.DataFrame(columns=["UnderlyingPrice", "Payoff"])

    strikes = [leg["strike"] for leg in strategy_legs]
    if price_range is None:
        mid = np.mean(strikes)
        spread = max(max(strikes) - min(strikes), mid * 0.1)
        lo = min(strikes) - spread * 1.5
        hi = max(strikes) + spread * 1.5
        price_range = (max(lo, 0.0), hi)

    prices = np.linspace(price_range[0], price_range[1], n_points)
    payoffs = np.zeros(n_points)

    for leg in strategy_legs:
        for i, s in enumerate(prices):
            payoffs[i] += _leg_payoff(
                S=s,
                strike=leg["strike"],
                option_type=leg["option_type"],
                position=leg["position"],
                premium=leg.get("premium", 0.0),
                quantity=leg.get("quantity", 1),
            )

    return pd.DataFrame({
        "UnderlyingPrice": prices,
        "Payoff": payoffs,
    })
