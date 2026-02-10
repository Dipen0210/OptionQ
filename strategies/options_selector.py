# strategies/options_selector.py
"""Option contract selector based on strategy rules."""
from __future__ import annotations

import pandas as pd


def select_option_contract(
    strategy_name: str,
    chain_df: pd.DataFrame,
) -> dict | None:
    """
    Select the optimal single-leg contract for a given strategy.

    Parameters
    ----------
    strategy_name : str
        Strategy name (e.g. "Momentum", "Value").
    chain_df : DataFrame
        Option chain enriched with Greeks (must have 'delta', 'type').

    Returns
    -------
    dict | None
        Selected contract row as a dict, or None if no suitable contract found.
    """
    if chain_df.empty or "delta" not in chain_df.columns:
        return None

    # Filter for valid open interest to ensure liquidity
    liquid = chain_df[chain_df["openInterest"] > 10].copy()
    if liquid.empty:
        liquid = chain_df  # Fallback if no liquidity data

    target_contract = None

    name = strategy_name.lower()
    
    if name in ["momentum", "growth"]:
        # BULLISH: Long Call (Delta ~0.30 to 0.50)
        # Look for calls
        calls = liquid[liquid["type"] == "call"]
        if not calls.empty:
            # Sort by distance from Delta 0.40
            calls = calls.copy()  # explicit copy to avoid SettingWithCopyWarning
            calls["delta_dist"] = abs(calls["delta"] - 0.40)
            target_contract = calls.sort_values("delta_dist").iloc[0]

    elif name in ["value", "mean reversion"]:
        # MEAN REVERSION / VALUE: Often bullish (buying the dip) or neutral-bullish
        # We'll map "Value" to conservative Long Call (Delta ~0.70) aka ITM substitution
        # Or you could do Short Put (Delta ~ -0.30) if account allows
        # For simplicity in Phase 1: Deep ITM Call (proxy for stock but with leverage)
        calls = liquid[liquid["type"] == "call"]
        if not calls.empty:
            # Sort by distance from Delta 0.75
            calls = calls.copy()
            calls["delta_dist"] = abs(calls["delta"] - 0.75)
            target_contract = calls.sort_values("delta_dist").iloc[0]

    elif name == "quality":
        # QUALITY: Low Volatility → Income focus (Covered Call proxy = ITM Call)
        # OR slightly OTM call (Delta 0.50 ATM)
        calls = liquid[liquid["type"] == "call"]
        if not calls.empty:
            calls = calls.copy()
            calls["delta_dist"] = abs(calls["delta"] - 0.50)
            target_contract = calls.sort_values("delta_dist").iloc[0]

    if target_contract is not None:
        return target_contract.to_dict()
    
    return None
