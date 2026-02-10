# hardfilters/equities.py
"""Filter the equity universe by risk level (market-cap proxy) and sector."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# Market-cap thresholds (in billions) mapped from risk label
_CAP_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Low":    (10.0, float("inf")),   # Large Cap ≥ $10B
    "Medium": (2.0, 10.0),            # Mid Cap $2B–$10B
    "High":   (0.0, 2.0),             # Small Cap < $2B
}


def filter_universe_by_risk_and_sector(
    company_df: pd.DataFrame,
    sectors: list[str] | None = None,
    risk_level: str = "Medium",
    top_n_per_sector: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Slice the company universe into per-sector DataFrames filtered by
    market-cap range (risk proxy) and optional sector whitelist.

    Parameters
    ----------
    company_df : DataFrame
        Must contain columns: 'ticker', 'sector', 'market cap'.
    sectors : list[str] | None
        If provided, keep only these sectors. Otherwise use all.
    risk_level : str
        One of 'Low', 'Medium', 'High'.
    top_n_per_sector : int
        Maximum number of stocks to keep per sector (by market cap desc).

    Returns
    -------
    dict[str, DataFrame]
        Mapping of sector name → filtered DataFrame.
    """
    if company_df is None or company_df.empty:
        return {}

    required = {"ticker", "sector", "market cap"}
    if not required.issubset(set(company_df.columns)):
        return {}

    df = company_df.copy()
    df["market cap"] = pd.to_numeric(df["market cap"], errors="coerce")
    df = df.dropna(subset=["ticker", "sector", "market cap"])

    # Apply market-cap filter
    lo, hi = _CAP_THRESHOLDS.get(risk_level, _CAP_THRESHOLDS["Medium"])
    cap_in_billions = df["market cap"] / 1e9
    df = df[(cap_in_billions >= lo) & (cap_in_billions < hi)]

    if df.empty:
        # Fall back: if overly restrictive, return top-N by cap regardless
        df = company_df.copy()
        df["market cap"] = pd.to_numeric(df["market cap"], errors="coerce")
        df = df.dropna(subset=["ticker", "sector", "market cap"])

    # Sector filter
    if sectors:
        df = df[df["sector"].isin(sectors)]

    if df.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for sector, group in df.groupby("sector"):
        top = (
            group.sort_values("market cap", ascending=False)
            .head(top_n_per_sector)
            .reset_index(drop=True)
        )
        if not top.empty:
            result[str(sector)] = top

    return result
