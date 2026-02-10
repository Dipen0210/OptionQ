# strategies/momentum.py
"""Momentum strategy — ranks tickers by cumulative return over trailing window."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_momentum(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score each ticker by its cumulative return over the available price history.

    Parameters
    ----------
    price_data : dict[str, DataFrame]
        Mapping of ticker -> OHLCV DataFrame with a 'Close' column.

    Returns
    -------
    DataFrame with columns ['Ticker', 'Strategy_Score', 'Rank']
        Sorted by Rank ascending (1 = best momentum).
    """
    records: list[dict] = []
    for ticker, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            continue
        cumulative_return = (close.iloc[-1] / close.iloc[0]) - 1.0
        records.append({
            "Ticker": ticker,
            "Strategy_Score": float(cumulative_return),
        })

    if not records:
        return pd.DataFrame(columns=["Ticker", "Strategy_Score", "Rank"])

    result = pd.DataFrame(records)
    result = result.sort_values("Strategy_Score", ascending=False).reset_index(drop=True)
    result["Rank"] = range(1, len(result) + 1)
    return result
