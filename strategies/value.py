# strategies/value.py
"""Value strategy — ranks tickers by price-to-high ratio (discount from peak)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_value(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score each ticker by how far below its trailing high the current price sits.

    A lower price-to-high ratio indicates a deeper discount (better value).
    The score is inverted so that the deepest discounts rank highest.

    Parameters
    ----------
    price_data : dict[str, DataFrame]
        Mapping of ticker -> OHLCV DataFrame with a 'Close' column.

    Returns
    -------
    DataFrame with columns ['Ticker', 'Strategy_Score', 'Rank']
    """
    records: list[dict] = []
    for ticker, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            continue

        trailing_high = close.max()
        if trailing_high <= 0:
            continue

        price_to_high = close.iloc[-1] / trailing_high
        # Invert: deeper discount (lower ratio) → higher score
        score = 1.0 - float(price_to_high)
        records.append({
            "Ticker": ticker,
            "Strategy_Score": score,
        })

    if not records:
        return pd.DataFrame(columns=["Ticker", "Strategy_Score", "Rank"])

    result = pd.DataFrame(records)
    result = result.sort_values("Strategy_Score", ascending=False).reset_index(drop=True)
    result["Rank"] = range(1, len(result) + 1)
    return result
