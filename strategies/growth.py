# strategies/growth.py
"""Growth strategy — ranks tickers by trailing return + positive-return consistency."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_growth(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score each ticker using a composite of trailing return and the fraction
    of days with positive returns (consistency).

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
        if len(close) < 5:
            continue

        daily_returns = close.pct_change().dropna()
        if daily_returns.empty:
            continue

        trailing_return = (close.iloc[-1] / close.iloc[0]) - 1.0
        positive_ratio = float((daily_returns > 0).sum()) / len(daily_returns)

        # Composite: 60% trailing return + 40% consistency
        score = 0.6 * trailing_return + 0.4 * positive_ratio
        records.append({
            "Ticker": ticker,
            "Strategy_Score": float(score),
        })

    if not records:
        return pd.DataFrame(columns=["Ticker", "Strategy_Score", "Rank"])

    result = pd.DataFrame(records)
    result = result.sort_values("Strategy_Score", ascending=False).reset_index(drop=True)
    result["Rank"] = range(1, len(result) + 1)
    return result
