# strategies/mean_reversion.py
"""Mean-reversion strategy — ranks tickers by z-score distance from rolling mean."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_mean_reversion(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score each ticker by how far below its rolling mean the current price sits.

    A more negative z-score indicates a deeper reversion opportunity, which is
    scored higher (inverted so that the most oversold stocks rank first).

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

        window = min(20, len(close))
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std(ddof=1)

        last_mean = rolling_mean.iloc[-1]
        last_std = rolling_std.iloc[-1]
        last_close = close.iloc[-1]

        if np.isnan(last_std) or last_std < 1e-9:
            continue

        z_score = (last_close - last_mean) / last_std
        # Invert: lower z-score = more oversold = better opportunity
        score = -float(z_score)
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
