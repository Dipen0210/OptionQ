# strategies/quality.py
"""Quality strategy — ranks tickers by low-volatility + Sharpe composite."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_quality(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score each ticker using a risk-adjusted quality composite:
    - Annualized Sharpe ratio (higher is better)
    - Inverse annualized volatility (lower vol is better)

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
        if len(close) < 10:
            continue

        daily_returns = close.pct_change().dropna()
        if daily_returns.empty or daily_returns.std(ddof=1) < 1e-9:
            continue

        ann_return = daily_returns.mean() * 252
        ann_vol = daily_returns.std(ddof=1) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        inv_vol = 1.0 / ann_vol if ann_vol > 0 else 0.0

        # Composite: 60% Sharpe + 40% inverse vol (normalised later via ranking)
        score = 0.6 * sharpe + 0.4 * inv_vol
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
