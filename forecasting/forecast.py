# forecasting/forecast.py
"""Compute expected (mean) returns from historical price data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expected_returns(
    price_data: dict[str, pd.DataFrame],
    window: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Estimate annualised expected returns from trailing log-return means.

    Parameters
    ----------
    price_data : dict[str, DataFrame]
        Ticker -> OHLCV DataFrame with a 'Close' column.
    window : int
        Number of trailing observations to use (capped at available data).

    Returns
    -------
    (expected_returns_df, expected_returns_series)
        DataFrame with columns ['Ticker', 'Expected_Return'] and a
        Series indexed by ticker.
    """
    records: list[dict] = []
    for ticker, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            continue

        # Use min(window, available) observations
        effective = close.iloc[-min(window, len(close)):]
        log_returns = np.log(effective / effective.shift(1)).dropna()
        if log_returns.empty:
            continue

        daily_mean = float(log_returns.mean())
        annualised = daily_mean * 252
        records.append({
            "Ticker": ticker,
            "Expected_Return": annualised,
        })

    if not records:
        empty_df = pd.DataFrame(columns=["Ticker", "Expected_Return"])
        empty_series = pd.Series(dtype=float, name="Expected_Return")
        return empty_df, empty_series

    result_df = pd.DataFrame(records)
    result_series = (
        result_df.set_index("Ticker")["Expected_Return"]
        .rename("Expected_Return")
    )
    return result_df, result_series
