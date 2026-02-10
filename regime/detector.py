# regime/detector.py
"""
Market regime detection — classifies volatility and trend regimes
from price data to inform strategy selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Volatility Regime
# ------------------------------------------------------------------

def detect_volatility_regime(
    price_series: pd.Series,
    window: int = 21,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
) -> str:
    """
    Classify the current volatility regime.

    Uses annualised realised volatility over a trailing window.

    Parameters
    ----------
    price_series : Series
        Close prices (must have >= *window* + 1 observations).
    window : int
        Rolling window in trading days (default 21 ≈ 1 month).
    low_threshold : float
        Below this annualised vol → "low".
    high_threshold : float
        Above this annualised vol → "high".

    Returns
    -------
    str
        One of "low", "normal", "high".
    """
    close = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(close) < window + 1:
        return "normal"

    log_returns = np.log(close / close.shift(1)).dropna()
    trailing = log_returns.iloc[-window:]
    realised_vol = float(trailing.std(ddof=1) * np.sqrt(252))

    if realised_vol < low_threshold:
        return "low"
    elif realised_vol > high_threshold:
        return "high"
    return "normal"


# ------------------------------------------------------------------
# Trend Regime
# ------------------------------------------------------------------

def detect_trend_regime(
    price_series: pd.Series,
    fast: int = 20,
    slow: int = 50,
) -> str:
    """
    Classify trend regime using dual moving-average crossover.

    Parameters
    ----------
    price_series : Series
        Close prices.
    fast : int
        Fast moving-average window.
    slow : int
        Slow moving-average window.

    Returns
    -------
    str
        "bullish" (fast > slow), "bearish" (fast < slow), or "neutral".
    """
    close = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(close) < slow:
        return "neutral"

    ma_fast = float(close.iloc[-fast:].mean())
    ma_slow = float(close.iloc[-slow:].mean())

    # Use a 1% dead-zone to avoid whipsaws
    diff_pct = (ma_fast - ma_slow) / ma_slow if ma_slow != 0 else 0.0
    if diff_pct > 0.01:
        return "bullish"
    elif diff_pct < -0.01:
        return "bearish"
    return "neutral"


# ------------------------------------------------------------------
# Combined Regime
# ------------------------------------------------------------------

def classify_regime(
    price_series: pd.Series,
    vol_window: int = 21,
    fast: int = 20,
    slow: int = 50,
) -> dict:
    """
    Combined regime classification.

    Parameters
    ----------
    price_series : Series
        Close prices.
    vol_window, fast, slow : int
        Configuration for sub-detectors.

    Returns
    -------
    dict
        Keys: 'volatility_regime', 'trend_regime', 'combined_label'.

    The combined_label follows the pattern "<trend>_<vol>", e.g.
    "bullish_low", "bearish_high", "neutral_normal".
    """
    vol_regime = detect_volatility_regime(price_series, window=vol_window)
    trend_regime = detect_trend_regime(price_series, fast=fast, slow=slow)
    return {
        "volatility_regime": vol_regime,
        "trend_regime": trend_regime,
        "combined_label": f"{trend_regime}_{vol_regime}",
    }
