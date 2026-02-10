# risk/risk_pipeline.py
"""High-level risk report builder consumed by the rebalance controller."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_models import compute_log_returns


def build_risk_report(
    price_matrix: pd.DataFrame,
    weights: pd.Series,
    mu: pd.Series | None = None,
    confidence: float = 0.95,
) -> dict:
    """
    Construct a comprehensive risk snapshot for the given portfolio.

    Parameters
    ----------
    price_matrix : DataFrame
        Close prices (index=date, columns=tickers).
    weights : Series
        Portfolio weights indexed by ticker (should sum ≈ 1).
    mu : Series | None
        Expected returns per ticker (annualised). If None, computed from data.
    confidence : float
        Confidence level for VaR calculations (default 95%).

    Returns
    -------
    dict
        Keys: annualized_vol, annualized_mean, sharpe, parametric_VaR,
        historical_VaR, stress_uniform_minus5_pct_portfolio_return.
    """
    if price_matrix.empty or weights.empty:
        return {}

    common = price_matrix.columns.intersection(weights.index)
    if common.empty:
        return {}

    prices = price_matrix[common]
    w = weights.reindex(common).fillna(0.0)
    w = w / w.sum() if w.sum() > 0 else w

    returns = compute_log_returns(prices)
    if returns.empty or returns.shape[0] < 2:
        return {}

    # Fill any remaining NaNs with 0 for matrix ops
    returns_clean = returns.fillna(0.0)

    # Portfolio daily returns
    port_returns = returns_clean.dot(w)
    ann_vol = float(port_returns.std(ddof=1) * np.sqrt(252))
    ann_mean = float(port_returns.mean() * 252)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan

    # Parametric VaR (Gaussian)
    z = float(pd.Series([confidence]).apply(
        lambda c: abs(np.percentile(np.random.standard_normal(100_000), (1 - c) * 100))
    ).iloc[0])
    # Use scipy for exact z-score if available
    try:
        from scipy.stats import norm
        z = abs(float(norm.ppf(1 - confidence)))
    except ImportError:
        z = 1.645  # 95% fallback
    daily_vol = float(port_returns.std(ddof=1))
    daily_mean = float(port_returns.mean())
    parametric_var = -(daily_mean - z * daily_vol)

    # Historical VaR
    historical_var = -float(np.percentile(port_returns, (1 - confidence) * 100))

    # Stress test: uniform -5 % shock
    stress_returns = pd.Series(-0.05, index=common)
    stress_portfolio = float(stress_returns.dot(w))

    return {
        "annualized_vol": ann_vol,
        "annualized_mean": ann_mean,
        "sharpe": sharpe,
        "parametric_VaR": parametric_var,
        "historical_VaR": historical_var,
        "stress_uniform_minus5_pct_portfolio_return": stress_portfolio,
    }
