# optimization/weight_optimizer.py
"""Mean-variance portfolio optimisation using scipy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def mean_variance_optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_level: str = "Medium",
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """
    Optimise portfolio weights using mean-variance framework.

    Parameters
    ----------
    mu : Series
        Expected returns (annualised), indexed by ticker.
    cov : DataFrame
        Covariance matrix (annualised or daily — consistent with *mu*).
    risk_level : str
        'Low' → min-variance, 'Medium' → max Sharpe, 'High' → max return.
    risk_free_rate : float
        Annualised risk-free rate (used in Sharpe objective).

    Returns
    -------
    Series
        Optimal weights indexed by ticker, summing to 1.
    """
    tickers = mu.index.tolist()
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float, name="weight")
    if n == 1:
        return pd.Series([1.0], index=tickers, name="weight")

    mu_arr = mu.values.astype(float)
    cov_arr = cov.reindex(index=tickers, columns=tickers).values.astype(float)

    # Ensure positive semi-definite via small ridge
    cov_arr += np.eye(n) * 1e-8

    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = np.ones(n) / n

    if risk_level == "Low":
        # Minimum variance
        def objective(w):
            return float(w @ cov_arr @ w)
    elif risk_level == "High":
        # Maximise expected return (negative for minimiser)
        def objective(w):
            return -float(w @ mu_arr)
    else:
        # Maximum Sharpe ratio (Medium / default)
        def objective(w):
            port_return = float(w @ mu_arr)
            port_vol = float(np.sqrt(w @ cov_arr @ w))
            if port_vol < 1e-10:
                return 1e6
            return -(port_return - risk_free_rate) / port_vol

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        # Fall back to equal weight
        weights = np.ones(n) / n
    else:
        weights = result.x

    # Clamp tiny negatives to zero and renormalise
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > 0:
        weights /= total

    return pd.Series(weights, index=tickers, name="weight")
