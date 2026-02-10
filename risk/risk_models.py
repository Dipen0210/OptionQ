# risk/risk_models.py
"""Core risk model building blocks: log returns and covariance estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price matrix (index=date, columns=tickers).

    Parameters
    ----------
    price_matrix : DataFrame
        Close prices with DatetimeIndex and one column per ticker.

    Returns
    -------
    DataFrame
        Log returns of the same shape (minus the first row).
    """
    if price_matrix.empty:
        return pd.DataFrame()
    numeric = price_matrix.apply(pd.to_numeric, errors="coerce")
    log_ret = np.log(numeric / numeric.shift(1)).iloc[1:]
    return log_ret


def compute_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Estimate the covariance matrix of asset returns.

    Parameters
    ----------
    returns : DataFrame
        Daily returns (index=date, columns=tickers).
    method : str
        'sample' for the plain sample covariance, 'ledoit_wolf' for
        Ledoit-Wolf shrinkage.

    Returns
    -------
    DataFrame
        Covariance matrix indexed and columned by ticker.
    """
    if returns.empty:
        return pd.DataFrame()

    clean = returns.dropna(how="all").dropna(axis=1, how="all")
    if clean.shape[0] < 2 or clean.shape[1] < 1:
        return pd.DataFrame()

    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(clean.fillna(0).values)
            cov = pd.DataFrame(
                lw.covariance_,
                index=clean.columns,
                columns=clean.columns,
            )
            return cov
        except Exception:
            # Fall through to sample covariance
            pass

    cov = clean.cov(ddof=1)
    return cov
