# pricing/engine.py
"""Option chain pricing engine — calculates Greeks for an entire chain."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .black_scholes import implied_volatility
from .greeks import delta, gamma, vega, theta, rho


def enrich_chain_with_greeks(
    chain_df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float = 0.045,  # 4.5% approx
) -> pd.DataFrame:
    """
    Calculate Greeks for every contract in the option chain.

    Parameters
    ----------
    chain_df : DataFrame
        Must contain [strike, type, expiration, impliedVolatility].
    underlying_price : float
        Current price of the underlying asset (S).
    risk_free_rate : float
        Annualized risk-free rate (r).

    Returns
    -------
    DataFrame
        Original chain with added columns: [delta, gamma, vega, theta, rho, calculated_IV].
    """
    if chain_df.empty:
        return chain_df

    df = chain_df.copy()
    
    # Pre-calculate time to expiration (T) in years
    today = datetime.now().date()
    
    def _calc_row(row):
        try:
            exp_date = datetime.strptime(str(row["expiration"]), "%Y-%m-%d").date()
            days = (exp_date - today).days
            T = max(days / 365.0, 0.001)  # Avoid T=0 division errors
            
            K = float(row["strike"])
            # Use provided IV if valid, else fallback (0.0 means yfinance missing data)
            sigma = float(row["impliedVolatility"])
            if sigma <= 0.001:
                sigma = 0.2  # simple fallback if data missing
                
            opt_type = str(row["type"]).lower()
            
            d = delta(underlying_price, K, T, risk_free_rate, sigma, opt_type)
            g = gamma(underlying_price, K, T, risk_free_rate, sigma)
            v = vega(underlying_price, K, T, risk_free_rate, sigma)
            t = theta(underlying_price, K, T, risk_free_rate, sigma, opt_type)
            r = rho(underlying_price, K, T, risk_free_rate, sigma, opt_type)
            
            return pd.Series([d, g, v, t, r, sigma])
        except Exception:
            return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    cols = ["delta", "gamma", "vega", "theta", "rho", "effective_iv"]
    df[cols] = df.apply(_calc_row, axis=1)
    
    return df
