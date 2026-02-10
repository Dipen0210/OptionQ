# pricing/black_scholes.py
"""
Black-Scholes option pricing model.

Provides closed-form European call/put pricing and implied-volatility solver.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ------------------------------------------------------------------
# Pricing
# ------------------------------------------------------------------

def bs_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    European call price under the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free interest rate (e.g. 0.05 for 5%).
    sigma : float
        Annualised volatility of the underlying (e.g. 0.20 for 20%).

    Returns
    -------
    float
        Theoretical call option price.
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def bs_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    European put price under the Black-Scholes model.

    Uses put-call parity: P = C - S + K * exp(-rT).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Same as :func:`bs_call_price`.

    Returns
    -------
    float
        Theoretical put option price.
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ------------------------------------------------------------------
# Implied Volatility
# ------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_vol: float = 5.0,
) -> float:
    """
    Solve for the implied volatility that reproduces *market_price*.

    Uses Brent's method over the interval [1e-6, max_vol].

    Parameters
    ----------
    market_price : float
        Observed option market price.
    S, K, T, r : float
        Underlying price, strike, time to expiry, risk-free rate.
    option_type : str
        'call' or 'put'.
    tol : float
        Solver tolerance.
    max_vol : float
        Upper bound for volatility search.

    Returns
    -------
    float
        Implied volatility (annualised).

    Raises
    ------
    ValueError
        If no root is found in the search interval.
    """
    pricer = bs_call_price if option_type.lower() == "call" else bs_put_price

    def objective(sigma: float) -> float:
        return pricer(S, K, T, r, sigma) - market_price

    try:
        iv = brentq(objective, 1e-6, max_vol, xtol=tol)
    except ValueError as exc:
        raise ValueError(
            f"Could not find implied volatility in [0, {max_vol}] for "
            f"market_price={market_price}, S={S}, K={K}, T={T}, r={r}, "
            f"type={option_type}"
        ) from exc
    return float(iv)
