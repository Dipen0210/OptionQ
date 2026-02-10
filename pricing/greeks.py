# pricing/greeks.py
"""
Black-Scholes Greeks — first- and second-order sensitivities for European options.

All Greeks use the standard Black-Scholes closed-form expressions.
"""
from __future__ import annotations

import math

from scipy.stats import norm

from .black_scholes import _d1, _d2


# ------------------------------------------------------------------
# First-order Greeks
# ------------------------------------------------------------------

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Option delta — sensitivity of price to underlying price change.

    Call delta ∈ [0, 1], Put delta ∈ [-1, 0].
    """
    if T <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def vega(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Option vega — sensitivity of price to 1% change in volatility.

    Same value for calls and puts. Returns per-percentage-point change
    (i.e. value for a 0.01 increase in σ).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * math.sqrt(T) * 0.01)


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Option theta — time decay per calendar day.

    Typically negative (options lose value as time passes).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    shared = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type.lower() == "call":
        term = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term = r * K * math.exp(-r * T) * norm.cdf(-d2)
    # Convert from per-year to per-calendar-day
    return float((shared + term) / 365.0)


def rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Option rho — sensitivity of price to 1% change in interest rate.

    Returns the change for a 0.01 increase in r.
    """
    if T <= 0:
        return 0.0
    d2 = _d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(K * T * math.exp(-r * T) * norm.cdf(d2) * 0.01)
    return float(-K * T * math.exp(-r * T) * norm.cdf(-d2) * 0.01)


# ------------------------------------------------------------------
# Second-order Greeks
# ------------------------------------------------------------------

def gamma(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Option gamma — rate of change of delta w.r.t. underlying price.

    Same value for calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def vanna(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Vanna — cross sensitivity of delta to volatility (∂Δ/∂σ).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return float(-norm.pdf(d1) * d2 / sigma)


def charm(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Charm — rate of change of delta w.r.t. time (∂Δ/∂T), a.k.a. delta decay.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    term = 2.0 * r * T - d2 * sigma * math.sqrt(T)
    denom = 2.0 * T * sigma * math.sqrt(T)
    if option_type.lower() == "call":
        return float(-pdf_d1 * term / denom)
    return float(-pdf_d1 * term / denom)
