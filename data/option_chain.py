# data/option_chain.py
"""Option chain data fetching via yfinance."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


def get_nearest_expiration(ticker: str, min_days: int = 15) -> Optional[str]:
    """
    Find the nearest expiration date that is at least *min_days* away.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    min_days : int
        Minimum days to expiration (to avoid expiring-tomorrow contracts).

    Returns
    -------
    str | None
        Expiration date string (YYYY-MM-DD), or None if no valid expiration found.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        # Force a refresh if needed (sometimes .options is cached empty)
        if not yf_ticker.options:
            return None
        
        today = datetime.now().date()
        valid_expirations = []
        for exp_str in yf_ticker.options:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            if days_to_exp >= min_days:
                valid_expirations.append((days_to_exp, exp_str))
        
        if not valid_expirations:
            return None
        
        # Sort by days to expiry and return the nearest one
        valid_expirations.sort()
        return valid_expirations[0][1]
    except Exception:
        return None


def get_option_chain(ticker: str, expiration: str | None = None) -> pd.DataFrame:
    """
    Fetch call and put option chains for a given *expiration*.

    If *expiration* is None, uses the nearest expiration >= 30 days out.

    Returns
    -------
    DataFrame
        Columns: [contractSymbol, strike, type, bid, ask, lastPrice, volume, openInterest, impliedVolatility, expiration]
    """
    try:
        if expiration is None:
            expiration = get_nearest_expiration(ticker, min_days=30)
            if expiration is None:
                return pd.DataFrame()

        yf_ticker = yf.Ticker(ticker)
        chain = yf_ticker.option_chain(expiration)
        
        calls = chain.calls.copy()
        calls["type"] = "call"
        
        puts = chain.puts.copy()
        puts["type"] = "put"
        
        df = pd.concat([calls, puts], ignore_index=True)
        df["expiration"] = expiration
        df["underlying"] = ticker
        
        # Ensure standardized columns
        cols = [
            "contractSymbol", "strike", "type", "bid", "ask", 
            "lastPrice", "volume", "openInterest", "impliedVolatility", 
            "expiration", "underlying"
        ]
        # Some yfinance versions might miss bid/ask if market closed, fill with 0
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
                
        return df[cols]
    except Exception as e:
        print(f"Error fetching option chain for {ticker}: {e}")
        return pd.DataFrame()
