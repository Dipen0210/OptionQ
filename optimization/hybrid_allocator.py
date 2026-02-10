# optimization/hybrid_allocator.py
"""Blend rank-based and optimizer-based portfolio weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def allocate_hybrid_weights(
    final_portfolio_df: pd.DataFrame,
    optimized_weights: pd.Series,
    rank_blend: float = 0.5,
) -> pd.DataFrame:
    """
    Produce blended weights combining rank-based intuition with mean-variance
    optimiser outputs.

    Parameters
    ----------
    final_portfolio_df : DataFrame
        Must contain columns 'Ticker' and 'Strategy_Score' (higher = better).
    optimized_weights : Series
        Optimised weights indexed by ticker.
    rank_blend : float
        Weight given to rank-based allocation (0-1). Default 0.5 = equal blend.

    Returns
    -------
    DataFrame
        Columns: Ticker, Rank_Weight, Opt_Weight, Final_Weight.
    """
    tickers = final_portfolio_df["Ticker"].unique().tolist()
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Rank_Weight", "Opt_Weight", "Final_Weight"])

    # --- Rank-based weights (proportional to strategy score) ---
    score_map: dict[str, float] = {}
    for _, row in final_portfolio_df.iterrows():
        t = row["Ticker"]
        s = float(row.get("Strategy_Score", 0.0))
        score_map[t] = max(s, 0.0)

    total_score = sum(score_map.values())
    if total_score > 0:
        rank_weights = {t: s / total_score for t, s in score_map.items()}
    else:
        n = len(tickers)
        rank_weights = {t: 1.0 / n for t in tickers}

    # --- Optimiser weights ---
    opt_weights: dict[str, float] = {}
    for t in tickers:
        opt_weights[t] = float(optimized_weights.get(t, 0.0))
    opt_total = sum(opt_weights.values())
    if opt_total > 0:
        opt_weights = {t: v / opt_total for t, v in opt_weights.items()}
    else:
        n = len(tickers)
        opt_weights = {t: 1.0 / n for t in tickers}

    # --- Blend ---
    rows: list[dict] = []
    for t in tickers:
        rw = rank_weights.get(t, 0.0)
        ow = opt_weights.get(t, 0.0)
        fw = rank_blend * rw + (1.0 - rank_blend) * ow
        rows.append({
            "Ticker": t,
            "Rank_Weight": rw,
            "Opt_Weight": ow,
            "Final_Weight": fw,
        })

    result = pd.DataFrame(rows)
    # Renormalise Final_Weight to sum to 1
    fw_sum = result["Final_Weight"].sum()
    if fw_sum > 0:
        result["Final_Weight"] = result["Final_Weight"] / fw_sum
    return result
