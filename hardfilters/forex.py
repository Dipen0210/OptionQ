# hardfilters/forex.py
"""Forex universe utilities — load pairs and filter by base currency."""
from __future__ import annotations

import os

import pandas as pd


_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "forex.csv",
)


def load_forex_universe(path: str | None = None) -> pd.DataFrame:
    """
    Load the FX pair universe CSV.

    Expected columns include at least 'BASE', 'QUOTE', 'YF_SYMBOL'.
    """
    filepath = path or _DEFAULT_PATH
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Forex universe file not found: {filepath}"
        )
    df = pd.read_csv(filepath)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def available_base_currencies(data: pd.DataFrame) -> list[str]:
    """Return sorted list of unique base currencies present in the dataset."""
    if data is None or data.empty or "BASE" not in data.columns:
        return []
    return sorted(data["BASE"].dropna().unique().tolist())


def list_pairs_for_bases(
    bases: list[str],
    data: pd.DataFrame | None = None,
    path: str | None = None,
) -> pd.DataFrame:
    """
    Return all FX pairs whose base currency is in *bases*.

    Parameters
    ----------
    bases : list[str]
        Base currencies to include (e.g. ['USD', 'EUR']).
    data : DataFrame | None
        Pre-loaded FX universe. If None, loads from disk.
    path : str | None
        Override CSV path (only used when *data* is None).

    Returns
    -------
    DataFrame
        Filtered rows with at least columns: BASE, QUOTE, YF_SYMBOL.
    """
    if data is None:
        data = load_forex_universe(path)
    if data.empty or "BASE" not in data.columns:
        return pd.DataFrame(columns=["BASE", "QUOTE", "YF_SYMBOL"])
    mask = data["BASE"].isin([b.upper() for b in bases])
    return data[mask].reset_index(drop=True)
