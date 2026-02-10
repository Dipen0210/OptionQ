# hardfilters/data_utils.py
"""Data loading utilities for the company universe."""
from __future__ import annotations

import os

import pandas as pd


_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "cleaned_companies.csv",
)


def load_company_universe(path: str | None = None) -> pd.DataFrame:
    """
    Load and return the cleaned company universe CSV.

    Parameters
    ----------
    path : str | None
        Override path to the CSV file. Defaults to
        ``data/processed/cleaned_companies.csv`` relative to the project root.

    Returns
    -------
    DataFrame
        The company universe with columns such as 'ticker', 'sector', 'market cap'.
    """
    filepath = path or _DEFAULT_PATH
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Company universe file not found: {filepath}"
        )
    df = pd.read_csv(filepath)
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    return df
