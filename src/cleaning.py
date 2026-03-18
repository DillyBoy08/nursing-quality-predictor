"""
cleaning.py
-----------
Data cleaning and type coercion for nursing home provider data.
Handles missing values, outliers, and categorical encoding using
Pandas and NumPy.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns expected to be numeric after ingestion
NUMERIC_COLS = [
    "overall_rating",
    "staffing_rating",
    "quality_rating",
    "health_inspection_rating",
    "num_beds",
    "num_residents",
    "total_nursing_hrs",
    "rn_hrs",
    "aide_hrs",
    "num_deficiencies",
]

# Acceptable ranges for domain validation
VALID_RANGES = {
    "overall_rating":           (1, 5),
    "staffing_rating":          (1, 5),
    "quality_rating":           (1, 5),
    "health_inspection_rating": (1, 5),
    "num_beds":                 (1, 1_500),
    "num_residents":            (0, 1_500),
    "total_nursing_hrs":        (0.5, 24),
    "rn_hrs":                   (0.0, 24),
    "aide_hrs":                 (0.0, 24),
    "num_deficiencies":         (0, 200),
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for raw nursing home data.

    Steps
    -----
    1. Drop columns with >60 % missing values.
    2. Coerce numeric columns; non-parseable values become NaN.
    3. Clamp values to valid domain ranges (winsorisation).
    4. Impute remaining numeric NaNs with column medians.
    5. Strip and normalise the ownership_type categorical.
    6. Drop rows still missing the target label (overall_rating).
    7. Reset index.

    Parameters
    ----------
    df : pd.DataFrame  Raw dataframe from ingestion module.

    Returns
    -------
    pd.DataFrame  Cleaned dataframe ready for feature engineering.
    """
    logger.info(f"Cleaning started — {len(df):,} rows, {df.shape[1]} columns.")

    df = df.copy()

    # ── 1. Drop columns that are mostly empty ───────────────────────────────
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.6].index.tolist()
    if drop_cols:
        logger.info(f"  Dropping {len(drop_cols)} high-null columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # ── 2. Coerce numerics ──────────────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3. Clamp to valid domain ranges (winsorise outliers) ────────────────
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            logger.debug(f"  Clamped '{col}' to [{lo}, {hi}].")

    # ── 4. Impute numeric NaNs with column medians ──────────────────────────
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    medians = df[numeric_present].median()
    missing_before = df[numeric_present].isnull().sum().sum()
    df[numeric_present] = df[numeric_present].fillna(medians)
    logger.info(f"  Imputed {missing_before:,} numeric NaNs with column medians.")

    # ── 5. Normalise ownership_type ─────────────────────────────────────────
    if "ownership_type" in df.columns:
        df["ownership_type"] = (
            df["ownership_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
        # Collapse rare categories into 'other'
        top_types = df["ownership_type"].value_counts().head(5).index
        df["ownership_type"] = np.where(
            df["ownership_type"].isin(top_types), df["ownership_type"], "other"
        )

    # ── 6. Drop rows without a target label ─────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["overall_rating"])
    dropped = before - len(df)
    if dropped:
        logger.info(f"  Dropped {dropped:,} rows with missing overall_rating.")

    df = df.reset_index(drop=True)
    logger.info(f"Cleaning complete — {len(df):,} rows remaining.")
    return df
