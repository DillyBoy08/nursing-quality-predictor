"""
analysis.py
-----------
Exploratory and statistical analysis of cleaned nursing home data.
Covers:
  - Descriptive statistics
  - IQR-based outlier detection
  - Distribution summary (skewness, kurtosis)
  - Correlation with the target variable
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ANALYSIS_COLS = [
    "total_nursing_hrs",
    "rn_hrs",
    "aide_hrs",
    "num_beds",
    "num_residents",
    "num_deficiencies",
]


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return extended descriptive statistics for key numeric columns.
    Augments the standard describe() output with skewness and kurtosis.
    """
    numeric = df.select_dtypes(include=np.number)
    desc = numeric.describe().T
    desc["skewness"] = numeric.skew()
    desc["kurtosis"] = numeric.kurt()
    return desc


def detect_outliers_iqr(df: pd.DataFrame, cols: list = None) -> Dict[str, pd.Series]:
    """
    Flag outliers using the interquartile range (IQR) method.

    A value is flagged if it falls below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

    Parameters
    ----------
    df   : pd.DataFrame
    cols : list  Columns to analyse. Defaults to ANALYSIS_COLS.

    Returns
    -------
    dict  Mapping column name -> boolean Series (True = outlier).
    """
    cols = cols or [c for c in ANALYSIS_COLS if c in df.columns]
    outlier_flags: Dict[str, pd.Series] = {}

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags = (df[col] < lower) | (df[col] > upper)
        outlier_flags[col] = flags
        logger.info(
            f"  Outliers in '{col}': {flags.sum():,} "
            f"({flags.mean() * 100:.1f}%)  bounds=[{lower:.2f}, {upper:.2f}]"
        )

    return outlier_flags


def distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the distribution of each analysis column:
    mean, std, median, skewness, kurtosis, and a Shapiro-Wilk normality p-value
    (on a random 500-row sample to keep computation fast).
    """
    rows = []
    sample = df.sample(min(500, len(df)), random_state=0)

    for col in ANALYSIS_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        _, sw_p = stats.shapiro(sample[col].dropna())
        rows.append({
            "column":    col,
            "mean":      series.mean(),
            "std":       series.std(),
            "median":    series.median(),
            "skewness":  series.skew(),
            "kurtosis":  series.kurt(),
            "shapiro_p": round(sw_p, 4),
            "normal?":   "yes" if sw_p > 0.05 else "no",
        })

    return pd.DataFrame(rows).set_index("column")


def target_correlation(df: pd.DataFrame, target: str = "overall_rating") -> pd.Series:
    """
    Pearson correlation between each numeric feature and the target column,
    sorted by absolute magnitude descending.
    """
    numeric = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
    corr = numeric.corrwith(df[target]).sort_values(key=abs, ascending=False)
    return corr


def run_analysis(df: pd.DataFrame) -> None:
    """
    Run and log the full statistical analysis suite.
    """
    logger.info("── Descriptive Statistics ──────────────────────────────────")
    stats_df = descriptive_stats(df)
    logger.info(f"\n{stats_df[['mean','std','min','50%','max','skewness']].to_string()}")

    logger.info("\n── IQR Outlier Detection ───────────────────────────────────")
    outlier_flags = detect_outliers_iqr(df)
    total_outliers = sum(f.sum() for f in outlier_flags.values())
    logger.info(f"  Total outlier instances across all columns: {total_outliers:,}")

    logger.info("\n── Distribution Summary ────────────────────────────────────")
    dist_df = distribution_summary(df)
    logger.info(f"\n{dist_df.to_string()}")

    logger.info("\n── Correlation with Overall Rating ─────────────────────────")
    corr = target_correlation(df)
    logger.info(f"\n{corr.to_string()}")

    return outlier_flags, stats_df, corr
