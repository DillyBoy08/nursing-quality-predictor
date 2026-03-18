"""
ingestion.py
------------
Downloads real nursing home provider data from the CMS (Centers for Medicare
& Medicaid Services) public dataset via direct CSV download.

Data source
-----------
CMS Nursing Home Compare — Provider Information
Updated monthly. ~15,000 U.S. nursing facilities.
https://data.cms.gov/provider-data/dataset/4pq5-n9py
"""

import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Direct CSV download — real CMS Nursing Home Compare data (updated monthly)
CMS_CSV_URL = (
    "https://data.cms.gov/provider-data/api/1/datastore/query/"
    "4pq5-n9py/0/download?format=csv"
)

# Real CMS column names -> standardised internal names
CMS_COLUMN_MAP = {
    "Overall Rating":                                        "overall_rating",
    "Staffing Rating":                                       "staffing_rating",
    "QM Rating":                                             "quality_rating",
    "Health Inspection Rating":                              "health_inspection_rating",
    "Number of Certified Beds":                              "num_beds",
    "Average Number of Residents per Day":                   "num_residents",
    "Reported Total Nurse Staffing Hours per Resident per Day": "total_nursing_hrs",
    "Reported RN Staffing Hours per Resident per Day":       "rn_hrs",
    "Reported Nurse Aide Staffing Hours per Resident per Day": "aide_hrs",
    "Ownership Type":                                        "ownership_type",
    "Rating Cycle 1 Total Number of Health Deficiencies":    "num_deficiencies",
}


def fetch_cms_data(max_records: int = 15_000) -> pd.DataFrame:
    """
    Download the CMS Nursing Home Provider Information dataset as CSV.

    Returns real data for ~15,000 U.S. nursing facilities including
    star ratings, staffing hours, deficiency counts, and ownership type.
    Falls back to synthetic data only if the download fails.

    Parameters
    ----------
    max_records : int
        Cap on rows returned (the full dataset is ~15,000 facilities).

    Returns
    -------
    pd.DataFrame
        Provider data with standardised internal column names.
    """
    logger.info("Downloading CMS Nursing Home Provider data (real dataset)...")

    try:
        response = requests.get(CMS_CSV_URL, timeout=60)
        response.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text), low_memory=False)

        # Keep only the columns we need
        available = {k: v for k, v in CMS_COLUMN_MAP.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)
        df = df.head(max_records)
        df["data_source"] = "cms_real"

        logger.info(f"Real CMS data loaded: {len(df):,} facilities.")
        return df

    except requests.RequestException as exc:
        logger.warning(f"CMS download failed: {exc}")
        logger.warning("Falling back to synthetic dataset.")
        return _generate_synthetic_data()


def _generate_synthetic_data(n: int = 8_000, seed: int = 42) -> pd.DataFrame:
    """
    Fallback: generate synthetic nursing home data using NumPy.
    Only used if the CMS download is unavailable.
    """
    rng = np.random.default_rng(seed)
    quality = rng.normal(0, 1, n)

    def _star_rating(factor, noise=0.6):
        raw = factor + rng.normal(0, noise, n)
        return np.clip(np.round(np.interp(raw, [-2.5, 2.5], [1, 5])).astype(int), 1, 5)

    num_beds      = rng.integers(20, 350, n)
    occupancy     = rng.beta(8, 2, n)
    num_residents = np.round(num_beds * occupancy).astype(float)
    total_nursing = np.clip(rng.normal(3.8, 0.9, n) + quality * 0.4, 1.2, 9.0)
    rn_hrs        = np.clip(total_nursing * rng.beta(2, 6, n), 0.1, 4.0)
    aide_hrs      = np.clip(total_nursing - rn_hrs + rng.normal(0, 0.3, n), 0.3, 7.0)

    return pd.DataFrame({
        "overall_rating":           _star_rating(quality),
        "staffing_rating":          _star_rating(quality),
        "quality_rating":           _star_rating(quality),
        "health_inspection_rating": _star_rating(-quality),
        "num_beds":                 num_beds,
        "num_residents":            num_residents,
        "total_nursing_hrs":        total_nursing,
        "rn_hrs":                   rn_hrs,
        "aide_hrs":                 aide_hrs,
        "ownership_type":           rng.choice(
            ["For profit - Corporation", "Non profit - Corporation",
             "Government - State/County"], n, p=[0.65, 0.25, 0.10]
        ),
        "num_deficiencies":         np.clip(
            rng.poisson(8, n) - (quality * 2.5).astype(int), 0, 60
        ),
        "data_source": "synthetic",
    })
