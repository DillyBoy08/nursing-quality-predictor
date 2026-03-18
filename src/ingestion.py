"""
ingestion.py
------------
Pulls nursing home provider data from the CMS (Centers for Medicare & Medicaid
Services) public Socrata API. Falls back to a NumPy-generated synthetic dataset
if the API is unavailable, ensuring the pipeline can always run end-to-end.
"""

import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# CMS Provider Information dataset (Nursing Home Compare)
CMS_API_URL = "https://data.cms.gov/resource/4pq5-n9py.json"
BATCH_SIZE = 5_000

# Map CMS column names -> standardised internal names used throughout pipeline
CMS_COLUMN_MAP = {
    "overall_rating": "overall_rating",
    "staffing_rating": "staffing_rating",
    "quality_rating": "quality_rating",
    "health_inspection_rating": "health_inspection_rating",
    "number_of_certified_beds": "num_beds",
    "average_number_of_residents_per_day": "num_residents",
    "total_nursing_hours_per_resident_per_day": "total_nursing_hrs",
    "reported_rn_hours_per_resident_per_day": "rn_hrs",
    "reported_total_nurse_aide_hours_per_resident_per_day": "aide_hrs",
    "ownership_type": "ownership_type",
    "total_number_of_health_deficiencies": "num_deficiencies",
}


def fetch_cms_data(max_records: int = 15_000) -> pd.DataFrame:
    """
    Fetch nursing home provider records from the CMS Socrata API.

    Parameters
    ----------
    max_records : int
        Maximum number of rows to retrieve (API is paginated via $limit/$offset).

    Returns
    -------
    pd.DataFrame
        Raw provider data with standardised column names.
    """
    records = []
    offset = 0

    logger.info("Fetching data from CMS API...")

    while offset < max_records:
        batch_limit = min(BATCH_SIZE, max_records - offset)
        params = {"$limit": batch_limit, "$offset": offset}

        try:
            response = requests.get(CMS_API_URL, params=params, timeout=30)
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            records.extend(batch)
            offset += len(batch)
            logger.info(f"  Retrieved {len(records):,} records...")

            if len(batch) < batch_limit:
                break

        except requests.RequestException as exc:
            logger.warning(f"API request failed at offset {offset}: {exc}")
            break

    if records:
        df = pd.DataFrame(records)
        df = df.rename(columns={k: v for k, v in CMS_COLUMN_MAP.items() if k in df.columns})
        df["data_source"] = "cms_api"
        logger.info(f"CMS API returned {len(df):,} rows.")
        return df

    logger.warning("CMS API unavailable — generating synthetic dataset.")
    return _generate_synthetic_data()


def _generate_synthetic_data(n: int = 8_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic nursing home dataset using NumPy.

    A latent quality factor drives correlated star ratings and staffing metrics,
    mirroring the structure found in real CMS data.

    Parameters
    ----------
    n    : int   Number of synthetic facilities to generate.
    seed : int   RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the same schema as CMS data.
    """
    rng = np.random.default_rng(seed)

    # Latent quality score — higher = better facility
    quality = rng.normal(0, 1, n)

    def _star_rating(factor: np.ndarray, noise: float = 0.6) -> np.ndarray:
        """Convert a continuous factor to a 1–5 integer star rating."""
        raw = factor + rng.normal(0, noise, n)
        return np.clip(np.round(np.interp(raw, [-2.5, 2.5], [1, 5])).astype(int), 1, 5)

    overall_rating       = _star_rating(quality)
    staffing_rating      = _star_rating(quality)
    quality_rating       = _star_rating(quality)
    health_inspection    = _star_rating(-quality)  # more deficiencies → lower inspection score

    num_beds      = rng.integers(20, 350, n)
    occupancy     = rng.beta(8, 2, n)               # most facilities run near capacity
    num_residents = np.round(num_beds * occupancy).astype(float)

    total_nursing_hrs = np.clip(rng.normal(3.8, 0.9, n) + quality * 0.4, 1.2, 9.0)
    rn_hrs            = np.clip(total_nursing_hrs * rng.beta(2, 6, n), 0.1, 4.0)
    aide_hrs          = np.clip(total_nursing_hrs - rn_hrs + rng.normal(0, 0.3, n), 0.3, 7.0)

    ownership_type = rng.choice(
        ["For profit - Corporation", "Non profit - Corporation", "Government - State/County"],
        n,
        p=[0.65, 0.25, 0.10],
    )

    # Deficiencies negatively correlated with quality
    num_deficiencies = np.clip(
        rng.poisson(8, n) - (quality * 2.5).astype(int), 0, 60
    )

    return pd.DataFrame({
        "overall_rating":        overall_rating,
        "staffing_rating":       staffing_rating,
        "quality_rating":        quality_rating,
        "health_inspection_rating": health_inspection,
        "num_beds":              num_beds,
        "num_residents":         num_residents,
        "total_nursing_hrs":     total_nursing_hrs,
        "rn_hrs":                rn_hrs,
        "aide_hrs":              aide_hrs,
        "ownership_type":        ownership_type,
        "num_deficiencies":      num_deficiencies,
        "data_source":           "synthetic",
    })
