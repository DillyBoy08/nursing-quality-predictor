"""
model.py
--------
Feature engineering, model training, and evaluation for the nursing home
quality classifier.

Task
----
Binary classification: predict whether a nursing home will achieve a
HIGH overall rating (4-5 stars) vs LOW (1-2 stars). Three-star facilities
are excluded — they are the ambiguous mid-point that would add noise.

Models compared
---------------
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost

Evaluation
----------
  Stratified 5-fold cross-validation, optimised for F1-score (macro)
  to handle the class imbalance between high and low-rated facilities.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Fixed ownership encoding — used consistently in training and the live predictor
OWNERSHIP_MAP = {
    "for profit - corporation": 0,
    "non profit - corporation": 1,
    "government - state/county": 2,
    "other": 3,
}

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "staffing_rating",
    "quality_rating",
    "health_inspection_rating",
    "num_beds",
    "num_residents",
    "total_nursing_hrs",
    "rn_hrs",
    "aide_hrs",
    "num_deficiencies",
    "occupancy_rate",
    "rn_share",
    "ownership_encoded",
]

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=1,        # adjusted below after computing class ratio
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features and encode categoricals.

    New features
    ------------
    occupancy_rate  : num_residents / num_beds  (facility utilisation)
    rn_share        : rn_hrs / total_nursing_hrs  (RN proportion of care hours)
    ownership_encoded : label-encoded ownership_type
    """
    df = df.copy()

    # Derived ratios
    df["occupancy_rate"] = np.where(
        df["num_beds"] > 0, df["num_residents"] / df["num_beds"], np.nan
    )
    df["rn_share"] = np.where(
        df["total_nursing_hrs"] > 0, df["rn_hrs"] / df["total_nursing_hrs"], np.nan
    )

    # Fill any NaNs introduced by division
    df["occupancy_rate"] = df["occupancy_rate"].fillna(df["occupancy_rate"].median())
    df["rn_share"]       = df["rn_share"].fillna(df["rn_share"].median())

    # Encode ownership type using fixed map for consistency across train/predict
    if "ownership_type" in df.columns:
        df["ownership_encoded"] = (
            df["ownership_type"].map(OWNERSHIP_MAP).fillna(3).astype(int)
        )
    else:
        df["ownership_encoded"] = 0

    return df


def build_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Binarise overall_rating:
        4–5 stars → 1  (high quality)
        1–2 stars → 0  (low quality)
        3 stars   → excluded (ambiguous)

    Returns
    -------
    X : pd.DataFrame  Feature matrix.
    y : pd.Series     Binary target.
    """
    mask = df["overall_rating"] != 3
    filtered = df[mask].copy()
    y = (filtered["overall_rating"] >= 4).astype(int)
    available = [c for c in FEATURE_COLS if c in filtered.columns]
    X = filtered[available]
    logger.info(
        f"Target distribution — High (1): {y.sum():,}  Low (0): {(1 - y).sum():,}"
    )
    return X, y


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compare models using stratified 5-fold cross-validation (F1 macro).

    Returns
    -------
    dict  Model name -> mean CV F1 score.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Adjust XGBoost scale_pos_weight for class imbalance
    neg, pos = (y == 0).sum(), (y == 1).sum()
    MODELS["XGBoost"].set_params(scale_pos_weight=neg / max(pos, 1))

    for name, model in MODELS.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        results[name] = scores.mean()
        logger.info(
            f"  {name:25s}  CV F1-macro: {scores.mean():.4f} ± {scores.std():.4f}"
        )

    return results


def train_best_model(X: pd.DataFrame, y: pd.Series, cv_results: dict) -> tuple:
    """
    Retrain the best-performing model on a 80/20 stratified train-test split,
    then evaluate on the held-out test set.

    Returns
    -------
    model        : fitted estimator
    X_test       : pd.DataFrame
    y_test       : pd.Series
    y_pred       : np.ndarray
    y_proba      : np.ndarray  (probability of class 1)
    feature_cols : list
    """
    best_name = max(cv_results, key=cv_results.get)
    logger.info(f"\nBest model: {best_name}  (CV F1={cv_results[best_name]:.4f})")

    model = MODELS[best_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else np.zeros(len(y_test))
    )

    logger.info("\n── Classification Report (test set) ────────────────────────")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Low','High'])}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    return model, X_test, y_test, y_pred, y_proba, list(X.columns)


def save_model(model, output_dir: str = "outputs/models") -> None:
    """Serialise the trained model to disk with joblib."""
    path = Path(output_dir) / "best_model.joblib"
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def get_feature_importance(model, feature_cols: list) -> pd.Series:
    """
    Extract feature importances from tree-based or linear models.
    Returns a Series sorted descending by importance magnitude.
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.Series(dtype=float)

    return pd.Series(importances, index=feature_cols).sort_values(ascending=False)
