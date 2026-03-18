"""
pipeline.py
-----------
End-to-end orchestration for the Nursing Home Quality Predictor.

Run
---
    python pipeline.py

What it does
------------
1. Fetch nursing home provider data from the CMS public API
   (falls back to synthetic data if the API is unreachable).
2. Clean and validate the data with Pandas / NumPy.
3. Run statistical analysis and outlier detection.
4. Engineer features and train / compare three classifiers
   (Logistic Regression, Random Forest, XGBoost).
5. Evaluate the best model on a held-out test set.
6. Generate and save all Matplotlib visualisations.
7. Serialise the best model to disk with joblib.
"""

import logging
import sys
import time
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────────────────────
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
_stream_handler.setFormatter(_fmt)
_file_handler = logging.FileHandler("outputs/pipeline.log", mode="w", encoding="utf-8")
_file_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger(__name__)

# ── Project imports ──────────────────────────────────────────────────────────
from src.ingestion      import fetch_cms_data
from src.cleaning       import clean
from src.analysis       import run_analysis
from src.model          import (
    engineer_features,
    build_target,
    evaluate_models,
    train_best_model,
    save_model,
    get_feature_importance,
)
from src.visualizations import (
    plot_rating_distribution,
    plot_correlation_heatmap,
    plot_staffing_vs_rating,
    plot_outlier_summary,
    plot_feature_importance,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_cv_comparison,
)

Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  Nursing Home Quality Predictor — Pipeline Start")
    logger.info("=" * 60)

    # ── Step 1: Ingest ───────────────────────────────────────────────────────
    logger.info("\n[1/6] Data Ingestion")
    raw_df = fetch_cms_data(max_records=15_000)
    logger.info(f"  Loaded {len(raw_df):,} rows  |  {raw_df.shape[1]} columns")

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    logger.info("\n[2/6] Data Cleaning")
    clean_df = clean(raw_df)

    # ── Step 3: Analysis ─────────────────────────────────────────────────────
    logger.info("\n[3/6] Statistical Analysis")
    outlier_flags, stats_df, corr = run_analysis(clean_df)

    logger.info("\n  Generating EDA visualisations...")
    plot_rating_distribution(clean_df)
    plot_correlation_heatmap(clean_df)
    plot_staffing_vs_rating(clean_df)
    plot_outlier_summary(outlier_flags)

    # ── Step 4: Feature Engineering ──────────────────────────────────────────
    logger.info("\n[4/6] Feature Engineering")
    feat_df = engineer_features(clean_df)
    X, y = build_target(feat_df)
    logger.info(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # ── Step 5: Model Selection ──────────────────────────────────────────────
    logger.info("\n[5/6] Model Training & Cross-Validation")
    cv_results = evaluate_models(X, y)
    plot_cv_comparison(cv_results)

    # ── Step 6: Final Evaluation ─────────────────────────────────────────────
    logger.info("\n[6/6] Final Model Evaluation")
    model, X_test, y_test, y_pred, y_proba, feature_cols = train_best_model(X, y, cv_results)

    importances = get_feature_importance(model, feature_cols)
    if not importances.empty:
        plot_feature_importance(importances)

    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)

    save_model(model)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Plots saved to  → outputs/plots/")
    logger.info(f"  Model saved to  → outputs/models/best_model.joblib")
    logger.info(f"  Log saved to    → outputs/pipeline.log")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
