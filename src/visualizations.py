"""
visualizations.py
-----------------
Matplotlib-based plots for the nursing home quality predictor.
All plots are saved to outputs/plots/.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

logger = logging.getLogger(__name__)

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colour palette
PALETTE = sns.color_palette("Blues_d", 5)
sns.set_theme(style="whitegrid", palette="Blues_d")


def _save(fig: plt.Figure, filename: str) -> None:
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def plot_rating_distribution(df: pd.DataFrame) -> None:
    """Bar chart of overall star-rating distribution."""
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["overall_rating"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title("Distribution of Overall Star Ratings", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Star Rating (1 = Lowest, 5 = Highest)")
    ax.set_ylabel("Number of Facilities")
    ax.set_xticks([1, 2, 3, 4, 5])
    _save(fig, "rating_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap for numeric features."""
    numeric = df.select_dtypes(include=np.number).drop(
        columns=["ownership_encoded", "data_source"], errors="ignore"
    )
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    _save(fig, "correlation_heatmap.png")


def plot_staffing_vs_rating(df: pd.DataFrame) -> None:
    """Box plots: nursing hours per resident per day by star rating."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    cols = {
        "total_nursing_hrs": "Total Nursing Hrs / Resident / Day",
        "rn_hrs":            "RN Hrs / Resident / Day",
        "aide_hrs":          "Aide Hrs / Resident / Day",
    }

    for ax, (col, label) in zip(axes, cols.items()):
        if col not in df.columns:
            continue
        df.boxplot(column=col, by="overall_rating", ax=ax, grid=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[2], color="navy"),
                   medianprops=dict(color="white", linewidth=2))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Star Rating")
        ax.set_ylabel("")

    fig.suptitle("Staffing Levels by Overall Star Rating", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "staffing_by_rating.png")


def plot_outlier_summary(outlier_flags: dict) -> None:
    """Horizontal bar chart showing outlier percentage per column."""
    cols  = list(outlier_flags.keys())
    pcts  = [flags.mean() * 100 for flags in outlier_flags.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cols, pcts, color=sns.color_palette("Reds_d", len(cols)), edgecolor="white")

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Outlier Prevalence by Feature (IQR Method)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("% Flagged as Outlier")
    ax.invert_yaxis()
    _save(fig, "outlier_summary.png")


def plot_feature_importance(importances: pd.Series) -> None:
    """Horizontal bar chart of model feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5))
    imp = importances.sort_values(ascending=True)
    colors = sns.color_palette("Blues_d", len(imp))
    ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.set_title("Feature Importances — Best Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score")
    _save(fig, "feature_importance.png")


def plot_roc_curve(y_test, y_proba) -> None:
    """ROC curve with AUC for the best model."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE[3], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE[3])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Best Model", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    _save(fig, "roc_curve.png")


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Low Quality", "High Quality"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix — Best Model", fontsize=13, fontweight="bold", pad=12)
    _save(fig, "confusion_matrix.png")


def plot_cv_comparison(cv_results: dict) -> None:
    """Bar chart comparing CV F1 scores across all models."""
    names  = list(cv_results.keys())
    scores = list(cv_results.values())
    colors = sns.color_palette("Blues_d", len(names))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.8)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CV F1-Macro Score")
    ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=13, fontweight="bold", pad=12)
    _save(fig, "model_comparison.png")
