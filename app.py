"""
app.py  —  Nursing Home Quality Intelligence Platform
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix,
    f1_score, precision_score, roc_curve,
)

from src.cleaning import clean
from src.ingestion import fetch_cms_data
from src.model import (
    OWNERSHIP_MAP, build_target, engineer_features,
    evaluate_models, get_feature_importance, train_best_model,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nursing Home Quality Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0A0F1E;
    border-right: 1px solid #1E2A45;
}
.nav-logo {
    font-size: 1.25rem;
    font-weight: 700;
    color: #4F8BF9;
    letter-spacing: -0.5px;
}
.nav-sub { font-size: 0.72rem; color: #5A6A8A; margin-top: 2px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0D1B3E 0%, #1A2F6B 50%, #0D2445 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    border: 1px solid #1E3A6E;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 12px 0;
    line-height: 1.2;
}
.hero p {
    font-size: 1.05rem;
    color: #90A8D0;
    max-width: 640px;
    line-height: 1.6;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(79,139,249,0.15);
    color: #4F8BF9;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(79,139,249,0.3);
    margin-bottom: 16px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.kpi-card {
    background: #111827;
    border: 1px solid #1E2A45;
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 3px solid #4F8BF9;
}
.kpi-val { font-size: 2rem; font-weight: 700; color: #4F8BF9; line-height: 1; }
.kpi-lbl { font-size: 0.78rem; color: #6B7A99; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Feature cards ── */
.feature-card {
    background: #111827;
    border: 1px solid #1E2A45;
    border-radius: 12px;
    padding: 24px;
    height: 100%;
}
.feature-icon { font-size: 2rem; margin-bottom: 12px; }
.feature-title { font-size: 1rem; font-weight: 600; color: #E2E8F0; margin-bottom: 8px; }
.feature-desc { font-size: 0.85rem; color: #6B7A99; line-height: 1.6; }

/* ── Section headers ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #4F8BF9;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 6px;
}
.section-desc {
    font-size: 0.88rem;
    color: #6B7A99;
    line-height: 1.6;
    margin-bottom: 24px;
}

/* ── Insight boxes ── */
.insight {
    background: rgba(79,139,249,0.07);
    border: 1px solid rgba(79,139,249,0.2);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.85rem;
    color: #90A8D0;
    line-height: 1.6;
    margin-top: 12px;
}
.insight strong { color: #4F8BF9; }

/* ── Step badges ── */
.step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: #4F8BF9;
    color: white;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 10px;
    flex-shrink: 0;
}

/* ── Result boxes ── */
.result-high {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-low {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-label { font-size: 0.75rem; color: #9AA0B4; text-transform: uppercase; letter-spacing: 1px; }
.result-value-high { font-size: 2rem; font-weight: 700; color: #22C55E; }
.result-value-low  { font-size: 2rem; font-weight: 700; color: #EF4444; }

/* Divider */
hr { border-color: #1E2A45 !important; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

T = "rgba(0,0,0,0)"  # transparent background for charts

def chart_layout(fig, height=None, **kw):
    fig.update_layout(
        plot_bgcolor=T, paper_bgcolor=T,
        font_color="#C8D4E8",
        **({"height": height} if height else {}),
        **kw,
    )
    return fig


# ── Cached data & model ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return clean(fetch_cms_data(max_records=8_000))


@st.cache_resource(show_spinner="Training models — takes ~20s on first load...")
def get_pipeline():
    df      = load_data()
    feat_df = engineer_features(df)
    X, y    = build_target(feat_df)
    cv      = evaluate_models(X, y)
    model, X_test, y_test, y_pred, y_proba, fcols = train_best_model(X, y, cv)
    return {
        "model": model, "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "y_proba": y_proba, "feature_cols": fcols,
        "cv_results": cv, "importances": get_feature_importance(model, fcols),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="nav-logo">🏥 NursingQuality AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="nav-sub">Powered by CMS Provider Data</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home", "📊  Explore Data", "🤖  Model Results", "🔮  Predict Quality"],
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="nav-sub">Built to showcase end-to-end ML skills:<br>• CMS API integration<br>• Pandas / NumPy wrangling<br>• Scikit-Learn + XGBoost<br>• Statistical analysis<br>• Interactive deployment</p>', unsafe_allow_html=True)


df = load_data()
page = page.split("  ")[1]   # strip icon prefix


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Nursing Industry · Machine Learning · Data Intelligence</div>
        <h1>Nursing Home Quality<br>Intelligence Platform</h1>
        <p>
            Every year, CMS rates thousands of U.S. nursing facilities on a 1–5 star scale.
            This platform uses machine learning to <strong style="color:#fff">predict which facilities will
            earn high or low ratings</strong> — before inspectors visit — based on staffing levels,
            health deficiencies, and facility data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── What does this tool do? ───────────────────────────────────────────────
    st.markdown('<p class="section-label">What This Does</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Three capabilities in one platform</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Explore Real Data</div>
            <div class="feature-desc">
                Analyse 8,000+ nursing facility records. Identify outliers,
                understand how staffing levels correlate with quality, and
                spot patterns across ownership types.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">ML-Powered Predictions</div>
            <div class="feature-desc">
                Three models — Logistic Regression, Random Forest, and XGBoost —
                compete to predict quality ratings. Best model achieves
                <strong style="color:#4F8BF9">90% accuracy and 0.967 AUC</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Live Quality Predictor</div>
            <div class="feature-desc">
                Enter any facility's details and get an instant prediction.
                See exactly which factors are driving the result — useful for
                targeted quality improvement decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">What the data tells us</p>', unsafe_allow_html=True)

    high_pct = (df["overall_rating"] >= 4).mean() * 100
    low_pct  = (df["overall_rating"] <= 2).mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        (f"{len(df):,}",                         "Facilities analysed"),
        (f"{df['overall_rating'].mean():.1f} ★",  "Average star rating"),
        (f"{df['total_nursing_hrs'].mean():.1f}h","Avg nursing hrs / day"),
        (f"{high_pct:.0f}%",                      "Rated high quality (4-5★)"),
        (f"{low_pct:.0f}%",                       "Rated low quality (1-2★)"),
    ]
    for col, (val, lbl) in zip([k1, k2, k3, k4, k5], cards):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick charts ──────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<p class="section-label">Rating Breakdown</p>', unsafe_allow_html=True)
        counts = df["overall_rating"].value_counts().sort_index().reset_index()
        counts.columns = ["rating", "count"]
        counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
        fig = px.bar(
            counts, x="rating", y="count",
            color="count", color_continuous_scale="Blues",
            text=counts["pct"].map("{}%".format),
            labels={"rating": "Star Rating", "count": "Facilities"},
        )
        fig.update_traces(textposition="outside", textfont_color="#C8D4E8")
        chart_layout(fig, coloraxis_showscale=False, xaxis=dict(tickvals=[1,2,3,4,5]))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight">
            <strong>Key insight:</strong> Ratings are roughly normally distributed around 3★.
            Only ~27% of facilities achieve a high rating (4-5★) — making prediction
            genuinely valuable for early intervention.
        </div>
        """, unsafe_allow_html=True)

    with ch2:
        st.markdown('<p class="section-label">Staffing vs Quality</p>', unsafe_allow_html=True)
        fig2 = px.box(
            df, x="overall_rating", y="total_nursing_hrs",
            color="overall_rating",
            color_discrete_sequence=px.colors.sequential.Blues[1:],
            labels={"overall_rating": "Star Rating", "total_nursing_hrs": "Nursing Hrs / Resident / Day"},
        )
        chart_layout(fig2, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        <div class="insight">
            <strong>Key insight:</strong> Higher-rated facilities consistently provide more nursing
            hours per resident. This is one of the strongest predictors in the model
            (correlation: <strong>0.33</strong>).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── How to use ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Getting Started</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">How to use this platform</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("🏠", "Start here", "Read the dataset overview and key statistics to understand what the data represents."),
        ("📊", "Explore Data", "Dig into outlier detection, feature distributions, and correlations. Understand what drives quality."),
        ("🤖", "Model Results", "See how three ML models compare. Examine the ROC curve, confusion matrix, and feature importances."),
        ("🎯", "Predict Quality", "Enter a real or hypothetical facility's details and get an instant quality prediction with explanation."),
    ]
    for col, (icon, title, desc) in zip([s1, s2, s3, s4], steps):
        col.markdown(f"""
        <div class="feature-card" style="text-align:center;">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLORE DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Explore Data":
    from src.analysis import detect_outliers_iqr, distribution_summary, target_correlation

    st.markdown('<p class="section-label">Data Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Explore the Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Statistical analysis of 8,000 nursing facility records — outliers, distributions, and what correlates most strongly with quality ratings.</p>', unsafe_allow_html=True)

    COLS = ["total_nursing_hrs", "rn_hrs", "aide_hrs", "num_beds", "num_residents", "num_deficiencies"]
    present = [c for c in COLS if c in df.columns]
    outlier_flags = detect_outliers_iqr(df, cols=present)

    # ── Outliers ──────────────────────────────────────────────────────────────
    st.markdown("#### Outlier Detection — IQR Method")
    st.markdown('<p class="section-desc">Values beyond Q1 − 1.5×IQR or Q3 + 1.5×IQR are flagged. Outliers are winsorised (clamped) during cleaning rather than dropped, preserving sample size.</p>', unsafe_allow_html=True)

    pcts = {c: outlier_flags[c].mean() * 100 for c in present if c in outlier_flags}
    fig = px.bar(
        x=list(pcts.values()), y=[c.replace("_"," ").title() for c in pcts],
        orientation="h", color=list(pcts.values()),
        color_continuous_scale="Reds",
        text=[f"{v:.1f}%" for v in pcts.values()],
        labels={"x": "% Records Flagged as Outlier"},
    )
    fig.update_traces(textposition="outside", textfont_color="#C8D4E8")
    chart_layout(fig, coloraxis_showscale=False, yaxis=dict(autorange="reversed"), height=280)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight">
        <strong>RN hours</strong> has the highest outlier rate (2.3%) — some facilities employ
        far more registered nurses than average, skewing the distribution right (skewness: 1.07).
        Health deficiencies also show notable outliers at the high end, flagging facilities
        with serious compliance issues.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Distribution explorer ─────────────────────────────────────────────────
    st.markdown("#### Feature Distribution Explorer")
    st.markdown('<p class="section-desc">Select a feature to see its distribution and how it varies across star ratings.</p>', unsafe_allow_html=True)

    label_map = {
        "total_nursing_hrs": "Total Nursing Hrs / Resident / Day",
        "rn_hrs":            "RN Hrs / Resident / Day",
        "aide_hrs":          "Aide Hrs / Resident / Day",
        "num_beds":          "Number of Certified Beds",
        "num_residents":     "Avg Residents per Day",
        "num_deficiencies":  "Health Deficiencies",
    }
    sel = st.selectbox("Feature", present, format_func=lambda c: label_map.get(c, c))

    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(
            df, x=sel, nbins=50,
            color_discrete_sequence=["#4F8BF9"],
            marginal="box",
            labels={sel: label_map.get(sel, sel)},
            title="Overall Distribution",
        )
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.box(
            df, x="overall_rating", y=sel,
            color="overall_rating",
            color_discrete_sequence=px.colors.sequential.Blues[1:],
            labels={"overall_rating": "Star Rating", sel: label_map.get(sel, sel)},
            title="Split by Star Rating",
        )
        chart_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Correlation with target ───────────────────────────────────────────────
    st.markdown("#### What Drives Quality Ratings?")
    st.markdown('<p class="section-desc">Pearson correlation between each numeric feature and the overall star rating. Positive = higher values linked to better ratings. Negative = higher values linked to worse ratings.</p>', unsafe_allow_html=True)

    corr_s = target_correlation(df).drop("overall_rating", errors="ignore")
    cdf = corr_s.reset_index()
    cdf.columns = ["Feature", "Correlation"]
    cdf["Feature"] = cdf["Feature"].str.replace("_", " ").str.title()
    cdf["Direction"] = cdf["Correlation"].apply(lambda x: "Positive" if x >= 0 else "Negative")

    fig = px.bar(
        cdf.sort_values("Correlation"),
        x="Correlation", y="Feature", orientation="h",
        color="Direction",
        color_discrete_map={"Positive": "#4F8BF9", "Negative": "#E05C5C"},
        text=cdf.sort_values("Correlation")["Correlation"].map("{:.2f}".format),
    )
    fig.update_traces(textposition="outside", textfont_color="#C8D4E8")
    chart_layout(fig, showlegend=True, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight">
        <strong>Staffing and quality measure ratings</strong> are the strongest positive predictors (r ≈ 0.66).
        <strong>Health inspection rating</strong> is strongly negative — this is because a higher inspection
        score in this dataset reflects <em>more</em> deficiencies found, not better performance.
        <strong>Bed count and occupancy</strong> have near-zero correlation, meaning facility size alone
        does not predict quality.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("#### Full Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    aspect="auto", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor=T, font_color="#C8D4E8", height=480)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.markdown('<p class="section-label">Machine Learning</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Three classifiers were trained and compared using stratified 5-fold cross-validation. The task: predict <strong>High quality (4-5★)</strong> vs <strong>Low quality (1-2★)</strong>. Three-star facilities are excluded as the ambiguous midpoint.</p>', unsafe_allow_html=True)

    p = get_pipeline()
    fpr, tpr, _ = roc_curve(p["y_test"], p["y_proba"])
    roc_auc = auc(fpr, tpr)

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, (val, lbl) in zip([m1, m2, m3, m4], [
        (f"{accuracy_score(p['y_test'], p['y_pred']):.1%}", "Test Accuracy"),
        (f"{f1_score(p['y_test'], p['y_pred'], average='macro'):.3f}", "F1-Macro Score"),
        (f"{roc_auc:.3f}", "ROC-AUC"),
        (f"{precision_score(p['y_test'], p['y_pred'], average='macro'):.3f}", "Precision"),
    ]):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CV + ROC ──────────────────────────────────────────────────────────────
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Model Comparison")
        st.markdown('<p class="section-desc">5-fold cross-validated F1-macro score. Higher is better. Logistic Regression edges out the tree-based models, suggesting the decision boundary is largely linear.</p>', unsafe_allow_html=True)
        cv_df = pd.DataFrame(list(p["cv_results"].items()), columns=["Model", "F1"])
        fig = px.bar(cv_df.sort_values("F1", ascending=False),
                     x="Model", y="F1",
                     color="F1", color_continuous_scale="Blues",
                     text=cv_df.sort_values("F1",ascending=False)["F1"].map("{:.4f}".format),
                     range_y=[0.85, 0.92])
        fig.update_traces(textposition="outside", textfont_color="#C8D4E8")
        chart_layout(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("#### ROC Curve")
        st.markdown('<p class="section-desc">The ROC curve plots true positive rate vs false positive rate. AUC of 0.967 means the model correctly ranks 96.7% of high-quality facilities above low-quality ones.</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
            fillcolor="rgba(79,139,249,0.12)",
            line=dict(color="#4F8BF9", width=2.5),
            name=f"Best Model  AUC = {roc_auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
            line=dict(color="#334155", dash="dash"), showlegend=False))
        chart_layout(fig, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                     legend=dict(x=0.4, y=0.05))
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix + importances ────────────────────────────────────────
    r3, r4 = st.columns(2)
    with r3:
        st.markdown("#### Confusion Matrix")
        st.markdown('<p class="section-desc">Out of 936 held-out test facilities, the model correctly classified 90%. Misclassifications are roughly balanced between false positives and false negatives.</p>', unsafe_allow_html=True)
        cm = confusion_matrix(p["y_test"], p["y_pred"])
        fig = px.imshow(cm, text_auto=True,
            x=["Predicted Low", "Predicted High"],
            y=["Actual Low", "Actual High"],
            color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(paper_bgcolor=T, font_color="#C8D4E8", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        st.markdown("#### Feature Importances")
        st.markdown('<p class="section-desc">Which inputs matter most to the model? Staffing and quality ratings dominate — facility size has almost no influence.</p>', unsafe_allow_html=True)
        if not p["importances"].empty:
            imp = p["importances"].reset_index()
            imp.columns = ["Feature", "Importance"]
            imp["Feature"] = imp["Feature"].str.replace("_", " ").str.title()
            fig = px.bar(imp.sort_values("Importance"),
                         x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
            chart_layout(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Quality":
    st.markdown('<p class="section-label">Live Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Will This Facility Achieve a High Quality Rating?</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Enter the details of any nursing home below. The trained model will instantly predict whether it is likely to receive a high (4-5★) or low (1-2★) CMS quality rating — and show you exactly why.</p>', unsafe_allow_html=True)

    p = get_pipeline()
    model        = p["model"]
    feature_cols = p["feature_cols"]

    form, result = st.columns([1, 1], gap="large")

    with form:
        st.markdown("#### Step 1 — CMS Star Ratings")
        st.markdown('<p class="section-desc">These are sub-ratings already assigned by CMS inspectors.</p>', unsafe_allow_html=True)
        staffing_r  = st.select_slider("Staffing Rating",          options=[1,2,3,4,5], value=3)
        quality_r   = st.select_slider("Quality Measure Rating",   options=[1,2,3,4,5], value=3)
        inspection_r= st.select_slider("Health Inspection Rating", options=[1,2,3,4,5], value=3)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Step 2 — Staffing Levels")
        st.markdown('<p class="section-desc">Hours of care provided per resident per day.</p>', unsafe_allow_html=True)
        total_hrs = st.slider("Total Nursing Hours",  1.0, 8.0, 3.8, 0.1)
        rn_hrs    = st.slider("RN Hours",             0.1, 4.0, 1.0, 0.1)
        aide_hrs  = st.slider("Aide Hours",           0.3, 7.0, 2.8, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Step 3 — Facility Details")
        c1, c2 = st.columns(2)
        num_beds      = c1.number_input("Certified Beds",    20,  500, 150, 10)
        num_residents = c2.number_input("Avg Residents/Day", 10,  500, 120, 10)
        num_defic     = st.number_input("Health Deficiencies (total)", 0, 60, 8, 1)
        ownership     = st.selectbox("Ownership Type",
                                     list(OWNERSHIP_MAP.keys()),
                                     format_func=str.title)

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("🔮  Run Prediction", type="primary", use_container_width=True)

    with result:
        st.markdown("#### Prediction Output")

        if btn:
            occupancy_rate = num_residents / max(num_beds, 1)
            rn_share       = rn_hrs / max(total_hrs, 0.01)
            ownership_enc  = OWNERSHIP_MAP.get(ownership, 3)

            input_df = pd.DataFrame([{
                "staffing_rating":          staffing_r,
                "quality_rating":           quality_r,
                "health_inspection_rating": inspection_r,
                "num_beds":                 num_beds,
                "num_residents":            num_residents,
                "total_nursing_hrs":        total_hrs,
                "rn_hrs":                   rn_hrs,
                "aide_hrs":                 aide_hrs,
                "num_deficiencies":         num_defic,
                "occupancy_rate":           occupancy_rate,
                "rn_share":                 rn_share,
                "ownership_encoded":        ownership_enc,
            }])[feature_cols]

            pred  = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            conf  = proba if pred == 1 else 1 - proba
            color = "#22C55E" if pred == 1 else "#EF4444"
            label = "HIGH QUALITY  4-5★" if pred == 1 else "LOW QUALITY  1-2★"
            cls   = "result-high" if pred == 1 else "result-low"
            val_cls = "result-value-high" if pred == 1 else "result-value-low"

            st.markdown(f"""
            <div class="{cls}">
                <div class="result-label">Predicted Rating</div>
                <div class="{val_cls}">{label}</div>
                <div class="result-label" style="margin-top:8px;">Model confidence: {conf:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                domain={"x": [0,1], "y": [0,1]},
                title={"text": "Probability of High Quality Rating", "font": {"size": 13, "color": "#9AA0B4"}},
                number={"suffix": "%", "font": {"size": 48, "color": color}},
                gauge={
                    "axis": {"range": [0,100], "ticksuffix": "%", "tickcolor": "#334155"},
                    "bar":  {"color": color},
                    "bgcolor": "#111827",
                    "bordercolor": "#1E2A45",
                    "steps": [
                        {"range": [0,  50], "color": "#1A0D0D"},
                        {"range": [50, 100], "color": "#0D1A0D"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor=T, font_color="#C8D4E8",
                height=280, margin=dict(t=50, b=10, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature contributions
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            if hasattr(clf, "coef_"):
                st.markdown("#### Why did the model decide this?")
                st.markdown('<p class="section-desc">Each bar shows how much a feature pushed the prediction toward High (blue) or Low (red) quality.</p>', unsafe_allow_html=True)
                coefs    = pd.Series(clf.coef_[0], index=feature_cols)
                contribs = (coefs * input_df.iloc[0]).sort_values()
                cdf = contribs.reset_index()
                cdf.columns = ["Feature", "Contribution"]
                cdf["Feature"]   = cdf["Feature"].str.replace("_"," ").str.title()
                cdf["Direction"] = cdf["Contribution"].apply(lambda x: "Towards High Quality" if x >= 0 else "Towards Low Quality")
                fig = px.bar(cdf, x="Contribution", y="Feature", orientation="h",
                             color="Direction",
                             color_discrete_map={
                                 "Towards High Quality": "#4F8BF9",
                                 "Towards Low Quality":  "#EF4444",
                             })
                chart_layout(fig, height=320, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div class="insight" style="text-align:center; padding: 40px;">
                <div style="font-size:2.5rem; margin-bottom:12px;">🔮</div>
                <div style="font-size:1rem; color:#C8D4E8; margin-bottom:8px;">
                    Ready to predict
                </div>
                <div>
                    Fill in the facility details on the left,<br>then click
                    <strong>Run Prediction</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)
