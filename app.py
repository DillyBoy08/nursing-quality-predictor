"""
app.py
------
Streamlit dashboard for the Nursing Home Quality Predictor.

Pages
-----
  Overview         — KPI cards, rating distribution, ownership breakdown, data table
  Data Analysis    — Outlier detection, distributions, correlation heatmap
  Model Performance — CV comparison, ROC curve, confusion matrix, feature importances
  Live Predictor   — Real-time quality prediction with probability gauge
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_curve,
)

from src.cleaning import clean
from src.ingestion import fetch_cms_data
from src.model import (
    MODELS,
    OWNERSHIP_MAP,
    build_target,
    engineer_features,
    evaluate_models,
    get_feature_importance,
    train_best_model,
)

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nursing Home Quality Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* hide default Streamlit menu */
    #MainMenu, footer { visibility: hidden; }

    /* metric card styling */
    div[data-testid="metric-container"] {
        background: #1E2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4F8BF9;
    }

    /* sidebar header */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4F8BF9;
        margin-bottom: 4px;
    }
    .sidebar-sub {
        font-size: 0.78rem;
        color: #9AA0B4;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating dataset...")
def load_data() -> pd.DataFrame:
    raw = fetch_cms_data(max_records=8_000)
    return clean(raw)


@st.cache_resource(show_spinner="Training models — first run only, ~20s...")
def get_pipeline() -> dict:
    df = load_data()
    feat_df = engineer_features(df)
    X, y = build_target(feat_df)
    cv_results = evaluate_models(X, y)
    model, X_test, y_test, y_pred, y_proba, feature_cols = train_best_model(
        X, y, cv_results
    )
    importances = get_feature_importance(model, feature_cols)
    return {
        "model":        model,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_pred":       y_pred,
        "y_proba":      y_proba,
        "feature_cols": feature_cols,
        "cv_results":   cv_results,
        "importances":  importances,
    }


# ── Shared plot defaults ──────────────────────────────────────────────────────
TRANSPARENT = "rgba(0,0,0,0)"
BLUE_SEQ = px.colors.sequential.Blues


def _layout(fig, **kwargs):
    fig.update_layout(
        plot_bgcolor=TRANSPARENT,
        paper_bgcolor=TRANSPARENT,
        **kwargs,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">🏥 NursingQuality AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">CMS Nursing Home Predictor</p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Data Analysis", "Model Performance", "Live Predictor"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: CMS Nursing Home Compare")
    st.caption("Models: Logistic Regression · Random Forest · XGBoost")


# ── Load data (always needed) ─────────────────────────────────────────────────
df = load_data()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Nursing Home Quality Predictor")
    st.markdown(
        "Predicts whether a U.S. nursing facility achieves a **high star rating (4-5★)** "
        "based on staffing levels, health inspections, and facility characteristics."
    )
    st.divider()

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Facilities", f"{len(df):,}")
    k2.metric("Avg Star Rating", f"{df['overall_rating'].mean():.2f} ★")
    k3.metric("Avg Nursing Hrs / Day", f"{df['total_nursing_hrs'].mean():.2f} hrs")
    high_pct = (df["overall_rating"] >= 4).mean() * 100
    k4.metric("High Quality (4-5★)", f"{high_pct:.1f}%")

    st.divider()
    col_l, col_r = st.columns(2)

    # Rating distribution
    with col_l:
        st.subheader("Overall Rating Distribution")
        counts = df["overall_rating"].value_counts().sort_index().reset_index()
        counts.columns = ["rating", "count"]
        fig = px.bar(
            counts, x="rating", y="count",
            color="count", color_continuous_scale="Blues",
            text="count",
            labels={"rating": "Star Rating", "count": "Facilities"},
        )
        fig.update_traces(textposition="outside")
        _layout(fig, coloraxis_showscale=False, xaxis=dict(tickvals=[1, 2, 3, 4, 5]))
        st.plotly_chart(fig, use_container_width=True)

    # Ownership donut
    with col_r:
        st.subheader("Ownership Type")
        own = df["ownership_type"].value_counts().reset_index()
        own.columns = ["type", "count"]
        own["type"] = own["type"].str.title()
        fig2 = px.pie(
            own, values="count", names="type",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.45,
        )
        fig2.update_layout(
            paper_bgcolor=TRANSPARENT,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Staffing by rating
    st.subheader("Staffing Hours by Star Rating")
    melt = df[["overall_rating", "total_nursing_hrs", "rn_hrs", "aide_hrs"]].melt(
        id_vars="overall_rating",
        var_name="Metric",
        value_name="Hours",
    )
    label_map = {
        "total_nursing_hrs": "Total Nursing",
        "rn_hrs":            "RN Hours",
        "aide_hrs":          "Aide Hours",
    }
    melt["Metric"] = melt["Metric"].map(label_map)
    fig3 = px.box(
        melt, x="overall_rating", y="Hours", color="Metric",
        color_discrete_sequence=["#4F8BF9", "#7EC8E3", "#B0D4F1"],
        labels={"overall_rating": "Star Rating", "Hours": "Hrs / Resident / Day"},
    )
    _layout(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Data Sample")
    st.dataframe(df.head(200), use_container_width=True, height=280)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Data Analysis":
    from src.analysis import (
        detect_outliers_iqr,
        distribution_summary,
        target_correlation,
    )

    st.title("Statistical Analysis")
    st.divider()

    ANALYSIS_COLS = [
        "total_nursing_hrs", "rn_hrs", "aide_hrs",
        "num_beds", "num_residents", "num_deficiencies",
    ]
    present_cols = [c for c in ANALYSIS_COLS if c in df.columns]
    outlier_flags = detect_outliers_iqr(df, cols=present_cols)

    # ── Outlier summary ───────────────────────────────────────────────────────
    st.subheader("Outlier Prevalence — IQR Method")
    pcts = {col: outlier_flags[col].mean() * 100 for col in present_cols}
    fig = px.bar(
        x=list(pcts.values()), y=list(pcts.keys()),
        orientation="h",
        color=list(pcts.values()),
        color_continuous_scale="Reds",
        text=[f"{v:.1f}%" for v in pcts.values()],
        labels={"x": "% Flagged as Outlier", "y": "Feature"},
    )
    fig.update_traces(textposition="outside")
    _layout(fig, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Distribution explorer ─────────────────────────────────────────────────
    st.subheader("Feature Distribution Explorer")
    selected = st.selectbox("Select feature", present_cols, format_func=lambda c: c.replace("_", " ").title())

    d_left, d_right = st.columns(2)
    with d_left:
        fig = px.histogram(
            df, x=selected, nbins=50,
            color_discrete_sequence=["#4F8BF9"],
            marginal="box",
            labels={selected: selected.replace("_", " ").title()},
            title="Distribution",
        )
        _layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with d_right:
        fig = px.box(
            df, x="overall_rating", y=selected,
            color="overall_rating",
            color_discrete_sequence=px.colors.sequential.Blues,
            labels={
                "overall_rating": "Star Rating",
                selected: selected.replace("_", " ").title(),
            },
            title="By Star Rating",
        )
        _layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="Blues",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig.update_layout(paper_bgcolor=TRANSPARENT, height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Correlation with target ───────────────────────────────────────────────
    st.subheader("Correlation with Overall Star Rating")
    corr_series = target_correlation(df).drop("overall_rating", errors="ignore")
    corr_df = corr_series.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    corr_df["color"] = corr_df["Correlation"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    fig = px.bar(
        corr_df.sort_values("Correlation"),
        x="Correlation", y="Feature",
        orientation="h",
        color="color",
        color_discrete_map={"Positive": "#4F8BF9", "Negative": "#E05C5C"},
    )
    _layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Distribution summary table ────────────────────────────────────────────
    st.subheader("Distribution Summary")
    dist_df = distribution_summary(df)
    st.dataframe(
        dist_df.style.background_gradient(cmap="Blues", subset=["skewness", "kurtosis"]),
        use_container_width=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        "Binary classification: **High quality (4-5★)** vs **Low quality (1-2★)**. "
        "Three-star facilities are excluded as the ambiguous midpoint."
    )
    st.divider()

    pipeline = get_pipeline()
    cv_results   = pipeline["cv_results"]
    y_test       = pipeline["y_test"]
    y_pred       = pipeline["y_pred"]
    y_proba      = pipeline["y_proba"]
    importances  = pipeline["importances"]

    # ── Top metrics ───────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.1%}")
    m2.metric("F1-Macro",  f"{f1_score(y_test, y_pred, average='macro'):.4f}")
    m3.metric("ROC-AUC",   f"{roc_auc:.4f}")
    m4.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.4f}")

    st.divider()
    row1_l, row1_r = st.columns(2)

    # CV comparison
    with row1_l:
        st.subheader("Model Comparison — 5-Fold CV F1")
        cv_df = pd.DataFrame(
            list(cv_results.items()), columns=["Model", "F1 Score"]
        ).sort_values("F1 Score", ascending=False)
        fig = px.bar(
            cv_df, x="Model", y="F1 Score",
            color="F1 Score", color_continuous_scale="Blues",
            text=cv_df["F1 Score"].map("{:.4f}".format),
            range_y=[0.85, 0.92],
        )
        fig.update_traces(textposition="outside")
        _layout(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ROC curve
    with row1_r:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            fill="tozeroy", fillcolor="rgba(79,139,249,0.15)",
            line=dict(color="#4F8BF9", width=2.5),
            name=f"AUC = {roc_auc:.3f}",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False,
        ))
        _layout(
            fig,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    # Confusion matrix
    with row2_l:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            x=["Predicted Low", "Predicted High"],
            y=["Actual Low", "Actual High"],
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(paper_bgcolor=TRANSPARENT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importances
    with row2_r:
        st.subheader("Feature Importances")
        if not importances.empty:
            imp_df = importances.reset_index()
            imp_df.columns = ["Feature", "Importance"]
            imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ").str.title()
            fig = px.bar(
                imp_df.sort_values("Importance"),
                x="Importance", y="Feature",
                orientation="h",
                color="Importance", color_continuous_scale="Blues",
            )
            _layout(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Live Predictor":
    st.title("Live Quality Predictor")
    st.markdown(
        "Enter nursing home details below to get an instant quality prediction "
        "from the trained model."
    )
    st.divider()

    pipeline = get_pipeline()
    model        = pipeline["model"]
    feature_cols = pipeline["feature_cols"]

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        st.subheader("Facility Details")

        staffing_rating          = st.slider("Staffing Rating (CMS, 1-5)",          1, 5, 3)
        quality_rating           = st.slider("Quality Measure Rating (CMS, 1-5)",    1, 5, 3)
        health_inspection_rating = st.slider("Health Inspection Rating (CMS, 1-5)",  1, 5, 3)

        st.markdown("---")
        total_nursing_hrs = st.slider("Total Nursing Hrs / Resident / Day", 1.0, 8.0, 3.8, 0.1)
        rn_hrs            = st.slider("RN Hours / Resident / Day",          0.1, 4.0, 1.0, 0.1)
        aide_hrs          = st.slider("Aide Hours / Resident / Day",        0.3, 7.0, 2.8, 0.1)

        st.markdown("---")
        num_beds         = st.number_input("Certified Beds",          20,  500, 150, step=10)
        num_residents    = st.number_input("Avg Residents per Day",   10,  500, 120, step=10)
        num_deficiencies = st.number_input("Health Deficiencies",      0,   60,   8, step=1)
        ownership_type   = st.selectbox(
            "Ownership Type",
            list(OWNERSHIP_MAP.keys()),
            format_func=lambda x: x.title(),
        )

        predict_btn = st.button("Predict Quality Rating", type="primary", use_container_width=True)

    with result_col:
        st.subheader("Prediction Result")

        # Build the gauge placeholder or real result
        def _gauge(value: float, color: str = "#4F8BF9") -> go.Figure:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(value, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Probability of High Quality (4-5★)", "font": {"size": 14}},
                number={"suffix": "%", "font": {"size": 44}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#9AA0B4"},
                    "bar":  {"color": color},
                    "bgcolor": "#1E2130",
                    "bordercolor": "#1E2130",
                    "steps": [
                        {"range": [0,  50], "color": "#2D1B33"},
                        {"range": [50, 100], "color": "#1B2D33"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor=TRANSPARENT,
                font_color="#FAFAFA",
                height=320,
                margin=dict(t=60, b=20, l=20, r=20),
            )
            return fig

        if predict_btn:
            occupancy_rate = num_residents / max(num_beds, 1)
            rn_share       = rn_hrs / max(total_nursing_hrs, 0.01)
            ownership_enc  = OWNERSHIP_MAP.get(ownership_type, 3)

            input_df = pd.DataFrame([{
                "staffing_rating":          staffing_rating,
                "quality_rating":           quality_rating,
                "health_inspection_rating": health_inspection_rating,
                "num_beds":                 num_beds,
                "num_residents":            num_residents,
                "total_nursing_hrs":        total_nursing_hrs,
                "rn_hrs":                   rn_hrs,
                "aide_hrs":                 aide_hrs,
                "num_deficiencies":         num_deficiencies,
                "occupancy_rate":           occupancy_rate,
                "rn_share":                 rn_share,
                "ownership_encoded":        ownership_enc,
            }])[feature_cols]

            pred   = model.predict(input_df)[0]
            proba  = model.predict_proba(input_df)[0][1]
            color  = "#4F8BF9" if pred == 1 else "#E05C5C"

            st.plotly_chart(_gauge(proba * 100, color=color), use_container_width=True)

            if pred == 1:
                st.success(
                    f"**HIGH QUALITY** — Predicted 4-5★ rating.  "
                    f"Confidence: **{proba:.1%}**"
                )
            else:
                st.error(
                    f"**LOW QUALITY** — Predicted 1-2★ rating.  "
                    f"Confidence: **{1 - proba:.1%}**"
                )

            # Feature contribution breakdown (for linear models)
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            if hasattr(clf, "coef_"):
                st.markdown("##### Key drivers")
                coefs = pd.Series(clf.coef_[0], index=feature_cols)
                contribs = (coefs * input_df.iloc[0]).sort_values(ascending=False)
                top = pd.concat([contribs.head(3), contribs.tail(3)])
                contrib_df = top.reset_index()
                contrib_df.columns = ["Feature", "Contribution"]
                contrib_df["Feature"] = contrib_df["Feature"].str.replace("_", " ").str.title()
                contrib_df["Direction"] = contrib_df["Contribution"].apply(
                    lambda x: "Positive" if x >= 0 else "Negative"
                )
                fig = px.bar(
                    contrib_df.sort_values("Contribution"),
                    x="Contribution", y="Feature", orientation="h",
                    color="Direction",
                    color_discrete_map={"Positive": "#4F8BF9", "Negative": "#E05C5C"},
                )
                _layout(fig, showlegend=False, height=280,
                        margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(_gauge(50, color="#555"), use_container_width=True)
            st.info("Adjust the sliders on the left and click **Predict Quality Rating**.")
