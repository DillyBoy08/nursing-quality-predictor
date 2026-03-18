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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
BG        = "#0F172A"
SURFACE   = "#1E293B"
BORDER    = "#334155"
TEXT_PRI  = "#F1F5F9"
TEXT_SEC  = "#94A3B8"
ACCENT    = "#3B82F6"
SUCCESS   = "#10B981"
DANGER    = "#EF4444"
T         = "rgba(0,0,0,0)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* App background */
.stApp {{ background: {BG}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding: 24px 16px; }}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* Typography */
h1 {{ font-size: 1.5rem !important; font-weight: 700 !important; color: {TEXT_PRI} !important; margin: 0 0 4px 0 !important; letter-spacing: -0.3px; }}
h2 {{ font-size: 1.1rem !important; font-weight: 600 !important; color: {TEXT_PRI} !important; margin: 0 0 4px 0 !important; }}
h3 {{ font-size: 0.95rem !important; font-weight: 600 !important; color: {TEXT_PRI} !important; margin: 0 !important; }}
p, li {{ color: {TEXT_SEC} !important; font-size: 0.875rem !important; line-height: 1.6 !important; }}

/* Metric cards */
div[data-testid="metric-container"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
}}
div[data-testid="metric-container"] label {{
    color: {TEXT_SEC} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: {TEXT_PRI} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}}

/* Inputs */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {{
    color: {TEXT_SEC} !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}}

/* Buttons */
div[data-testid="stButton"] button[kind="primary"] {{
    background: {ACCENT};
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.875rem;
    padding: 10px 20px;
    color: white;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: #2563EB;
}}

/* Divider */
hr {{ border-color: {BORDER} !important; margin: 24px 0 !important; }}

/* Selectbox / radio */
div[data-testid="stRadio"] label {{ color: {TEXT_SEC} !important; font-size: 0.875rem !important; }}
div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] + div {{
    color: {TEXT_PRI} !important;
    font-weight: 600 !important;
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Plotly chart template ─────────────────────────────────────────────────────
def chart(fig, height=320, legend=False, **kw):
    fig.update_layout(
        plot_bgcolor=T,
        paper_bgcolor=T,
        font=dict(family="Inter", color=TEXT_SEC, size=12),
        title_font=dict(color=TEXT_PRI, size=13),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
        margin=dict(t=32, b=8, l=8, r=8),
        showlegend=legend,
        height=height,
        **kw,
    )
    return fig


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    return clean(fetch_cms_data(max_records=8_000))


@st.cache_resource(show_spinner="Training models...")
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


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <p style="font-size:1rem;font-weight:700;color:{TEXT_PRI};margin:0 0 2px 0;">
            Nursing Home Quality
        </p>
        <p style="font-size:0.72rem;color:{TEXT_SEC};margin:0 0 24px 0;">
            Quality Intelligence Platform
        </p>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Data Analysis", "Model Performance", "Predict Quality"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""
        <p style="font-size:0.72rem;color:{TEXT_SEC};line-height:1.8;">
            Data source: CMS Nursing Home Compare<br>
            Records: 8,000 facilities<br>
            Models: Logistic Regression · Random Forest · XGBoost
        </p>
    """, unsafe_allow_html=True)


df = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
        <p style="font-size:0.7rem;font-weight:600;color:{ACCENT};text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;">
            Dashboard
        </p>
    """, unsafe_allow_html=True)
    st.title("Nursing Home Quality Overview")
    st.markdown("Predicts whether a U.S. nursing facility achieves a high CMS star rating based on staffing, inspection, and facility data.")
    st.divider()

    # KPIs
    high_pct = (df["overall_rating"] >= 4).mean() * 100
    low_pct  = (df["overall_rating"] <= 2).mean() * 100
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Facilities", f"{len(df):,}")
    k2.metric("Avg Star Rating", f"{df['overall_rating'].mean():.2f}")
    k3.metric("Avg Nursing Hrs", f"{df['total_nursing_hrs'].mean():.1f} / day")
    k4.metric("High Quality (4-5★)", f"{high_pct:.0f}%")
    k5.metric("Low Quality (1-2★)", f"{low_pct:.0f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Rating Distribution")
        counts = df["overall_rating"].value_counts().sort_index().reset_index()
        counts.columns = ["rating", "count"]
        counts["pct"] = (counts["count"] / len(df) * 100).round(1)
        fig = px.bar(counts, x="rating", y="count",
                     text=counts["pct"].map("{}%".format),
                     color_discrete_sequence=[ACCENT])
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_SEC, size=11))
        chart(fig, xaxis=dict(tickvals=[1,2,3,4,5], gridcolor=BORDER,
                              linecolor=BORDER, tickcolor=BORDER, title="Star Rating"),
              yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
                         title="Facilities"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Staffing Hours by Rating")
        fig = px.box(df, x="overall_rating", y="total_nursing_hrs",
                     color_discrete_sequence=[ACCENT],
                     labels={"overall_rating": "Star Rating",
                             "total_nursing_hrs": "Nursing Hrs / Resident / Day"})
        chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Deficiencies vs Rating")
        avg_def = df.groupby("overall_rating")["num_deficiencies"].mean().reset_index()
        fig = px.bar(avg_def, x="overall_rating", y="num_deficiencies",
                     text=avg_def["num_deficiencies"].round(1),
                     color_discrete_sequence=[DANGER],
                     labels={"overall_rating": "Star Rating",
                             "num_deficiencies": "Avg Health Deficiencies"})
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_SEC, size=11))
        chart(fig, xaxis=dict(tickvals=[1,2,3,4,5], gridcolor=BORDER,
                              linecolor=BORDER, tickcolor=BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### Ownership Type Breakdown")
        own = df["ownership_type"].value_counts().reset_index()
        own.columns = ["type", "count"]
        own["type"] = own["type"].str.title()
        fig = px.pie(own, values="count", names="type", hole=0.55,
                     color_discrete_sequence=[ACCENT, "#1D4ED8", "#60A5FA", "#BFDBFE"])
        fig.update_traces(textfont=dict(color=TEXT_PRI, size=11))
        fig.update_layout(paper_bgcolor=T, font_color=TEXT_SEC, height=320,
                          legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
                          margin=dict(t=32, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analysis":
    from src.analysis import detect_outliers_iqr, distribution_summary, target_correlation

    st.markdown(f"""
        <p style="font-size:0.7rem;font-weight:600;color:{ACCENT};text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;">
            Statistical Analysis
        </p>
    """, unsafe_allow_html=True)
    st.title("Data Analysis")
    st.markdown("Outlier detection, feature distributions, and correlation analysis across 8,000 nursing facility records.")
    st.divider()

    COLS    = ["total_nursing_hrs", "rn_hrs", "aide_hrs", "num_beds", "num_residents", "num_deficiencies"]
    present = [c for c in COLS if c in df.columns]
    LABELS  = {
        "total_nursing_hrs": "Total Nursing Hrs",
        "rn_hrs":            "RN Hours",
        "aide_hrs":          "Aide Hours",
        "num_beds":          "Certified Beds",
        "num_residents":     "Avg Residents / Day",
        "num_deficiencies":  "Health Deficiencies",
    }

    outlier_flags = detect_outliers_iqr(df, cols=present)

    # Outlier chart
    st.markdown("#### Outlier Prevalence — IQR Method")
    pcts = {LABELS.get(c, c): outlier_flags[c].mean() * 100
            for c in present if c in outlier_flags}
    fig = px.bar(x=list(pcts.values()), y=list(pcts.keys()),
                 orientation="h",
                 text=[f"{v:.1f}%" for v in pcts.values()],
                 color=list(pcts.values()),
                 color_continuous_scale=[[0, SURFACE], [1, DANGER]])
    fig.update_traces(textposition="outside", textfont=dict(color=TEXT_SEC, size=11))
    chart(fig, height=280, coloraxis_showscale=False,
          yaxis=dict(autorange="reversed", gridcolor=BORDER,
                     linecolor=BORDER, tickcolor=BORDER),
          xaxis=dict(title="% Flagged", gridcolor=BORDER,
                     linecolor=BORDER, tickcolor=BORDER))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Distribution explorer
    st.markdown("#### Feature Distribution")
    sel = st.selectbox("Select feature", present, format_func=lambda c: LABELS.get(c, c))

    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(df, x=sel, nbins=50,
                           color_discrete_sequence=[ACCENT],
                           marginal="box",
                           labels={sel: LABELS.get(sel, sel)})
        chart(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        fig = px.box(df, x="overall_rating", y=sel,
                     color_discrete_sequence=[ACCENT],
                     labels={"overall_rating": "Star Rating",
                             sel: LABELS.get(sel, sel)})
        chart(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Correlation with target
    st.markdown("#### Correlation with Overall Star Rating")
    corr_s = target_correlation(df).drop("overall_rating", errors="ignore")
    cdf = corr_s.reset_index()
    cdf.columns = ["Feature", "Correlation"]
    cdf["Feature"] = cdf["Feature"].str.replace("_", " ").str.title()
    cdf["color"] = cdf["Correlation"].apply(lambda x: ACCENT if x >= 0 else DANGER)

    fig = go.Figure(go.Bar(
        x=cdf.sort_values("Correlation")["Correlation"],
        y=cdf.sort_values("Correlation")["Feature"],
        orientation="h",
        marker_color=cdf.sort_values("Correlation")["color"],
        text=cdf.sort_values("Correlation")["Correlation"].map("{:+.2f}".format),
        textposition="outside",
        textfont=dict(color=TEXT_SEC, size=11),
    ))
    chart(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Heatmap
    st.markdown("#### Correlation Matrix")
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    aspect="auto", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor=T, font_color=TEXT_SEC, height=460,
                      margin=dict(t=16, b=8, l=8, r=8))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Distribution summary table
    st.markdown("#### Distribution Summary")
    dist_df = distribution_summary(df)
    st.dataframe(dist_df.style.format(precision=3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""
        <p style="font-size:0.7rem;font-weight:600;color:{ACCENT};text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;">
            Machine Learning
        </p>
    """, unsafe_allow_html=True)
    st.title("Model Performance")
    st.markdown("Binary classification — High quality (4-5★) vs Low quality (1-2★). Three-star facilities excluded. Evaluated with stratified 5-fold cross-validation.")
    st.divider()

    p = get_pipeline()
    fpr, tpr, _ = roc_curve(p["y_test"], p["y_proba"])
    roc_auc = auc(fpr, tpr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy",  f"{accuracy_score(p['y_test'], p['y_pred']):.1%}")
    m2.metric("F1-Macro",       f"{f1_score(p['y_test'], p['y_pred'], average='macro'):.3f}")
    m3.metric("ROC-AUC",        f"{roc_auc:.3f}")
    m4.metric("Precision",      f"{precision_score(p['y_test'], p['y_pred'], average='macro'):.3f}")

    st.divider()

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("#### Cross-Validation F1-Macro")
        cv_df = pd.DataFrame(list(p["cv_results"].items()), columns=["Model", "F1"])
        fig = px.bar(cv_df.sort_values("F1", ascending=False),
                     x="Model", y="F1",
                     text=cv_df.sort_values("F1", ascending=False)["F1"].map("{:.4f}".format),
                     color_discrete_sequence=[ACCENT],
                     range_y=[0.85, 0.92])
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_SEC, size=11))
        chart(fig, yaxis=dict(title="F1-Macro Score", gridcolor=BORDER,
                              linecolor=BORDER, tickcolor=BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("#### ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, fill="tozeroy",
            fillcolor=f"rgba(59,130,246,0.1)",
            line=dict(color=ACCENT, width=2),
            name=f"AUC = {roc_auc:.3f}",
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(color=BORDER, dash="dash", width=1),
            showlegend=False,
        ))
        chart(fig, legend=True,
              xaxis=dict(title="False Positive Rate", gridcolor=BORDER,
                         linecolor=BORDER, tickcolor=BORDER),
              yaxis=dict(title="True Positive Rate", gridcolor=BORDER,
                         linecolor=BORDER, tickcolor=BORDER),
              legend_font_color=TEXT_SEC)
        st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)

    with r3:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(p["y_test"], p["y_pred"])
        fig = px.imshow(cm, text_auto=True,
                        x=["Predicted Low", "Predicted High"],
                        y=["Actual Low", "Actual High"],
                        color_continuous_scale=[[0, SURFACE], [1, ACCENT]],
                        aspect="auto")
        fig.update_layout(paper_bgcolor=T, font_color=TEXT_SEC,
                          coloraxis_showscale=False, height=320,
                          margin=dict(t=32, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        st.markdown("#### Feature Importances")
        if not p["importances"].empty:
            imp = p["importances"].reset_index()
            imp.columns = ["Feature", "Importance"]
            imp["Feature"] = imp["Feature"].str.replace("_", " ").str.title()
            fig = px.bar(imp.sort_values("Importance"),
                         x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[ACCENT])
            chart(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Quality":
    st.markdown(f"""
        <p style="font-size:0.7rem;font-weight:600;color:{ACCENT};text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0;">
            Live Prediction
        </p>
    """, unsafe_allow_html=True)
    st.title("Predict Facility Quality Rating")
    st.markdown("Enter nursing home parameters to get an instant quality prediction from the trained model.")
    st.divider()

    p            = get_pipeline()
    model        = p["model"]
    feature_cols = p["feature_cols"]

    form, result = st.columns([1, 1], gap="large")

    with form:
        st.markdown("#### CMS Sub-Ratings")
        staffing_r   = st.select_slider("Staffing Rating",          options=[1,2,3,4,5], value=3)
        quality_r    = st.select_slider("Quality Measure Rating",   options=[1,2,3,4,5], value=3)
        inspection_r = st.select_slider("Health Inspection Rating", options=[1,2,3,4,5], value=3)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Staffing Hours per Resident per Day")
        total_hrs = st.slider("Total Nursing Hours", 1.0, 8.0, 3.8, 0.1)
        rn_hrs    = st.slider("RN Hours",            0.1, 4.0, 1.0, 0.1)
        aide_hrs  = st.slider("Aide Hours",          0.3, 7.0, 2.8, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Facility Details")
        fc1, fc2 = st.columns(2)
        num_beds      = fc1.number_input("Certified Beds",    20,  500, 150, 10)
        num_residents = fc2.number_input("Avg Residents/Day", 10,  500, 120, 10)
        num_defic     = st.number_input("Health Deficiencies", 0, 60, 8, 1)
        ownership     = st.selectbox("Ownership Type", list(OWNERSHIP_MAP.keys()),
                                     format_func=str.title)

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("Run Prediction", type="primary", use_container_width=True)

    with result:
        st.markdown("#### Result")

        if btn:
            occupancy_rate = num_residents / max(num_beds, 1)
            rn_share       = rn_hrs / max(total_hrs, 0.01)
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
                "ownership_encoded":        OWNERSHIP_MAP.get(ownership, 3),
            }])[feature_cols]

            pred  = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            conf  = proba if pred == 1 else 1 - proba
            color = SUCCESS if pred == 1 else DANGER
            label = "High Quality — 4 to 5 Stars" if pred == 1 else "Low Quality — 1 to 2 Stars"

            # Result banner
            st.markdown(f"""
            <div style="
                background: {'rgba(16,185,129,0.08)' if pred == 1 else 'rgba(239,68,68,0.08)'};
                border: 1px solid {color};
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 20px;
            ">
                <p style="font-size:0.7rem;font-weight:600;color:{TEXT_SEC};text-transform:uppercase;
                           letter-spacing:1px;margin:0 0 6px 0;">Predicted Rating</p>
                <p style="font-size:1.6rem;font-weight:700;color:{color};margin:0 0 4px 0;">{label}</p>
                <p style="font-size:0.8rem;color:{TEXT_SEC};margin:0;">Model confidence: {conf:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Inter"}},
                title={"text": "Probability of High Quality Rating",
                       "font": {"size": 12, "color": TEXT_SEC, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%",
                             "tickcolor": BORDER, "tickfont": {"color": TEXT_SEC}},
                    "bar":  {"color": color},
                    "bgcolor": SURFACE,
                    "bordercolor": BORDER,
                    "steps": [
                        {"range": [0,  50], "color": "#1A0D0D"},
                        {"range": [50, 100], "color": "#0D1A10"},
                    ],
                    "threshold": {"line": {"color": TEXT_PRI, "width": 2},
                                  "thickness": 0.75, "value": 50},
                },
                domain={"x": [0,1], "y": [0,1]},
            ))
            fig.update_layout(paper_bgcolor=T, font_color=TEXT_SEC,
                              height=260, margin=dict(t=48, b=0, l=16, r=16))
            st.plotly_chart(fig, use_container_width=True)

            # Feature contributions
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            if hasattr(clf, "coef_"):
                st.markdown("#### Feature Contributions")
                st.markdown("How each input pushed the prediction.")
                coefs    = pd.Series(clf.coef_[0], index=feature_cols)
                contribs = (coefs * input_df.iloc[0]).sort_values()
                cdf = contribs.reset_index()
                cdf.columns = ["Feature", "Contribution"]
                cdf["Feature"] = cdf["Feature"].str.replace("_", " ").str.title()
                cdf["color"]   = cdf["Contribution"].apply(lambda x: ACCENT if x >= 0 else DANGER)

                fig = go.Figure(go.Bar(
                    x=cdf["Contribution"], y=cdf["Feature"],
                    orientation="h",
                    marker_color=cdf["color"],
                    text=cdf["Contribution"].map("{:+.3f}".format),
                    textposition="outside",
                    textfont=dict(color=TEXT_SEC, size=10),
                ))
                chart(fig, height=300, margin=dict(t=8, b=8, l=8, r=48))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(f"""
            <div style="
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 48px 24px;
                text-align: center;
            ">
                <p style="color:{TEXT_SEC};font-size:0.875rem;margin:0;">
                    Configure the facility parameters on the left and click
                    <strong style="color:{TEXT_PRI};">Run Prediction</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
