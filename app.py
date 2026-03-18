"""
app.py  —  Nursing Home Quality Intelligence Platform
Light theme · Interactive visualisations · Titled charts
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

st.set_page_config(
    page_title="Nursing Home Quality Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG       = "#F8FAFC"
SURFACE  = "#FFFFFF"
BORDER   = "#E2E8F0"
TEXT_PRI = "#0F172A"
TEXT_SEC = "#64748B"
ACCENT   = "#3B82F6"
ACCENT2  = "#6366F1"
SUCCESS  = "#10B981"
DANGER   = "#EF4444"
WARN     = "#F59E0B"
T        = "rgba(0,0,0,0)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background: {BG}; }}
section[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {BORDER}; }}
#MainMenu, footer, header, .stDeployButton {{ visibility:hidden; display:none; }}
h1 {{ font-size:1.5rem !important; font-weight:700 !important; color:{TEXT_PRI} !important; margin:0 0 4px 0 !important; letter-spacing:-0.3px; }}
p, li {{ color:{TEXT_SEC} !important; font-size:0.875rem !important; line-height:1.6 !important; }}
div[data-testid="metric-container"] {{
    background:{SURFACE}; border:1px solid {BORDER}; border-top:3px solid {ACCENT};
    border-radius:8px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.06);
}}
div[data-testid="metric-container"] label {{
    color:{TEXT_SEC} !important; font-size:0.7rem !important; font-weight:600 !important;
    text-transform:uppercase !important; letter-spacing:0.6px !important;
}}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color:{TEXT_PRI} !important; font-size:1.6rem !important; font-weight:700 !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background:{ACCENT}; border:none; border-radius:6px; font-weight:600;
    font-size:0.875rem; color:white; box-shadow:0 1px 2px rgba(59,130,246,0.3);
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{ background:#2563EB; }}
hr {{ border-color:{BORDER} !important; margin:24px 0 !important; }}
div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
.page-eyebrow {{ font-size:0.7rem; font-weight:600; color:{ACCENT}; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0; }}
</style>
""", unsafe_allow_html=True)


# ── Chart helper — applies consistent styling + embeds title in figure ────────
def chart(fig, title="", subtitle="", height=340, show_legend=False, **kw):
    """Apply consistent theme and embed title/subtitle inside the Plotly figure."""
    title_text = f"<b>{title}</b>" if title else ""
    if subtitle:
        title_text += f"<br><span style='font-size:11px;font-weight:400;color:{TEXT_SEC}'>{subtitle}</span>"

    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
                     tickfont=dict(color=TEXT_SEC, size=11))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
                     tickfont=dict(color=TEXT_SEC, size=11))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13, color=TEXT_PRI, family="Inter"),
                   x=0, xanchor="left", pad=dict(b=8)),
        plot_bgcolor=SURFACE, paper_bgcolor=T,
        font=dict(family="Inter", color=TEXT_SEC, size=12),
        margin=dict(t=56 if title else 20, b=12, l=12, r=12),
        showlegend=show_legend,
        height=height,
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT_PRI,
                        font_family="Inter", bordercolor=BORDER),
        **kw,
    )
    return fig


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading CMS data...")
def load_data():
    return clean(fetch_cms_data(max_records=15_000))


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


LABELS = {
    "total_nursing_hrs":        "Total Nursing Hrs",
    "rn_hrs":                   "RN Hours",
    "aide_hrs":                 "Aide Hours",
    "num_beds":                 "Certified Beds",
    "num_residents":            "Avg Residents / Day",
    "num_deficiencies":         "Health Deficiencies",
    "overall_rating":           "Overall Rating",
    "staffing_rating":          "Staffing Rating",
    "quality_rating":           "Quality Rating",
    "health_inspection_rating": "Health Inspection Rating",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <p style="font-size:1rem;font-weight:700;color:{TEXT_PRI};margin:0 0 2px 0;">Nursing Home Quality</p>
        <p style="font-size:0.72rem;color:{TEXT_SEC};margin:0 0 24px 0;">Quality Intelligence Platform</p>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["Overview", "Data Analysis", "Model Performance", "Predict Quality", "Raw Data"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown(f"""
        <p style="font-size:0.72rem;color:{TEXT_SEC};line-height:1.9;">
            Data: CMS Nursing Home Compare<br>
            Facilities: ~14,700 real U.S. nursing homes<br>
            Models: Logistic Regression · Random Forest · XGBoost
        </p>
    """, unsafe_allow_html=True)


df = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<p class="page-eyebrow">Dashboard</p>', unsafe_allow_html=True)
    st.title("Nursing Home Quality Overview")
    st.markdown("Real CMS data for U.S. nursing facilities covering star ratings, staffing levels, and health deficiencies.")
    st.divider()

    # Filters
    f1, f2, f3 = st.columns(3)
    ownership_opts = ["All"] + sorted(df["ownership_type"].dropna().unique().tolist())
    sel_own    = f1.selectbox("Ownership type", ownership_opts)
    rating_rng = f2.select_slider("Star rating range", options=[1,2,3,4,5], value=(1,5))
    nursing_min = float(df["total_nursing_hrs"].min())
    nursing_max = float(df["total_nursing_hrs"].max())
    nursing_rng = f3.slider("Nursing hrs / day", nursing_min, nursing_max,
                            (nursing_min, nursing_max))

    fdf = df.copy()
    if sel_own != "All":
        fdf = fdf[fdf["ownership_type"] == sel_own]
    fdf = fdf[
        fdf["overall_rating"].between(rating_rng[0], rating_rng[1]) &
        fdf["total_nursing_hrs"].between(nursing_rng[0], nursing_rng[1])
    ]
    st.caption(f"{len(fdf):,} facilities match current filters")
    st.divider()

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Facilities",          f"{len(fdf):,}")
    k2.metric("Avg Star Rating",     f"{fdf['overall_rating'].mean():.2f}")
    k3.metric("Avg Nursing Hrs/Day", f"{fdf['total_nursing_hrs'].mean():.1f}")
    k4.metric("High Quality (4-5★)", f"{(fdf['overall_rating'] >= 4).mean():.0%}")
    k5.metric("Avg Deficiencies",    f"{fdf['num_deficiencies'].mean():.1f}")
    st.divider()

    # Row 1
    c1, c2 = st.columns(2)

    with c1:
        counts = fdf["overall_rating"].value_counts().sort_index().reset_index()
        counts.columns = ["rating", "count"]
        counts["pct"] = (counts["count"] / len(fdf) * 100).round(1)
        fig = px.bar(counts, x="rating", y="count",
                     text=counts["pct"].map("{}%".format),
                     color="count",
                     color_continuous_scale=[[0,"#DBEAFE"],[1,ACCENT]],
                     custom_data=["pct"],
                     labels={"rating": "Star Rating", "count": "Facilities"})
        fig.update_traces(
            textposition="outside", textfont=dict(color=TEXT_SEC, size=11),
            hovertemplate="<b>%{x} Stars</b><br>Facilities: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        )
        fig.update_xaxes(tickvals=[1,2,3,4,5], title="Star Rating")
        fig.update_yaxes(title="Number of Facilities")
        chart(fig,
              title="Overall Star Rating Distribution",
              subtitle="How facilities are distributed across CMS's 1-5 star scale",
              coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("CMS assigns each nursing home an overall star rating from 1 (much below average) to 5 (much above average). This chart shows how the filtered facilities spread across those ratings. A large share of 1-2 star facilities in a region is a red flag for care quality.")

    with c2:
        fig = px.scatter(
            fdf.sample(min(2000, len(fdf)), random_state=42),
            x="total_nursing_hrs", y="num_deficiencies",
            color="overall_rating",
            color_continuous_scale=px.colors.sequential.Blues,
            opacity=0.55,
            hover_data={"overall_rating": True, "total_nursing_hrs": ":.2f",
                        "num_deficiencies": True, "num_beds": True},
            labels={"total_nursing_hrs": "Nursing Hrs / Resident / Day",
                    "num_deficiencies":  "Health Deficiencies",
                    "overall_rating":    "Star Rating"},
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]:.0f} Stars</b><br>Nursing hrs: %{x:.2f}<br>Deficiencies: %{y}<br>Beds: %{customdata[3]}<extra></extra>"
        )
        chart(fig,
              title="Staffing Hours vs Health Deficiencies",
              subtitle="Each dot is one facility. Colour shows star rating.",
              coloraxis_colorbar=dict(title="Stars", thickness=12, len=0.6))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Facilities with more nursing hours per resident per day tend to have fewer health deficiencies. Each point is one facility. Darker blue = higher rated. Hover over any point for details.")

    # Row 2
    c3, c4 = st.columns(2)

    with c3:
        avg_def = fdf.groupby("overall_rating")["num_deficiencies"].agg(
            mean="mean", std="std", count="count"
        ).reset_index()
        avg_def["se"] = avg_def["std"] / np.sqrt(avg_def["count"])
        fig = go.Figure(go.Bar(
            x=avg_def["overall_rating"], y=avg_def["mean"],
            error_y=dict(type="data", array=avg_def["se"], visible=True,
                         color=TEXT_SEC, thickness=1.5),
            marker_color=[DANGER, WARN, "#94A3B8", SUCCESS, "#059669"],
            text=avg_def["mean"].map("{:.1f}".format),
            textposition="outside",
            hovertemplate="<b>%{x} Stars</b><br>Avg deficiencies: %{y:.1f}<br>±%{error_y.array:.2f} SE<extra></extra>",
        ))
        fig.update_xaxes(tickvals=[1,2,3,4,5], title="Star Rating")
        fig.update_yaxes(title="Avg Health Deficiencies")
        chart(fig,
              title="Average Health Deficiencies by Star Rating",
              subtitle="Error bars show standard error. Lower deficiencies correlate with higher ratings.")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Health deficiencies are violations found during state inspections (e.g. medication errors, inadequate hygiene, staffing gaps). This confirms that CMS star ratings reflect real inspection outcomes: 1-star homes average nearly three times as many deficiencies as 5-star homes.")

    with c4:
        own_rating = (
            fdf.groupby(["ownership_type", "overall_rating"])
            .size().reset_index(name="count")
        )
        own_rating["ownership_type"] = own_rating["ownership_type"].str.title()
        fig = px.bar(own_rating, x="overall_rating", y="count",
                     color="ownership_type", barmode="group",
                     color_discrete_sequence=[ACCENT, ACCENT2, SUCCESS, WARN],
                     labels={"overall_rating": "Star Rating", "count": "Facilities",
                             "ownership_type": "Ownership"},
                     custom_data=["ownership_type"])
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>Rating: %{x}★<br>Facilities: %{y:,}<extra></extra>"
        )
        fig.update_xaxes(tickvals=[1,2,3,4,5], title="Star Rating")
        fig.update_layout(legend=dict(orientation="h", y=-0.25, font=dict(size=11)))
        chart(fig,
              title="Star Rating Distribution by Ownership Type",
              subtitle="For-profit facilities dominate but differ in quality spread",
              show_legend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("About 65% of U.S. nursing homes are for-profit corporations. This chart shows whether ownership type affects the rating spread. Non-profit and government facilities tend to have a higher proportion of 4-5 star ratings in the real data.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analysis":
    from src.analysis import detect_outliers_iqr, distribution_summary, target_correlation

    st.markdown('<p class="page-eyebrow">Statistical Analysis</p>', unsafe_allow_html=True)
    st.title("Data Analysis")
    st.markdown("Explore distributions, outliers, and feature correlations across real CMS nursing facility records."
)
    st.divider()

    NUM_COLS = ["total_nursing_hrs", "rn_hrs", "aide_hrs",
                "num_beds", "num_residents", "num_deficiencies"]
    present = [c for c in NUM_COLS if c in df.columns]

    fa, fb = st.columns([2, 3])
    sel_ratings = fa.multiselect("Filter by star rating", [1,2,3,4,5], default=[1,2,3,4,5])
    sel_own     = fb.multiselect("Filter by ownership",
                                 df["ownership_type"].dropna().unique().tolist(),
                                 default=df["ownership_type"].dropna().unique().tolist(),
                                 format_func=str.title)
    adf = df[df["overall_rating"].isin(sel_ratings) & df["ownership_type"].isin(sel_own)]
    st.caption(f"{len(adf):,} facilities in current selection")
    st.divider()

    # Distribution explorer
    d_col, d_split = st.columns([2, 1])
    sel_feat = d_col.selectbox("Feature to explore", present, format_func=lambda c: LABELS.get(c, c))
    split_by = d_split.selectbox("Split by", ["None", "Star Rating", "Ownership Type"])

    feat_label = LABELS.get(sel_feat, sel_feat)
    dh1, dh2 = st.columns(2)

    with dh1:
        if split_by == "Star Rating":
            fig = px.histogram(adf, x=sel_feat, color="overall_rating",
                               nbins=40, barmode="overlay", opacity=0.7,
                               color_discrete_sequence=px.colors.sequential.Blues[1:],
                               labels={sel_feat: feat_label, "overall_rating": "Stars"})
            chart(fig, title=f"{feat_label}: Distribution by Star Rating",
                  subtitle="Overlapping histograms per rating group",
                  show_legend=True, height=320)
        elif split_by == "Ownership Type":
            adf2 = adf.copy()
            adf2["ownership_type"] = adf2["ownership_type"].str.title()
            fig = px.histogram(adf2, x=sel_feat, color="ownership_type",
                               nbins=40, barmode="overlay", opacity=0.7,
                               color_discrete_sequence=[ACCENT, ACCENT2, SUCCESS, WARN],
                               labels={sel_feat: feat_label, "ownership_type": "Ownership"})
            chart(fig, title=f"{feat_label}: Distribution by Ownership Type",
                  subtitle="Overlapping histograms per ownership group",
                  show_legend=True, height=320)
        else:
            fig = px.histogram(adf, x=sel_feat, nbins=50, marginal="box",
                               color_discrete_sequence=[ACCENT],
                               labels={sel_feat: feat_label})
            fig.update_traces(
                hovertemplate=f"{feat_label}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
            )
            chart(fig, title=f"{feat_label}: Overall Distribution",
                  subtitle="Histogram with box plot showing median and spread",
                  height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Distribution of {feat_label} across the selected facilities. The shape of the histogram tells you where most facilities cluster and whether the data is skewed.")

    with dh2:
        fig = px.violin(adf, x="overall_rating", y=sel_feat,
                        box=True, points="outliers",
                        color="overall_rating",
                        color_discrete_sequence=px.colors.sequential.Blues[1:],
                        labels={"overall_rating": "Star Rating", sel_feat: feat_label})
        fig.update_traces(
            hovertemplate=f"Rating: %{{x}}★<br>{feat_label}: %{{y:.2f}}<extra></extra>"
        )
        fig.update_xaxes(tickvals=[1,2,3,4,5])
        chart(fig, title=f"{feat_label}: Spread by Star Rating",
              subtitle="Violin shows full distribution shape; box shows median and IQR",
              show_legend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each violin shows the full spread of values for that star rating. The wider the shape at a given value, the more facilities land there. The inner box marks the median and interquartile range.")

    st.divider()

    # Scatter matrix
    st.markdown("#### Feature Relationships: Scatter Matrix")
    sm_cols = st.multiselect(
        "Select 2–5 features to compare",
        present, default=["total_nursing_hrs", "rn_hrs", "num_deficiencies"],
        format_func=lambda c: LABELS.get(c, c),
    )
    if len(sm_cols) >= 2:
        fig = px.scatter_matrix(
            adf.sample(min(1500, len(adf)), random_state=0),
            dimensions=sm_cols, color="overall_rating",
            color_continuous_scale=px.colors.sequential.Blues,
            labels=LABELS, opacity=0.5,
            title="<b>Scatter Matrix: Pairwise Feature Relationships</b><br>"
                  "<span style='font-size:11px;font-weight:400;color:#64748B'>"
                  "Each panel shows the relationship between two features, coloured by star rating</span>",
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
        fig.update_layout(
            paper_bgcolor=T, plot_bgcolor=SURFACE, font_color=TEXT_SEC,
            height=520, coloraxis_colorbar=dict(title="Stars", thickness=12),
            title_font=dict(size=13, color=TEXT_PRI, family="Inter"),
            margin=dict(t=72, b=12, l=12, r=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each small panel shows the relationship between two features. A diagonal pattern means they move together (positive correlation). A scattered cloud means little or no relationship. Colour shows star rating.")
    else:
        st.info("Select at least 2 features to display the scatter matrix.")

    st.divider()

    # Correlation heatmap
    corr = adf.select_dtypes(include=np.number).corr().round(2)
    corr.index   = [LABELS.get(c, c.replace("_"," ").title()) for c in corr.index]
    corr.columns = [LABELS.get(c, c.replace("_"," ").title()) for c in corr.columns]
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_traces(hovertemplate="%{x} × %{y}<br>Correlation: %{z:.2f}<extra></extra>")
    fig.update_layout(
        title=dict(
            text="<b>Feature Correlation Matrix</b><br>"
                 "<span style='font-size:11px;font-weight:400;color:#64748B'>"
                 "Pearson correlation coefficients. Red = negative, blue = positive.</span>",
            font=dict(size=13, color=TEXT_PRI, family="Inter"), x=0, xanchor="left",
        ),
        paper_bgcolor=T, font_color=TEXT_SEC, height=480,
        margin=dict(t=72, b=12, l=12, r=12),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows the Pearson correlation between every pair of numeric features. A value of +1 means both metrics rise together. A value of -1 means one rises as the other falls. Values near 0 mean no linear relationship. For example, total nursing hours and RN hours are strongly correlated because RN hours are a subset of total hours.")

    st.divider()

    # Outlier chart
    outlier_flags = detect_outliers_iqr(adf, cols=present)
    pcts = {LABELS.get(c, c): outlier_flags[c].mean() * 100
            for c in present if c in outlier_flags}
    fig = go.Figure(go.Bar(
        x=list(pcts.values()), y=list(pcts.keys()),
        orientation="h",
        marker=dict(color=list(pcts.values()),
                    colorscale=[[0,"#FEF3C7"],[0.5,WARN],[1,DANGER]],
                    showscale=True,
                    colorbar=dict(title="%", thickness=12, len=0.7)),
        text=[f"{v:.1f}%" for v in pcts.values()],
        textposition="outside",
        hovertemplate="%{y}<br>Outlier rate: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title="% of Records Flagged as Outlier")
    fig.update_yaxes(autorange="reversed")
    chart(fig,
          title="Outlier Prevalence by Feature (IQR Method)",
          subtitle="Values beyond Q1 - 1.5xIQR or Q3 + 1.5xIQR are flagged",
          height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Outliers are records with unusually extreme values for that metric. A high outlier rate does not necessarily mean bad data — it can reflect real-world extremes like very large facilities or severe understaffing. These records are kept in the dataset but are worth investigating.")

    st.divider()
    st.markdown("#### Distribution Summary Table")
    dist_df = distribution_summary(adf)
    st.dataframe(dist_df.style.format(precision=3)
                 .background_gradient(cmap="Blues", subset=["mean","std"])
                 .background_gradient(cmap="RdYlGn_r", subset=["skewness"]),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<p class="page-eyebrow">Machine Learning</p>', unsafe_allow_html=True)
    st.title("Model Performance")
    st.markdown("Binary classification: **High quality (4-5 stars)** vs **Low quality (1-2 stars)**. Three-star facilities excluded. Stratified 5-fold cross-validation.")
    st.divider()

    p = get_pipeline()
    fpr, tpr, thresholds = roc_curve(p["y_test"], p["y_proba"])
    roc_auc = auc(fpr, tpr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy",  f"{accuracy_score(p['y_test'], p['y_pred']):.1%}")
    m2.metric("F1-Macro",       f"{f1_score(p['y_test'], p['y_pred'], average='macro'):.3f}")
    m3.metric("ROC-AUC",        f"{roc_auc:.3f}")
    m4.metric("Precision",      f"{precision_score(p['y_test'], p['y_pred'], average='macro'):.3f}")
    st.divider()

    r1, r2 = st.columns(2)

    with r1:
        cv_df = pd.DataFrame(list(p["cv_results"].items()), columns=["Model", "F1"])
        cv_df = cv_df.sort_values("F1", ascending=False)
        fig = px.bar(cv_df, x="Model", y="F1",
                     color="Model",
                     color_discrete_sequence=[ACCENT, ACCENT2, SUCCESS],
                     text=cv_df["F1"].map("{:.4f}".format),
                     range_y=[0.85, 0.92],
                     labels={"F1": "F1-Macro Score"})
        fig.update_traces(
            textposition="outside", textfont=dict(color=TEXT_SEC, size=11),
            hovertemplate="<b>%{x}</b><br>F1-Macro: %{y:.4f}<extra></extra>",
        )
        fig.update_yaxes(title="F1-Macro Score")
        chart(fig,
              title="Model Comparison: 5-Fold Cross-Validation F1-Macro",
              subtitle="Higher is better. All three models compared on the same data split.",
              show_legend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Three different machine learning models were trained and evaluated on the same data using 5-fold cross-validation (the data is split into 5 equal parts; each part is used once as a test set). F1-Macro balances precision and recall across both classes. The top-scoring model is used for all predictions in the app.")

    with r2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            line=dict(color=ACCENT, width=2.5),
            name=f"AUC = {roc_auc:.3f}",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>Best Model</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(color=BORDER, dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        dist = np.sqrt((fpr**2) + ((1-tpr)**2))
        opt_idx = np.argmin(dist)
        fig.add_trace(go.Scatter(
            x=[fpr[opt_idx]], y=[tpr[opt_idx]],
            mode="markers+text",
            marker=dict(color=DANGER, size=9),
            text=[f"  Optimal threshold ({thresholds[opt_idx]:.2f})"],
            textposition="middle right",
            textfont=dict(size=10, color=TEXT_SEC),
            showlegend=False,
            hovertemplate=f"Optimal point<br>FPR: {fpr[opt_idx]:.3f}<br>TPR: {tpr[opt_idx]:.3f}<extra></extra>",
        ))
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        fig.update_layout(legend=dict(x=0.5, y=0.05, font=dict(size=11)))
        chart(fig,
              title="ROC Curve (Receiver Operating Characteristic)",
              subtitle=f"AUC = {roc_auc:.3f}. The closer to 1.0, the better the model separates the classes.",
              show_legend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The ROC curve plots how many true positives the model catches (Y axis) against how many false alarms it raises (X axis) at every possible decision threshold. A perfect model would reach the top-left corner. The red dot marks the threshold that best balances sensitivity and specificity.")

    r3, r4 = st.columns(2)

    with r3:
        cm = confusion_matrix(p["y_test"], p["y_pred"])
        pct = (cm / cm.sum() * 100).round(1)
        labels_cm = [[f"{cm[i][j]:,}<br>({pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Low Quality", "Predicted High Quality"],
            y=["Actual Low Quality", "Actual High Quality"],
            text=labels_cm, texttemplate="%{text}",
            colorscale=[[0,"#EFF6FF"],[1,ACCENT]],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text="<b>Confusion Matrix</b><br>"
                     "<span style='font-size:11px;font-weight:400;color:#64748B'>"
                     "Counts and percentages on held-out test set</span>",
                font=dict(size=13, color=TEXT_PRI, family="Inter"), x=0, xanchor="left",
            ),
            paper_bgcolor=T, plot_bgcolor=SURFACE, font_color=TEXT_SEC,
            height=340, margin=dict(t=72, b=12, l=12, r=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The confusion matrix shows what the model got right and wrong on unseen test data. The diagonal cells are correct predictions. Off-diagonal cells are errors: bottom-left = a low-quality facility predicted as high quality (the more dangerous mistake); top-right = a high-quality facility predicted as low quality.")

    with r4:
        if not p["importances"].empty:
            imp = p["importances"].reset_index()
            imp.columns = ["Feature", "Importance"]
            imp["Feature"] = imp["Feature"].str.replace("_", " ").str.title()
            imp = imp.sort_values("Importance", ascending=True)
            fig = go.Figure(go.Bar(
                x=imp["Importance"], y=imp["Feature"],
                orientation="h",
                marker=dict(color=imp["Importance"],
                            colorscale=[[0,"#DBEAFE"],[1,ACCENT]],
                            showscale=False),
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
            ))
            chart(fig,
                  title="Feature Importances: Best Model",
                  subtitle="Which inputs had the most influence on predictions",
                  height=340)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Longer bars = that feature had more influence on the model's decisions. Features like staffing rating and number of deficiencies typically dominate because they are directly tied to how CMS calculates the overall star rating.")

    st.divider()

    prob_df = pd.DataFrame({
        "Probability of High Quality": p["y_proba"],
        "Actual": p["y_test"].map({0: "Low Quality", 1: "High Quality"}),
    })
    fig = px.histogram(prob_df, x="Probability of High Quality",
                       color="Actual", nbins=50, barmode="overlay", opacity=0.75,
                       color_discrete_map={"High Quality": SUCCESS, "Low Quality": DANGER})
    fig.update_traces(hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra></extra>")
    fig.add_vline(x=0.5, line_dash="dash", line_color=TEXT_SEC, line_width=1.5,
                  annotation_text="Decision boundary (0.5)",
                  annotation_position="top right",
                  annotation_font_color=TEXT_SEC, annotation_font_size=11)
    fig.update_xaxes(title="Predicted Probability of High Quality")
    fig.update_yaxes(title="Number of Facilities")
    fig.update_layout(legend=dict(orientation="h", y=-0.25, font=dict(size=11)))
    chart(fig,
          title="Predicted Probability Distribution",
          subtitle="Well-separated peaks indicate the model distinguishes the two classes confidently",
          show_legend=True, height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This histogram shows the model's predicted probability of being high quality for every facility in the test set, split by actual outcome. Green = actually high quality, red = actually low quality. Well-separated peaks mean the model is confident and accurate. Overlap near 0.5 shows where the model is uncertain.")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Quality":
    st.markdown('<p class="page-eyebrow">Live Prediction</p>', unsafe_allow_html=True)
    st.title("Predict Facility Quality Rating")
    st.markdown("Configure a facility's parameters to get an instant quality prediction from the trained model."
)
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

        st.markdown("#### Staffing Hours per Resident per Day")
        total_hrs = st.slider("Total Nursing Hours", 1.0, 8.0, 3.8, 0.1)
        rn_hrs    = st.slider("RN Hours",            0.1, 4.0, 1.0, 0.1)
        aide_hrs  = st.slider("Aide Hours",          0.3, 7.0, 2.8, 0.1)

        st.markdown("#### Facility Details")
        fc1, fc2      = st.columns(2)
        num_beds      = fc1.number_input("Certified Beds",     20, 500, 150, 10)
        num_residents = fc2.number_input("Avg Residents / Day",10, 500, 120, 10)
        num_defic     = st.number_input("Health Deficiencies",  0,  60,   8,  1)
        ownership     = st.selectbox("Ownership Type", list(OWNERSHIP_MAP.keys()),
                                     format_func=str.title)
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
            label = "High Quality (4 to 5 Stars)" if pred == 1 else "Low Quality (1 to 2 Stars)"
            bg    = "rgba(16,185,129,0.06)" if pred == 1 else "rgba(239,68,68,0.06)"

            st.markdown(f"""
            <div style="background:{bg};border:1px solid {color};border-radius:8px;
                        padding:20px 24px;margin-bottom:16px;">
                <p style="font-size:0.7rem;font-weight:600;color:{TEXT_SEC};
                           text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0;">Predicted Rating</p>
                <p style="font-size:1.5rem;font-weight:700;color:{color};margin:0 0 4px 0;">{label}</p>
                <p style="font-size:0.8rem;color:{TEXT_SEC};margin:0;">Model confidence: {conf:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Inter"}},
                title={"text": "Probability of High Quality Rating",
                       "font": {"size": 12, "color": TEXT_SEC, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0,100], "ticksuffix": "%",
                             "tickcolor": BORDER, "tickfont": {"color": TEXT_SEC, "size": 10}},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": BG, "bordercolor": BORDER,
                    "steps": [{"range": [0,50], "color": "#FEF2F2"},
                               {"range": [50,100], "color": "#F0FDF4"}],
                    "threshold": {"line": {"color": TEXT_PRI, "width": 2},
                                  "thickness": 0.75, "value": 50},
                },
                domain={"x": [0,1], "y": [0,1]},
            ))
            fig.update_layout(paper_bgcolor=T, height=240,
                              margin=dict(t=40, b=0, l=16, r=16))
            st.plotly_chart(fig, use_container_width=True)

            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            if hasattr(clf, "coef_"):
                coefs    = pd.Series(clf.coef_[0], index=feature_cols)
                contribs = (coefs * input_df.iloc[0]).sort_values()
                cdf = contribs.reset_index()
                cdf.columns = ["Feature", "Contribution"]
                cdf["Feature"] = cdf["Feature"].str.replace("_", " ").str.title()
                cdf["color"]   = cdf["Contribution"].apply(lambda x: SUCCESS if x >= 0 else DANGER)
                fig = go.Figure(go.Bar(
                    x=cdf["Contribution"], y=cdf["Feature"],
                    orientation="h", marker_color=cdf["color"],
                    text=cdf["Contribution"].map("{:+.3f}".format),
                    textposition="outside",
                    textfont=dict(color=TEXT_SEC, size=10),
                    hovertemplate="%{y}<br>Contribution: %{x:+.3f}<extra></extra>",
                ))
                fig.add_vline(x=0, line_color=BORDER, line_width=1.5)
                chart(fig,
                      title="Feature Contributions to This Prediction",
                      subtitle="Green bars pushed the prediction toward High Quality. Red bars pushed it toward Low Quality.",
                      height=320, margin=dict(t=64, b=8, l=8, r=56))
                st.plotly_chart(fig, use_container_width=True)

            compare_cols = ["total_nursing_hrs", "rn_hrs", "aide_hrs", "num_deficiencies"]
            means = df[compare_cols].mean()
            this  = input_df.iloc[0]
            comp_df = pd.DataFrame({
                "Feature":        [LABELS.get(c, c) for c in compare_cols],
                "This Facility":  [this.get(c, np.nan) for c in compare_cols],
                "Dataset Average":[means[c] for c in compare_cols],
            }).melt(id_vars="Feature", var_name="Group", value_name="Value")
            fig = px.bar(comp_df, x="Feature", y="Value", color="Group",
                         barmode="group",
                         color_discrete_map={"This Facility": color, "Dataset Average": "#94A3B8"},
                         labels={"Value": "Value", "Feature": ""})
            fig.update_traces(hovertemplate="%{x}<br>%{data.name}: %{y:.2f}<extra></extra>")
            fig.update_layout(legend=dict(orientation="h", y=-0.28, font=dict(size=11)))
            chart(fig,
                  title="This Facility vs Dataset Average",
                  subtitle="How the entered values compare to the average across all 14,700 real facilities",
                  show_legend=True, height=300)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
                        padding:48px 24px;text-align:center;">
                <p style="color:{TEXT_SEC};font-size:0.875rem;margin:0;">
                    Configure the facility parameters on the left and click
                    <strong style="color:{TEXT_PRI};">Run Prediction</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Raw Data":
    st.markdown('<p class="page-eyebrow">Data Source</p>', unsafe_allow_html=True)
    st.title("Raw CMS Nursing Home Data")
    st.markdown(
        "This dataset is downloaded live from the **Centers for Medicare and Medicaid Services (CMS)** "
        "Nursing Home Compare public database. It is updated monthly and covers approximately 14,700 "
        "certified nursing facilities across the United States."
    )

    st.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-left:4px solid {ACCENT};
                border-radius:8px;padding:16px 20px;margin:12px 0 24px 0;">
        <p style="font-size:0.72rem;font-weight:600;color:{ACCENT};text-transform:uppercase;
                   letter-spacing:0.8px;margin:0 0 6px 0;">Official Data Source</p>
        <p style="color:{TEXT_PRI};font-weight:600;font-size:0.9rem;margin:0 0 4px 0;">
            CMS Nursing Home Compare — Provider Information
        </p>
        <p style="color:{TEXT_SEC};font-size:0.8rem;margin:0 0 8px 0;">
            Published by the U.S. Department of Health and Human Services.
            Data includes star ratings, staffing hours, health inspection results,
            and ownership type for every Medicare/Medicaid certified nursing facility.
        </p>
        <p style="color:{TEXT_SEC};font-size:0.75rem;margin:0;">
            Dataset ID: 4pq5-n9py &nbsp;&nbsp;|&nbsp;&nbsp;
            Portal: data.cms.gov &nbsp;&nbsp;|&nbsp;&nbsp;
            Updated: monthly
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Column reference
    col_info = {
        "overall_rating":           ("Overall Rating",              "CMS composite star rating (1 to 5)"),
        "staffing_rating":          ("Staffing Rating",             "Star rating based on staffing hours relative to case mix"),
        "quality_rating":           ("Quality Measure Rating",      "Star rating based on resident outcomes and quality measures"),
        "health_inspection_rating": ("Health Inspection Rating",    "Star rating based on state health inspection results"),
        "num_beds":                 ("Number of Certified Beds",    "Total licensed beds in the facility"),
        "num_residents":            ("Avg Residents per Day",       "Average daily census (occupied beds)"),
        "total_nursing_hrs":        ("Total Nursing Hours/Day",     "Reported total nurse staffing hours per resident per day"),
        "rn_hrs":                   ("RN Hours/Day",                "Registered nurse hours per resident per day"),
        "aide_hrs":                 ("Aide Hours/Day",              "Nurse aide hours per resident per day"),
        "num_deficiencies":         ("Health Deficiencies",         "Total deficiencies cited in the most recent inspection cycle"),
        "ownership_type":           ("Ownership Type",              "For-profit, non-profit, or government"),
    }
    ref_df = pd.DataFrame(
        [(col, info[0], info[1]) for col, info in col_info.items()],
        columns=["Column Name", "CMS Field", "Description"],
    )
    st.markdown("#### Column Reference")
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.divider()

    # Filters for the raw table
    st.markdown("#### Browse the Dataset")
    st.markdown("Use the filters below to explore the underlying records. Every row is one real U.S. nursing facility.")
    rc1, rc2, rc3 = st.columns(3)
    raw_own_opts = ["All"] + sorted(df["ownership_type"].dropna().unique().tolist())
    raw_own  = rc1.selectbox("Ownership type", raw_own_opts, key="raw_own")
    raw_rate = rc2.select_slider("Star rating", options=[1,2,3,4,5], value=(1,5), key="raw_rate")
    n_rows   = rc3.selectbox("Rows to display", [50, 100, 250, 500], index=1)

    rdf = df.copy()
    if raw_own != "All":
        rdf = rdf[rdf["ownership_type"] == raw_own]
    rdf = rdf[rdf["overall_rating"].between(raw_rate[0], raw_rate[1])]

    display_cols = [c for c in col_info.keys() if c in rdf.columns]
    rename_map   = {c: col_info[c][0] for c in display_cols}

    matched = len(rdf)
    st.markdown(f"**{min(n_rows, matched):,} of {matched:,} facilities** match the current filters.")

    table_df = rdf[display_cols].head(n_rows).rename(columns=rename_map).reset_index(drop=True)
    st.dataframe(table_df, use_container_width=True, height=500)

    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download this selection as CSV",
        data=csv_bytes,
        file_name="cms_nursing_homes.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown(f"""
    <p style="font-size:0.75rem;color:{TEXT_SEC};">
        Data downloaded from
        <code>data.cms.gov/provider-data/dataset/4pq5-n9py</code> via the CMS public API.
        No modifications are made to the source values — only null handling and type coercion
        are applied during ingestion. See <code>src/ingestion.py</code> and <code>src/cleaning.py</code>
        for the full pipeline.
    </p>
    """, unsafe_allow_html=True)
