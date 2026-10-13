import streamlit as st
import pandas as pd
from model import BASELINE_FEATURES, IMPROVED_FEATURES, explain_project, run_anomaly_detection

st.set_page_config(page_title="LEDGERLENS", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at top left,#e6f4f1 0,#f7f8f4 38%,#eef2f3 100%);color:#17212b}
.block-container{max-width:1200px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:1.7rem 1.9rem;border:1px solid #9ccbc2;border-radius:20px;background:linear-gradient(120deg,#0f766e,#155e75);margin-bottom:1.2rem;box-shadow:0 12px 30px rgba(15,118,110,.14)}
.hero-kicker{font-size:.78rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#bdeee7}
.hero-title{font-size:2.6rem;font-weight:850;letter-spacing:.06em;color:white}
.hero-subtitle{color:#e1f6f2;font-size:1rem;margin-top:.35rem}
[data-testid="stMetric"]{background:rgba(255,255,255,.92);border:1px solid #d2e1df;border-radius:16px;padding:.8rem 1rem;box-shadow:0 5px 18px rgba(36,58,61,.06)}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:#55706f!important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#17343b!important}
.stTabs [data-baseweb="tab-list"]{gap:.45rem}
.stTabs [data-baseweb="tab"]{background:#e8f1ef!important;color:#315458!important;border-radius:10px 10px 0 0;padding:.55rem 1rem}
.stTabs [data-baseweb="tab"] *{color:#315458!important}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:#0f766e!important;color:white!important}
.stTabs [data-baseweb="tab"][aria-selected="true"] *{color:white!important}
[data-testid="stSelectbox"] label,[data-testid="stSelectbox"] label *{color:#55706f!important}
[data-testid="stSelectbox"] div{color:#17343b}
.stMarkdown p,.stMarkdown span,.stCaption{color:#294347}
h1,h2,h3,h4{color:#17343b!important}
[data-testid="stMetricValue"]{overflow:visible!important}
[data-testid="stMetricValue"] div{white-space:nowrap!important}
[data-testid="stMetricValue"]{font-size:1.55rem!important}
.stTabs [role="tab"]{color:#315458!important;opacity:1!important}
.stTabs [role="tab"] *{color:#315458!important;opacity:1!important}
.stTabs [role="tab"][aria-selected="true"]{color:#0f766e!important;opacity:1!important}
.stTabs [role="tab"][aria-selected="true"] *{color:#0f766e!important;opacity:1!important}
.stTabs [data-baseweb="tab-highlight"]{background:#0f766e!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="hero-kicker">CIVIC AUDIT INTELLIGENCE</div>
<div class="hero-title">🔎 LEDGERLENS</div>
<div class="hero-subtitle">Spot unusual cost, spending and timeline behaviour before deeper review.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Feature Engineering Selection
st.sidebar.header("⚙️ Model Configuration")
feature_choice = st.sidebar.selectbox(
    "Feature Set",
    ["Improved Features (Peer-Aware & Unclipped)", "Baseline Features (Original)"],
    help="Select the feature engineering strategy used by Isolation Forest."
)
feature_mode = "improved" if "Improved" in feature_choice else "baseline"

@st.cache_data
def get_results(mode: str):
    return run_anomaly_detection(feature_set=mode)

results = get_results(feature_mode)
data = results["data"]

a, b, c, d = st.columns(4)
a.metric("Projects", f"{len(data):,}")
b.metric("Anomalies", f"{int((data['anomaly'] == -1).sum()):,}")
c.metric("Avg Utilisation", f"{data['fund_utilisation'].mean():.1%}")
d.metric("Active Model", "Improved (Issue #2)" if feature_mode == "improved" else "Baseline (Original)")
st.divider()

overview, review, details, comp_tab = st.tabs(["Overview", "Review Queue", "Project Details", "Feature Comparison"])

with overview:
    st.subheader("Project Overview")
    left, right = st.columns(2)
    with left:
        st.markdown("### Sector Distribution")
        st.bar_chart(data["sector"].value_counts())
    with right:
        st.markdown("### Anomaly Score Distribution")
        hist = pd.cut(data["anomaly_score"], bins=10).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)

    if feature_mode == "improved":
        st.markdown("### Budget Burn vs Physical Progress (True Expenditure)")
        st.caption("True total budget expensed (funds used / estimated cost) vs physical completion percentage.")
        chart_data = data.assign(budget_spent_pct=data["budget_spent_ratio"] * 100)
        st.scatter_chart(chart_data, x="budget_spent_pct", y="progress_pct", color="sector")
    else:
        st.markdown("### Tranche Utilisation vs Physical Progress (Baseline)")
        st.caption("Tranche absorption (funds used / released) vs progress percentage.")
        chart_data = data.assign(fund_utilisation_pct=data["fund_utilisation"] * 100)
        st.scatter_chart(chart_data, x="fund_utilisation_pct", y="progress_pct", color="sector")

with review:
    st.subheader("Projects Flagged for Review")
    flagged = data[data["anomaly"] == -1].sort_values("anomaly_score").copy()
    st.caption("Lower anomaly scores indicate stronger model-level evidence of unusual behaviour.")
    
    if feature_mode == "improved":
        cols = ["id", "name", "sector", "district", "est_cost", "actual_cost", "cost_overrun", "budget_burn_vs_progress", "idle_funds_ratio", "anomaly_score"]
        display_df = flagged[cols].rename(columns={
            "id": "Project",
            "name": "Project Name",
            "sector": "Sector",
            "district": "District",
            "est_cost": "Est Cost (₹)",
            "actual_cost": "Actual Cost (₹)",
            "cost_overrun": "Cost Overrun",
            "budget_burn_vs_progress": "Burn vs Progress Gap",
            "idle_funds_ratio": "Idle Funds Ratio",
            "anomaly_score": "Anomaly Score"
        })
    else:
        cols = ["id", "name", "sector", "district", "est_cost", "actual_cost", "fund_utilisation", "progress_pct", "anomaly_score"]
        display_df = flagged[cols].rename(columns={
            "id": "Project",
            "name": "Project Name",
            "sector": "Sector",
            "district": "District",
            "est_cost": "Est Cost (₹)",
            "actual_cost": "Actual Cost (₹)",
            "fund_utilisation": "Fund Utilisation",
            "progress_pct": "Progress",
            "anomaly_score": "Anomaly Score"
        })
    st.dataframe(display_df, use_container_width=True)

with details:
    st.subheader("Project Details")
    project_id = st.selectbox("Select project", data["id"].tolist())
    row = data[data["id"] == project_id].iloc[0]
    st.write(f"### {row['name']}")
    st.write(f"**Sector:** {row['sector']}  ·  **District:** {row['district']}")
    
    cols = st.columns(4)
    cols[0].metric("Estimated Cost", f"₹{row['est_cost']:,.0f}")
    cols[1].metric("Actual Cost", f"₹{row['actual_cost']:,.0f}")
    cols[2].metric("Cost Overrun", f"{row['cost_overrun']:.1%}")
    cols[3].metric("Physical Progress", f"{row['progress_pct']:.1f}%")

    cols2 = st.columns(4)
    cols2[0].metric("Tranche Absorption", f"{row['fund_utilisation']:.1%}")
    cols2[1].metric("Budget Spent", f"{row['budget_spent_ratio']:.1%}")
    cols2[2].metric("Idle Funds", f"{row['idle_funds_ratio']:.1%}")
    cols2[3].metric("Time Elapsed", f"{row['time_elapsed']:.2f}x")

    if row["anomaly"] == -1:
        st.error(f"Flagged for review by the {feature_mode} detector (Anomaly score: {row['anomaly_score']:.3f}).")
    else:
        st.success(f"Not flagged by the {feature_mode} detector (Anomaly score: {row['anomaly_score']:.3f}).")

    st.markdown("### Why was it flagged?")
    reasons = explain_project(row, data, feature_set=feature_mode)
    if reasons:
        for reason in reasons:
            st.warning(reason)
    else:
        st.info("No strong rule-based signal was found for this project.")

    st.markdown("### All Engineered Indicators")
    metric_cols = [
        "est_cost", "actual_cost", "funds_released", "funds_used",
        "planned_days", "elapsed_days", "progress_pct",
        "cost_overrun", "cost_overrun_peer_gap",
        "fund_utilisation", "fund_util_peer_gap",
        "funds_released_ratio", "budget_burn_vs_progress",
        "time_elapsed", "schedule_slippage", "schedule_overrun",
        "idle_funds_ratio", "spend_vs_progress"
    ]
    st.dataframe(row[metric_cols].to_frame("Value"), use_container_width=True)

with comp_tab:
    st.subheader("Feature Set Comparison (Issue #2)")
    st.markdown("""
    **Baseline vs Improved Feature Architecture:**
    - **Baseline Flaw Resolved:** The baseline feature `spend_vs_progress = fund_utilisation - progress` confused *disbursement absorption* with *budget spent*. Early-stage projects that used 90% of a small initial 20% advance were wrongly flagged as runaway spenders (+70% spending lead).
    - **Improved `budget_burn_vs_progress`:** Replaces this with total public capital expended (`funds_used / est_cost - progress`), eliminating false positives on healthy early projects.
    - **Unclipped Timeline:** Baseline clipped `time_elapsed` at 1.0, blinding the model to severe delays among the 66 overdue projects. `schedule_slippage` leaves elapsed time unclipped.
    - **Sector Peer Medians:** `cost_overrun_peer_gap` and `fund_util_peer_gap` measure deviation relative to robust sector norms.
    - **Parked Idle Capital:** `idle_funds_ratio` flags public funds sitting unutilised in agency accounts.
    """)
    st.info("Run `python evaluate_features.py` to generate the complete statistical audit in `docs/feature_engineering_report.md`.")
