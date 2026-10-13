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
.feature-pill{display:inline-block;background:#d1fae5;color:#065f46;border-radius:99px;font-size:.72rem;font-weight:600;padding:.15rem .65rem;margin:.1rem .15rem .1rem 0;border:1px solid #a7f3d0}
.feature-pill-base{background:#fef3c7;color:#92400e;border:1px solid #fde68a}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="hero-kicker">CIVIC AUDIT INTELLIGENCE</div>
<div class="hero-title">🔎 LEDGERLENS</div>
<div class="hero-subtitle">Spot unusual cost, spending and timeline behaviour before deeper review.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: feature-set selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Detector Settings")
    feature_set = st.radio(
        "Feature Set",
        options=["improved", "baseline"],
        format_func=lambda x: "✨ Improved (Issue #2)" if x == "improved" else "📋 Baseline (original)",
        index=0,
        help=(
            "**Improved** uses peer-normalized and unclipped indicators.\n\n"
            "**Baseline** uses the original 6 prototype features."
        ),
    )
    active_features = IMPROVED_FEATURES if feature_set == "improved" else BASELINE_FEATURES
    pill_class = "feature-pill" if feature_set == "improved" else "feature-pill feature-pill-base"
    pills_html = "".join(f'<span class="{pill_class}">{f}</span>' for f in active_features)
    st.markdown(f"**Active features ({len(active_features)}):**", unsafe_allow_html=False)
    st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "LEDGERLENS is a screening tool. An anomaly flags a project for "
        "closer human review — it does not prove wrongdoing."
    )


# ---------------------------------------------------------------------------
# Load results (cached per feature_set selection)
# ---------------------------------------------------------------------------
@st.cache_data
def get_results(fs: str):
    return run_anomaly_detection(feature_set=fs)


results = get_results(feature_set)
data = results["data"]

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
a, b, c, d = st.columns(4)
a.metric("Projects", f"{len(data):,}")
b.metric("Flagged for Review", f"{int((data['anomaly'] == -1).sum()):,}")
c.metric("Avg Fund Utilisation", f"{data['fund_utilisation'].mean():.1%}")
d.metric("Active Model", "Improved (Issue #2)" if feature_set == "improved" else "Baseline (Original)")
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
overview, review, details, compare = st.tabs(
    ["Overview", "Review Queue", "Project Details", "Feature Comparison"]
)

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

    if feature_set == "improved":
        st.markdown("### Budget Burn vs Physical Progress (True Expenditure)")
        st.caption(
            "Points above 0 % have spent more total budget than physical progress warrants. "
            "Points below 0 % indicate physical progress outpacing expenditure."
        )
        chart_data = data.assign(budget_burn_pct=data["budget_burn_vs_progress"] * 100)
        st.scatter_chart(
            chart_data, x="progress_pct", y="budget_burn_pct", color="sector",
            x_label="Physical Progress (%)", y_label="Budget Burn Ahead of Progress (%)"
        )
    else:
        st.markdown("### Tranche Utilisation vs Physical Progress (Baseline)")
        st.caption("Tranche absorption (funds used / released) vs progress percentage.")
        chart_data = data.assign(fund_utilisation_pct=data["fund_utilisation"] * 100)
        st.scatter_chart(
            chart_data, x="fund_utilisation_pct", y="progress_pct", color="sector"
        )

with review:
    st.subheader("Projects Flagged for Review")
    flagged = data[data["anomaly"] == -1].sort_values("anomaly_score").copy()
    st.caption(
        "Lower anomaly scores indicate stronger model-level evidence of unusual behaviour. "
        f"Feature set: **{feature_set}**."
    )
    if feature_set == "improved":
        cols = [
            "id", "name", "sector", "district",
            "est_cost", "actual_cost",
            "cost_overrun", "budget_burn_vs_progress",
            "idle_funds_ratio", "schedule_slippage",
            "cost_overrun_peer_gap", "anomaly_score",
        ]
        rename_map = {
            "id": "Project", "name": "Project Name",
            "est_cost": "Est. Cost (₹)", "actual_cost": "Actual Cost (₹)",
            "cost_overrun": "Cost Overrun",
            "budget_burn_vs_progress": "Burn vs Progress Gap",
            "idle_funds_ratio": "Idle Funds Ratio",
            "schedule_slippage": "Schedule Slippage",
            "cost_overrun_peer_gap": "Cost Overrun Peer Gap",
            "anomaly_score": "Anomaly Score",
        }
    else:
        cols = [
            "id", "name", "sector", "district",
            "est_cost", "actual_cost",
            "fund_utilisation", "progress_pct", "anomaly_score",
        ]
        rename_map = {
            "id": "Project", "name": "Project Name",
            "est_cost": "Estimated Cost", "actual_cost": "Actual Cost",
            "fund_utilisation": "Fund Utilisation",
            "progress_pct": "Progress", "anomaly_score": "Anomaly Score",
        }
    st.dataframe(
        flagged[cols].rename(columns=rename_map),
        use_container_width=True,
    )

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

    if feature_set == "improved":
        cols2 = st.columns(4)
        cols2[0].metric("Tranche Absorption", f"{row['fund_utilisation']:.1%}")
        cols2[1].metric("Budget Spent", f"{row['budget_spent_ratio']:.1%}")
        cols2[2].metric("Idle Funds", f"{row['idle_funds_ratio']:.1%}")
        cols2[3].metric("Time Elapsed", f"{row['time_elapsed']:.2f}x")

    if row["anomaly"] == -1:
        st.error(
            f"🚩 Flagged for review by the {feature_set} detector "
            f"(Anomaly score: {row['anomaly_score']:.3f})."
        )
    else:
        st.success(
            f"✅ Not flagged by the {feature_set} detector "
            f"(Anomaly score: {row['anomaly_score']:.3f})."
        )

    st.markdown("### Why was it flagged?")
    reasons = explain_project(row, data, feature_set=feature_set)
    if reasons:
        for reason in reasons:
            st.warning(reason)
    else:
        st.info("No strong rule-based signal was found for this project.")

    st.markdown("### All Engineered Indicators")
    if feature_set == "improved":
        metric_cols = [
            "est_cost", "actual_cost", "funds_released", "funds_used",
            "planned_days", "elapsed_days", "progress_pct",
            "cost_overrun", "cost_overrun_peer_gap",
            "fund_utilisation", "fund_util_peer_gap",
            "budget_spent_ratio", "budget_burn_vs_progress",
            "schedule_slippage", "schedule_overrun",
            "idle_funds_ratio", "fund_release_lead",
        ]
    else:
        metric_cols = [
            "est_cost", "actual_cost", "funds_released", "funds_used",
            "planned_days", "elapsed_days", "progress_pct",
            "cost_overrun", "fund_utilisation", "funds_released_ratio",
            "time_elapsed", "progress_vs_time", "spend_vs_progress",
        ]
    st.dataframe(row[metric_cols].to_frame("Value"), use_container_width=True)

with compare:
    st.subheader("Baseline vs Improved Features — Side-by-Side")
    st.caption(
        "Runs both detectors simultaneously so you can compare which projects "
        "each feature set flags and why the improved indicators provide richer signals."
    )
    st.markdown("""
    **Key architectural differences resolved by Issue #2:**
    - **`spend_vs_progress` flaw fixed:** Baseline subtracted progress from *tranche absorption* (not total budget), incorrectly flagging healthy early-stage projects absorbing their mobilization advance.
    - **Unclipped timeline:** Baseline clipped `time_elapsed` at 1.0, hiding severity of overdue delays. `schedule_slippage` is unclipped.
    - **Sector peer medians:** `cost_overrun_peer_gap` and `fund_util_peer_gap` measure deviation relative to sector norms, not the whole portfolio.
    - **Parked idle capital:** `idle_funds_ratio` flags public funds released by treasury but sitting unutilised.
    """)
    st.info(
        "Run `python evaluate_features.py` to regenerate the full statistical audit in "
        "`docs/feature_engineering_report.md` with live case studies."
    )

    @st.cache_data
    def get_both_results():
        base = run_anomaly_detection(feature_set="baseline")
        impr = run_anomaly_detection(feature_set="improved")
        return base, impr

    base_res, impr_res = get_both_results()
    base_flagged = set(base_res["data"][base_res["data"]["anomaly"] == -1]["id"])
    impr_flagged = set(impr_res["data"][impr_res["data"]["anomaly"] == -1]["id"])

    only_base = base_flagged - impr_flagged
    only_impr = impr_flagged - base_flagged
    both = base_flagged & impr_flagged

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Flagged by Both", len(both))
    col_b.metric("Only Baseline Flags", len(only_base))
    col_c.metric("Only Improved Flags", len(only_impr))

    st.markdown("#### Projects flagged only by the Improved detector")
    st.caption("These are projects the baseline missed — caught by idle-funds, unclipped slippage, or peer-gap signals.")
    if only_impr:
        impr_data = impr_res["data"]
        subset = impr_data[impr_data["id"].isin(only_impr)].sort_values("anomaly_score")
        st.dataframe(
            subset[["id", "name", "sector", "district",
                    "cost_overrun", "budget_burn_vs_progress",
                    "idle_funds_ratio", "cost_overrun_peer_gap",
                    "schedule_slippage", "anomaly_score"]],
            use_container_width=True,
        )
    else:
        st.info("No projects are uniquely flagged by the improved detector.")

    st.markdown("#### Projects flagged only by the Baseline detector")
    st.caption("These projects lose their flag with peer-normalized features — likely early-stage lifecycle artefacts.")
    if only_base:
        base_data = base_res["data"]
        subset = base_data[base_data["id"].isin(only_base)].sort_values("anomaly_score")
        st.dataframe(
            subset[["id", "name", "sector", "district",
                    "cost_overrun", "fund_utilisation",
                    "progress_vs_time", "spend_vs_progress", "anomaly_score"]],
            use_container_width=True,
        )
    else:
        st.info("No projects are uniquely flagged by the baseline detector.")
