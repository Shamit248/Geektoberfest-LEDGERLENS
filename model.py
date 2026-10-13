"""LEDGERLENS Anomaly Detection Model and Feature Engineering.

This module provides data loading, data validation/sanitization, feature engineering,
and Isolation Forest anomaly detection for civic public development projects.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).resolve().parent / "sample_projects.csv"

# Baseline 6 indicators from original prototype
BASELINE_FEATURES = [
    "cost_overrun",
    "fund_utilisation",
    "funds_released_ratio",
    "time_elapsed",
    "progress_vs_time",
    "spend_vs_progress",
]

# Improved indicators (Issue #2): peer-normalized, unclipped, and domain-grounded
IMPROVED_FEATURES = [
    "cost_overrun",
    "cost_overrun_peer_gap",
    "fund_util_peer_gap",
    "budget_burn_vs_progress",
    "schedule_slippage",
    "idle_funds_ratio",
]

# Default to improved features while maintaining FEATURES alias for backward compatibility
FEATURES = IMPROVED_FEATURES


def load_data(filepath: Optional[Union[str, Path]] = None, sanitize: bool = True) -> pd.DataFrame:
    """Load, validate, sanitize and engineer features for public development projects.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the project CSV. Defaults to sample_projects.csv.
    sanitize : bool, default=True
        Whether to explicitly sanitize missing, inconsistent, and extreme values.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns, data hygiene, and both baseline
        and improved engineered features.
    """
    path = Path(filepath) if filepath else DATASET_PATH
    data = pd.read_csv(path)

    # Standardize column naming
    rename = {
        "project_id": "id",
        "project_name": "name",
        "estimated_cost": "est_cost",
        "funds_utilised": "funds_used",
        "planned_days": "planned_days",
        "elapsed_days": "elapsed_days",
        "progress": "progress_pct",
    }
    data = data.rename(columns=rename)

    # 1. Handle missing string/category identifiers
    if "id" not in data.columns:
        data["id"] = [f"PRJ-{i + 2000}" for i in range(len(data))]
    else:
        data["id"] = data["id"].fillna(pd.Series([f"PRJ-{i + 2000}" for i in range(len(data))]))

    if "name" not in data.columns:
        data["name"] = data["id"]
    else:
        data["name"] = data["name"].fillna(data["id"])

    if "sector" not in data.columns:
        data["sector"] = "General"
    else:
        data["sector"] = data["sector"].fillna("General")

    if "district" not in data.columns:
        data["district"] = "Unknown"
    else:
        data["district"] = data["district"].fillna("Unknown")

    # 2. Convert and validate numeric attributes
    numeric_cols = [
        "est_cost",
        "actual_cost",
        "funds_released",
        "funds_used",
        "planned_days",
        "elapsed_days",
        "progress_pct",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    if sanitize:
        # Handle non-positive or missing estimated costs
        est_valid = data["est_cost"].dropna()
        est_median = est_valid[est_valid > 0].median() if len(est_valid[est_valid > 0]) > 0 else 1.0
        data["est_cost"] = data["est_cost"].fillna(est_median).clip(lower=0.0)

        # Actual costs and disbursements
        data["actual_cost"] = data["actual_cost"].fillna(data["est_cost"]).clip(lower=0.0)
        data["funds_released"] = data["funds_released"].fillna(0.0).clip(lower=0.0)
        data["funds_used"] = data["funds_used"].fillna(0.0).clip(lower=0.0)

        # Planned and elapsed project durations
        planned_valid = data["planned_days"].dropna()
        planned_median = planned_valid[planned_valid > 0].median() if len(planned_valid[planned_valid > 0]) > 0 else 365.0
        data["planned_days"] = data["planned_days"].fillna(planned_median).clip(lower=1.0)
        data["elapsed_days"] = data["elapsed_days"].fillna(0.0).clip(lower=0.0)

        # Normalize progress scale: if [0, 1.5], scale to percentage [0, 100]
        prog = data["progress_pct"].fillna(0.0)
        if prog.max(skipna=True) <= 1.5:
            prog = prog * 100.0
        data["progress_pct"] = prog.clip(0.0, 100.0)

    # Safe denominator series avoiding zero-division
    safe_est_cost = np.where(data["est_cost"] > 0, data["est_cost"], np.nan)
    safe_funds_released = np.where(data["funds_released"] > 0, data["funds_released"], np.nan)
    safe_planned_days = np.where(data["planned_days"] > 0, data["planned_days"], np.nan)

    # =========================================================================
    # BASELINE INDICATORS (Original formulas maintained for 1:1 comparison)
    # =========================================================================
    # 1. Cost overrun ratio
    data["cost_overrun"] = (data["actual_cost"] - data["est_cost"]) / safe_est_cost
    data["cost_overrun"] = data["cost_overrun"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. Fund utilisation (tranche absorption rate) clipped at 1.0
    data["fund_utilisation"] = data["funds_used"] / safe_funds_released
    data["fund_utilisation"] = (
        data["fund_utilisation"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    )

    # 3. Released ratio relative to estimate
    data["funds_released_ratio"] = data["funds_released"] / safe_est_cost
    data["funds_released_ratio"] = data["funds_released_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4. Elapsed time ratio
    data["time_elapsed"] = data["elapsed_days"] / safe_planned_days
    data["time_elapsed"] = data["time_elapsed"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 5. Progress vs time (baseline clips time_elapsed at 1.0)
    data["progress_vs_time"] = data["progress_pct"] / 100.0 - data["time_elapsed"].clip(upper=1.0)

    # 6. Spend vs progress (baseline subtracts progress from tranche absorption)
    data["spend_vs_progress"] = data["fund_utilisation"] - data["progress_pct"] / 100.0

    # =========================================================================
    # REVISED / IMPROVED INDICATORS (Issue #2)
    # =========================================================================
    # 1. True Budget Burn vs Physical Progress:
    # (Funds expended / est_cost) - (progress_pct / 100)
    # Measures the true divergence between total budget expended and physical delivery,
    # eliminating the lifecycle-conflation of baseline spend_vs_progress.
    data["budget_spent_ratio"] = (data["funds_used"] / safe_est_cost).fillna(0.0)
    data["budget_burn_vs_progress"] = data["budget_spent_ratio"] - data["progress_pct"] / 100.0

    # 2. Unclipped Schedule Slippage:
    # (progress_pct / 100) - time_elapsed
    # Removes the artificial 1.0 ceiling, preserving visibility into severe project delays.
    data["schedule_slippage"] = data["progress_pct"] / 100.0 - data["time_elapsed"]

    # 3. Schedule Overrun:
    # max(0, (elapsed_days - planned_days) / planned_days)
    # Measures magnitude of project duration exceeding the approved deadline.
    data["schedule_overrun"] = (data["elapsed_days"] - data["planned_days"]) / safe_planned_days
    data["schedule_overrun"] = (
        data["schedule_overrun"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    )

    # 4. Idle Funds Ratio (Parked Capital):
    # (funds_released - funds_used) / est_cost
    # Measures treasury funds released but sitting unspent in implementing agency accounts.
    data["idle_funds_ratio"] = ((data["funds_released"] - data["funds_used"]) / safe_est_cost).fillna(0.0)

    # 5. Fund Release Lead (Front-loading Risk):
    # (funds_released / est_cost) - (progress_pct / 100)
    # Captures advance release of public money ahead of verified physical progress.
    data["fund_release_lead"] = data["funds_released_ratio"] - data["progress_pct"] / 100.0

    # 6. Sector Peer-Normalized Deviations:
    # Normalizes cost overrun and tranche absorption against robust sector medians.
    port_cost_med = data["cost_overrun"].median()
    port_util_med = data["fund_utilisation"].median()

    sector_cost_meds = data.groupby("sector")["cost_overrun"].transform("median")
    sector_util_meds = data.groupby("sector")["fund_utilisation"].transform("median")

    data["cost_overrun_peer_gap"] = data["cost_overrun"] - sector_cost_meds.fillna(port_cost_med)
    data["fund_util_peer_gap"] = data["fund_utilisation"] - sector_util_meds.fillna(port_util_med)

    # Bounded extreme values / winsorization for numerical stability
    data["cost_overrun"] = data["cost_overrun"].clip(-1.0, 10.0)
    data["cost_overrun_peer_gap"] = data["cost_overrun_peer_gap"].clip(-5.0, 10.0)
    data["budget_burn_vs_progress"] = data["budget_burn_vs_progress"].clip(-2.0, 2.0)
    data["schedule_slippage"] = data["schedule_slippage"].clip(-10.0, 2.0)
    data["idle_funds_ratio"] = data["idle_funds_ratio"].clip(0.0, 2.0)

    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_anomaly_detection(
    contamination: float = 0.12,
    feature_set: Union[str, List[str]] = "improved",
    data: Optional[pd.DataFrame] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Execute Isolation Forest anomaly detection on project data.

    Parameters
    ----------
    contamination : float, default=0.12
        Expected proportion of anomalies in the dataset.
    feature_set : str or list of str, default="improved"
        "improved" for revised feature set, "baseline" for original 6 features,
        or an explicit list of feature column names.
    data : pd.DataFrame, optional
        Pre-loaded project DataFrame. If None, loaded via load_data().
    random_state : int, default=42
        Seed for deterministic Isolation Forest training.

    Returns
    -------
    dict
        Dictionary containing scored DataFrame, fitted model, fitted scaler,
        feature list, and feature_set name.
    """
    if data is None:
        df = load_data()
    else:
        df = data.copy()

    if feature_set == "baseline":
        features = list(BASELINE_FEATURES)
    elif feature_set == "improved":
        features = list(IMPROVED_FEATURES)
    elif isinstance(feature_set, (list, tuple)):
        features = list(feature_set)
    else:
        features = list(IMPROVED_FEATURES)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=random_state,
    )
    df["anomaly"] = model.fit_predict(X)
    df["anomaly_score"] = model.decision_function(X)
    df["anomaly_strength"] = (-df["anomaly_score"]).clip(lower=0.0)

    return {
        "data": df,
        "model": model,
        "scaler": scaler,
        "features": features,
        "feature_set": feature_set if isinstance(feature_set, str) else "custom",
    }


def explain_project(
    row: pd.Series,
    data: pd.DataFrame,
    feature_set: str = "improved",
) -> List[str]:
    """Generate evidence-based civic audit explanations for a project row.

    Parameters
    ----------
    row : pd.Series
        Single project row from scored DataFrame.
    data : pd.DataFrame
        Full dataset for peer comparison calculations.
    feature_set : str, default="improved"
        Whether to generate explanations using "improved" or "baseline" logic.

    Returns
    -------
    list of str
        Human-readable diagnostic reasons highlighting specific project deviations.
    """
    peers = data[data["sector"] == row["sector"]]
    reasons = []

    if feature_set == "baseline":
        peer_cost = peers["cost_overrun"].median()
        peer_util = peers["fund_utilisation"].median()
        peer_gap = peers["spend_vs_progress"].median()

        if row["cost_overrun"] > peer_cost + 0.15:
            reasons.append(
                f"Cost overrun is {row['cost_overrun']:.0%}, above the sector peer median of {peer_cost:.0%}."
            )
        if row["fund_utilisation"] > peer_util + 0.15:
            reasons.append(
                f"Fund utilisation is {row['fund_utilisation']:.0%}, above the sector peer median of {peer_util:.0%}."
            )
        if row["spend_vs_progress"] > peer_gap + 0.20:
            reasons.append("Spending is running ahead of reported physical progress.")
        if row["progress_vs_time"] < -0.20:
            reasons.append("Physical progress is materially behind the elapsed timeline.")
    else:
        peer_cost = peers["cost_overrun"].median()
        peer_util = peers["fund_utilisation"].median()

        # 1. Cost overrun (peer deviation and absolute)
        peer_gap = row.get("cost_overrun_peer_gap", row["cost_overrun"] - peer_cost)
        if peer_gap > 0.15:
            reasons.append(
                f"Cost overrun is {row['cost_overrun']:.1%}, which is {peer_gap:+.1%} above "
                f"the {row['sector']} sector median ({peer_cost:.1%})."
            )
        elif row["cost_overrun"] > 0.40:
            reasons.append(f"Substantial absolute cost overrun of {row['cost_overrun']:.1%}.")

        # 2. Budget burn vs physical delivery
        burn_gap = row.get("budget_burn_vs_progress", (row["funds_used"] / row["est_cost"]) - (row["progress_pct"] / 100.0))
        spent_ratio = row.get("budget_spent_ratio", row["funds_used"] / row["est_cost"])
        if burn_gap > 0.20:
            reasons.append(
                f"Budget expenditure ({spent_ratio:.1%}) is running materially ahead of "
                f"delivered physical progress ({row['progress_pct']:.1f}%)."
            )
        elif burn_gap < -0.30:
            reasons.append(
                f"Unusual progress-expenditure gap: physical progress ({row['progress_pct']:.1f}%) "
                f"greatly exceeds recorded budget spending ({spent_ratio:.1%})."
            )

        # 3. Schedule slippage
        slippage = row.get("schedule_slippage", (row["progress_pct"] / 100.0) - row["time_elapsed"])
        if slippage < -0.20:
            reasons.append(
                f"Physical progress ({row['progress_pct']:.1f}%) is severely lagging behind "
                f"elapsed timeline ({row['time_elapsed']:.1%} of planned duration)."
            )

        # 4. Idle funds / parked capital
        idle_ratio = row.get("idle_funds_ratio", (row["funds_released"] - row["funds_used"]) / row["est_cost"])
        if idle_ratio > 0.15:
            reasons.append(
                f"High idle funds: {idle_ratio:.1%} of total project budget has been released "
                f"by treasury but remains unutilised."
            )

        # 5. Peer fund absorption lag
        util_gap = row.get("fund_util_peer_gap", row["fund_utilisation"] - peer_util)
        if util_gap < -0.15:
            reasons.append(
                f"Tranche fund absorption is {row['fund_utilisation']:.1%}, lagging {row['sector']} "
                f"sector peer median ({peer_util:.1%}) by {util_gap:.1%}."
            )

    if not reasons and row.get("anomaly", 1) == -1:
        reasons.append(
            "Multivariate anomaly: The combination of project metrics is unusual "
            "relative to the overall portfolio."
        )

    return reasons
