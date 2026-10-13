"""Unit tests for LEDGERLENS feature engineering and data sanitization (Issue #2)."""
import numpy as np
import pandas as pd
import pytest

from evaluate_features import compare_feature_sets, evaluate_feature_set
from model import (
    BASELINE_FEATURES,
    IMPROVED_FEATURES,
    explain_project,
    load_data,
    run_anomaly_detection,
)


def test_improved_features_columns_present():
    """Verify that all improved features are computed and present in load_data()."""
    data = load_data()
    for feat in IMPROVED_FEATURES:
        assert feat in data.columns, f"Missing improved feature: {feat}"
    assert data[IMPROVED_FEATURES].isna().sum().sum() == 0, "NaNs found in improved features"


def test_division_by_zero_safety(tmp_path):
    """Verify that zero denominators do not produce inf or NaN."""
    bad_df = pd.DataFrame({
        "project_id": ["TEST-1", "TEST-2"],
        "project_name": ["Bad Project 1", "Bad Project 2"],
        "sector": ["Roads", "Roads"],
        "district": ["DistA", "DistB"],
        "estimated_cost": [0.0, -100.0],
        "actual_cost": [100.0, 50.0],
        "funds_released": [0.0, 0.0],
        "funds_utilised": [0.0, 0.0],
        "planned_days": [0, -10],
        "elapsed_days": [100, 50],
        "progress": [0.5, 1.2],
    })
    csv_file = tmp_path / "bad_projects.csv"
    bad_df.to_csv(csv_file, index=False)

    sanitized = load_data(filepath=csv_file, sanitize=True)
    assert not np.isinf(sanitized[IMPROVED_FEATURES].values).any()
    assert not np.isnan(sanitized[IMPROVED_FEATURES].values).any()
    assert (sanitized["planned_days"] >= 1.0).all()
    assert (sanitized["est_cost"] >= 0.0).all()


def test_progress_scaling_normalization(tmp_path):
    """Verify that fractional progress in [0, 1] is correctly scaled to [0, 100]."""
    test_df = pd.DataFrame({
        "project_id": ["TEST-1"],
        "project_name": ["Fractional Progress Project"],
        "sector": ["Water"],
        "district": ["DistA"],
        "estimated_cost": [1000000.0],
        "actual_cost": [1000000.0],
        "funds_released": [500000.0],
        "funds_utilised": [400000.0],
        "planned_days": [300],
        "elapsed_days": [150],
        "progress": [0.65],  # Fractional progress 65%
    })
    csv_file = tmp_path / "frac_progress.csv"
    test_df.to_csv(csv_file, index=False)

    df = load_data(filepath=csv_file)
    assert df["progress_pct"].iloc[0] == pytest.approx(65.0, abs=1e-3)


def test_budget_burn_vs_progress_math():
    """Verify mathematical definition of budget_burn_vs_progress."""
    data = load_data()
    expected = (data["funds_used"] / data["est_cost"]) - (data["progress_pct"] / 100.0)
    assert np.allclose(data["budget_burn_vs_progress"], expected.clip(-2.0, 2.0), atol=1e-5)


def test_unclipped_schedule_slippage_unbounded():
    """Verify that schedule_slippage is unclipped and can capture delays past 1.0."""
    data = load_data()
    overdue_mask = data["time_elapsed"] > 1.0
    assert overdue_mask.sum() > 0, "Dataset should contain overdue projects"
    # Unclipped slippage should be strictly lower than baseline's clipped progress_vs_time
    assert (data.loc[overdue_mask, "schedule_slippage"] < data.loc[overdue_mask, "progress_vs_time"]).any()


def test_idle_funds_non_negative():
    """Verify that idle funds ratio is non-negative for well-behaved sample data."""
    data = load_data()
    assert (data["idle_funds_ratio"] >= 0.0).all()


def test_sector_peer_gaps_mean_near_zero():
    """Verify that sector peer gaps center around zero for each sector."""
    data = load_data()
    for sector in data["sector"].unique():
        sec_data = data[data["sector"] == sector]
        median_cost_gap = sec_data["cost_overrun_peer_gap"].median()
        assert abs(median_cost_gap) < 1e-4


def test_feature_set_switching():
    """Verify that run_anomaly_detection works for both 'improved' and 'baseline'."""
    res_imp = run_anomaly_detection(feature_set="improved")
    res_base = run_anomaly_detection(feature_set="baseline")

    assert res_imp["features"] == IMPROVED_FEATURES
    assert res_base["features"] == BASELINE_FEATURES
    assert len(res_imp["data"]) == 260
    assert len(res_base["data"]) == 260
    assert int((res_imp["data"]["anomaly"] == -1).sum()) == 32
    assert int((res_base["data"]["anomaly"] == -1).sum()) == 32


def test_explain_project_with_improved_features():
    """Verify that explain_project returns informative diagnostic strings."""
    data = load_data()
    # Pick a project with known cost overrun
    row = data.sort_values("cost_overrun", ascending=False).iloc[0]
    reasons = explain_project(row, data, feature_set="improved")
    assert isinstance(reasons, list)
    assert len(reasons) > 0
    assert any("Cost overrun" in r or "overrun" in r for r in reasons)


def test_evaluation_comparison_jaccard_overlap():
    """Verify that baseline and improved flag sets have substantial Jaccard overlap."""
    base_eval, imp_eval, summary, diff_df = compare_feature_sets()
    assert summary["baseline_flagged"] == 32
    assert summary["improved_flagged"] == 32
    assert summary["jaccard_similarity"] > 0.75, "Jaccard overlap should be substantial (> 75%)"
    assert len(diff_df) == summary["dropped_count"] + summary["newly_flagged_count"]
