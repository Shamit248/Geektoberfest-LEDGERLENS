"""Tests for improved feature engineering (Issue #2).

Covers:
- Feature presence, numeric types, no nulls/inf
- Domain validity and winsorization bounds
- Peer-normalization (sector median near zero)
- budget_burn_vs_progress formula consistency
- Unclipped schedule_slippage vs baseline progress_vs_time
- Division-by-zero safety on degenerate inputs
- Progress scaling normalization (fractional vs percentage)
- run_anomaly_detection with baseline, improved, and custom feature_set
- explain_project: synthetic rows trigger expected rules
- Jaccard overlap comparison (via evaluate_features if available)
"""
import numpy as np
import pandas as pd
import pytest

from model import (
    BASELINE_FEATURES,
    IMPROVED_FEATURES,
    explain_project,
    load_data,
    run_anomaly_detection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    """Shared loaded dataset (sanitized, all features engineered)."""
    return load_data()


@pytest.fixture(scope="module")
def baseline_result():
    return run_anomaly_detection(feature_set="baseline")


@pytest.fixture(scope="module")
def improved_result():
    return run_anomaly_detection(feature_set="improved")


# ---------------------------------------------------------------------------
# 1. Feature presence and types
# ---------------------------------------------------------------------------

class TestFeaturePresence:
    """All engineered columns must exist and be numeric after load_data."""

    BASELINE_COLS = [
        "cost_overrun", "fund_utilisation", "funds_released_ratio",
        "time_elapsed", "progress_vs_time", "spend_vs_progress",
    ]
    IMPROVED_COLS = [
        "budget_spent_ratio", "budget_burn_vs_progress",
        "schedule_slippage", "schedule_overrun",
        "idle_funds_ratio", "fund_release_lead",
        "cost_overrun_peer_gap", "fund_util_peer_gap",
    ]

    def test_baseline_columns_present(self, data):
        missing = [c for c in self.BASELINE_COLS if c not in data.columns]
        assert missing == [], f"Missing baseline columns: {missing}"

    def test_improved_columns_present(self, data):
        missing = [c for c in self.IMPROVED_COLS if c not in data.columns]
        assert missing == [], f"Missing improved columns: {missing}"

    def test_all_feature_cols_are_numeric(self, data):
        all_cols = self.BASELINE_COLS + self.IMPROVED_COLS
        for col in all_cols:
            assert pd.api.types.is_numeric_dtype(data[col]), \
                f"Column '{col}' is not numeric (dtype={data[col].dtype})"

    def test_no_nulls_or_inf_in_features(self, data):
        all_cols = self.BASELINE_COLS + self.IMPROVED_COLS
        subset = data[all_cols]
        assert subset.isna().sum().sum() == 0, "NaN found in engineered features"
        assert not np.isinf(subset.values).any(), "Inf found in engineered features"


# ---------------------------------------------------------------------------
# 2. Feature value range / domain correctness
# ---------------------------------------------------------------------------

class TestFeatureDomains:
    """Improved indicators must stay within their designed bounds."""

    def test_idle_funds_ratio_non_negative(self, data):
        assert (data["idle_funds_ratio"] >= 0).all(), \
            "idle_funds_ratio should be >= 0 (winsorized)"

    def test_schedule_overrun_non_negative(self, data):
        assert (data["schedule_overrun"] >= 0).all(), \
            "schedule_overrun is clipped to lower=0"

    def test_cost_overrun_winsorized(self, data):
        assert (data["cost_overrun"] >= -1.0).all()
        assert (data["cost_overrun"] <= 10.0).all()

    def test_budget_burn_vs_progress_winsorized(self, data):
        assert (data["budget_burn_vs_progress"] >= -2.0).all()
        assert (data["budget_burn_vs_progress"] <= 2.0).all()

    def test_schedule_slippage_winsorized(self, data):
        assert (data["schedule_slippage"] >= -10.0).all()
        assert (data["schedule_slippage"] <= 2.0).all()

    def test_fund_utilisation_clipped_0_to_1(self, data):
        assert (data["fund_utilisation"] >= 0.0).all()
        assert (data["fund_utilisation"] <= 1.0).all()

    def test_progress_pct_in_range(self, data):
        assert (data["progress_pct"] >= 0).all()
        assert (data["progress_pct"] <= 100).all()


# ---------------------------------------------------------------------------
# 3. Peer-gap indicators are zero-mean within each sector
# ---------------------------------------------------------------------------

class TestPeerNormalization:
    """cost_overrun_peer_gap and fund_util_peer_gap should have near-zero
    median within each sector (by construction from sector medians)."""

    def test_cost_overrun_peer_gap_sector_median_near_zero(self, data):
        sector_meds = data.groupby("sector")["cost_overrun_peer_gap"].median()
        for sector, med in sector_meds.items():
            assert abs(med) < 0.05, \
                f"Sector '{sector}' cost_overrun_peer_gap median = {med:.4f} (expected ≈ 0)"

    def test_fund_util_peer_gap_sector_median_near_zero(self, data):
        sector_meds = data.groupby("sector")["fund_util_peer_gap"].median()
        for sector, med in sector_meds.items():
            assert abs(med) < 0.05, \
                f"Sector '{sector}' fund_util_peer_gap median = {med:.4f} (expected ≈ 0)"

    def test_all_sectors_represented_in_peer_gap(self, data):
        expected_sectors = {"Roads", "Water", "Schools", "Health", "Power"}
        assert set(data["sector"].unique()) == expected_sectors


# ---------------------------------------------------------------------------
# 4. Budget burn vs progress: formula and relationship checks
# ---------------------------------------------------------------------------

class TestBudgetBurnIndicator:
    """budget_burn_vs_progress = budget_spent_ratio - progress_pct/100."""

    def test_budget_burn_formula_consistency(self, data):
        expected = data["budget_spent_ratio"] - data["progress_pct"] / 100.0
        delta = (data["budget_burn_vs_progress"] - expected.clip(-2.0, 2.0)).abs()
        assert (delta < 1e-9).all(), "budget_burn_vs_progress formula mismatch"

    def test_budget_spent_ratio_non_negative(self, data):
        assert (data["budget_spent_ratio"] >= 0).all()

    def test_budget_burn_math_standalone(self):
        """Verify mathematical definition of budget_burn_vs_progress."""
        data = load_data()
        expected = (data["funds_used"] / data["est_cost"]) - (data["progress_pct"] / 100.0)
        assert np.allclose(
            data["budget_burn_vs_progress"], expected.clip(-2.0, 2.0), atol=1e-5
        )


# ---------------------------------------------------------------------------
# 5. Unclipped schedule slippage vs baseline progress_vs_time
# ---------------------------------------------------------------------------

class TestUnclippedScheduleSlippage:
    """schedule_slippage must differ from baseline progress_vs_time for
    projects where time_elapsed > 1.0 (i.e., already past deadline)."""

    def test_overdue_projects_differ_from_baseline(self, data):
        overdue = data[data["time_elapsed"] > 1.01]
        if len(overdue) == 0:
            pytest.skip("No overdue projects found in dataset")
        worse = (overdue["schedule_slippage"] < overdue["progress_vs_time"]).sum()
        assert worse > 0, \
            "Unclipped schedule_slippage should be more negative for overdue projects"

    def test_overdue_projects_exist(self, data):
        overdue_mask = data["time_elapsed"] > 1.0
        assert overdue_mask.sum() > 0, "Dataset should contain overdue projects"


# ---------------------------------------------------------------------------
# 6. Data hygiene: division-by-zero and progress scaling
# ---------------------------------------------------------------------------

class TestDataHygiene:

    def test_division_by_zero_safety(self, tmp_path):
        """Zero denominators must not produce inf or NaN."""
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

    def test_progress_scaling_normalization(self, tmp_path):
        """Fractional progress in [0, 1] is correctly scaled to [0, 100]."""
        test_df = pd.DataFrame({
            "project_id": ["TEST-1"],
            "project_name": ["Fractional Progress Project"],
            "sector": ["Water"],
            "district": ["DistA"],
            "estimated_cost": [1_000_000.0],
            "actual_cost": [1_000_000.0],
            "funds_released": [500_000.0],
            "funds_utilised": [400_000.0],
            "planned_days": [300],
            "elapsed_days": [150],
            "progress": [0.65],
        })
        csv_file = tmp_path / "frac_progress.csv"
        test_df.to_csv(csv_file, index=False)

        df = load_data(filepath=csv_file)
        assert df["progress_pct"].iloc[0] == pytest.approx(65.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 7. Isolation Forest: feature_set selection and determinism
# ---------------------------------------------------------------------------

class TestAnomalyDetectionFeatureSets:

    def test_baseline_result_uses_baseline_features(self, baseline_result):
        assert set(baseline_result["features"]) == set(BASELINE_FEATURES)
        assert baseline_result["feature_set"] == "baseline"

    def test_improved_result_uses_improved_features(self, improved_result):
        assert set(improved_result["features"]) == set(IMPROVED_FEATURES)
        assert improved_result["feature_set"] == "improved"

    def test_both_flag_exactly_32_anomalies(self, baseline_result, improved_result):
        n_base = int((baseline_result["data"]["anomaly"] == -1).sum())
        n_impr = int((improved_result["data"]["anomaly"] == -1).sum())
        assert n_base == 32, f"Baseline flagged {n_base}, expected 32"
        assert n_impr == 32, f"Improved flagged {n_impr}, expected 32"

    def test_improved_and_baseline_flag_different_projects(self, baseline_result, improved_result):
        base_ids = set(
            baseline_result["data"][baseline_result["data"]["anomaly"] == -1]["id"]
        )
        impr_ids = set(
            improved_result["data"][improved_result["data"]["anomaly"] == -1]["id"]
        )
        assert base_ids != impr_ids, \
            "Improved features should flag at least one different project than baseline"

    def test_feature_set_switching_sizes(self):
        """Verify feature lists are correct for each mode."""
        res_imp = run_anomaly_detection(feature_set="improved")
        res_base = run_anomaly_detection(feature_set="baseline")
        assert res_imp["features"] == IMPROVED_FEATURES
        assert res_base["features"] == BASELINE_FEATURES
        assert len(res_imp["data"]) == 260
        assert len(res_base["data"]) == 260

    def test_custom_feature_list_accepted(self):
        custom = ["cost_overrun", "schedule_slippage", "idle_funds_ratio"]
        result = run_anomaly_detection(feature_set=custom)
        assert result["features"] == custom
        assert result["feature_set"] == "custom"

    def test_result_is_deterministic(self):
        r1 = run_anomaly_detection(feature_set="improved", random_state=0)
        r2 = run_anomaly_detection(feature_set="improved", random_state=0)
        assert (r1["data"]["anomaly"] == r2["data"]["anomaly"]).all()

    def test_result_contains_required_keys(self, improved_result):
        assert {"data", "model", "scaler", "features", "feature_set"}.issubset(
            improved_result.keys()
        )

    def test_anomaly_strength_non_negative(self, improved_result):
        assert (improved_result["data"]["anomaly_strength"] >= 0).all()

    def test_jaccard_overlap_substantial(self, baseline_result, improved_result):
        """Baseline and improved flag sets should share >75% Jaccard similarity."""
        base_ids = set(
            baseline_result["data"][baseline_result["data"]["anomaly"] == -1]["id"]
        )
        impr_ids = set(
            improved_result["data"][improved_result["data"]["anomaly"] == -1]["id"]
        )
        intersection = len(base_ids & impr_ids)
        union = len(base_ids | impr_ids)
        jaccard = intersection / union if union > 0 else 0
        assert jaccard > 0.75, \
            f"Jaccard overlap should be >75%, got {jaccard:.1%}"


# ---------------------------------------------------------------------------
# 8. explain_project: improved path produces richer context
# ---------------------------------------------------------------------------

class TestExplainProject:

    def test_returns_list_for_normal_project(self, improved_result):
        data = improved_result["data"]
        normal = data[data["anomaly"] == 1].iloc[0]
        result = explain_project(normal, data, feature_set="improved")
        assert isinstance(result, list)

    def test_flagged_project_has_at_least_one_reason_or_fallback(self, improved_result):
        data = improved_result["data"]
        flagged = data[data["anomaly"] == -1]
        for _, row in flagged.iterrows():
            reasons = explain_project(row, data, feature_set="improved")
            assert len(reasons) >= 1, \
                f"Project {row['id']} flagged but explain_project returned no reasons"

    def test_baseline_explain_returns_list(self, baseline_result):
        data = baseline_result["data"]
        row = data.iloc[0]
        result = explain_project(row, data, feature_set="baseline")
        assert isinstance(result, list)

    def test_worst_cost_overrun_triggers_reason(self):
        """Project with highest cost overrun must get a cost-related reason."""
        data = load_data()
        row = data.sort_values("cost_overrun", ascending=False).iloc[0]
        reasons = explain_project(row, data, feature_set="improved")
        assert isinstance(reasons, list)
        assert len(reasons) > 0
        assert any("overrun" in r.lower() or "cost" in r.lower() for r in reasons)

    def test_explain_high_idle_funds(self):
        """Synthetic row with very high idle funds must trigger the idle-funds rule."""
        data = load_data()
        row = data.iloc[0].copy()
        row["idle_funds_ratio"] = 0.50
        row["cost_overrun_peer_gap"] = 0.00
        row["budget_burn_vs_progress"] = 0.00
        row["schedule_slippage"] = 0.00
        row["fund_util_peer_gap"] = 0.00
        row["anomaly"] = -1
        reasons = explain_project(row, data, feature_set="improved")
        idle_reasons = [r for r in reasons if "idle" in r.lower()]
        assert len(idle_reasons) >= 1, \
            f"Expected idle-funds reason, got: {reasons}"

    def test_explain_cost_overrun_peer_gap(self):
        """Synthetic row with large cost_overrun_peer_gap must trigger cost rule."""
        data = load_data()
        row = data.iloc[0].copy()
        row["cost_overrun_peer_gap"] = 0.50
        row["cost_overrun"] = 0.60
        row["budget_burn_vs_progress"] = 0.00
        row["schedule_slippage"] = 0.00
        row["idle_funds_ratio"] = 0.00
        row["fund_util_peer_gap"] = 0.00
        row["anomaly"] = -1
        reasons = explain_project(row, data, feature_set="improved")
        cost_reasons = [r for r in reasons if "cost overrun" in r.lower()]
        assert len(cost_reasons) >= 1, \
            f"Expected cost-overrun reason, got: {reasons}"
