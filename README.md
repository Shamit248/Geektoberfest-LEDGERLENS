# 🔎 LEDGERLENS

**Public Development Project Anomaly Detection**

LEDGERLENS is a Streamlit machine-learning prototype that uses exploratory analysis, engineered project metrics and Isolation Forest to flag unusual public development project behaviour for further human review.

## Live Demo

https://ledgerlens-e8hoghpl88mynxsm2jfnw2.streamlit.app/

## Problem

Public development projects produce information about project cost, spending, fund release, utilisation, progress and timelines. Unusual combinations of these values can help identify projects for further review.

LEDGERLENS is a **screening tool, not a fraud or corruption detector**. An anomaly means the project's behaviour is unusual relative to the portfolio; it does not prove wrongdoing.

## What the prototype does

- Overview and EDA dashboard
- Isolation Forest anomaly detection
- Review queue for flagged projects
- Project-level anomaly explanations
- Sector-aware peer comparisons

## ML Pipeline

Project data → Feature engineering → Standardization → Isolation Forest → Anomaly score → Human review

## Feature Engineering

LEDGERLENS provides both the original baseline indicators and a revised, peer-normalized feature set (Issue #2):

### Baseline Indicators
- `cost_overrun`: `(actual_cost - est_cost) / est_cost`
- `fund_utilisation`: `(funds_used / funds_released).clip(0, 1)`
- `funds_released_ratio`: `funds_released / est_cost`
- `time_elapsed`: `elapsed_days / planned_days`
- `progress_vs_time`: `progress_pct / 100 - time_elapsed.clip(upper=1)`
- `spend_vs_progress`: `fund_utilisation - progress_pct / 100`

### Improved Indicators (Issue #2)
- `budget_burn_vs_progress`: `(funds_used / est_cost) - (progress_pct / 100)` (true budget expensed vs physical progress)
- `cost_overrun_peer_gap`: `cost_overrun - median(cost_overrun_sector)` (sector peer-normalized cost escalation)
- `fund_util_peer_gap`: `fund_utilisation - median(fund_utilisation_sector)` (sector peer-normalized tranche absorption)
- `schedule_slippage`: `progress_pct / 100 - time_elapsed` (unclipped timeline execution gap)
- `idle_funds_ratio`: `(funds_released - funds_used) / est_cost` (parked public treasury funds)
- `cost_overrun`: `(actual_cost - est_cost) / est_cost` (absolute financial cost inflation)

## Dataset

The repository contains a reproducible **260-project synthetic starter dataset**. It is designed to simulate public development project behaviour and includes sectors, districts, estimated and actual cost, funds released and utilised, planned and elapsed days, and progress. It must not be presented as official government data.

## Tech Stack

Python • Pandas • NumPy • scikit-learn • Isolation Forest • Streamlit • GitHub

## Run locally
 
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To run the feature engineering evaluation and regenerate `docs/feature_engineering_report.md`:

```bash
python evaluate_features.py
python -m pytest
```

## Competition Workflow

The base repository is intentionally a working baseline. Participants should:

1. Pick one open GitHub Issue.
2. Clone the repository and reproduce the baseline.
3. Make the improvement on their own branch.
4. Compare the result with the baseline.
5. Open a Pull Request linked to the issue.
6. Explain what changed, why it helps, and how it was evaluated.

### Open Challenges

- **Easy:** [Audit anomaly results and establish evaluation baseline](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/1)
- **Medium:** [Improve project feature engineering](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/2)
- **Medium:** [Improve anomaly explanations with evidence](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/3)
- **Hard:** [Compare anomaly-detection methods](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/4)

Anomalies are candidates for human review, not proof of fraud or misconduct. Since the starter dataset is synthetic, contributors should explain what their evaluation can and cannot establish.

## Project Structure

```
LEDGERLENS/
├── app.py
├── model.py
├── evaluate_features.py
├── sample_projects.csv
├── requirements.txt
├── CONTRIBUTING.md
├── PRESENTATION.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── feature_engineering_report.md
│   ├── baseline_flagged_projects.csv
│   ├── improved_flagged_projects.csv
│   └── feature_diff_cases.csv
└── tests/
    ├── test_model.py
    └── test_feature_engineering.py
```
