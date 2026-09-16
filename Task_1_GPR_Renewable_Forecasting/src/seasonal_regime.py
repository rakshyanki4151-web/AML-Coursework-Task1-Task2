"""Cross-seasonal regime analysis: compares PSO-GPR accuracy in the dry
winter months (Oct-Dec) vs the monsoon months (Jun-Aug), using the
per-fold results already produced by walk_forward_expanding.py (Design
2), which stores test_start/test_end dates and RMSE per fold.

IMPORTANT CAVEAT (state this in the report): your chronological split
(config.py) trains on Jan-Aug 2025 and tests on Oct-Dec 2025 only, and
walk_forward_expanding's folds are carved from the same is_daylight-
filtered dataframe stepping backward from the end of the series -- so
depending on n_folds/test_days, your actual fold windows may ALL fall
within Oct-Dec (dry season) with none reaching back into Jun-Aug
(monsoon), because Jun-Aug was in the TRAINING period for the main
split. This script reports whichever months your folds actually cover
and says so explicitly, rather than assuming both regimes are present.
If no fold touches Jun-Aug, that itself is worth reporting: it means a
true monsoon-vs-winter comparison would require re-running
walk_forward_expanding with more/larger folds (or a different split)
that actually reach back into the monsoon period.

Run standalone after the main pipeline has produced
outputs/walk_forward_design2_results.json:
    python -m src.seasonal_regime
"""
import json

import pandas as pd

from . import config

DRY_WINTER_MONTHS = {10, 11, 12}   # Oct-Dec
MONSOON_MONTHS = {6, 7, 8}         # Jun-Aug


def _month_of(date_str):
    return pd.Timestamp(date_str).month


def _classify_fold(fold):
    start_month = _month_of(fold["test_start"])
    end_month = _month_of(fold["test_end"])
    months_covered = {start_month, end_month}
    if months_covered <= DRY_WINTER_MONTHS:
        return "dry_winter"
    if months_covered <= MONSOON_MONTHS:
        return "monsoon"
    return "other_or_mixed"


def analyse_target(target, design2_summary):
    rows = []
    for fold in design2_summary["folds"]:
        regime = _classify_fold(fold)
        for kernel_kind, rmse in fold["frozen_kernel_RMSE"].items():
            rows.append({
                "target": target,
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "regime": regime,
                "kernel": kernel_kind,
                "RMSE": rmse,
                "linear_regression_RMSE": fold["linear_regression_RMSE"],
            })
    return rows


def main():
    print("--- Cross-Seasonal Regime Analysis (Dry Winter vs Monsoon) ---\n")
    path = config.OUTPUTS_DIR / "walk_forward_design2_results.json"
    if not path.exists():
        print(f"{path} not found. Run the main pipeline first (python -m src.pipeline), "
              "which produces walk_forward_design2_results.json.")
        return

    with open(path) as f:
        design2_results = json.load(f)

    all_rows = []
    for target, summary in design2_results.items():
        all_rows.extend(analyse_target(target, summary))

    if not all_rows:
        print("No folds found in walk_forward_design2_results.json.")
        return

    df = pd.DataFrame(all_rows)
    regimes_present = sorted(df["regime"].unique())
    print(f"Regimes actually present in the available folds: {regimes_present}\n")

    if "monsoon" not in regimes_present:
        print("WARNING: no fold falls in the Jun-Aug monsoon window. Your current "
              "expanding-window folds step backward from the end of 2025 and Jun-Aug "
              "was inside the TRAINING period for the main split, so it is not covered "
              "by walk_forward_expanding's test folds as currently configured.")
        print("To get a genuine monsoon-vs-winter comparison you have two options:")
        print("  1. Re-run walk_forward_expanding.run_expanding_window with more folds "
              "(increase n_folds) or larger test_days so folds reach back into Jun-Aug.")
        print("  2. Report this as a limitation: with only one year of hourly data and "
              "a single chronological split, the dry-season/monsoon comparison could only "
              "be done for the season(s) that actually fall in the test-adjacent period.")
        print()

    summary_table = (
        df.groupby(["target", "kernel", "regime"])["RMSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print(summary_table.to_string(index=False))

    out_path = config.OUTPUTS_DIR / "seasonal_regime_summary.csv"
    summary_table.to_csv(out_path, index=False)
    df.to_csv(config.OUTPUTS_DIR / "seasonal_regime_detail.csv", index=False)
    print(f"\nSaved summary to {out_path}")
    print(f"Saved per-fold detail to {config.OUTPUTS_DIR / 'seasonal_regime_detail.csv'}")


if __name__ == "__main__":
    main()
