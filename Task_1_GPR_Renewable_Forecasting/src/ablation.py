"""Ablation study: systematically removes/swaps one design choice at a
time from the PSO-tuned RBF GPR pipeline and re-measures test RMSE, to
show which design decisions actually matter rather than just asserting
they do. Kept as a SEPARATE script from pipeline.py deliberately -- it
changes multiple settings systematically and touches feature sets, so
bolting it into the main pipeline risked breaking the main run's clean
single-responsibility flow, and pipeline.py needs NO changes to support
this file. Run standalone: ``python -m src.ablation`` after
``python -m src.pipeline`` has produced data/processed/features.csv.

Six variants, each answering ONE question that nothing else in this
project already answers (kernel-choice RBF-vs-Matern-vs-Periodic and
PSO-vs-default-L-BFGS are deliberately NOT repeated here -- those are
already fully answered by pipeline.py's gpr_rbf_default / gpr_matern_
default / gpr_rbf_pso / gpr_matern_pso / gpr_periodic_pso rows in
metrics_table.csv, and re-running them here would just duplicate that
table under a different name):

1. full_pipeline -- PCA + residual learning + full feature set + PSO-RBF.
   Reference row every other row is compared against; not itself an
   ablation.
2. no_pca -- raw scaled features fed directly to GPR instead of the
   PCA-reduced components. Answers: was PCA worth its accuracy cost, or
   did it just buy dimensionality reduction / hardware efficiency at a
   real accuracy price? (Your own prior run found ~3.7-5% R2 loss --
   this variant is what produces that number.)
3. no_residual_learning -- GPR learns the raw target directly instead of
   (target - physical baseline). Answers: does the Lambert's-law solar /
   wind-momentum physical prior actually help, or does GPR alone do just
   as well without it?
4. no_cloud_features -- drops cloud_amount and its lag/rolling
   derivatives. Answers: what is the actual measured value of adding the
   NASA POWER CLOUD_AMT variable?
5. no_physics_features -- drops air_mass, solar_zenith, wind_speed_cubed
   (the hand-engineered physical features), keeping raw met variables +
   calendar + lags/rolling/cloud. Answers: does the "physics-informed"
   feature engineering actually add anything beyond what raw
   lags/calendar already capture -- this is the direct evidence for
   that claim, not just an assertion of it.
6. calendar_features_only (optional/extreme case) -- strips everything
   down to calendar + physics only, zero autoregressive history at all.
   Answers: how much does the model degrade with no lagged/rolling
   history whatsoever -- the floor-level sanity check.

Each PSO-based ablation uses a SINGLE PSO run (not the 5-seed multiseed
sweep) with a fixed seed, since the point here is comparing design
choices against each other under matched conditions, not re-litigating
PSO's own stability (already covered by fit_pso_gpr_multiseed in
gpr_models.py, run on the main split).
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config, splits, features, gpr_models, evaluate


def _prepare_common(feat_df, target):
    daylight_only = target in config.DAYLIGHT_FILTER_TARGETS
    train_full, val_full, test_full = splits.chronological_split(feat_df)
    if daylight_only:
        train_full = train_full[train_full["is_daylight"]].reset_index(drop=True)
        val_full = val_full[val_full["is_daylight"]].reset_index(drop=True)
        test_full = test_full[test_full["is_daylight"]].reset_index(drop=True)
    return train_full, val_full, test_full, daylight_only


def _baseline_for(target, df):
    if target == "solar_irradiance":
        return gpr_models.compute_solar_baseline(df)
    return gpr_models.compute_wind_baseline(df)


def _run_variant(train, val, test, target, feature_cols, use_pca, use_residual):
    target_next = f"{target}_next"
    train_sub = splits.gpr_subsample(train)

    X_train = train_sub[feature_cols].to_numpy()
    X_val = val[feature_cols].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_train = train_sub[target_next].to_numpy()
    y_val = val[target_next].to_numpy()
    y_test = test[target_next].to_numpy()

    scaler = gpr_models.build_scaler(X_train)
    Xs_train, Xs_val, Xs_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    if use_pca:
        pca = PCA(n_components=0.95, random_state=config.RANDOM_SEED)
        Xp_train = pca.fit_transform(Xs_train)
        Xp_val = pca.transform(Xs_val)
        Xp_test = pca.transform(Xs_test)
    else:
        Xp_train, Xp_val, Xp_test = Xs_train, Xs_val, Xs_test

    if use_residual:
        base_train = _baseline_for(target, train_sub).to_numpy()
        base_val = _baseline_for(target, val).to_numpy()
        base_test = _baseline_for(target, test).to_numpy()
        y_train_target = y_train - base_train
        y_val_target = y_val - base_val
    else:
        base_train = np.zeros(len(train_sub))
        base_val = np.zeros(len(val))
        base_test = np.zeros(len(test))
        y_train_target = y_train
        y_val_target = y_val

    gpr, pso_result, transformer = gpr_models.fit_pso_gpr(
        Xp_train, y_train_target, Xp_val, y_val_target, y_val, base_val,
        kernel_kind="rbf", seed=config.RANDOM_SEED,
    )
    mean_t = gpr.predict(Xp_test)
    mean_delta = transformer.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    preds = base_test + mean_delta
    return evaluate.regression_metrics(y_test, preds), pso_result.best_score


def run_ablation(feat_df, target):
    train, val, test, daylight_only = _prepare_common(feat_df, target)
    full_cols = list(features.FEATURE_COLUMNS)

    calendar_and_physics = ("solar_irradiance", "wind_speed", "temperature", "humidity", "cloud_amount",
                             "doy_sin", "doy_cos", "hour_sin", "hour_cos", "solar_zenith",
                             "air_mass", "wind_speed_cubed")
    calendar_only_cols = [c for c in full_cols if c in calendar_and_physics]

    no_cloud_cols = [c for c in full_cols if "cloud_amount" not in c]
    physics_derived = ("air_mass", "solar_zenith", "wind_speed_cubed")
    no_physics_cols = [c for c in full_cols if c not in physics_derived]

    variants = [
        ("full_pipeline", full_cols, True, True),
        ("no_pca", full_cols, False, True),
        ("no_residual_learning", full_cols, True, False),
        ("no_cloud_features", no_cloud_cols, True, True),
        ("no_physics_features", no_physics_cols, True, True),
        ("calendar_features_only", calendar_only_cols, True, True),
    ]

    results = {}
    for name, cols, use_pca, use_residual in variants:
        print(f"[{target}] ablation: {name} ({len(cols)} features, PCA={use_pca}, residual={use_residual})")
        t0 = time.time()
        metrics, val_rmse = _run_variant(train, val, test, target, cols, use_pca, use_residual)
        elapsed = time.time() - t0
        metrics["val_rmse_at_pso_optimum"] = val_rmse
        metrics["n_features"] = len(cols)
        metrics["elapsed_seconds"] = round(elapsed, 1)
        results[name] = metrics
        print(f"  -> test RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f} "
              f"({elapsed:.1f}s)")

    return results


def main():
    feat_csv = config.PROCESSED_DIR / "features.csv"
    if not feat_csv.exists():
        print(f"Error: {feat_csv} not found. Run the main pipeline first (python -m src.pipeline).")
        return
    feat_df = pd.read_csv(feat_csv, parse_dates=["date"])

    all_results = {}
    for target in config.TARGETS:
        all_results[target] = run_ablation(feat_df, target)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results written to {out_path}")

    # Compact summary table for the report
    rows = []
    for target, variants in all_results.items():
        for name, m in variants.items():
            rows.append({"target": target, "variant": name, "RMSE": m["RMSE"], "MAE": m["MAE"], "R2": m["R2"],
                         "n_features": m["n_features"]})
    summary_df = pd.DataFrame(rows)
    print("\nAblation summary:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(config.OUTPUTS_DIR / "ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
