"""Expanding-window walk-forward validation (Design 2): freezes the kernel
hyperparameters already found by the main split's PSO run (one GPR fit
per fold, no fresh search) and checks whether that fitted kernel structure
keeps performing as it's retrained on progressively more historical data.

This is the cheaper, more narrowly-scoped companion to walk_forward.py's
Design 1 (which re-runs the full PSO search on every fold). Design 2
answers a different, also-useful question: "given the kernel structure
PSO already found, does an operator get consistent value from simply
retraining on new data over time, without paying PSO's search cost
again?" -- which mirrors how a deployed system would actually be
operated (retrain periodically, don't re-tune from scratch every cycle).

Both designs are run from pipeline.main() and reported together, since
they test genuinely different things and neither alone is the complete
picture: Design 1 tests the whole method (including search stability)
across time; Design 2 isolates "does this specific found kernel structure
generalise across time" from "is the search itself reliable" (the latter
already covered separately by gpr_models.fit_pso_gpr_multiseed on the
main split).

UPGRADED to match Design 1's rigor (previously this only reported raw
RMSE per fold per kernel, with no way to tell whether an RMSE difference
was real or noise, and no uncertainty-calibration check at all -- an
inconsistency worth fixing since Design 1 already had both):
  - paired t-test/Wilcoxon AND Diebold-Mariano significance tests, each
    frozen kernel vs Linear Regression, per fold (same tests used
    throughout the rest of this project, via evaluate.py).
  - coverage_95 (evaluate.coverage_at_z) per fold per kernel, since a
    frozen kernel refit on new data can still report calibrated or
    miscalibrated uncertainty, and this project's whole premise is that
    GPR's confidence band is worth having -- that claim should be
    checked here too, not just on the main split.
  - an explicit best_kernel_beats_lr flag and a per-kernel win count
    across folds, mirroring Design 1's summary exactly, so the two
    designs' JSON outputs are directly comparable side by side.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PowerTransformer

from . import config, splits, features, gpr_models, baselines, evaluate


def _fit_and_predict_frozen(kernel_kind, position, Xs_train, y_train_res, Xs_test, base_test):
    """One GPR fit with a FROZEN kernel (optimizer=None, hyperparameters
    fixed at ``position``), returning test-set mean and a calibrated std
    (via the same lower/upper-inverse-transform trick used in pipeline.py,
    since PowerTransformer's inverse is nonlinear so std can't just be
    inverse-transformed directly).
    """
    base_cls = gpr_models.RBF if kernel_kind == "rbf" else (
        gpr_models.Matern if kernel_kind == "matern" else gpr_models.ExpSineSquared)
    base_kwargs = {} if kernel_kind != "matern" else {"nu": 1.5}
    kernel = gpr_models._make_kernel(base_cls, *position, **base_kwargs)

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True,
                                    random_state=config.RANDOM_SEED)
    transformer = PowerTransformer(method="yeo-johnson")
    y_train_t = transformer.fit_transform(y_train_res.reshape(-1, 1)).ravel()
    gpr.fit(Xs_train, y_train_t)

    mean_t, std_t = gpr.predict(Xs_test, return_std=True)
    mean_res = transformer.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    mean = base_test + mean_res
    lower_res = transformer.inverse_transform((mean_t - std_t).reshape(-1, 1)).ravel()
    upper_res = transformer.inverse_transform((mean_t + std_t).reshape(-1, 1)).ravel()
    lower, upper = base_test + lower_res, base_test + upper_res
    std_final = np.clip((upper - lower) / 2.0, a_min=1e-6, a_max=None)
    return mean, std_final


def run_expanding_window(feat_df, target, kernel_positions, n_folds=3, test_days=30):
    """``kernel_positions``: dict mapping kernel_kind ("rbf"/"matern"/
    "periodic") -> (length_scale, signal_variance, noise) tuple, i.e. the
    best_position already found by the main split's PSO run for that
    kernel (pipeline.py stores these in results["models"][...]).
    """
    daylight_only = target in config.DAYLIGHT_FILTER_TARGETS
    df = feat_df[feat_df["is_daylight"]].reset_index(drop=True) if daylight_only else feat_df
    target_next = f"{target}_next"

    fold_results = []
    for i, (train, test) in enumerate(splits.expanding_window_splits(df, n_folds=n_folds, test_days=test_days)):
        train_sub = splits.gpr_subsample(train)
        X_train = train_sub[features.FEATURE_COLUMNS].to_numpy()
        y_train = train_sub[target_next].to_numpy()
        X_test = test[features.FEATURE_COLUMNS].to_numpy()
        y_test = test[target_next].to_numpy()

        scaler = gpr_models.build_scaler(X_train)
        Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

        if target == "solar_irradiance":
            base_train = gpr_models.compute_solar_baseline(train_sub)
            base_test = gpr_models.compute_solar_baseline(test)
        else:
            base_train = gpr_models.compute_wind_baseline(train_sub)
            base_test = gpr_models.compute_wind_baseline(test)

        y_train_res = (y_train - base_train).to_numpy()
        base_test_arr = base_test.to_numpy()

        # Linear Regression for the same fold, as a like-for-like generalisation comparison
        lr = baselines.fit_linear_regression(
            train[features.FEATURE_COLUMNS].to_numpy(), train[target_next].to_numpy()
        )
        pred_lr = lr.predict(X_test)
        lr_metrics = evaluate.regression_metrics(y_test, pred_lr)

        kernel_fold_results = {}
        best_overall = None  # (rmse, kernel_name, preds)
        for kernel_kind, position in kernel_positions.items():
            mean, std_final = _fit_and_predict_frozen(
                kernel_kind, position, Xs_train, y_train_res, Xs_test, base_test_arr
            )
            metrics = evaluate.regression_metrics(y_test, mean)
            metrics["NLPD"] = evaluate.nlpd(y_test, mean, std_final)
            metrics["coverage_95"] = evaluate.coverage_at_z(y_test, mean, std_final, z=1.96)

            sig = evaluate.paired_error_test(y_test, mean, pred_lr)
            dm = evaluate.diebold_mariano_test(y_test, mean, pred_lr)

            kernel_fold_results[kernel_kind] = {
                "RMSE": metrics["RMSE"], "MAE": metrics["MAE"], "R2": metrics["R2"],
                "NLPD": metrics["NLPD"], "coverage_95": metrics["coverage_95"],
                "beats_lr": bool(metrics["RMSE"] < lr_metrics["RMSE"]),
                "significance_vs_lr": sig,
                "dm_vs_lr": dm,
            }
            print(f"[{target}] fold {i} (train={len(train)}h) [{kernel_kind}, frozen]: "
                  f"RMSE={metrics['RMSE']:.4f} (LR={lr_metrics['RMSE']:.4f}), "
                  f"coverage_95={metrics['coverage_95']:.3f}, "
                  f"beats_lr={kernel_fold_results[kernel_kind]['beats_lr']}")

            if best_overall is None or metrics["RMSE"] < best_overall[0]:
                best_overall = (metrics["RMSE"], kernel_kind)

        best_rmse, best_kernel_name = best_overall
        fold_results.append({
            "fold": i,
            "train_hours": int(len(train)),
            "test_start": str(test["date"].min()),
            "test_end": str(test["date"].max()),
            "linear_regression_RMSE": lr_metrics["RMSE"],
            "frozen_kernel_results": kernel_fold_results,
            # kept for backward compatibility with anything reading the old flat-RMSE shape
            "frozen_kernel_RMSE": {k: v["RMSE"] for k, v in kernel_fold_results.items()},
            "best_kernel": best_kernel_name,
            "best_kernel_RMSE": best_rmse,
            "best_kernel_beats_lr": bool(best_rmse < lr_metrics["RMSE"]),
        })

    n_wins = sum(1 for r in fold_results if r["best_kernel_beats_lr"])
    summary = {
        "target": target,
        "n_folds_completed": len(fold_results),
        "pso_gpr_wins": n_wins,
        "folds": fold_results,
    }
    print(f"[{target}] Walk-forward (Design 2, frozen kernel) complete: "
          f"{len(fold_results)} expanding-window folds, "
          f"best frozen kernel beat Linear Regression in {n_wins}/{len(fold_results)} folds")
    return summary


def main():
    import json
    import pandas as pd
    feat_csv = config.PROCESSED_DIR / "features.csv"
    if not feat_csv.exists():
        print(f"Error: {feat_csv} not found. Run the main pipeline first (python -m src.pipeline).")
        return
    feat_df = pd.read_csv(feat_csv, parse_dates=["date"])

    metrics_json = config.OUTPUTS_DIR / "metrics.json"
    if not metrics_json.exists():
        print(f"Error: {metrics_json} not found. Run the main pipeline first.")
        return
    with open(metrics_json, "r") as f:
        metrics_data = json.load(f)

    walk_forward_expanding_results = {}
    for target in config.TARGETS:
        models = metrics_data[target]["models"]
        kernel_positions = {
            "rbf": tuple(models["gpr_rbf_pso"]["best_position"]),
            "matern": tuple(models["gpr_matern_pso"]["best_position"]),
            "periodic": tuple(models["gpr_periodic_pso"]["best_position"]),
        }
        walk_forward_expanding_results[target] = run_expanding_window(
            feat_df, target, kernel_positions, n_folds=3
        )

    out_file = config.OUTPUTS_DIR / "walk_forward_design2_results.json"
    with open(out_file, "w") as f:
        json.dump(walk_forward_expanding_results, f, indent=2)
    print(f"\nWalk-forward Design 2 results saved to {out_file}")


if __name__ == "__main__":
    main()

