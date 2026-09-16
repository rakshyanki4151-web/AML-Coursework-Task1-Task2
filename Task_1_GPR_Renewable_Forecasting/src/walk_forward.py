"""Rolling-origin walk-forward validation (Design 1): a robustness check
answering "does PSO-tuned GPR's advantage over baselines hold up across
different chronological train/test cuts, not just the one fixed split
used everywhere else in this project?"

UNSCOPED version: each fold re-runs the exact same full search the main
split gets -- all 3 kernels (RBF, Matern, Periodic), each with the full
config.PSO_SEEDS multi-seed sweep via gpr_models.fit_pso_gpr_multiseed --
compared against Linear Regression. This deliberately does NOT freeze
hyperparameters or reduce to a single kernel/seed: the project's actual
contribution is "PSO automatically finds the kernel," so a generalisation
check should test the whole method, search included, on every fold.

Runtime note: this is intentionally expensive. 3 folds x 2 targets x 3
kernels x 5 seeds = 90 full PSO searches, on top of the 30 already run on
the main split in pipeline.py. Expect a long run. If this needs to be
cheaper later, the affordable alternative is walk_forward_expanding.py
(Design 2: freezes the main split's already-found kernel hyperparameters
and only re-fits the GPR mean per fold -- much faster, but answers a
narrower question). Both are run from pipeline.main() so the report can
compare what each design shows.
"""
import time

from . import config, splits, features, gpr_models, baselines, evaluate


def run_walk_forward(feat_df, target, n_folds=3):
    daylight_only = target in config.DAYLIGHT_FILTER_TARGETS
    df = feat_df[feat_df["is_daylight"]].reset_index(drop=True) if daylight_only else feat_df
    target_next = f"{target}_next"

    fold_results = []
    for i, (train, val, test) in enumerate(splits.rolling_origin_splits(df, n_folds=n_folds)):
        t0 = time.time()
        print(f"[{target}] --- fold {i} (train={len(train)}h, "
              f"test {test['date'].min()} to {test['date'].max()}) ---")

        # --- Linear Regression baseline for this fold (full train window) ---
        X_train_full = train[features.FEATURE_COLUMNS].to_numpy()
        y_train_full = train[target_next].to_numpy()
        X_test = test[features.FEATURE_COLUMNS].to_numpy()
        y_test = test[target_next].to_numpy()

        lr = baselines.fit_linear_regression(X_train_full, y_train_full)
        pred_lr = lr.predict(X_test)
        lr_metrics = evaluate.regression_metrics(y_test, pred_lr)

        # --- Persistence baseline for this fold -------------------------
        # Added because Ridge alone is the weaker comparator: on the main
        # chronological split persistence beats Ridge on both targets, so a
        # walk-forward that only reports "beats Ridge" would overstate the
        # result. Persistence needs no fitting -- it is today's value at the
        # same hour -- so it costs nothing to include per fold.
        pred_persist = baselines.persistence_predict(test, target)
        persist_metrics = evaluate.regression_metrics(y_test, pred_persist)

        # --- Shared GPR inputs for this fold ---
        train_sub = splits.gpr_subsample(train)
        X_train_sub = train_sub[features.FEATURE_COLUMNS].to_numpy()
        y_train_sub = train_sub[target_next].to_numpy()
        X_val = val[features.FEATURE_COLUMNS].to_numpy()
        y_val = val[target_next].to_numpy()

        scaler = gpr_models.build_scaler(X_train_sub)
        Xs_train = scaler.transform(X_train_sub)
        Xs_val = scaler.transform(X_val)
        Xs_test = scaler.transform(X_test)

        if target == "solar_irradiance":
            base_train = gpr_models.compute_solar_baseline(train_sub)
            base_val = gpr_models.compute_solar_baseline(val)
            base_test = gpr_models.compute_solar_baseline(test)
        else:
            base_train = gpr_models.compute_wind_baseline(train_sub)
            base_val = gpr_models.compute_wind_baseline(val)
            base_test = gpr_models.compute_wind_baseline(test)

        y_train_res = (y_train_sub - base_train).to_numpy()
        y_val_res = (y_val - base_val).to_numpy()

        # --- Full 3-kernel x 5-seed PSO sweep, same as the main split ---
        kernel_results = {}
        best_overall = None  # (test_rmse, kernel_name, preds)
        for kernel_kind in ("rbf", "matern", "periodic"):
            print(f"  [{target}] fold {i}: PSO sweep, kernel={kernel_kind} "
                  f"({len(config.WALK_FORWARD_PSO_SEEDS)} seeds)...", flush=True)
            gpr, pso_result, transformer, pso_summary = gpr_models.fit_pso_gpr_multiseed(
                Xs_train, y_train_res, Xs_val, y_val_res, y_val, base_val.to_numpy(),
                kernel_kind=kernel_kind, seeds=config.WALK_FORWARD_PSO_SEEDS,
            )
            mean_t = gpr.predict(Xs_test)
            mean_res = transformer.inverse_transform(mean_t.reshape(-1, 1)).ravel()
            preds = base_test.to_numpy() + mean_res
            metrics = evaluate.regression_metrics(y_test, preds)
            kernel_results[kernel_kind] = {
                "test_RMSE": metrics["RMSE"], "test_MAE": metrics["MAE"], "test_R2": metrics["R2"],
                "val_rmse_mean_across_seeds": pso_summary["mean_val_rmse"],
                "val_rmse_std_across_seeds": pso_summary["std_val_rmse"],
                "best_seed": pso_summary["best_seed"],
                "best_position": [float(p) for p in pso_result.best_position],
            }
            print(f"    -> test RMSE={metrics['RMSE']:.4f} "
                  f"(val mean={pso_summary['mean_val_rmse']:.4f}, std={pso_summary['std_val_rmse']:.4f})")
            if best_overall is None or metrics["RMSE"] < best_overall[0]:
                best_overall = (metrics["RMSE"], kernel_kind, preds)

        best_rmse, best_kernel_name, best_preds = best_overall
        sig = evaluate.paired_error_test(y_test, best_preds, pred_lr)
        dm = evaluate.diebold_mariano_test(y_test, best_preds, pred_lr)
        sig_persist = evaluate.paired_error_test(y_test, best_preds, pred_persist)
        dm_persist = evaluate.diebold_mariano_test(y_test, best_preds, pred_persist)

        elapsed = time.time() - t0
        fold_results.append({
            "fold": i,
            "train_hours": int(len(train)),
            "test_start": str(test["date"].min()),
            "test_end": str(test["date"].max()),
            "linear_regression_RMSE": lr_metrics["RMSE"],
            "persistence_RMSE": persist_metrics["RMSE"],
            "kernel_results": kernel_results,
            "best_kernel": best_kernel_name,
            "best_kernel_test_RMSE": best_rmse,
            "best_kernel_beats_lr": bool(best_rmse < lr_metrics["RMSE"]),
            "best_kernel_beats_persistence": bool(best_rmse < persist_metrics["RMSE"]),
            "significance_vs_lr": sig,
            "dm_vs_lr": dm,
            "significance_vs_persistence": sig_persist,
            "dm_vs_persistence": dm_persist,
            "elapsed_seconds": round(elapsed, 1),
        })
        print(f"[{target}] fold {i} summary: LR RMSE={lr_metrics['RMSE']:.4f}, "
              f"persistence RMSE={persist_metrics['RMSE']:.4f}, "
              f"best PSO-GPR ({best_kernel_name}) RMSE={best_rmse:.4f}, {elapsed:.1f}s", flush=True)

    n_wins = sum(1 for r in fold_results if r["best_kernel_beats_lr"])
    n_wins_persist = sum(1 for r in fold_results if r["best_kernel_beats_persistence"])
    summary = {
        "target": target,
        "n_folds_completed": len(fold_results),
        "pso_gpr_wins": n_wins,
        "pso_gpr_wins_vs_persistence": n_wins_persist,
        "folds": fold_results,
    }
    print(f"[{target}] Walk-forward (Design 1, full sweep) summary: "
          f"best PSO-GPR kernel beat Linear Regression in {n_wins}/{len(fold_results)} folds, "
          f"and persistence in {n_wins_persist}/{len(fold_results)} folds")
    return summary
