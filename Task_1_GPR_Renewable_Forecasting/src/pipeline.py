"""End-to-end pipeline: data -> features -> splits -> baselines -> GPR
(default + PSO-tuned) -> GP classification -> metrics + figures + saved
models. Run with ``python run.py`` from the project root.
"""
import json
import time

try:
    import resource
    def _peak_memory_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    def _peak_memory_mb():
        return 0.0

import joblib
import numpy as np
import pandas as pd

from . import config, data_prep, features, splits, baselines, gpr_models, classification, evaluate, plotting, walk_forward, walk_forward_expanding

SHORT_NAME = {"solar_irradiance": "solar", "wind_speed": "wind"}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def run_target(feat_df, target):
    short = SHORT_NAME[target]
    target_next = f"{target}_next"
    daylight_only = target in config.DAYLIGHT_FILTER_TARGETS
    results = {"target": target, "daylight_only": daylight_only, "models": {}}
    _t_target_start = time.time()
    _mem_target_start = _peak_memory_mb()

    train_full, val_full, test_full = splits.chronological_split(feat_df)
    if daylight_only:
        train = train_full[train_full["is_daylight"]].reset_index(drop=True)
        val = val_full[val_full["is_daylight"]].reset_index(drop=True)
        test = test_full[test_full["is_daylight"]].reset_index(drop=True)
        _log(f"[{short}] Daylight filter applied: {len(train)}/{len(train_full)} train, "
             f"{len(val)}/{len(val_full)} val, {len(test)}/{len(test_full)} test hours kept")
    else:
        train, val, test = train_full, val_full, test_full

    X_train_full = train[features.FEATURE_COLUMNS].to_numpy()
    y_train_full = train[target_next].to_numpy()
    X_val = val[features.FEATURE_COLUMNS].to_numpy()
    y_val = val[target_next].to_numpy()
    X_test = test[features.FEATURE_COLUMNS].to_numpy()
    y_test = test[target_next].to_numpy()

    # --- Baselines --------------------------------------------------
    _log(f"[{short}] Persistence baseline")
    pred_persist = baselines.persistence_predict(test, target)
    results["models"]["persistence"] = evaluate.regression_metrics(y_test, pred_persist)

    _log(f"[{short}] Linear Regression baseline (full training window)")
    lr = baselines.fit_linear_regression(X_train_full, y_train_full)
    pred_lr = lr.predict(X_test)
    results["models"]["linear_regression"] = evaluate.regression_metrics(y_test, pred_lr)

    _log(f"[{short}] SARIMAX baseline (walk-forward, 24h-seasonal, continuous series)")
    eval_today_full = pd.concat([val_full[target], test_full[target]], ignore_index=True)
    sarimax_preds_full = baselines.rolling_forecast_sarimax(train_full[target], eval_today_full)
    sarimax_preds_test_full = sarimax_preds_full[len(val_full):]
    if daylight_only:
        pred_arima = sarimax_preds_test_full[test_full["is_daylight"].to_numpy()]
    else:
        pred_arima = sarimax_preds_test_full
    results["models"]["sarimax"] = evaluate.regression_metrics(y_test, pred_arima)

    # --- GPR family (subsampled training window; see config for why) --
    from sklearn.decomposition import PCA

    train_sub = splits.gpr_subsample(train)
    X_train_sub = train_sub[features.FEATURE_COLUMNS].to_numpy()
    y_train_sub = train_sub[target_next].to_numpy()
    scaler = gpr_models.build_scaler(X_train_sub)
    Xs_train_sub = scaler.transform(X_train_sub)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test)

    # Dimensionality reduction, controlled by config.USE_PCA (default False
    # -- see the long explanation there for why PCA was switched off).
    # When disabled, a pass-through reducer keeps the downstream API and the
    # saved model bundle identical, so nothing else has to change.
    if config.USE_PCA:
        pca = PCA(n_components=config.PCA_VARIANCE_RETAINED, random_state=config.RANDOM_SEED)
    else:
        pca = gpr_models.IdentityReducer()
    Xsp_train_sub = pca.fit_transform(Xs_train_sub)
    Xsp_val = pca.transform(Xs_val)
    Xsp_test = pca.transform(Xs_test)

    results["use_pca"] = bool(config.USE_PCA)
    results["pca_n_components"] = int(pca.n_components_)
    if config.USE_PCA:
        _log(f"[{short}] PCA reduced features from {Xs_train_sub.shape[1]} to {Xsp_train_sub.shape[1]} components")
        results["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        results["pca_cumulative_variance"] = float(np.sum(pca.explained_variance_ratio_))
        plotting.plot_pca_scree(
            pca.explained_variance_ratio_,
            title=f"{target}: PCA cumulative explained variance",
            filename=f"{short}_pca_scree.png",
        )
    else:
        _log(f"[{short}] PCA disabled (config.USE_PCA=False); "
             f"using all {Xsp_train_sub.shape[1]} scaled features directly")

    # Compute physical baselines for residual learning
    if target == "solar_irradiance":
        baseline_train_sub = gpr_models.compute_solar_baseline(train_sub)
        baseline_val = gpr_models.compute_solar_baseline(val)
        baseline_test = gpr_models.compute_solar_baseline(test)
    else:
        baseline_train_sub = gpr_models.compute_wind_baseline(train_sub)
        baseline_val = gpr_models.compute_wind_baseline(val)
        baseline_test = gpr_models.compute_wind_baseline(test)

    y_train_sub_residual = (y_train_sub - baseline_train_sub).to_numpy()
    y_val_residual = (y_val - baseline_val).to_numpy()

    _log(f"[{short}] Default GPR (RBF kernel, sklearn L-BFGS optimiser)")
    gpr_rbf, transformer_rbf = gpr_models.fit_default_gpr(Xsp_train_sub, y_train_sub_residual, "rbf")
    mean_t, std_t = gpr_models.predict_with_uncertainty(gpr_rbf, Xsp_test)
    # Inverse transform GPR residual predictions and add baseline back
    mean_residual = transformer_rbf.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    mean_rbf = baseline_test + mean_residual
    lower_residual = transformer_rbf.inverse_transform((mean_t - std_t).reshape(-1, 1)).ravel()
    upper_residual = transformer_rbf.inverse_transform((mean_t + std_t).reshape(-1, 1)).ravel()
    lower_rbf = baseline_test + lower_residual
    upper_rbf = baseline_test + upper_residual
    std_rbf = np.clip((upper_rbf - lower_rbf) / 2.0, a_min=1e-6, a_max=None)
    
    m = evaluate.regression_metrics(y_test, mean_rbf)
    m["NLPD"] = evaluate.nlpd(y_test, mean_rbf, std_rbf)
    m["coverage_95"] = evaluate.coverage_at_z(y_test, mean_rbf, std_rbf, z=1.96)
    results["models"]["gpr_rbf_default"] = m

    _log(f"[{short}] Default GPR (Matern kernel, sklearn L-BFGS optimiser)")
    gpr_matern, transformer_mat = gpr_models.fit_default_gpr(Xsp_train_sub, y_train_sub_residual, "matern")
    mean_t, std_t = gpr_models.predict_with_uncertainty(gpr_matern, Xsp_test)
    # Inverse transform GPR residual predictions and add baseline back
    mean_residual = transformer_mat.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    mean_mat = baseline_test + mean_residual
    lower_residual = transformer_mat.inverse_transform((mean_t - std_t).reshape(-1, 1)).ravel()
    upper_residual = transformer_mat.inverse_transform((mean_t + std_t).reshape(-1, 1)).ravel()
    lower_mat = baseline_test + lower_residual
    upper_mat = baseline_test + upper_residual
    std_mat = np.clip((upper_mat - lower_mat) / 2.0, a_min=1e-6, a_max=None)
    
    m = evaluate.regression_metrics(y_test, mean_mat)
    m["NLPD"] = evaluate.nlpd(y_test, mean_mat, std_mat)
    m["coverage_95"] = evaluate.coverage_at_z(y_test, mean_mat, std_mat, z=1.96)
    results["models"]["gpr_matern_default"] = m

    _log(f"[{short}] PSO-tuned GPR (RBF kernel, {config.PSO_N_PARTICLES} particles x "
         f"{config.PSO_N_ITERATIONS} iterations x {len(config.PSO_SEEDS)} seeds)")
    gpr_pso, pso_result, transformer_pso, pso_summary_rbf = gpr_models.fit_pso_gpr_multiseed(
        Xsp_train_sub, y_train_sub_residual, Xsp_val, y_val_residual, y_val, baseline_val, "rbf"
    )
    _log(f"[{short}] RBF PSO across {len(config.PSO_SEEDS)} seeds: "
         f"val RMSE mean={pso_summary_rbf['mean_val_rmse']:.4f} std={pso_summary_rbf['std_val_rmse']:.4f} "
         f"(best seed={pso_summary_rbf['best_seed']})")
    mean_t, std_t = gpr_models.predict_with_uncertainty(gpr_pso, Xsp_test)
    # Inverse transform GPR residual predictions and add baseline back
    mean_residual = transformer_pso.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    mean_pso = baseline_test + mean_residual
    lower_residual = transformer_pso.inverse_transform((mean_t - std_t).reshape(-1, 1)).ravel()
    upper_residual = transformer_pso.inverse_transform((mean_t + std_t).reshape(-1, 1)).ravel()
    lower_pso = baseline_test + lower_residual
    upper_pso = baseline_test + upper_residual
    std_pso = np.clip((upper_pso - lower_pso) / 2.0, a_min=1e-6, a_max=None)
    
    m = evaluate.regression_metrics(y_test, mean_pso)
    m["NLPD"] = evaluate.nlpd(y_test, mean_pso, std_pso)
    m["coverage_95"] = evaluate.coverage_at_z(y_test, mean_pso, std_pso, z=1.96)
    m["best_length_scale"] = float(pso_result.best_position[0])
    m["best_signal_variance"] = float(pso_result.best_position[1])
    m["best_noise"] = float(pso_result.best_position[2])
    m["best_position"] = [float(p) for p in pso_result.best_position]
    results["models"]["gpr_rbf_pso"] = m
    results["pso_convergence"] = pso_result.history
    results["pso_multiseed_rbf"] = pso_summary_rbf

    # PSO-tuned Matern kernel, mirroring the RBF PSO run above. Added so
    # the RBF-vs-Matern comparison is made under PSO tuning on both sides,
    # not just "PSO-RBF vs default-Matern" (which would be an unfair
    # comparison since the default Matern above still benefits from
    # sklearn's own L-BFGS optimiser).
    _log(f"[{short}] PSO-tuned GPR (Matern kernel, {config.PSO_N_PARTICLES} particles x "
         f"{config.PSO_N_ITERATIONS} iterations x {len(config.PSO_SEEDS)} seeds)")
    gpr_pso_mat, pso_result_mat, transformer_pso_mat, pso_summary_matern = gpr_models.fit_pso_gpr_multiseed(
        Xsp_train_sub, y_train_sub_residual, Xsp_val, y_val_residual, y_val, baseline_val, "matern"
    )
    _log(f"[{short}] Matern PSO across {len(config.PSO_SEEDS)} seeds: "
         f"val RMSE mean={pso_summary_matern['mean_val_rmse']:.4f} std={pso_summary_matern['std_val_rmse']:.4f} "
         f"(best seed={pso_summary_matern['best_seed']})")
    mean_t, std_t = gpr_models.predict_with_uncertainty(gpr_pso_mat, Xsp_test)
    mean_residual = transformer_pso_mat.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    mean_pso_mat = baseline_test + mean_residual
    lower_residual = transformer_pso_mat.inverse_transform((mean_t - std_t).reshape(-1, 1)).ravel()
    upper_residual = transformer_pso_mat.inverse_transform((mean_t + std_t).reshape(-1, 1)).ravel()
    lower_pso_mat = baseline_test + lower_residual
    upper_pso_mat = baseline_test + upper_residual
    std_pso_mat = np.clip((upper_pso_mat - lower_pso_mat) / 2.0, a_min=1e-6, a_max=None)

    m = evaluate.regression_metrics(y_test, mean_pso_mat)
    m["NLPD"] = evaluate.nlpd(y_test, mean_pso_mat, std_pso_mat)
    m["coverage_95"] = evaluate.coverage_at_z(y_test, mean_pso_mat, std_pso_mat, z=1.96)
    m["best_length_scale"] = float(pso_result_mat.best_position[0])
    m["best_signal_variance"] = float(pso_result_mat.best_position[1])
    m["best_noise"] = float(pso_result_mat.best_position[2])
    m["best_position"] = [float(p) for p in pso_result_mat.best_position]
    results["models"]["gpr_matern_pso"] = m
    results["pso_convergence_matern"] = pso_result_mat.history
    results["pso_multiseed_matern"] = pso_summary_matern

    # PSO-tuned periodic kernel (ExpSineSquared x RBF envelope). Previously
    # this kernel was implemented in gpr_models._make_kernel but never
    # actually used anywhere -- a periodic kernel is a natural fit for the
    # 24h diurnal solar/wind cycle, so it's included here as a third
    # PSO-tuned option alongside RBF and Matern.
    _log(f"[{short}] PSO-tuned GPR (Periodic kernel, {config.PSO_N_PARTICLES} particles x "
         f"{config.PSO_N_ITERATIONS} iterations x {len(config.PSO_SEEDS)} seeds)")
    gpr_pso_per, pso_result_per, transformer_pso_per, pso_summary_periodic = gpr_models.fit_pso_gpr_multiseed(
        Xsp_train_sub, y_train_sub_residual, Xsp_val, y_val_residual, y_val, baseline_val, "periodic"
    )
    _log(f"[{short}] Periodic PSO across {len(config.PSO_SEEDS)} seeds: "
         f"val RMSE mean={pso_summary_periodic['mean_val_rmse']:.4f} std={pso_summary_periodic['std_val_rmse']:.4f} "
         f"(best seed={pso_summary_periodic['best_seed']})")
    mean_t, std_t = gpr_models.predict_with_uncertainty(gpr_pso_per, Xsp_test)
    mean_residual = transformer_pso_per.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    mean_pso_per = baseline_test + mean_residual
    lower_residual = transformer_pso_per.inverse_transform((mean_t - std_t).reshape(-1, 1)).ravel()
    upper_residual = transformer_pso_per.inverse_transform((mean_t + std_t).reshape(-1, 1)).ravel()
    lower_pso_per = baseline_test + lower_residual
    upper_pso_per = baseline_test + upper_residual
    std_pso_per = np.clip((upper_pso_per - lower_pso_per) / 2.0, a_min=1e-6, a_max=None)

    m = evaluate.regression_metrics(y_test, mean_pso_per)
    m["NLPD"] = evaluate.nlpd(y_test, mean_pso_per, std_pso_per)
    m["coverage_95"] = evaluate.coverage_at_z(y_test, mean_pso_per, std_pso_per, z=1.96)
    m["best_length_scale"] = float(pso_result_per.best_position[0])
    m["best_signal_variance"] = float(pso_result_per.best_position[1])
    m["best_noise"] = float(pso_result_per.best_position[2])
    m["best_position"] = [float(p) for p in pso_result_per.best_position]
    results["models"]["gpr_periodic_pso"] = m
    results["pso_convergence_periodic"] = pso_result_per.history
    results["pso_multiseed_periodic"] = pso_summary_periodic

    # --- Significance test: is PSO-RBF actually better than the best
    # baseline, or could the RMSE gap be noise? Paired on the same test
    # hours since every model is scored on an identical y_test.
    best_baseline_name = min(
        ("persistence", "linear_regression", "sarimax"),
        key=lambda n: results["models"][n]["RMSE"],
    )
    best_baseline_pred = {"persistence": pred_persist, "linear_regression": pred_lr, "sarimax": pred_arima}[best_baseline_name]
    results["significance_vs_best_baseline"] = evaluate.paired_error_test(y_test, mean_pso, best_baseline_pred)
    results["significance_vs_best_baseline"]["compared_against"] = best_baseline_name
    results["significance_pso_rbf_vs_default_rbf"] = evaluate.paired_error_test(y_test, mean_pso, mean_rbf)
    results["significance_pso_rbf_vs_pso_matern"] = evaluate.paired_error_test(y_test, mean_pso, mean_pso_mat)
    results["significance_pso_rbf_vs_pso_periodic"] = evaluate.paired_error_test(y_test, mean_pso, mean_pso_per)

    # Diebold-Mariano: the same four comparisons as above, but using a
    # test built for autocorrelated time-series forecast errors instead
    # of assuming independence (which the paired t-test/Wilcoxon above
    # technically do, and hourly forecast errors technically violate).
    # Reported alongside, not instead of, the paired tests -- three
    # converging significance tests is stronger evidence than any one.
    results["dm_vs_best_baseline"] = evaluate.diebold_mariano_test(y_test, mean_pso, best_baseline_pred)
    results["dm_vs_best_baseline"]["compared_against"] = best_baseline_name
    results["dm_pso_rbf_vs_default_rbf"] = evaluate.diebold_mariano_test(y_test, mean_pso, mean_rbf)
    results["dm_pso_rbf_vs_pso_matern"] = evaluate.diebold_mariano_test(y_test, mean_pso, mean_pso_mat)
    results["dm_pso_rbf_vs_pso_periodic"] = evaluate.diebold_mariano_test(y_test, mean_pso, mean_pso_per)

    # --- Classification (Low / Medium / High) -----------------------
    _log(f"[{short}] Gaussian Process Classification on thresholded output")
    q1, q2 = classification.compute_thresholds(y_train_full)
    y_class_train_sub = classification.to_classes(y_train_sub, q1, q2)
    y_class_val = classification.to_classes(y_val, q1, q2)
    y_class_test = classification.to_classes(y_test, q1, q2)

    # Untuned GPC (fixed RBF, sklearn's own restart optimiser) -- kept as
    # the "default" comparison point.
    gpc = classification.fit_gpc(Xsp_train_sub, y_class_train_sub)
    pred_class_val = gpc.predict(Xsp_val)
    pred_class_test = gpc.predict(Xsp_test)
    results["classification_val"] = evaluate.classification_metrics(y_class_val, pred_class_val)
    results["classification"] = evaluate.classification_metrics(y_class_test, pred_class_test)
    results["classification"]["thresholds"] = {"q1": float(q1), "q2": float(q2)}

    # PSO-tuned GPC -- previously classification was the one stage PSO
    # never touched, despite PSO being the project's headline method.
    # Tuned on validation macro-F1, refit on train+val, evaluated on test,
    # reported alongside the untuned GPC above rather than replacing it.
    _log(f"[{short}] PSO-tuned Gaussian Process Classification ({config.GPC_PSO_N_PARTICLES} particles x "
         f"{config.GPC_PSO_N_ITERATIONS} iterations)")
    gpc_pso, gpc_pso_result = classification.fit_gpc_pso(Xsp_train_sub, y_class_train_sub, Xsp_val, y_class_val)
    pred_class_test_pso = gpc_pso.predict(Xsp_test)
    results["classification_pso"] = evaluate.classification_metrics(y_class_test, pred_class_test_pso)
    results["classification_pso"]["thresholds"] = {"q1": float(q1), "q2": float(q2)}
    results["classification_pso"]["best_length_scale"] = float(gpc_pso_result.best_position[0])
    results["classification_pso"]["best_constant_value"] = float(gpc_pso_result.best_position[1])
    results["classification_pso_convergence"] = gpc_pso_result.history

    # --- Figures -----------------------------------------------------
    plotting.plot_forecast_with_uncertainty(
        test["date"], y_test, mean_pso, std_pso,
        title=f"{target}: PSO-tuned GPR {config.FORECAST_HORIZON_HOURS}h-ahead forecast vs actual (test set)",
        filename=f"{short}_forecast_uncertainty.png",
        n_points=240,
    )
    plotting.plot_reliability_diagram(
        y_test, mean_pso, std_pso,
        title=f"{target}: PSO-GPR (RBF) calibration reliability diagram",
        filename=f"{short}_reliability_diagram.png",
    )
    plotting.plot_pso_convergence(
        pso_result.history, title=f"{target}: PSO convergence (validation RMSE)",
        filename=f"{short}_pso_convergence.png",
    )
    model_order = ["persistence", "linear_regression", "sarimax", "gpr_rbf_default", "gpr_matern_default",
                   "gpr_rbf_pso", "gpr_matern_pso", "gpr_periodic_pso"]
    plotting.plot_model_comparison(
        [n.replace("_", " ") for n in model_order],
        [results["models"][n]["RMSE"] for n in model_order],
        title=f"{target}: test RMSE across models",
        filename=f"{short}_model_comparison.png",
    )
    plotting.plot_confusion_matrix(
        results["classification"]["confusion_matrix"], config.CLASS_LABELS,
        title=f"{target}: GP classification confusion matrix (untuned)",
        filename=f"{short}_confusion_matrix.png",
    )
    plotting.plot_confusion_matrix(
        results["classification_pso"]["confusion_matrix"], config.CLASS_LABELS,
        title=f"{target}: GP classification confusion matrix (PSO-tuned)",
        filename=f"{short}_confusion_matrix_pso.png",
    )
    plotting.plot_pso_convergence(
        pso_result_mat.history, title=f"{target}: PSO convergence, Matern kernel (validation RMSE)",
        filename=f"{short}_pso_convergence_matern.png",
    )
    # Default (sklearn L-BFGS) vs PSO-tuned hyperparameters, side by side --
    # this is the direct evidence for objective 2 ("automate kernel tuning
    # with PSO"), previously only saved as numbers in metrics.json and
    # never actually plotted or tabulated against the default values.
    default_rbf_length_scale = float(gpr_rbf.kernel_.k1.k2.length_scale)
    default_matern_length_scale = float(gpr_matern.kernel_.k1.k2.length_scale)
    plotting.plot_hyperparameter_comparison(
        {
            "RBF length-scale": (default_rbf_length_scale, results["models"]["gpr_rbf_pso"]["best_length_scale"]),
            "Matern length-scale": (default_matern_length_scale, results["models"]["gpr_matern_pso"]["best_length_scale"]),
        },
        title=f"{target}: default (sklearn) vs PSO-tuned kernel length-scale",
        filename=f"{short}_hyperparam_comparison.png",
    )

    # --- Persist fitted models for the Streamlit app / reproducibility --
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler, "pca": pca, "linear_regression": lr,
            "gpr_rbf_default": gpr_rbf, "transformer_rbf": transformer_rbf,
            "gpr_matern_default": gpr_matern, "transformer_mat": transformer_mat,
            "gpr_rbf_pso": gpr_pso, "transformer_pso": transformer_pso,
            "gpr_matern_pso": gpr_pso_mat, "transformer_pso_matern": transformer_pso_mat,
            "gpr_periodic_pso": gpr_pso_per, "transformer_pso_periodic": transformer_pso_per,
            "gpc": gpc, "gpc_pso": gpc_pso, "class_thresholds": (q1, q2), "feature_columns": features.FEATURE_COLUMNS,
            "daylight_only": daylight_only,
        },
        config.MODELS_DIR / f"{short}_models.joblib",
    )

    # --- Compute cost bookkeeping: wall-clock time and peak RSS memory for
    # this target's full run (baselines + all PSO/default GPR + GPC).
    results["compute_stats"] = {
        "elapsed_seconds": round(time.time() - _t_target_start, 1),
        "peak_memory_mb_delta": round(_peak_memory_mb() - _mem_target_start, 1),
        "peak_memory_mb_total": round(_peak_memory_mb(), 1),
    }
    _log(f"[{short}] Done in {results['compute_stats']['elapsed_seconds']}s, "
         f"peak memory so far {results['compute_stats']['peak_memory_mb_total']} MB")

    return results


def main():
    _t_main_start = time.time()
    _log("Loading and cleaning NASA POWER hourly data")
    raw_df, n_missing = data_prep.load_clean()
    _log(f"{len(raw_df)} rows loaded; {n_missing} missing cells interpolated")

    _log("Engineering features (calendar, hour-of-day, solar zenith, lags, rolling means, 24h-ahead targets)")
    feat_df = features.build_feature_frame(raw_df)
    _log(f"{len(feat_df)} rows remain after feature engineering; "
         f"{len(features.FEATURE_COLUMNS)} input features per target; "
         f"{int(feat_df['is_daylight'].sum())} daylight hours")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(config.PROCESSED_DIR / "features.csv", index=False)

    all_results = {}
    for target in config.TARGETS:
        all_results[target] = run_target(feat_df, target)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for target, res in all_results.items():
        for model_name, m in res["models"].items():
            row = {"target": target, "model": model_name}
            row.update({k: v for k, v in m.items() if not isinstance(v, (list, dict))})
            rows.append(row)
    pd.DataFrame(rows).to_csv(config.OUTPUTS_DIR / "metrics_table.csv", index=False)

    # --- Walk-forward robustness check, Design 1 (rolling-origin, full PSO sweep per fold)
    _log("Running walk-forward Design 1 (rolling-origin, full PSO sweep per fold)...")
    walk_forward_results = {}
    for target in config.TARGETS:
        walk_forward_results[target] = walk_forward.run_walk_forward(feat_df, target, n_folds=3)
    with open(config.OUTPUTS_DIR / "walk_forward_design1_results.json", "w") as f:
        json.dump(walk_forward_results, f, indent=2)
    _log(f"Walk-forward Design 1 results written to "
         f"{config.OUTPUTS_DIR / 'walk_forward_design1_results.json'}")

    # --- Walk-forward robustness check, Design 2 (expanding-window, frozen kernel)
    _log("Running walk-forward Design 2 (expanding-window, frozen kernel)...")
    walk_forward_expanding_results = {}
    for target in config.TARGETS:
        kernel_positions = {
            "rbf": tuple(all_results[target]["models"]["gpr_rbf_pso"]["best_position"]),
            "matern": tuple(all_results[target]["models"]["gpr_matern_pso"]["best_position"]),
            "periodic": tuple(all_results[target]["models"]["gpr_periodic_pso"]["best_position"]),
        }
        walk_forward_expanding_results[target] = walk_forward_expanding.run_expanding_window(
            feat_df, target, kernel_positions, n_folds=3
        )
    with open(config.OUTPUTS_DIR / "walk_forward_design2_results.json", "w") as f:
        json.dump(walk_forward_expanding_results, f, indent=2)
    _log(f"Walk-forward Design 2 results written to "
         f"{config.OUTPUTS_DIR / 'walk_forward_design2_results.json'}")

    total_elapsed = time.time() - _t_main_start
    _log(f"Done. Total pipeline runtime: {total_elapsed / 60:.1f} min. "
         f"Metrics written to {config.OUTPUTS_DIR / 'metrics.json'} and "
         f"{config.OUTPUTS_DIR / 'metrics_table.csv'}. Figures in {config.FIGURES_DIR}.")
    return all_results


if __name__ == "__main__":
    main()
