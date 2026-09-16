"""Permutation Feature Importance for GPR models.

Loads the saved models, calculates the importance of each of the 41 engineered features
on the test set using a PCA pipeline wrapper, and generates figures comparing feature impact.
"""
import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

# Load config and features modules from src
from . import config, splits, features


def _run_permutation_importance(wrapper, X_test, y_test):
    """permutation_importance(..., n_jobs=-1) parallelises across CPU
    cores via joblib, shuffling+predicting for each feature/repeat in a
    worker process. This can fail silently-ish (falling back to serial,
    or raising a pickling error) if the custom PCAGPRWrapper or the GPR
    object it holds doesn't pickle cleanly across processes -- which
    happens more often with GPR than with simpler sklearn estimators.
    Wrapped here with timing (to eyeball whether it actually ran in
    parallel: n_jobs=-1 on a multi-core machine should be noticeably
    faster than n_jobs=1 for the same n_repeats) and a safe fallback to
    n_jobs=1 if the parallel attempt raises.
    """
    t0 = time.time()
    try:
        result = permutation_importance(
            wrapper, X_test, y_test, scoring="neg_root_mean_squared_error",
            n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=-1
        )
        elapsed = time.time() - t0
        print(f"  permutation_importance (n_jobs=-1) completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"  n_jobs=-1 failed ({e!r}); falling back to n_jobs=1 (serial, slower but safe)")
        t0 = time.time()
        result = permutation_importance(
            wrapper, X_test, y_test, scoring="neg_root_mean_squared_error",
            n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=1
        )
        elapsed = time.time() - t0
        print(f"  permutation_importance (n_jobs=1, fallback) completed in {elapsed:.1f}s")
    return result


class PCAGPRWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to allow permutation importance to shuffle raw 41 features
    and propagate the changes through Scaler and PCA into the GPR predictions.
    """
    def __init__(self, scaler, pca, model, transformer):
        self.scaler = scaler
        self.pca = pca
        self.model = model
        self.transformer = transformer

    def fit(self, X, y):
        return self

    def predict(self, X):
        # 1. Scale raw 41 features
        Xs = self.scaler.transform(X)
        # 2. Project to 9 PCA components
        Xsp = self.pca.transform(Xs)
        # 3. Predict in transformed target space
        preds_t = self.model.predict(Xsp)
        # 4. Inverse transform back to original scale
        return self.transformer.inverse_transform(preds_t.reshape(-1, 1)).ravel()


def run_importance():
    print("--- Running Permutation Feature Importance ---")
    
    # Load processed features
    feat_csv = os.path.join("data", "processed", "features.csv")
    if not os.path.exists(feat_csv):
        print(f"Error: Processed features not found at {feat_csv}. Please run pipeline first.")
        return
        
    feat_df = pd.read_csv(feat_csv)
    _, _, test_full = splits.chronological_split(feat_df)
    
    # Run for both targets
    for target in config.TARGETS:
        short = "solar" if target == "solar_irradiance" else "wind"
        model_path = os.path.join("outputs", "models", f"{short}_models.joblib")
        
        if not os.path.exists(model_path):
            print(f"Error: Saved models not found at {model_path}.")
            continue
            
        print(f"\nProcessing target: {target}")
        saved_data = joblib.load(model_path)
        
        scaler = saved_data["scaler"]
        pca = saved_data["pca"]
        daylight_only = saved_data["daylight_only"]

        # Filter test data similar to training
        if daylight_only:
            test = test_full[test_full["is_daylight"]].reset_index(drop=True)
        else:
            test = test_full.reset_index(drop=True)

        X_test = test[features.FEATURE_COLUMNS].to_numpy()
        y_test = test[f"{target}_next"].to_numpy()

        # Run importance for BOTH PSO-tuned kernels, not just RBF -- with
        # the Matern kernel now also PSO-tuned (see gpr_models.py /
        # pipeline.py), a fair comparison needs to check whether the two
        # kernels are actually leaning on the same features or not.
        model_variants = {
            "rbf_pso": (saved_data["gpr_rbf_pso"], saved_data["transformer_pso"]),
        }
        if "gpr_matern_pso" in saved_data:
            model_variants["matern_pso"] = (saved_data["gpr_matern_pso"], saved_data["transformer_pso_matern"])

        for variant_name, (model, transformer) in model_variants.items():
            print(f"Computing permutation importance for {short} ({variant_name})...")
            wrapper = PCAGPRWrapper(scaler, pca, model, transformer)
            result = _run_permutation_importance(wrapper, X_test, y_test)

            sorted_importances_idx = result.importances_mean.argsort()[::-1]
            importance_data = {
                "Feature": [features.FEATURE_COLUMNS[i] for i in sorted_importances_idx],
                "Importance_Mean": result.importances_mean[sorted_importances_idx],
                "Importance_Std": result.importances_std[sorted_importances_idx]
            }
            df_importance = pd.DataFrame(importance_data)

            csv_path = os.path.join("outputs", f"feature_importance_{short}_{variant_name}.csv")
            df_importance.to_csv(csv_path, index=False)
            print(f"Saved feature importance table to {csv_path}")

            print(f"Top 10 features for {short} GPR ({variant_name}):")
            print(df_importance.head(10).to_string(index=False))

            plt.figure(figsize=(10, 6))
            top_n = df_importance.head(15).iloc[::-1]
            plt.barh(top_n["Feature"], top_n["Importance_Mean"], xerr=top_n["Importance_Std"],
                     color="skyblue", edgecolor="gray")
            plt.xlabel("Decrease in Test RMSE Score")
            plt.title(f"Permutation Feature Importance: {target} Forecast ({variant_name})")
            plt.tight_layout()

            plot_path = os.path.join("outputs", "figures", f"{short}_feature_importance_{variant_name}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved feature importance plot to {plot_path}")

if __name__ == "__main__":
    run_importance()
