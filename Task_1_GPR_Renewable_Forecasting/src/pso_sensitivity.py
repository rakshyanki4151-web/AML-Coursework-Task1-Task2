"""PSO Swarm Hyperparameter Sensitivity Analysis.

Runs the PSO hyperparameter optimization with different particle counts
and logs their convergence performance over iterations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import config, splits, features, gpr_models

def run_sensitivity():
    print("--- Running PSO Hyperparameter Sensitivity Analysis ---")
    
    # Load processed features
    feat_csv = os.path.join("data", "processed", "features.csv")
    if not os.path.exists(feat_csv):
        print(f"Error: Processed features not found at {feat_csv}. Please run pipeline first.")
        return
        
    feat_df = pd.read_csv(feat_csv)
    
    # We will use solar_irradiance for the sensitivity test
    target = "solar_irradiance"
    target_next = f"{target}_next"
    
    # Filter daylight hours
    train_full, val_full, _ = splits.chronological_split(feat_df)
    train = train_full[train_full["is_daylight"]].reset_index(drop=True)
    val = val_full[val_full["is_daylight"]].reset_index(drop=True)
    
    X_train_sub = splits.gpr_subsample(train)
    X_train = X_train_sub[features.FEATURE_COLUMNS].to_numpy()
    y_train = X_train_sub[target_next].to_numpy()
    
    X_val = val[features.FEATURE_COLUMNS].to_numpy()
    y_val = val[target_next].to_numpy()
    
    # Scale features
    scaler = gpr_models.build_scaler(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val)

    # gpr_models.fit_pso_gpr() scores particles on the RESIDUAL from a
    # physical baseline (matching pipeline.py's residual-learning setup),
    # not on the raw target. Compute the same solar clear-sky baseline
    # here and convert y_train/y_val into residuals before calling it --
    # otherwise the function signature (which now expects
    # y_val_actual/y_val_baseline too) does not match and either crashes
    # or silently scores against the wrong quantity.
    baseline_train = gpr_models.compute_solar_baseline(X_train_sub)
    baseline_val = gpr_models.compute_solar_baseline(val)
    y_train_residual = y_train - baseline_train.to_numpy()
    y_val_residual = y_val - baseline_val.to_numpy()

    # Swarm sizes to test
    particle_settings = [5, 15, 30]
    iterations = 20 # Bounded to keep it fast
    
    plt.figure(figsize=(10, 6))
    
    results = []
    
    for n_particles in particle_settings:
        print(f"Running PSO optimization with {n_particles} particles...")
        # Fit GPR model using custom particle count. Pass residual targets
        # plus the actual/baseline values fit_pso_gpr needs to reconstruct
        # final predictions (baseline + residual) for its fitness function.
        _, pso_result, _ = gpr_models.fit_pso_gpr(
            Xs_train, y_train_residual, Xs_val, y_val_residual,
            y_val_actual=y_val, y_val_baseline=baseline_val.to_numpy(),
            kernel_kind="rbf", n_particles=n_particles, n_iterations=iterations
        )
        
        # Plot convergence line
        plt.plot(range(1, len(pso_result.history) + 1), pso_result.history, 
                 marker='o', label=f"Swarm Size: {n_particles} Particles")
        
        results.append({
            "Particles": n_particles,
            "Best_Val_RMSE": pso_result.best_score,
            "Final_Pos": pso_result.best_position
        })
        
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation RMSE")
    plt.title("PSO Convergence Sensitivity Analysis (Solar Irradiance)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    fig_dir = os.path.join("outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, "pso_sensitivity_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved sensitivity plot to {plot_path}")
    
    # Print results summary
    print("\nPSO Sensitivity Analysis Summary Table:")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    # Save results table
    csv_path = os.path.join("outputs", "pso_sensitivity_results.csv")
    df_res.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_sensitivity()
